"""
training_pairs.py
Deterministic training pairs from the existing blocker and pair features.

Flow:
    stratified Source-1 sample
    -> blocking.py candidate generation
    -> features.featurize_candidate_pairs
    -> features.attach_true_match_label
    -> hard negatives plus ordinary negatives

Blocking passes, document-frequency caps, and pair features are not
reimplemented. For the sampled queries, exact / token / postal / number
postings are built with the same `_accumulate` cap rule as a full index:
a key the queries will look up gets the same posting list it would have
in the full country index. Character TF-IDF uses the existing search on a
deterministic name sample (plus every ground-truth match of the sampled
Source-1 rows) so the name list for a whole country does not have to sit
in RAM. Set `tfidf_index_cap=None` to search every target name.

`negative_role` is sampling metadata, not a model feature.
Country values are whatever the file contains. Nothing here is limited
to a fixed country list.
"""
from __future__ import annotations

import gc
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .blocking import (
    BlockingConfig,
    _Index,
    _accumulate,
    _add,
    _char_tfidf_search,
    _freeze,
    _is_target_id,
    _iter_raw_tsv,
    _load_truth_for,
    _mask_to_rules,
    _query_record,
    ground_truth_map,
    make_record,
)
from .features import FEATURE_COLUMNS, attach_true_match_label, featurize_candidate_pairs
from .normalization import build_name_views

SEED = 42
BUCKET_QUOTA = {"0": 60, "1": 90, "2plus": 60}
NONLATIN_FLOOR = {"0": 8, "1": 20, "2plus": 15}
HARD_SCORE_MIN = 0.40
MAX_HARD_PER_S1 = 12
MAX_ORDINARY_PER_S1 = 8
MAX_CANDIDATES_PER_S1 = 2_000
TFIDF_INDEX_CAP = 200_000
TFIDF_BIT = 1 << 8
# Exact name, transliteration, premises number, postal, character TF-IDF.
_STRONG_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 6) | (1 << 7) | TFIDF_BIT

_SCORE_WEIGHTS = (
    ("name_ratio_core", 0.28, False),
    ("name_ratio_translit", 0.18, False),
    ("name_token_jaccard", 0.14, False),
    ("addr_ratio_basic", 0.12, False),
    ("addr_token_jaccard", 0.10, False),
    ("addr_number_agree", 0.06, False),
    ("addr_postal_agree", 0.06, False),
    ("n_blocking_rules", 0.06, True),
)

SAVED_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "is_true_match",
    *FEATURE_COLUMNS,
    "negative_role",
    "country",
)


def _log(message: str) -> None:
    print(message, flush=True)


def _stable_offset(seed: int, text: str) -> int:
    total = seed % 1_000_003
    for char in text:
        total = (total * 131 + ord(char)) % 1_000_003
    return total


def _bucket(n_matches: int) -> str:
    if n_matches <= 0:
        return "0"
    if n_matches == 1:
        return "1"
    return "2plus"


def _has_non_ascii_letter(text: str) -> bool:
    for char in text:
        if ord(char) > 127 and char.isalpha():
            return True
    return False


def _parse_matches(cell: str) -> list:
    if not cell:
        return []
    return [part for part in cell.split(",") if part]


def _reservoir_add(pool: list, item: str, seen: int, limit: int, rng: random.Random) -> None:
    if len(pool) < limit:
        pool.append(item)
        return
    slot = rng.randrange(seen)
    if slot < limit:
        pool[slot] = item


def _even_indices(n: int, k: int) -> list:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    chosen = []
    seen = set()
    for i in range(k):
        index = int(round(i * (n - 1) / (k - 1)))
        if index not in seen:
            seen.add(index)
            chosen.append(index)
    return chosen


def _count_targets(paths) -> dict:
    """Row counts, country counts, and ids whose business name has a non-ASCII letter."""
    counts = Counter()
    countries = Counter()
    nonlatin = set()
    for path in paths:
        n = 0
        file_countries = Counter()
        with Path(path).open(encoding="utf-8") as handle:
            header = handle.readline()
            if "entity_id" not in header:
                raise ValueError(f"unexpected header in {path}: {header!r}")
            for line in handle:
                entity_id, _, rest = line.rstrip("\n").partition("\t")
                name, _, rest = rest.partition("\t")
                _, _, country = rest.partition("\t")
                country = country.strip()
                n += 1
                file_countries[country] += 1
                if _has_non_ascii_letter(name):
                    nonlatin.add(entity_id)
                if n % 1_000_000 == 0:
                    _log(f"  scanned {n:,} {Path(path).name}")
        counts[Path(path).name] = n
        countries.update(file_countries)
        _log(f"  {Path(path).name}: {n:,} rows, {len(file_countries)} countries")
    return {"rows": dict(counts), "countries": dict(countries), "nonlatin_ids": nonlatin}


def select_training_source1(
    source1_path,
    ground_truth_path,
    nonlatin_ids: set,
    *,
    bucket_quota: dict | None = None,
    nonlatin_floor: dict | None = None,
    seed: int = SEED,
) -> dict:
    """Stratified Source-1 ids. Quota is applied per country found in the file."""
    quota = dict(bucket_quota or BUCKET_QUOTA)
    floor = dict(nonlatin_floor or NONLATIN_FLOOR)
    rng = random.Random(seed)

    country_of = {}
    country_counts = Counter()
    s1_nonlatin = set()
    with Path(source1_path).open(encoding="utf-8") as handle:
        header = handle.readline()
        if "entity_id" not in header:
            raise ValueError(f"unexpected header in {source1_path}: {header!r}")
        for line in handle:
            entity_id, _, rest = line.rstrip("\n").partition("\t")
            name, _, rest = rest.partition("\t")
            _, _, country = rest.partition("\t")
            country = country.strip()
            country_of[entity_id] = country
            country_counts[country] += 1
            if _has_non_ascii_letter(name):
                s1_nonlatin.add(entity_id)
    _log(f"  source 1: {len(country_of):,} rows, countries={dict(country_counts)}")

    pools = defaultdict(list)
    seen = Counter()
    match_hist = Counter()
    nonlatin_s1 = 0
    missing_country = 0
    with Path(ground_truth_path).open(encoding="utf-8") as handle:
        header = handle.readline()
        if "source1_entity_id" not in header:
            raise ValueError(f"unexpected ground-truth header in {ground_truth_path}: {header!r}")
        for line in handle:
            source_id, _, cell = line.rstrip("\n").partition("\t")
            country = country_of.get(source_id)
            if country is None:
                missing_country += 1
                continue
            matches = _parse_matches(cell)
            n_matches = len(matches)
            match_hist[n_matches if n_matches < 10 else "10+"] += 1
            bucket = _bucket(n_matches)
            nonlatin = source_id in s1_nonlatin or any(match in nonlatin_ids for match in matches)
            if nonlatin:
                nonlatin_s1 += 1
            key = (country, bucket, nonlatin)
            seen[key] += 1
            _reservoir_add(pools[key], source_id, seen[key], quota[bucket], rng)

    selected = []
    stratum_counts = {}
    for country in sorted(country_counts):
        for bucket in ("0", "1", "2plus"):
            nonlatin_pool = sorted(pools.get((country, bucket, True), []))
            latin_pool = sorted(pools.get((country, bucket, False), []))
            take_nonlatin = nonlatin_pool[: min(floor[bucket], len(nonlatin_pool))]
            latin_ids = set(latin_pool)
            chosen = list(take_nonlatin)
            need = quota[bucket] - len(chosen)
            if need > 0:
                chosen.extend(latin_pool[:need])
                need = quota[bucket] - len(chosen)
            if need > 0:
                already = set(chosen)
                chosen.extend([entity_id for entity_id in nonlatin_pool if entity_id not in already][:need])
            selected.extend(chosen)
            stratum_counts[f"{country}|match_{bucket}"] = {
                "selected": len(chosen),
                "nonlatin_in_selected": sum(1 for entity_id in chosen if entity_id not in latin_ids),
                "eligible_nonlatin": int(seen[(country, bucket, True)]),
                "eligible_other": int(seen[(country, bucket, False)]),
            }
    selected = sorted(set(selected))
    _log(f"  selected {len(selected):,} source-1 ids across {len(country_counts)} countries")
    return {
        "source1_ids": selected,
        "corpus": {
            "n_s1": len(country_of),
            "s1_countries": dict(country_counts),
            "match_hist": {str(key): int(value) for key, value in sorted(match_hist.items(), key=lambda item: str(item[0]))},
            "s1_with_nonlatin_name_or_match": nonlatin_s1,
            "gt_rows_missing_from_s1": missing_country,
        },
        "strata": stratum_counts,
    }


def _watch_maps(queries: list) -> dict:
    watches = {}
    for rec in queries:
        watch = watches.setdefault(rec.country, {
            "exact_name": set(),
            "exact_core": set(),
            "exact_sorted": set(),
            "translit": set(),
            "name_token": set(),
            "addr_token": set(),
            "number": set(),
            "postal": set(),
        })
        if rec.basic:
            watch["exact_name"].add(rec.basic)
        if rec.core:
            watch["exact_core"].add(rec.core)
        if rec.sorted_name:
            watch["exact_sorted"].add(rec.sorted_name)
        if rec.trans:
            watch["translit"].add(rec.trans)
        watch["name_token"].update(rec.name_toks)
        watch["addr_token"].update(rec.addr_toks)
        if rec.number_key is not None:
            watch["number"].add(rec.number_key)
        watch["postal"].update(rec.postals)
    return watches


def _new_buckets() -> dict:
    return {
        "exact_name": {},
        "exact_core": {},
        "exact_sorted": {},
        "translit_name": {},
        "name_token": {},
        "address_token": {},
        "address_number": {},
        "address_postal": {},
    }


def _index_from_buckets(buckets: dict, overflow: dict) -> _Index:
    index = _Index()
    index.exact_name = _freeze(buckets["exact_name"])
    index.exact_core = _freeze(buckets["exact_core"])
    index.exact_sorted = _freeze(buckets["exact_sorted"])
    index.translit_name = _freeze(buckets["translit_name"])
    index.name_token = _freeze(buckets["name_token"])
    index.address_token = _freeze(buckets["address_token"])
    index.address_number = _freeze(buckets["address_number"])
    index.address_postal = _freeze(buckets["address_postal"])
    index.overflow = {name: counter[0] for name, counter in overflow.items()}
    return index


def _accumulate_target(buckets: dict, overflow: dict, rec, watch: dict, cfg: BlockingConfig) -> None:
    if rec.basic in watch["exact_name"]:
        _accumulate(buckets["exact_name"], rec.basic, rec.eid, cfg.exact_max_fanout, overflow["exact_name"])
    if rec.core in watch["exact_core"]:
        _accumulate(buckets["exact_core"], rec.core, rec.eid, cfg.exact_max_fanout, overflow["exact_core"])
    if rec.sorted_name in watch["exact_sorted"]:
        _accumulate(buckets["exact_sorted"], rec.sorted_name, rec.eid, cfg.exact_max_fanout, overflow["exact_sorted"])
    if rec.trans in watch["translit"]:
        _accumulate(buckets["translit_name"], rec.trans, rec.eid, cfg.exact_max_fanout, overflow["translit_name"])
    name_watch = watch["name_token"]
    for token in rec.name_toks:
        if token in name_watch:
            _accumulate(buckets["name_token"], token, rec.eid, cfg.name_token_max_df, overflow["name_token"])
    addr_watch = watch["addr_token"]
    for token in rec.addr_toks:
        if token in addr_watch:
            _accumulate(buckets["address_token"], token, rec.eid, cfg.addr_token_max_df, overflow["address_token"])
    if rec.number_key is not None and rec.number_key in watch["number"]:
        _accumulate(
            buckets["address_number"], rec.number_key, rec.eid,
            cfg.address_number_max_df, overflow["address_number"],
        )
    postal_watch = watch["postal"]
    for postal in rec.postals:
        if postal in postal_watch:
            _accumulate(buckets["address_postal"], postal, rec.eid, cfg.postal_max_df, overflow["address_postal"])


def _trim_candidates(found: dict, truth_ids: set, limit: int) -> tuple:
    """Keep every ground-truth id, then strong blocking evidence, then a stride of the rest."""
    if len(found) <= limit:
        return found, 0
    kept = {}
    strong = []
    rest = []
    for candidate_id, mask in found.items():
        if candidate_id in truth_ids:
            kept[candidate_id] = mask
        elif mask & _STRONG_MASK or mask.bit_count() >= 2:
            strong.append(candidate_id)
        else:
            rest.append(candidate_id)
    strong.sort()
    rest.sort()
    for candidate_id in strong:
        if len(kept) >= limit and candidate_id not in truth_ids:
            break
        kept[candidate_id] = found[candidate_id]
    need = limit - len(kept)
    if need > 0:
        if len(rest) > need:
            step = len(rest) / need
            chosen = [rest[int(i * step)] for i in range(need)]
        else:
            chosen = rest
        for candidate_id in chosen:
            kept[candidate_id] = found[candidate_id]
    return kept, len(found) - len(kept)


def _collect_tfidf_names(iterate_targets, country: str, cap: int | None, seed: int, force_ids: set) -> tuple:
    rng = random.Random(seed)
    kept = []
    seen = 0
    forced = {}
    for entity_id, name, _address, row_country in iterate_targets():
        if row_country != country or not _is_target_id(entity_id):
            continue
        text = build_name_views(name)["business_name_transliterated"]
        if not text:
            continue
        if entity_id in force_ids:
            forced[entity_id] = text
        if cap is None:
            kept.append((entity_id, text))
            continue
        seen += 1
        if len(kept) < cap:
            kept.append((entity_id, text))
        else:
            slot = rng.randrange(seen)
            if slot < cap:
                kept[slot] = (entity_id, text)
        if seen % 1_000_000 == 0:
            _log(f"  tf-idf names seen {seen:,} ({country})")
    have = {entity_id for entity_id, _text in kept}
    for entity_id, text in forced.items():
        if entity_id not in have:
            kept.append((entity_id, text))
    if not kept:
        return [], []
    ids, texts = zip(*kept)
    return list(ids), list(texts)


def generate_query_candidates(
    queries: list,
    iterate_targets,
    *,
    config: BlockingConfig | None = None,
    truth_by_source: dict | None = None,
    tfidf_index_cap: int | None = TFIDF_INDEX_CAP,
    tfidf_seed: int = SEED,
    max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1,
) -> tuple:
    """Candidate pairs for `queries` using blocking.py's caps and query routine.

    `iterate_targets()` must be restartable and yield
    (entity_id, business_name, business_address, country) for Source 2 and 3.
    """
    cfg = config or BlockingConfig()
    truth_by_source = truth_by_source or {}
    by_country = defaultdict(list)
    for rec in queries:
        by_country[rec.country].append(rec)
    watches = _watch_maps(queries)

    source_ids = []
    cand_ids = []
    rules = []
    dropped = 0
    trimmed_queries = 0
    n_index = Counter()
    overflow = Counter()

    for country in sorted(by_country):
        watch = watches[country]
        buckets = _new_buckets()
        overflow_lists = {name: [0] for name in buckets}
        kept = 0
        _log(f"[{country}] indexing targets for {len(by_country[country]):,} queries")
        for entity_id, name, address, row_country in iterate_targets():
            if row_country != country or not _is_target_id(entity_id):
                continue
            rec = make_record(entity_id, name, address, row_country, cfg)
            _accumulate_target(buckets, overflow_lists, rec, watch, cfg)
            kept += 1
            if kept % 500_000 == 0:
                _log(f"  indexed {kept:,} {country}")
        n_index[country] = kept
        for name, counter in overflow_lists.items():
            overflow[f"{country}:{name}"] = counter[0]
        index = _index_from_buckets(buckets, overflow_lists)
        _log(f"[{country}] indexed {kept:,} overflow={index.overflow}")
        partial = [(rec, _query_record(rec, index, (), cfg)) for rec in by_country[country]]
        del index, buckets, overflow_lists
        gc.collect()

        if cfg.enable_tfidf:
            _log(f"[{country}] character tf-idf (cap={tfidf_index_cap})")
            force_ids = set()
            for rec in by_country[country]:
                force_ids.update(truth_by_source.get(rec.eid, ()))
            name_ids, name_texts = _collect_tfidf_names(
                iterate_targets,
                country,
                tfidf_index_cap,
                _stable_offset(tfidf_seed, country),
                force_ids,
            )
            if name_texts:
                neighbours = _char_tfidf_search(
                    name_ids, name_texts, [rec.trans for rec, _found in partial], cfg,
                )
                for (rec, found), tfidf_ids in zip(partial, neighbours):
                    _add(found, tfidf_ids, TFIDF_BIT, budget=None, force=True)
                    if rec.eid in found:
                        del found[rec.eid]
            del name_ids, name_texts
            gc.collect()

        for rec, found in partial:
            truth_ids = truth_by_source.get(rec.eid, set())
            trimmed, n_dropped = _trim_candidates(found, truth_ids, max_candidates_per_s1)
            if n_dropped:
                trimmed_queries += 1
                dropped += n_dropped
            if not trimmed:
                continue
            rule_strings = {mask: _mask_to_rules(mask) for mask in set(trimmed.values())}
            for candidate_id, mask in trimmed.items():
                source_ids.append(rec.eid)
                cand_ids.append(candidate_id)
                rules.append(rule_strings[mask])
        del partial
        gc.collect()
        _log(f"[{country}] pairs so far {len(source_ids):,}")

    frame = pd.DataFrame({
        "source1_entity_id": source_ids,
        "candidate_entity_id": cand_ids,
        "rules": rules,
    })
    stats = {
        "n_index_by_country": dict(n_index),
        "overflow_keys": {key: int(value) for key, value in overflow.items() if value},
        "candidates_dropped_by_per_s1_cap": int(dropped),
        "queries_trimmed": int(trimmed_queries),
        "tfidf_index_cap": tfidf_index_cap,
    }
    return frame, stats


def hard_negative_score(frame: pd.DataFrame) -> pd.Series:
    """Rank blocked negatives. Higher means more evidence of a confusing pair."""
    score = pd.Series(0.0, index=frame.index)
    for column, weight, is_rule_count in _SCORE_WEIGHTS:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if is_rule_count:
            values = values.clip(upper=4.0) / 4.0
        score = score + weight * values
    return score


def sample_hard_negatives(
    labeled: pd.DataFrame,
    *,
    max_hard_per_s1: int = MAX_HARD_PER_S1,
    max_ordinary_per_s1: int = MAX_ORDINARY_PER_S1,
    hard_score_min: float = HARD_SCORE_MIN,
) -> pd.DataFrame:
    """Keep every positive. Keep high-evidence negatives and a spread of the rest.

    Grouping columns stay on every row. Sampling is inside each
    source1_entity_id, so a later grouped split is still possible.
    """
    if labeled.empty:
        empty = labeled.copy()
        empty["negative_role"] = pd.Series(dtype=str)
        return empty

    work = labeled.copy()
    work["_score"] = hard_negative_score(work)
    pieces = []
    for _source_id, group in work.groupby("source1_entity_id", sort=False):
        positives = group[group["is_true_match"] == 1]
        if len(positives):
            positives = positives.copy()
            positives["negative_role"] = "positive"
            pieces.append(positives)
        negatives = group[group["is_true_match"] != 1]
        if negatives.empty:
            continue
        ranked = negatives.sort_values(["_score", "candidate_entity_id"], ascending=[False, True])
        hard_mask = ranked["_score"] >= hard_score_min
        hard = ranked.loc[hard_mask].head(max_hard_per_s1).copy()
        hard_ids = set(hard["candidate_entity_id"])
        tail = ranked.loc[~ranked["candidate_entity_id"].isin(hard_ids)]
        take_at = _even_indices(len(tail), min(max_ordinary_per_s1, len(tail)))
        ordinary = tail.iloc[take_at].copy() if take_at else tail.iloc[0:0].copy()
        if len(hard):
            hard["negative_role"] = "hard_negative"
            pieces.append(hard)
        if len(ordinary):
            ordinary["negative_role"] = "ordinary_negative"
            pieces.append(ordinary)
    sampled = pd.concat(pieces, ignore_index=True) if pieces else work.iloc[0:0].copy()
    if "_score" in sampled.columns:
        sampled = sampled.drop(columns=["_score"])
    return sampled


def _candidate_count_stats(pairs: pd.DataFrame, source1_ids: list) -> dict:
    counts = Counter(pairs["source1_entity_id"]) if len(pairs) else Counter()
    values = [counts.get(entity_id, 0) for entity_id in source1_ids]
    if not values:
        return {"n_s1": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return float(ordered[0])
        position = (n - 1) * p
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return float(ordered[low])
        weight = position - low
        return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)

    return {
        "n_s1": n,
        "mean": sum(ordered) / n,
        "median": pct(0.50),
        "p95": pct(0.95),
        "max": int(ordered[-1]),
        "s1_with_zero_candidates": int(sum(1 for value in ordered if value == 0)),
    }


def _load_entity_frame(paths, wanted: set) -> pd.DataFrame:
    rows = []
    remaining = set(wanted)
    for path in paths:
        if not remaining:
            break
        for entity_id, name, address, country in _iter_raw_tsv(Path(path)):
            if entity_id not in remaining:
                continue
            rows.append((entity_id, name, address, country))
            remaining.discard(entity_id)
            if not remaining:
                break
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def _match_hist(truth: dict) -> dict:
    hist = Counter()
    for matches in truth.values():
        n_matches = len(matches)
        hist[n_matches if n_matches < 10 else "10+"] += 1
    return {str(key): int(value) for key, value in sorted(hist.items(), key=lambda item: str(item[0]))}


def build_training_pairs(
    source1_path,
    source2_path,
    source3_path,
    ground_truth_path,
    *,
    bucket_quota: dict | None = None,
    nonlatin_floor: dict | None = None,
    seed: int = SEED,
    config: BlockingConfig | None = None,
    tfidf_index_cap: int | None = TFIDF_INDEX_CAP,
    max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1,
    max_hard_per_s1: int = MAX_HARD_PER_S1,
    max_ordinary_per_s1: int = MAX_ORDINARY_PER_S1,
    hard_score_min: float = HARD_SCORE_MIN,
    source1_ids: list | None = None,
) -> tuple:
    """Build the labeled training-pair table for a deterministic Source-1 subset."""
    started = time.perf_counter()
    cfg = config or BlockingConfig(verbose=False)
    target_paths = [Path(source2_path), Path(source3_path)]

    _log("counting source 2/3 and non-Latin names")
    target_profile = _count_targets(target_paths)
    nonlatin_ids = target_profile["nonlatin_ids"]

    if source1_ids is None:
        _log("selecting stratified source 1")
        selection = select_training_source1(
            source1_path,
            ground_truth_path,
            nonlatin_ids,
            bucket_quota=bucket_quota,
            nonlatin_floor=nonlatin_floor,
            seed=seed,
        )
        source1_ids = selection["source1_ids"]
    else:
        source1_ids = sorted(set(source1_ids))
        selection = {"source1_ids": source1_ids, "corpus": {}, "strata": {}}
    del nonlatin_ids
    gc.collect()

    wanted = set(source1_ids)
    truth_frame = _load_truth_for(Path(ground_truth_path), wanted)
    truth = ground_truth_map(truth_frame, source1_ids)
    _log(f"ground truth loaded for {len(truth):,} source-1 ids")

    source1 = _load_entity_frame([source1_path], wanted)
    country_of = dict(zip(source1["entity_id"], source1["country"]))
    queries = [
        make_record(row.entity_id, row.business_name, row.business_address, row.country, cfg)
        for row in source1.itertuples(index=False)
    ]
    _log(f"built {len(queries):,} query records")

    def iterate_targets():
        for path in target_paths:
            yield from _iter_raw_tsv(path)

    pairs, block_stats = generate_query_candidates(
        queries,
        iterate_targets,
        config=cfg,
        truth_by_source=truth,
        tfidf_index_cap=tfidf_index_cap,
        tfidf_seed=seed,
        max_candidates_per_s1=max_candidates_per_s1,
    )
    count_stats = _candidate_count_stats(pairs, source1_ids)
    _log(f"blocked pairs {len(pairs):,}  per-s1 mean {count_stats['mean']}")

    if pairs.empty:
        labeled = pd.DataFrame(columns=list(SAVED_COLUMNS))
        sampled = labeled
        feature_seconds = 0.0
    else:
        needed = set(pairs["source1_entity_id"]).union(pairs["candidate_entity_id"])
        _log(f"loading text for {len(needed):,} entities")
        source2 = _load_entity_frame([source2_path], needed)
        source3 = _load_entity_frame([source3_path], needed)
        feature_started = time.perf_counter()
        features = featurize_candidate_pairs(pairs, source1, source2, source3)
        labeled = attach_true_match_label(features, truth_frame)
        feature_seconds = time.perf_counter() - feature_started
        _log(f"featurized {len(labeled):,} pairs in {feature_seconds:.1f}s")
        sampled = sample_hard_negatives(
            labeled,
            max_hard_per_s1=max_hard_per_s1,
            max_ordinary_per_s1=max_ordinary_per_s1,
            hard_score_min=hard_score_min,
        )
        del features
        gc.collect()

    if len(sampled):
        sampled["country"] = sampled["source1_entity_id"].map(country_of).fillna("")
        sampled["is_true_match"] = sampled["is_true_match"].astype(int)
        sampled = sampled.loc[:, list(SAVED_COLUMNS)].reset_index(drop=True)
        positives_before = int((labeled["is_true_match"] == 1).sum()) if len(labeled) else 0
        positives_after = int((sampled["is_true_match"] == 1).sum())
        if positives_after != positives_before:
            raise RuntimeError(
                f"sampling dropped positives: {positives_before} before, {positives_after} after"
            )
    else:
        positives_before = 0

    true_links = sum(len(matches) for matches in truth.values())
    no_match_ids = [entity_id for entity_id, matches in truth.items() if not matches]
    no_match_in_pairs = set(sampled.loc[sampled["source1_entity_id"].isin(no_match_ids), "source1_entity_id"]) if len(sampled) else set()
    role_counts = Counter(sampled["negative_role"]) if len(sampled) else Counter()
    country_counts = Counter(sampled.drop_duplicates("source1_entity_id")["country"]) if len(sampled) else Counter()

    stats = {
        "runtime_seconds": time.perf_counter() - started,
        "feature_seconds": feature_seconds,
        "seed": seed,
        "bucket_quota": bucket_quota or BUCKET_QUOTA,
        "nonlatin_floor": nonlatin_floor or NONLATIN_FLOOR,
        "hard_score_min": hard_score_min,
        "max_hard_per_s1": max_hard_per_s1,
        "max_ordinary_per_s1": max_ordinary_per_s1,
        "max_candidates_per_s1": max_candidates_per_s1,
        "selection": {
            "strata": selection.get("strata", {}),
            "corpus": selection.get("corpus", {}),
        },
        "target_rows": target_profile["rows"],
        "target_countries": target_profile["countries"],
        "blocking": block_stats,
        "queried_s1": len(source1_ids),
        "blocked_pairs": int(len(pairs)),
        "candidate_counts": count_stats,
        "true_links_in_ground_truth": int(true_links),
        "true_links_retrieved": int(positives_before),
        "saved_rows": int(len(sampled)),
        "positive_pairs": int(role_counts.get("positive", 0)),
        "negative_pairs": int(role_counts.get("hard_negative", 0) + role_counts.get("ordinary_negative", 0)),
        "hard_negatives": int(role_counts.get("hard_negative", 0)),
        "ordinary_negatives": int(role_counts.get("ordinary_negative", 0)),
        "positive_rate": (role_counts.get("positive", 0) / len(sampled)) if len(sampled) else None,
        "no_match_s1_queried": len(no_match_ids),
        "no_match_s1_with_saved_pairs": len(no_match_in_pairs),
        "saved_s1": int(sampled["source1_entity_id"].nunique()) if len(sampled) else 0,
        "saved_country_s1": dict(country_counts),
        "saved_match_hist": _match_hist({
            entity_id: truth.get(entity_id, set())
            for entity_id in (sampled["source1_entity_id"].unique() if len(sampled) else [])
        }),
    }
    _log(
        "saved rows "
        f"{stats['saved_rows']:,} positives {stats['positive_pairs']:,} "
        f"hard {stats['hard_negatives']:,} ordinary {stats['ordinary_negatives']:,}"
    )
    return sampled, stats, {
        "source1": source1,
        "truth": truth,
    }


def save_training_pairs(frame: pd.DataFrame, path) -> str:
    """Write parquet when pyarrow/fastparquet is installed, otherwise gzipped TSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(destination, index=False)
        return str(destination)
    except (ImportError, ValueError):
        fallback = destination.with_suffix(".tsv.gz") if destination.suffix == ".parquet" else destination
        frame.to_csv(fallback, sep="\t", index=False, compression="gzip")
        return str(fallback)


def save_stats(stats: dict, path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


def lookup_texts(paths, wanted: set) -> dict:
    found = {}
    remaining = set(wanted)
    for path in paths:
        if not remaining:
            break
        for entity_id, name, address, country in _iter_raw_tsv(Path(path)):
            if entity_id not in remaining:
                continue
            found[entity_id] = (name, address, country)
            remaining.discard(entity_id)
            if not remaining:
                break
    return found


def example_rows(frame: pd.DataFrame, texts: dict, role: str, n: int = 8) -> list:
    subset = frame[frame["negative_role"] == role] if "negative_role" in frame.columns else frame
    if role == "positive":
        subset = frame[frame["is_true_match"] == 1]
    if subset.empty:
        return []
    ranked = subset.sort_values(
        ["name_ratio_core", "name_token_jaccard", "addr_ratio_basic", "n_blocking_rules", "candidate_entity_id"],
        ascending=[False, False, False, False, True],
    )
    picked = ranked.head(n)
    rows = []
    for record in picked.itertuples(index=False):
        left = texts.get(record.source1_entity_id, ("", "", ""))
        right = texts.get(record.candidate_entity_id, ("", "", ""))
        rows.append({
            "source1_entity_id": record.source1_entity_id,
            "candidate_entity_id": record.candidate_entity_id,
            "s1_name": left[0],
            "candidate_name": right[0],
            "s1_address": left[1],
            "candidate_address": right[1],
            "s1_country": left[2],
            "candidate_country": right[2],
            "is_true_match": int(record.is_true_match),
            "negative_role": getattr(record, "negative_role", ""),
            "name_ratio_core": None if pd.isna(record.name_ratio_core) else round(float(record.name_ratio_core), 3),
            "name_ratio_translit": None if pd.isna(record.name_ratio_translit) else round(float(record.name_ratio_translit), 3),
            "name_token_jaccard": None if pd.isna(record.name_token_jaccard) else round(float(record.name_token_jaccard), 3),
            "addr_ratio_basic": None if pd.isna(record.addr_ratio_basic) else round(float(record.addr_ratio_basic), 3),
            "addr_token_jaccard": None if pd.isna(record.addr_token_jaccard) else round(float(record.addr_token_jaccard), 3),
            "addr_number_agree": None if pd.isna(record.addr_number_agree) else round(float(record.addr_number_agree), 3),
            "addr_postal_agree": None if pd.isna(record.addr_postal_agree) else round(float(record.addr_postal_agree), 3),
            "n_blocking_rules": None if pd.isna(record.n_blocking_rules) else int(record.n_blocking_rules),
        })
    return rows


def _self_check() -> None:
    """Same pairs as generate_candidates on a tiny open-set frame when TF-IDF is off."""
    from .blocking import generate_candidates

    cfg = BlockingConfig(enable_tfidf=False, verbose=False, exact_max_fanout=50)
    source1 = pd.DataFrame([
        ["S1-yes", "ABC Technologies Pvt Ltd", "12 MG Road, Mumbai 400001", "India"],
        ["S1-multi", "Blue River Cafe", "9 Lake Ave, Madison, WI 53703", "US"],
        ["S1-none", "Quiet Books", "1 Oak Road, Dallas, TX 75201", "US"],
        ["S1-fr", "École Primaire Sainte", "22 Rue Descartes, Calais 62100", "France"],
    ], columns=["entity_id", "business_name", "business_address", "country"])
    source2 = pd.DataFrame([
        ["S2-yes", "ABC Technologies Mumbai", "12 MG Road Mumbai 400001", "India"],
        ["S2-hard", "ABC Technologies Delhi", "88 Ring Road, Delhi 110001", "India"],
        ["S2-cafe", "Cafe Blue River", "9 Lake Avenue, Madison, WI 53703", "US"],
        ["S2-other", "River Blue Supplies", "400 State St, Madison, WI 53703", "US"],
        ["S2-quiet", "Quiet Books LLC", "1 Oak Road, Dallas, TX 75201", "US"],
        ["S2-fr", "Ecole Primaire Sainte", "22 Rue Descartes Calais 62100", "France"],
    ], columns=["entity_id", "business_name", "business_address", "country"])
    source3 = pd.DataFrame([
        ["S3-yes", "ABC Technologies Private Limited", "Shop 12, MG Road, Mumbai 400001", "India"],
        ["S3-cafe", "Blue River Café LLC", "9 Lake Ave Madison WI 53703", "US"],
    ], columns=["entity_id", "business_name", "business_address", "country"])
    expected = generate_candidates(source1, source2, source3, cfg)
    rows = list(zip(source2["entity_id"], source2["business_name"], source2["business_address"], source2["country"]))
    rows.extend(zip(source3["entity_id"], source3["business_name"], source3["business_address"], source3["country"]))
    queries = [
        make_record(*row, cfg)
        for row in zip(source1["entity_id"], source1["business_name"], source1["business_address"], source1["country"])
    ]
    got, _stats = generate_query_candidates(
        queries, lambda: iter(rows), config=cfg, tfidf_index_cap=None, max_candidates_per_s1=100,
    )
    left = set(zip(expected["source1_entity_id"], expected["candidate_entity_id"], expected["rules"]))
    right = set(zip(got["source1_entity_id"], got["candidate_entity_id"], got["rules"]))
    if left != right:
        raise AssertionError(f"candidate mismatch\n only expected {left - right}\n only got {right - left}")

    truth = pd.DataFrame([
        ["S1-yes", "S2-yes,S3-yes"],
        ["S1-multi", "S2-cafe,S3-cafe"],
        ["S1-none", ""],
        ["S1-fr", "S2-fr"],
    ], columns=["source1_entity_id", "matched_entity_ids"])
    features = featurize_candidate_pairs(got, source1, source2, source3)
    labeled = attach_true_match_label(features, truth)
    sampled = sample_hard_negatives(labeled, max_hard_per_s1=4, max_ordinary_per_s1=2, hard_score_min=0.35)
    flags = dict(zip(
        zip(sampled["source1_entity_id"], sampled["candidate_entity_id"]),
        sampled["is_true_match"],
    ))
    assert flags[("S1-yes", "S2-yes")] == 1
    assert flags[("S1-yes", "S3-yes")] == 1
    assert ("S1-yes", "S2-hard") in flags and flags[("S1-yes", "S2-hard")] == 0
    assert sampled.loc[
        (sampled["source1_entity_id"] == "S1-yes") & (sampled["candidate_entity_id"] == "S2-hard"),
        "negative_role",
    ].iloc[0] == "hard_negative"
    none_rows = sampled[sampled["source1_entity_id"] == "S1-none"]
    assert len(none_rows) >= 1
    assert (none_rows["is_true_match"] == 0).all()
    multi = sampled[(sampled["source1_entity_id"] == "S1-multi") & (sampled["is_true_match"] == 1)]
    assert set(multi["candidate_entity_id"]) >= {"S2-cafe", "S3-cafe"}
    france = got[got["source1_entity_id"] == "S1-fr"]["candidate_entity_id"].tolist()
    assert france == ["S2-fr"] or set(france) == {"S2-fr"}
    assert not got["candidate_entity_id"].str.startswith("S1-").any()


# 1,250 Source-1 ids per country present (2,500 when train has two countries).
# Background rows are random seeks. True-match rows are an id lookup.
BOUNDED_QUOTA = {"0": 375, "1": 500, "2plus": 375}
BOUNDED_NONLATIN_FLOOR = {"0": 0, "1": 0, "2plus": 0}
BOUNDED_EXTRA_PER_FILE = 2_500
BOUNDED_NEGATIVES_PER_S1 = 30


def _parse_target_line(line: str):
    entity_id, _, rest = line.rstrip("\n").partition("\t")
    if not _is_target_id(entity_id):
        return None
    name, _, rest = rest.partition("\t")
    address, _, country = rest.partition("\t")
    return entity_id, name, address, country.strip()


def _seek_sample_rows(path, n: int, seed: int) -> list:
    """Deterministic rows from random offsets. Does not read the whole file."""
    path = Path(path)
    size = path.stat().st_size
    if size <= 1 or n <= 0:
        return []
    rng = random.Random(seed)
    rows = {}
    attempts = 0
    limit = max(n * 30, n + 10)
    with path.open("rb") as handle:
        while len(rows) < n and attempts < limit:
            attempts += 1
            handle.seek(rng.randrange(size))
            handle.readline()
            raw = handle.readline()
            if not raw:
                continue
            parsed = _parse_target_line(raw.decode("utf-8", errors="replace"))
            if parsed is None or parsed[0] in rows:
                continue
            rows[parsed[0]] = parsed
    return list(rows.values())


def _fetch_ids(paths, wanted: set) -> tuple:
    """Read only until every requested id has been seen. No normalization."""
    remaining = set(wanted)
    found = []
    nonlatin = set()
    lines_read = 0
    for path in paths:
        if not remaining:
            break
        with Path(path).open(encoding="utf-8") as handle:
            next(handle)
            for line in handle:
                lines_read += 1
                parsed = _parse_target_line(line)
                if parsed is None or parsed[0] not in remaining:
                    continue
                found.append(parsed)
                if _has_non_ascii_letter(parsed[1]):
                    nonlatin.add(parsed[0])
                remaining.discard(parsed[0])
                if not remaining:
                    break
    return found, nonlatin, remaining, lines_read


def _downsample_bounded(ids, country_of, truth, nonlatin_match_ids, quota, floor) -> list:
    groups = defaultdict(lambda: {"nonlatin": [], "other": []})
    for entity_id in ids:
        country = country_of.get(entity_id, "")
        matches = truth.get(entity_id, ())
        bucket = _bucket(len(matches))
        slot = "nonlatin" if any(match in nonlatin_match_ids for match in matches) else "other"
        groups[(country, bucket)][slot].append(entity_id)
    chosen = []
    strata = {}
    for (country, bucket), pools in sorted(groups.items()):
        nonlatin = sorted(pools["nonlatin"])
        other = sorted(pools["other"])
        take = nonlatin[: floor.get(bucket, 0)]
        need = quota.get(bucket, 0) - len(take)
        if need > 0:
            take.extend(other[:need])
            need = quota.get(bucket, 0) - len(take)
        if need > 0:
            already = set(take)
            take.extend([entity_id for entity_id in nonlatin if entity_id not in already][:need])
        chosen.extend(take)
        strata[f"{country}|match_{bucket}"] = {
            "selected": len(take),
            "nonlatin_in_selected": sum(1 for entity_id in take if entity_id in set(nonlatin)),
            "pool_nonlatin": len(nonlatin),
            "pool_other": len(other),
        }
    return sorted(set(chosen)), strata


def _cap_negatives(pairs: pd.DataFrame, truth: dict, limit: int) -> pd.DataFrame:
    """Keep every ground-truth candidate, then up to `limit` other candidates per Source-1."""
    if pairs.empty or limit <= 0:
        return pairs
    gt_flag = [
        candidate in truth.get(source, ())
        for source, candidate in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])
    ]
    work = pairs.copy()
    work["_gt"] = gt_flag
    work["_n"] = work["rules"].map(lambda text: (text.count("|") + 1) if text else 0)
    pieces = []
    for _source_id, group in work.groupby("source1_entity_id", sort=False):
        positives = group[group["_gt"]]
        negatives = group[~group["_gt"]].sort_values(["_n", "candidate_entity_id"], ascending=[False, True])
        pieces.append(positives)
        pieces.append(negatives.head(limit))
    capped = pd.concat(pieces, ignore_index=True) if pieces else work.iloc[0:0]
    return capped.drop(columns=["_gt", "_n"])


def build_bounded_training_pairs(
    source1_path,
    source2_path,
    source3_path,
    ground_truth_path,
    *,
    seed: int = SEED,
    deadline_s: float = 300.0,
) -> tuple:
    """Stratified Source-1 sample blocked with generate_candidates on a small target sample."""
    from .blocking import generate_candidates

    started = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - started

    def check(stage: str) -> None:
        if elapsed() > deadline_s:
            raise TimeoutError(f"stopped at {stage} after {elapsed():.1f}s (limit {deadline_s:.0f}s)")

    pool_quota = {bucket: count * BOUNDED_POOL_MULTIPLIER for bucket, count in BOUNDED_QUOTA.items()}
    _log(
        f"bounded selection pool quota per country {pool_quota} "
        f"final quota {BOUNDED_QUOTA} extra targets/country {BOUNDED_EXTRA_PER_COUNTRY}"
    )
    selection = select_training_source1(
        source1_path,
        ground_truth_path,
        set(),
        bucket_quota=pool_quota,
        nonlatin_floor={"0": 0, "1": 0, "2plus": 0},
        seed=seed,
    )
    pool_ids = selection["source1_ids"]
    _log(f"pool {len(pool_ids):,} source-1 ids in {elapsed():.1f}s")
    check("selection")

    truth_frame = _load_truth_for(Path(ground_truth_path), set(pool_ids))
    truth = ground_truth_map(truth_frame, pool_ids)
    force_ids = set()
    for matches in truth.values():
        force_ids.update(matches)
    _log(f"forced target ids {len(force_ids):,}")

    forced_rows, extra_rows, nonlatin_forced, scanned = _load_bounded_targets(
        [source2_path, source3_path], force_ids, BOUNDED_EXTRA_PER_COUNTRY, seed,
    )
    _log(
        f"loaded {len(forced_rows):,} matched targets + {len(extra_rows):,} extra "
        f"in {elapsed():.1f}s"
    )
    check("target load")

    source1 = _load_entity_frame([source1_path], set(pool_ids))
    country_of = dict(zip(source1["entity_id"], source1["country"]))
    selected, strata = _downsample_bounded(
        pool_ids, country_of, truth, nonlatin_forced, BOUNDED_QUOTA, BOUNDED_NONLATIN_FLOOR,
    )
    _log(f"selected {len(selected):,} source-1 ids")
    source1 = source1[source1["entity_id"].isin(selected)].reset_index(drop=True)
    country_of = dict(zip(source1["entity_id"], source1["country"]))
    truth = {entity_id: truth.get(entity_id, set()) for entity_id in selected}
    truth_frame = truth_frame[truth_frame["source1_entity_id"].isin(selected)].reset_index(drop=True)

    target_rows = forced_rows + extra_rows
    target_frame = pd.DataFrame(
        target_rows, columns=["entity_id", "business_name", "business_address", "country"],
    )
    target_frame = target_frame.drop_duplicates("entity_id")
    source2 = target_frame[target_frame["entity_id"].str.startswith("S2-")].reset_index(drop=True)
    source3 = target_frame[target_frame["entity_id"].str.startswith("S3-")].reset_index(drop=True)
    _log(f"blocking {len(source1):,} x ({len(source2):,} S2 + {len(source3):,} S3)")
    check("before blocking")
    block_started = time.perf_counter()
    pairs = generate_candidates(source1, source2, source3, BlockingConfig(verbose=False))
    block_seconds = time.perf_counter() - block_started
    _log(f"blocked {len(pairs):,} pairs in {block_seconds:.1f}s")
    check("blocking")
    count_stats = _candidate_count_stats(pairs, selected)

    featurize_pairs = _cap_negatives(pairs, truth, BOUNDED_NEGATIVES_PER_S1)
    _log(f"featurizing {len(featurize_pairs):,} pairs (cap {BOUNDED_NEGATIVES_PER_S1} negatives/S1)")
    feature_started = time.perf_counter()
    features = featurize_candidate_pairs(featurize_pairs, source1, source2, source3)
    labeled = attach_true_match_label(features, truth_frame)
    feature_seconds = time.perf_counter() - feature_started
    sampled = sample_hard_negatives(labeled)
    if len(sampled):
        sampled["country"] = sampled["source1_entity_id"].map(country_of).fillna("")
        sampled["is_true_match"] = sampled["is_true_match"].astype(int)
        sampled = sampled.loc[:, list(SAVED_COLUMNS)].reset_index(drop=True)
    positives_before = int((labeled["is_true_match"] == 1).sum()) if len(labeled) else 0
    if len(sampled) and int((sampled["is_true_match"] == 1).sum()) != positives_before:
        raise RuntimeError("sampling dropped a positive pair")

    role_counts = Counter(sampled["negative_role"]) if len(sampled) else Counter()
    no_match_ids = [entity_id for entity_id, matches in truth.items() if not matches]
    no_match_saved = (
        set(sampled.loc[sampled["source1_entity_id"].isin(no_match_ids), "source1_entity_id"])
        if len(sampled) else set()
    )
    country_counts = (
        Counter(sampled.drop_duplicates("source1_entity_id")["country"]) if len(sampled) else Counter()
    )
    stats = {
        "mode": "bounded_target_sample",
        "runtime_seconds": time.perf_counter() - started,
        "block_seconds": block_seconds,
        "feature_seconds": feature_seconds,
        "seed": seed,
        "bucket_quota": BOUNDED_QUOTA,
        "nonlatin_floor": BOUNDED_NONLATIN_FLOOR,
        "extra_targets_per_country": BOUNDED_EXTRA_PER_COUNTRY,
        "negatives_featurized_per_s1": BOUNDED_NEGATIVES_PER_S1,
        "selection": {"strata": strata, "corpus": selection.get("corpus", {})},
        "targets_scanned_by_country": scanned,
        "forced_targets": len(forced_rows),
        "extra_targets": len(extra_rows),
        "queried_s1": len(selected),
        "blocked_pairs": int(len(pairs)),
        "featurized_pairs": int(len(featurize_pairs)),
        "candidate_counts": count_stats,
        "true_links_in_ground_truth": int(sum(len(matches) for matches in truth.values())),
        "true_links_retrieved": positives_before,
        "saved_rows": int(len(sampled)),
        "positive_pairs": int(role_counts.get("positive", 0)),
        "negative_pairs": int(role_counts.get("hard_negative", 0) + role_counts.get("ordinary_negative", 0)),
        "hard_negatives": int(role_counts.get("hard_negative", 0)),
        "ordinary_negatives": int(role_counts.get("ordinary_negative", 0)),
        "positive_rate": (role_counts.get("positive", 0) / len(sampled)) if len(sampled) else None,
        "no_match_s1_queried": len(no_match_ids),
        "no_match_s1_with_saved_pairs": len(no_match_saved),
        "saved_s1": int(sampled["source1_entity_id"].nunique()) if len(sampled) else 0,
        "saved_country_s1": dict(country_counts),
        "saved_match_hist": _match_hist(truth),
        "note": (
            "generate_candidates ran on the selected Source-1 rows plus their ground-truth "
            "matches and a deterministic extra Source-2/3 sample. It did not index the full "
            "10M target set. Candidate counts are for that bounded index."
        ),
    }
    _log(f"done {stats['saved_rows']:,} rows in {stats['runtime_seconds']:.1f}s")
    return sampled, stats, {"source1": source1, "truth": truth}


SANITY_QUOTA = {"0": 100, "1": 180, "2plus": 120}
SANITY_SEEKS_PER_FILE = 400
_ROW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def build_sanity_pairs(source1_path, source2_path, source3_path, ground_truth_path, *, seed: int = SEED) -> tuple:
    """Pipeline check only. Target rows come from byte seeks, not a full S2/S3 read."""
    from .blocking import generate_candidates

    started = time.perf_counter()
    selection = select_training_source1(
        source1_path, ground_truth_path, set(),
        bucket_quota=SANITY_QUOTA,
        nonlatin_floor={"0": 0, "1": 0, "2plus": 0},
        seed=seed,
    )
    selected = selection["source1_ids"]
    _log(f"selected S1 count: {len(selected)}")
    source1 = _load_entity_frame([source1_path], set(selected))
    truth_frame = _load_truth_for(Path(ground_truth_path), set(selected))
    source2 = pd.DataFrame(_seek_sample_rows(source2_path, SANITY_SEEKS_PER_FILE, seed), columns=_ROW_COLUMNS)
    source3 = pd.DataFrame(
        _seek_sample_rows(source3_path, SANITY_SEEKS_PER_FILE, seed + 1), columns=_ROW_COLUMNS,
    )
    _log(f"S2 target-pool count: {len(source2)}")
    _log(f"S3 target-pool count: {len(source3)}")
    _log(f"total target-pool count: {len(source2) + len(source3)}")
    block_started = time.perf_counter()
    pairs = generate_candidates(source1, source2, source3, BlockingConfig(verbose=False))
    block_seconds = time.perf_counter() - block_started
    _log(f"blocked pairs: {len(pairs):,} in {block_seconds:.1f}s")
    truth = ground_truth_map(truth_frame, selected)
    kept = _cap_negatives(pairs, truth, 20) if len(pairs) > 40_000 else pairs
    features = featurize_candidate_pairs(kept, source1, source2, source3)
    labeled = attach_true_match_label(features, truth_frame)
    country_of = dict(zip(source1["entity_id"], source1["country"]))
    labeled["country"] = labeled["source1_entity_id"].map(country_of).fillna("")
    labeled["is_true_match"] = labeled["is_true_match"].astype(int)
    keep_cols = ["source1_entity_id", "candidate_entity_id", "is_true_match", *FEATURE_COLUMNS, "country"]
    labeled = labeled.loc[:, keep_cols].reset_index(drop=True)
    texts = {}
    for frame in (source1, source2, source3):
        for row in frame.itertuples(index=False):
            texts[row.entity_id] = (row.business_name, row.business_address, row.country)
    stats = {
        "mode": "pipeline_sanity",
        "warning": (
            "Byte-offset target sample. True matches whose S2/S3 rows were not "
            "sampled are missing. Not a recall or training-rate measurement."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "block_seconds": block_seconds,
        "queried_s1": len(selected),
        "s2_pool": int(len(source2)),
        "s3_pool": int(len(source3)),
        "target_pool": int(len(source2) + len(source3)),
        "blocked_pairs": int(len(pairs)),
        "saved_rows": int(len(labeled)),
        "positive_pairs": int((labeled["is_true_match"] == 1).sum()),
        "negative_pairs": int((labeled["is_true_match"] == 0).sum()),
        "candidate_counts": _candidate_count_stats(pairs, selected),
        "saved_country_s1": dict(Counter(
            labeled.drop_duplicates("source1_entity_id")["country"]
        )) if len(labeled) else {},
        "saved_match_hist": _match_hist({
            entity_id: truth.get(entity_id, set())
            for entity_id in (labeled["source1_entity_id"].unique() if len(labeled) else [])
        }),
        "strata": selection.get("strata", {}),
    }
    return labeled, stats, texts


REAL_QUOTA = {"0": 300, "1": 450, "2plus": 250}
REAL_EXTRA_TOTAL = 3_000
PAIR_EXPLOSION = 1_000_000


def build_real_pairs(source1_path, source2_path, source3_path, ground_truth_path, *, seed: int = SEED) -> tuple:
    """One S2 pass and one S3 pass for ground-truth targets, plus a seek sample of negatives."""
    from .blocking import generate_candidates

    timings = {}
    started = time.perf_counter()
    t0 = started
    selection = select_training_source1(
        source1_path, ground_truth_path, set(),
        bucket_quota=REAL_QUOTA,
        nonlatin_floor={"0": 0, "1": 0, "2plus": 0},
        seed=seed,
    )
    selected = selection["source1_ids"]
    source1 = _load_entity_frame([source1_path], set(selected))
    truth_frame = _load_truth_for(Path(ground_truth_path), set(selected))
    truth = ground_truth_map(truth_frame, selected)
    timings["selection_s"] = time.perf_counter() - t0

    zero = singleton = multi = 0
    required_s2, required_s3 = set(), set()
    for entity_id in selected:
        matches = truth.get(entity_id, set())
        n_matches = len(matches)
        if n_matches == 0:
            zero += 1
        elif n_matches == 1:
            singleton += 1
        else:
            multi += 1
        for match_id in matches:
            if str(match_id).startswith("S2-"):
                required_s2.add(match_id)
            elif str(match_id).startswith("S3-"):
                required_s3.add(match_id)
    _log(
        f"selected S1 {len(selected)} zero {zero} singleton {singleton} multi {multi} "
        f"required S2 {len(required_s2)} S3 {len(required_s3)}"
    )

    t0 = time.perf_counter()
    s2_rows, _, missing_s2, s2_lines = _fetch_ids([source2_path], required_s2)
    timings["s2_retrieval_s"] = time.perf_counter() - t0
    _log(f"S2 retrieved {len(s2_rows)} missing {len(missing_s2)} lines_read {s2_lines:,} in {timings['s2_retrieval_s']:.1f}s")
    t0 = time.perf_counter()
    s3_rows, _, missing_s3, s3_lines = _fetch_ids([source3_path], required_s3)
    timings["s3_retrieval_s"] = time.perf_counter() - t0
    _log(f"S3 retrieved {len(s3_rows)} missing {len(missing_s3)} lines_read {s3_lines:,} in {timings['s3_retrieval_s']:.1f}s")

    extra_each = REAL_EXTRA_TOTAL // 2
    if len(required_s2) + len(required_s3) > 15_000:
        extra_each = 500
    elif len(required_s2) + len(required_s3) > 8_000:
        extra_each = 1_000
    have = {row[0] for row in s2_rows}
    have.update(row[0] for row in s3_rows)
    extras = [
        row for row in _seek_sample_rows(source2_path, extra_each, seed + 7)
        + _seek_sample_rows(source3_path, extra_each, seed + 11)
        if row[0] not in have
    ]
    pool_rows = s2_rows + s3_rows + extras
    pool = pd.DataFrame(pool_rows, columns=_ROW_COLUMNS).drop_duplicates("entity_id")
    source2 = pool[pool["entity_id"].str.startswith("S2-")].reset_index(drop=True)
    source3 = pool[pool["entity_id"].str.startswith("S3-")].reset_index(drop=True)
    _log(f"target pool S2 {len(source2)} S3 {len(source3)} total {len(pool)}")

    t0 = time.perf_counter()
    pairs = generate_candidates(source1, source2, source3, BlockingConfig(verbose=False))
    timings["blocking_s"] = time.perf_counter() - t0
    _log(f"blocked pairs {len(pairs):,} in {timings['blocking_s']:.1f}s")
    pool_ids = set(pool["entity_id"])
    links_present = sum(1 for matches in truth.values() for match_id in matches if match_id in pool_ids)
    if len(pairs):
        links_found = sum(
            1 for source_id, candidate_id in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])
            if candidate_id in truth.get(source_id, ())
        )
    else:
        links_found = 0
    count_stats = _candidate_count_stats(pairs, selected)
    base = {
        "mode": "real_bounded_pool",
        "timings": timings,
        "selected_s1": len(selected),
        "zero_match_s1": zero,
        "singleton_s1": singleton,
        "multi_match_s1": multi,
        "required_s2": len(required_s2),
        "retrieved_s2": len(s2_rows),
        "missing_s2": sorted(missing_s2)[:8],
        "required_s3": len(required_s3),
        "retrieved_s3": len(s3_rows),
        "missing_s3": sorted(missing_s3)[:8],
        "s2_pool": int(len(source2)),
        "s3_pool": int(len(source3)),
        "target_pool": int(len(pool)),
        "blocked_pairs": int(len(pairs)),
        "true_links_present": int(links_present),
        "true_links_found": int(links_found),
        "candidate_recall": (links_found / links_present) if links_present else None,
        "candidate_counts": count_stats,
        "s2_lines_read": s2_lines,
        "s3_lines_read": s3_lines,
    }
    if len(pairs) >= PAIR_EXPLOSION:
        base["stopped"] = "candidate_explosion_before_features"
        base["runtime_seconds"] = time.perf_counter() - started
        _log(f"STOP before features: {len(pairs):,} pairs")
        return None, base, {}

    t0 = time.perf_counter()
    features = featurize_candidate_pairs(pairs, source1, source2, source3)
    labeled = attach_true_match_label(features, truth_frame)
    timings["features_s"] = time.perf_counter() - t0
    labeled["is_true_match"] = labeled["is_true_match"].astype(int)
    keep_cols = ["source1_entity_id", "candidate_entity_id", "is_true_match", *FEATURE_COLUMNS]
    labeled = labeled.loc[:, keep_cols].reset_index(drop=True)
    texts = {}
    for frame in (source1, source2, source3):
        for row in frame.itertuples(index=False):
            texts[row.entity_id] = (row.business_name, row.business_address, row.country)
    selected_set = set(selected)
    duplicate_pairs = int(labeled.duplicated(["source1_entity_id", "candidate_entity_id"]).sum())
    bad_labels = int((~labeled["is_true_match"].isin([0, 1])).sum())
    positive = labeled[labeled["is_true_match"] == 1]
    bad_s1 = int((~positive["source1_entity_id"].isin(selected_set)).sum()) if len(positive) else 0
    bad_cand = int((~labeled["candidate_entity_id"].isin(pool_ids)).sum()) if len(labeled) else 0
    base.update({
        "stopped": None,
        "runtime_seconds": time.perf_counter() - started,
        "saved_rows": int(len(labeled)),
        "positive_pairs": int(len(positive)),
        "negative_pairs": int((labeled["is_true_match"] == 0).sum()),
        "positive_rate": (len(positive) / len(labeled)) if len(labeled) else None,
        "s1_groups": int(labeled["source1_entity_id"].nunique()) if len(labeled) else 0,
        "s1_groups_with_positives": int(positive["source1_entity_id"].nunique()) if len(positive) else 0,
        "s1_groups_with_zero_candidates": int(count_stats.get("s1_with_zero_candidates") or 0),
        "duplicate_pairs": duplicate_pairs,
        "bad_labels": bad_labels,
        "positives_outside_selected_s1": bad_s1,
        "candidates_outside_pool": bad_cand,
    })
    base["timings"] = timings
    _log(f"features {timings['features_s']:.1f}s rows {len(labeled):,} positives {len(positive):,}")
    return labeled, base, texts


def main(argv=None) -> None:
    import argparse

    from .blocking import resolve_dataset_dir
    from .config import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Build a stratified training-pair file")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quota-0", type=int, default=BUCKET_QUOTA["0"])
    parser.add_argument("--quota-1", type=int, default=BUCKET_QUOTA["1"])
    parser.add_argument("--quota-2plus", type=int, default=BUCKET_QUOTA["2plus"])
    parser.add_argument("--tfidf-cap", type=int, default=TFIDF_INDEX_CAP)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bounded", action="store_true",
                        help="2k Source-1 sample blocked against a small target sample.")
    parser.add_argument("--sanity", action="store_true",
                        help="Small seek-sampled pipeline check. Does not scan S2/S3.")
    parser.add_argument("--real", action="store_true",
                        help="2k S1, one S2 pass, one S3 pass, seek negatives, then block and featurize.")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        _self_check()
        _log("self-check passed")
        return

    data_dir = resolve_dataset_dir(args.data_dir)
    train = data_dir / "train"
    if args.sanity:
        frame, stats, texts = build_sanity_pairs(
            train / "train_source1.tsv",
            train / "train_source2.tsv",
            train / "train_source3.tsv",
            train / "train_ground_truth.tsv",
            seed=args.seed,
        )
        output = Path(args.output) if args.output else PROJECT_ROOT / "data" / "processed" / "training_pairs_sanity.parquet"
        written = save_training_pairs(frame, output)
        stats["saved_path"] = written
        stats["examples_positive"] = example_rows(frame.assign(negative_role="positive"), texts, "positive", 5)
        stats["examples_negative"] = example_rows(
            frame.loc[frame["is_true_match"] == 0].assign(negative_role="hard_negative"),
            texts, "hard_negative", 5,
        )
        save_stats(stats, Path(written).with_suffix("").with_suffix(".stats.json") if not str(written).endswith(".gz") else Path(written).with_name("training_pairs_sanity.stats.json"))
        _log(f"wrote {written}")
        _log(json.dumps({
            key: stats[key] for key in (
                "mode", "warning", "runtime_seconds", "queried_s1", "s2_pool", "s3_pool",
                "target_pool", "blocked_pairs", "saved_rows", "positive_pairs", "negative_pairs",
                "candidate_counts", "saved_country_s1", "saved_match_hist",
            )
        }, indent=2))
        return

    if args.real:
        frame, stats, texts = build_real_pairs(
            train / "train_source1.tsv",
            train / "train_source2.tsv",
            train / "train_source3.tsv",
            train / "train_ground_truth.tsv",
            seed=args.seed,
        )
        if frame is None:
            stats_path = PROJECT_ROOT / "data" / "processed" / "training_pairs_real.stopped.json"
            save_stats(stats, stats_path)
            _log(f"stopped before save; wrote {stats_path}")
            _log(json.dumps(stats, indent=2))
            return
        output = Path(args.output) if args.output else PROJECT_ROOT / "data" / "processed" / "training_pairs_real.parquet"
        written = save_training_pairs(frame, output)
        stats["saved_path"] = written
        stats["examples_positive"] = example_rows(frame.assign(negative_role="positive"), texts, "positive", 5)
        stats["examples_negative"] = example_rows(
            frame.loc[frame["is_true_match"] == 0].assign(negative_role="hard_negative"),
            texts, "hard_negative", 5,
        )
        save_stats(stats, PROJECT_ROOT / "data" / "processed" / "training_pairs_real.stats.json")
        _log(f"wrote {written}")
        _log(json.dumps({key: stats[key] for key in stats if not str(key).startswith("examples")}, indent=2))
        return

    output = Path(args.output) if args.output else PROJECT_ROOT / "data" / "processed" / "training_pairs.parquet"
    builder = build_bounded_training_pairs if args.bounded else build_training_pairs
    build_kwargs = {"seed": args.seed}
    if not args.bounded:
        build_kwargs["bucket_quota"] = {"0": args.quota_0, "1": args.quota_1, "2plus": args.quota_2plus}
        build_kwargs["tfidf_index_cap"] = None if args.tfidf_cap < 0 else args.tfidf_cap
    frame, stats, _context = builder(
        train / "train_source1.tsv",
        train / "train_source2.tsv",
        train / "train_source3.tsv",
        train / "train_ground_truth.tsv",
        **build_kwargs,
    )
    written = save_training_pairs(frame, output)
    stats_path = Path(written).with_suffix("").with_suffix(".stats.json")
    if written.endswith(".gz"):
        stats_path = Path(written).with_name("training_pairs.stats.json")
    save_stats(stats, stats_path)

    positive_examples = example_rows(frame, {}, "positive", 8)
    hard_examples = example_rows(frame, {}, "hard_negative", 8)
    example_ids = set()
    for row in positive_examples + hard_examples:
        example_ids.add(row["source1_entity_id"])
        example_ids.add(row["candidate_entity_id"])
    texts = lookup_texts(
        [train / "train_source1.tsv", train / "train_source2.tsv", train / "train_source3.tsv"],
        example_ids,
    )
    positive_examples = example_rows(frame, texts, "positive", 8)
    hard_examples = example_rows(frame, texts, "hard_negative", 8)
    stats["examples_positive"] = positive_examples
    stats["examples_hard_negative"] = hard_examples
    stats["saved_path"] = written
    save_stats(stats, stats_path)
    _log(f"wrote {written}")
    _log(f"wrote {stats_path}")
    _log(json.dumps({key: stats[key] for key in (
        "runtime_seconds", "queried_s1", "blocked_pairs", "saved_rows",
        "positive_pairs", "negative_pairs", "hard_negatives", "ordinary_negatives",
        "positive_rate", "no_match_s1_queried", "no_match_s1_with_saved_pairs",
        "true_links_in_ground_truth", "true_links_retrieved", "saved_country_s1",
        "saved_match_hist", "candidate_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()

"""
blocking.py
High-recall multi-pass blocking and candidate generation.

Reads the normalized multi-view columns produced by normalization.py
(or builds them with the same functions when a frame is still raw) and
returns Source-1 → Source-2/Source-3 candidate pairs.

Country is an open set. Records are compared only inside the same
country label, whatever that label is — nothing here is restricted to
US or India, so France (and any later country) is handled the same way.

Passes (unioned, then deduplicated):
  1. exact_name       business_name_basic
  2. exact_core       business_name_core
  3. exact_sorted     business_name_sorted (word-order swaps)
  4. translit_name    business_name_transliterated (Unidecode view)
  5. name_token       rare informative name tokens, including the
                      transliterated token view
  6. address_token    rare informative address tokens
  7. address_number   premises-number + one distinctive address token
  8. address_postal   5/6-digit postal candidates from normalization
  9. char_tfidf       character n-gram TF-IDF, top-K per query,
                      hashed features + chunked sparse matmul

The output is a pair table. A submission-shaped candidate_pairs table
(one row per Source-1 id) is produced by to_candidate_pairs_frame().

Memory notes for Colab-sized inputs (millions of rows):
  * One country is indexed at a time, then released.
  * Source-1 is queried in chunks. Each chunk is unioned, deduplicated,
    written, and dropped before the next chunk starts. Candidate rows
    are not accumulated for the whole Source-1 file.
  * Posting lists that exceed a document-frequency cap are dropped
    instead of stored, so tokens like "street" never become a giant list.
  * Character TF-IDF never materialises a Source-1 × Source-2/3 dense
    matrix. Features live in a fixed hash space; common n-grams are
    zeroed; the index is scored in chunks, and queries are scored a
    chunk at a time.
  * Candidate recall is the objective. Caps exist only to keep a single
    posting list or a single query from exhausting RAM.
"""
from __future__ import annotations

import argparse
import array
import gc
import heapq
import math
import pickle
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .normalization import CORPORATE_SUFFIXES, build_address_views, build_name_views
from .text_utils import tokenize

# ---------------------------------------------------------------------------
# Rule names (stable order used in the `rules` column)
# ---------------------------------------------------------------------------

RULE_EXACT_NAME = "exact_name"
RULE_EXACT_CORE = "exact_core"
RULE_EXACT_SORTED = "exact_sorted"
RULE_TRANSLIT = "translit_name"
RULE_NAME_TOKEN = "name_token"
RULE_ADDRESS_TOKEN = "address_token"
RULE_ADDRESS_NUMBER = "address_number"
RULE_ADDRESS_POSTAL = "address_postal"
RULE_CHAR_TFIDF = "char_tfidf"

_RULES = (
    (1 << 0, RULE_EXACT_NAME),
    (1 << 1, RULE_EXACT_CORE),
    (1 << 2, RULE_EXACT_SORTED),
    (1 << 3, RULE_TRANSLIT),
    (1 << 4, RULE_NAME_TOKEN),
    (1 << 5, RULE_ADDRESS_TOKEN),
    (1 << 6, RULE_ADDRESS_NUMBER),
    (1 << 7, RULE_ADDRESS_POSTAL),
    (1 << 8, RULE_CHAR_TFIDF),
)

PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "rules"]
SUBMISSION_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

_OVERFLOW = ()  # sentinel: this key's posting list was too large to keep

_FUNCTION_STOPWORDS = frozenset({
    "the", "and", "of", "for", "com", "www",
})


def default_stopwords() -> frozenset:
    """Corporate suffixes from normalization.py, plus a few function words.

    Street generics ("road", "nagar", "street", ...) are intentionally NOT
    listed. They are dropped only when their document frequency exceeds the
    configured cap, which scales with whatever countries are in the file.
    """
    return frozenset(CORPORATE_SUFFIXES) | _FUNCTION_STOPWORDS


@dataclass
class BlockingConfig:
    """Knobs for the recall/RAM trade-off. Defaults were chosen against the
    training files: exact-name fan-out of true matches is small, while a
    handful of address/name tokens are extremely common and must be capped.
    """

    # Exact-string buckets larger than this are skipped (the token and
    # TF-IDF passes still have a chance at those rows).
    # On train, the fan-out of an exact key that hits a true match has
    # p99 ≈ 430 and max ≈ 1300, so 2000 keeps those buckets.
    exact_max_fanout: int = 2000

    min_name_token_len: int = 3
    min_addr_token_len: int = 4
    # Indexed only while the posting list stays within this cap.
    # Measured on train true pairs (document frequency inside the country):
    #   name OR address token df <= 5_000  covers ~90% of true links
    #   name OR address token df <= 20_000 covers ~97%
    # Caps sit a bit higher so city/locality tokens (often df 5k–40k)
    # still generate candidates. Common generics (road, nagar, delhi, …)
    # fall above the cap and are not stored.
    name_token_max_df: int = 10_000
    addr_token_max_df: int = 40_000
    # Per query, walk rarest tokens first. The budget is a candidate
    # ceiling for that family, not a second df cap: a posting is taken
    # in full or not at all.
    name_token_budget: int = 20_000
    addr_token_budget: int = 60_000
    # Tokens at or under this df are always taken (cheap, usually distinctive).
    always_include_df: int = 50
    max_name_tokens_per_query: int = 10
    max_addr_tokens_per_query: int = 12

    # One (premises number, distinctive address token) key per record.
    enable_address_number: bool = True
    address_number_max_df: int = 80
    # Postal codes are weak on this dataset (many rows have none) but cheap.
    postal_max_df: int = 400

    enable_tfidf: bool = True
    tfidf_top_k: int = 20
    tfidf_ngram_range: tuple = (3, 4)
    tfidf_n_features: int = 1 << 20
    tfidf_max_df: int = 4000
    tfidf_min_df: int = 1
    tfidf_min_score: float = 0.08
    tfidf_index_chunk: int = 200_000
    tfidf_query_batch: int = 512
    # Source-1 rows scored and written together. The full-file path never
    # keeps more than one chunk of query results resident.
    query_chunk_size: int = 5000

    stopwords: frozenset = field(default_factory=default_stopwords)
    verbose: bool = False


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _blank(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    return list(value)


class _Record:
    __slots__ = (
        "eid", "country", "basic", "core", "sorted_name", "trans",
        "name_toks", "addr_toks", "numbers", "postals", "number_key",
    )

    def __init__(
        self, eid, country, basic, core, sorted_name, trans,
        name_toks, addr_toks, numbers, postals, number_key,
    ):
        self.eid = eid
        self.country = country
        self.basic = basic
        self.core = core
        self.sorted_name = sorted_name
        self.trans = trans
        self.name_toks = name_toks
        self.addr_toks = addr_toks
        self.numbers = numbers
        self.postals = postals
        self.number_key = number_key


def _informative(tokens, *, min_len: int, stopwords: frozenset) -> tuple:
    seen = set()
    kept = []
    for token in tokens:
        if not token or len(token) < min_len:
            continue
        if token in stopwords or token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        kept.append(token)
    return tuple(kept)


def _primary_number(numbers: list) -> str:
    """Prefer a 2–6 digit run (house / plot / street number).

    5- and 6-digit postal codes are also stored separately; keeping them
    eligible here does not hurt, because the key is paired with a token.
    """
    usable = [n for n in numbers if 2 <= len(n) <= 6]
    if not usable:
        return ""
    return max(usable, key=len)


def _make_record_from_views(
    eid, country, name_views, addr_views, translit_name_tokens, translit_addr_tokens, cfg: BlockingConfig,
) -> _Record:
    name_toks = _informative(
        list(name_views["business_name_tokens"]) + list(translit_name_tokens),
        min_len=cfg.min_name_token_len,
        stopwords=cfg.stopwords,
    )
    addr_toks = _informative(
        list(addr_views["business_address_tokens"]) + list(translit_addr_tokens),
        min_len=cfg.min_addr_token_len,
        stopwords=cfg.stopwords,
    )
    numbers = [n for n in addr_views["address_numbers"] if n]
    postals = []
    seen_postal = set()
    for postal in addr_views["address_postal_candidates"]:
        if postal and postal not in seen_postal:
            seen_postal.add(postal)
            postals.append(postal)
    number = _primary_number(numbers)
    number_key = None
    if cfg.enable_address_number and number and addr_toks:
        distinctive = max(addr_toks, key=len)
        number_key = (number, distinctive)
    return _Record(
        eid=_blank(eid),
        country=_blank(country),
        basic=name_views["business_name_basic"] or "",
        core=name_views["business_name_core"] or "",
        sorted_name=name_views["business_name_sorted"] or "",
        trans=name_views["business_name_transliterated"] or "",
        name_toks=name_toks,
        addr_toks=addr_toks,
        numbers=tuple(numbers),
        postals=tuple(postals),
        number_key=number_key,
    )


def make_record(entity_id, business_name, business_address, country, cfg: BlockingConfig | None = None) -> _Record:
    """Build one blocking record from raw fields via normalization.py."""
    cfg = cfg or BlockingConfig()
    name_views = build_name_views(business_name)
    addr_views = build_address_views(business_address)
    return _make_record_from_views(
        entity_id,
        country,
        name_views,
        addr_views,
        tokenize(name_views["business_name_transliterated"]),
        tokenize(addr_views["business_address_transliterated"]),
        cfg,
    )


def _is_target_id(entity_id: str) -> bool:
    return entity_id.startswith("S2-") or entity_id.startswith("S3-")


_NORM_COLUMNS = (
    "business_name_basic",
    "business_name_core",
    "business_name_sorted",
    "business_name_transliterated",
    "business_name_tokens",
    "business_address_tokens",
    "business_address_transliterated",
    "address_numbers",
    "address_postal_candidates",
)


def _records_from_normalized_frame(df: pd.DataFrame, cfg: BlockingConfig) -> list:
    records = []
    cols = list(_NORM_COLUMNS) + ["entity_id", "country"]
    for row in df[cols].itertuples(index=False):
        name_views = {
            "business_name_basic": _blank(row.business_name_basic),
            "business_name_core": _blank(row.business_name_core),
            "business_name_sorted": _blank(row.business_name_sorted),
            "business_name_transliterated": _blank(row.business_name_transliterated),
            "business_name_tokens": _as_list(row.business_name_tokens),
        }
        addr_views = {
            "business_address_tokens": _as_list(row.business_address_tokens),
            "address_numbers": _as_list(row.address_numbers),
            "address_postal_candidates": _as_list(row.address_postal_candidates),
        }
        records.append(_make_record_from_views(
            row.entity_id,
            row.country,
            name_views,
            addr_views,
            tokenize(name_views["business_name_transliterated"]),
            tokenize(_blank(row.business_address_transliterated)),
            cfg,
        ))
    return records


def _records_from_raw_frame(df: pd.DataFrame, cfg: BlockingConfig) -> list:
    records = []
    for eid, name, addr, country in zip(
        df["entity_id"], df["business_name"], df["business_address"], df["country"],
    ):
        records.append(make_record(eid, name, addr, country, cfg))
    return records


def records_from_frame(df: pd.DataFrame, cfg: BlockingConfig | None = None) -> list:
    """Use Harshita's normalized columns when they are already present."""
    cfg = cfg or BlockingConfig()
    missing = [c for c in ("entity_id", "country") if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    if all(col in df.columns for col in _NORM_COLUMNS):
        return _records_from_normalized_frame(df, cfg)
    missing_raw = [c for c in ("business_name", "business_address") if c not in df.columns]
    if missing_raw:
        raise ValueError(
            "frame has neither the normalized view columns nor raw "
            f"business_name/business_address (missing {missing_raw})"
        )
    return _records_from_raw_frame(df, cfg)


# ---------------------------------------------------------------------------
# Inverted indexes
# ---------------------------------------------------------------------------

class _Index:
    __slots__ = (
        "exact_name", "exact_core", "exact_sorted", "translit_name",
        "name_token", "address_token", "address_number", "address_postal",
        "overflow",
    )

    def __init__(self):
        self.exact_name = {}
        self.exact_core = {}
        self.exact_sorted = {}
        self.translit_name = {}
        self.name_token = {}
        self.address_token = {}
        self.address_number = {}
        self.address_postal = {}
        self.overflow = {}


def _accumulate(buckets: dict, key, eid: str, cap: int, overflow_counter: list):
    """Append `eid` under `key` until `cap` is exceeded, then drop the key.

    Dropping the whole key (instead of keeping an arbitrary prefix) avoids
    a systematic bias toward whichever rows were seen first.
    """
    if not key:
        return
    current = buckets.get(key)
    if current is _OVERFLOW:
        return
    if current is None:
        buckets[key] = [eid]
        return
    if len(current) + 1 > cap:
        buckets[key] = _OVERFLOW
        overflow_counter[0] += 1
        return
    current.append(eid)


def _freeze(buckets: dict) -> dict:
    frozen = {}
    for key, value in buckets.items():
        if value is _OVERFLOW or not value:
            continue
        frozen[key] = tuple(value)
    return frozen


def _build_index(records: list, cfg: BlockingConfig) -> _Index:
    index = _Index()
    exact_name, exact_core, exact_sorted, translit = {}, {}, {}, {}
    name_token, addr_token = {}, {}
    number_keys, postal_keys = {}, {}
    overflow = defaultdict(lambda: [0])

    for rec in records:
        if not _is_target_id(rec.eid):
            continue
        _accumulate(exact_name, rec.basic, rec.eid, cfg.exact_max_fanout, overflow["exact_name"])
        _accumulate(exact_core, rec.core, rec.eid, cfg.exact_max_fanout, overflow["exact_core"])
        _accumulate(exact_sorted, rec.sorted_name, rec.eid, cfg.exact_max_fanout, overflow["exact_sorted"])
        _accumulate(translit, rec.trans, rec.eid, cfg.exact_max_fanout, overflow["translit_name"])
        for token in rec.name_toks:
            _accumulate(name_token, token, rec.eid, cfg.name_token_max_df, overflow["name_token"])
        for token in rec.addr_toks:
            _accumulate(addr_token, token, rec.eid, cfg.addr_token_max_df, overflow["address_token"])
        if rec.number_key is not None:
            _accumulate(
                number_keys, rec.number_key, rec.eid,
                cfg.address_number_max_df, overflow["address_number"],
            )
        for postal in rec.postals:
            _accumulate(postal_keys, postal, rec.eid, cfg.postal_max_df, overflow["address_postal"])

    index.exact_name = _freeze(exact_name)
    index.exact_core = _freeze(exact_core)
    index.exact_sorted = _freeze(exact_sorted)
    index.translit_name = _freeze(translit)
    index.name_token = _freeze(name_token)
    index.address_token = _freeze(addr_token)
    index.address_number = _freeze(number_keys)
    index.address_postal = _freeze(postal_keys)
    index.overflow = {name: counter[0] for name, counter in overflow.items()}
    return index


def _mask_to_rules(mask: int) -> str:
    return "|".join(name for bit, name in _RULES if mask & bit)


def _add(cands: dict, ids, bit: int, *, budget: int | None, force: bool) -> None:
    if not ids:
        return
    over = budget is not None and not force and len(cands) >= budget
    for eid in ids:
        previous = cands.get(eid)
        if previous is None:
            if over or (budget is not None and not force and len(cands) >= budget):
                continue
            cands[eid] = bit
        else:
            cands[eid] = previous | bit


def _selected_postings(tokens, index_map, *, max_tokens, budget, always_df):
    """Rarest tokens first. Each selected posting is consumed in full."""
    ranked = []
    for token in tokens:
        posting = index_map.get(token)
        if not posting:
            continue
        ranked.append((len(posting), posting))
    ranked.sort(key=lambda item: item[0])
    chosen = []
    running = 0
    for df, posting in ranked[:max_tokens]:
        if df <= always_df or running + df <= budget or running == 0:
            chosen.append(posting)
            running += df
            if running > budget and df > always_df:
                break
        else:
            break
    return chosen


def _query_record(rec: _Record, index: _Index, tfidf_ids, cfg: BlockingConfig) -> dict:
    cands = {}
    _add(cands, index.exact_name.get(rec.basic), 1 << 0, budget=None, force=True)
    _add(cands, index.exact_core.get(rec.core), 1 << 1, budget=None, force=True)
    _add(cands, index.exact_sorted.get(rec.sorted_name), 1 << 2, budget=None, force=True)
    _add(cands, index.translit_name.get(rec.trans), 1 << 3, budget=None, force=True)

    for posting in _selected_postings(
        rec.name_toks, index.name_token,
        max_tokens=cfg.max_name_tokens_per_query,
        budget=cfg.name_token_budget,
        always_df=cfg.always_include_df,
    ):
        _add(cands, posting, 1 << 4, budget=None, force=True)
    for posting in _selected_postings(
        rec.addr_toks, index.address_token,
        max_tokens=cfg.max_addr_tokens_per_query,
        budget=cfg.addr_token_budget,
        always_df=cfg.always_include_df,
    ):
        _add(cands, posting, 1 << 5, budget=None, force=True)

    if rec.number_key is not None:
        _add(cands, index.address_number.get(rec.number_key), 1 << 6, budget=None, force=True)
    for postal in rec.postals:
        _add(cands, index.address_postal.get(postal), 1 << 7, budget=None, force=True)

    if tfidf_ids:
        _add(cands, tfidf_ids, 1 << 8, budget=None, force=True)

    self_id = rec.eid
    if self_id in cands:
        del cands[self_id]
    return cands


# ---------------------------------------------------------------------------
# Character TF-IDF (hashed n-grams, chunked sparse top-K)
# ---------------------------------------------------------------------------

def _l2_rows(matrix):
    return normalize(matrix, norm="l2", copy=False)


def _transform_hashed(texts, vectorizer, idf: np.ndarray):
    matrix = vectorizer.transform(texts).tocsr().copy()
    if matrix.nnz:
        matrix.data = np.log1p(matrix.data) * idf[matrix.indices]
        matrix.eliminate_zeros()
    return _l2_rows(matrix)


def _update_topk(heaps, scores, cand_ids, k: int, min_score: float) -> None:
    scores = scores.tocsr()
    indptr = scores.indptr
    indices = scores.indices
    data = scores.data
    for row in range(scores.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if start == end:
            continue
        heap = heaps[row]
        for col, value in zip(indices[start:end], data[start:end]):
            score = float(value)
            if score < min_score:
                continue
            cand = cand_ids[int(col)]
            if len(heap) < k:
                heapq.heappush(heap, (score, cand))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, cand))


def _fit_char_tfidf(index_ids: list, index_texts: list, cfg: BlockingConfig):
    """Document-frequency and IDF for one country. None when TF-IDF is off.

    The returned object keeps the target ids and transliterated names so
    later query chunks can be scored without refitting. It does not store
    a dense similarity matrix.
    """
    if not cfg.enable_tfidf or len(index_texts) < 2:
        return None

    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.tfidf_ngram_range,
        n_features=cfg.tfidf_n_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
        dtype=np.float32,
    )
    doc_freq = np.zeros(cfg.tfidf_n_features, dtype=np.int32)
    chunk = max(1, cfg.tfidf_index_chunk)
    for start in range(0, len(index_texts), chunk):
        block = vectorizer.transform(index_texts[start:start + chunk])
        counted = np.bincount(block.indices, minlength=cfg.tfidf_n_features)
        doc_freq += counted.astype(np.int32, copy=False)
        del block

    n_docs = len(index_texts)
    max_df = cfg.tfidf_max_df
    if max_df >= n_docs:
        max_df = n_docs
    min_df = max(1, cfg.tfidf_min_df)
    idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
    drop = (doc_freq < min_df) | (doc_freq > max_df)
    idf[drop] = 0.0
    idf = idf.astype(np.float32, copy=False)
    return {
        "vectorizer": vectorizer,
        "idf": idf,
        "index_ids": index_ids,
        "index_texts": index_texts,
        "chunk": chunk,
    }


def _search_char_tfidf(prepared, query_texts: list, cfg: BlockingConfig, *, log: bool = True) -> list:
    """Top-K hits for this query slice. Aligned with `query_texts`."""
    if prepared is None or not query_texts:
        return [() for _ in query_texts]

    vectorizer = prepared["vectorizer"]
    idf = prepared["idf"]
    index_ids = prepared["index_ids"]
    index_texts = prepared["index_texts"]
    chunk = prepared["chunk"]
    heaps = [[] for _ in query_texts]
    q_batch = max(1, cfg.tfidf_query_batch)
    k = max(1, cfg.tfidf_top_k)
    n_chunks = math.ceil(len(index_texts) / chunk)
    for chunk_i, start in enumerate(range(0, len(index_texts), chunk), start=1):
        if log and cfg.verbose:
            _log(cfg, f"  char tf-idf chunk {chunk_i}/{n_chunks}")
        block_ids = index_ids[start:start + chunk]
        block_matrix = _transform_hashed(index_texts[start:start + chunk], vectorizer, idf)
        for q_start in range(0, len(query_texts), q_batch):
            query_matrix = _transform_hashed(query_texts[q_start:q_start + q_batch], vectorizer, idf)
            scores = query_matrix.dot(block_matrix.T)
            _update_topk(
                heaps[q_start:q_start + q_batch],
                scores,
                block_ids,
                k,
                cfg.tfidf_min_score,
            )
            del query_matrix, scores
        del block_matrix

    neighbours = []
    for heap in heaps:
        ordered = sorted(heap, key=lambda item: item[0], reverse=True)
        neighbours.append(tuple(cand for _, cand in ordered))
    return neighbours


def _char_tfidf_search(index_ids: list, index_texts: list, query_texts: list, cfg: BlockingConfig) -> list:
    """Top-K character TF-IDF hits for each query text. Aligned with `query_texts`."""
    if not query_texts:
        return []
    prepared = _fit_char_tfidf(index_ids, index_texts, cfg)
    return _search_char_tfidf(prepared, query_texts, cfg)


def char_tfidf_candidates(index_records: list, query_records: list, cfg: BlockingConfig) -> list:
    """Top-K character-TF-IDF neighbours for each query, within one country.

    Returns a list aligned with `query_records`. Each entry is a tuple of
    candidate entity ids (Source 2 / Source 3 only).
    """
    index_ids = []
    index_texts = []
    for rec in index_records:
        if rec.trans and _is_target_id(rec.eid):
            index_ids.append(rec.eid)
            index_texts.append(rec.trans)
    query_texts = [rec.trans for rec in query_records]
    return _char_tfidf_search(index_ids, index_texts, query_texts, cfg)


# ---------------------------------------------------------------------------
# Public generation API
# ---------------------------------------------------------------------------

def _log(cfg: BlockingConfig, message: str) -> None:
    if cfg.verbose:
        print(message, flush=True)


def _group_countries(records: list) -> dict:
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec.country].append(rec)
    return grouped


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """Union of every blocking pass, one row per candidate pair.

    Columns: source1_entity_id, candidate_entity_id, rules.
    `rules` lists the passes that proposed the pair, separated by `|`.
    candidate_entity_id is always a Source-2 or Source-3 id.

    Country labels are taken from the data. Each Source-1 row is compared
    only with Source-2/3 rows that carry the same country string.
    """
    cfg = config or BlockingConfig()
    queries = records_from_frame(source1, cfg)
    index_records = records_from_frame(source2, cfg)
    index_records.extend(records_from_frame(source3, cfg))
    return _candidates_from_records(queries, index_records, cfg)


def _candidates_from_records(queries: list, index_records: list, cfg: BlockingConfig) -> pd.DataFrame:
    by_query = _group_countries(queries)
    by_index = _group_countries(index_records)
    countries = sorted(set(by_query) | set(by_index))

    source_ids = []
    cand_ids = []
    rules = []
    search_same = 0
    n_index_kept = 0
    # Every queried id, including ones that receive zero candidates.
    # Dropping those would make recall look better than it is.
    queried_ids = [rec.eid for rec in queries]

    for country in countries:
        q_recs = by_query.get(country, [])
        i_recs = [rec for rec in by_index.get(country, []) if _is_target_id(rec.eid)]
        n_index_kept += len(i_recs)
        search_same += len(q_recs) * len(i_recs)
        if not q_recs or not i_recs:
            _log(cfg, f"[{country or '∅'}] queries={len(q_recs)} index={len(i_recs)} skipped")
            continue
        _log(cfg, f"[{country}] indexing {len(i_recs)} targets for {len(q_recs)} queries")
        index = _build_index(i_recs, cfg)
        _log(cfg, f"[{country}] overflow dropped keys: {index.overflow}")
        target_ids = []
        target_texts = []
        for rec in i_recs:
            if rec.trans and _is_target_id(rec.eid):
                target_ids.append(rec.eid)
                target_texts.append(rec.trans)
        prepared = _fit_char_tfidf(target_ids, target_texts, cfg)
        del target_ids, target_texts
        by_index.pop(country, None)
        del i_recs
        chunk_size = _chunk_size(cfg)
        for start in range(0, len(q_recs), chunk_size):
            chunk = q_recs[start:start + chunk_size]
            tfidf = _search_char_tfidf(
                prepared, [rec.trans for rec in chunk], cfg, log=False,
            )
            for rec, tfidf_ids in zip(chunk, tfidf):
                found = _query_record(rec, index, tfidf_ids, cfg)
                if not found:
                    continue
                rule_strings = {mask: _mask_to_rules(mask) for mask in set(found.values())}
                for cand, mask in found.items():
                    source_ids.append(rec.eid)
                    cand_ids.append(cand)
                    rules.append(rule_strings[mask])
            del chunk, tfidf
        del index, prepared, q_recs
        gc.collect()
        _log(cfg, f"[{country}] pairs so far {len(source_ids)}")

    frame = pd.DataFrame({
        "source1_entity_id": source_ids,
        "candidate_entity_id": cand_ids,
        "rules": rules,
    })
    n_s1 = len(queries)
    n_index = n_index_kept
    frame.attrs["blocking_stats"] = {
        "n_s1": n_s1,
        "n_index": n_index,
        "n_candidate_pairs": int(len(frame)),
        "search_space_same_country": int(search_same),
        "search_space_full_cartesian": int(n_s1 * n_index),
    }
    frame.attrs["queried_ids"] = queried_ids
    return frame


def to_candidate_pairs_frame(
    candidates: pd.DataFrame,
    source1_ids,
) -> pd.DataFrame:
    """Collapse pairs to the challenge candidate_pairs.tsv shape.

    One row per Source-1 id in `source1_ids` (order preserved). Entities
    with no candidates get an empty candidate_entity_ids cell. IDs inside
    a cell are deduplicated and sorted. Only the pair table's candidate
    ids are written; Source-1 ids are never emitted as candidates.
    """
    grouped = {}
    if len(candidates):
        for source_id, cand_id in zip(
            candidates["source1_entity_id"], candidates["candidate_entity_id"],
        ):
            if not _is_target_id(cand_id):
                continue
            grouped.setdefault(source_id, set()).add(cand_id)

    rows = []
    for source_id in source1_ids:
        ids = grouped.get(source_id)
        cell = ",".join(sorted(ids)) if ids else ""
        rows.append((source_id, cell))
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def write_candidate_pairs_tsv(frame: pd.DataFrame, path) -> None:
    """Write a candidate_pairs frame as UTF-8 TSV (no index column)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Evaluation against train_ground_truth.tsv
# ---------------------------------------------------------------------------

def _parse_id_list(cell) -> list:
    text = _blank(cell)
    if not text:
        return []
    return [part for part in text.split(",") if part]


def ground_truth_map(ground_truth: pd.DataFrame, source1_ids=None) -> dict:
    """{source1_entity_id: set(matched S2/S3 ids)} for the queried ids."""
    wanted = None if source1_ids is None else set(source1_ids)
    mapping = {}
    for source_id, cell in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        if wanted is not None and source_id not in wanted:
            continue
        mapping[source_id] = {mid for mid in _parse_id_list(cell) if _is_target_id(mid)}
    if wanted is not None:
        for source_id in wanted:
            mapping.setdefault(source_id, set())
    return mapping


def evaluate_blocking(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source1_ids=None,
) -> dict:
    """Candidate-recall report for the pairs in `candidates`.

    Pair recall is micro-averaged over true Source-2/3 links of the queried
    Source-1 ids. A link counts as found only when that exact id is in the
    candidate set. Per-pass recall counts a link when that pass is among
    the rules that generated the pair (passes overlap, so they do not sum
    to the union).

    Also reports S2 vs S3 recall, candidate-count distribution, and the
    reduction ratio against both the same-country search space and the full
    cartesian product. Search-space figures are read from
    `candidates.attrs["blocking_stats"]` when present.
    """
    if source1_ids is None:
        source1_ids = list(dict.fromkeys(ground_truth["source1_entity_id"].tolist()))
    else:
        source1_ids = list(source1_ids)

    truth = ground_truth_map(ground_truth, source1_ids)
    found = defaultdict(dict)  # s1 -> cand -> rules string
    if len(candidates):
        for source_id, cand_id, rule_text in zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
            candidates["rules"],
        ):
            if source_id in truth and _is_target_id(cand_id):
                found[source_id][cand_id] = rule_text

    rule_names = [name for _, name in _RULES]
    hits = {name: 0 for name in rule_names}
    exclusive = {name: 0 for name in rule_names}
    union_hits = 0
    s2_true = s2_hit = s3_true = s3_hit = 0
    true_links = 0
    missed = []

    per_s1_recall = []
    for source_id in source1_ids:
        true_ids = truth.get(source_id, set())
        if not true_ids:
            continue
        got = found.get(source_id, {})
        n_hit = len(true_ids & got.keys())
        per_s1_recall.append(n_hit / len(true_ids))
        for mid in true_ids:
            true_links += 1
            is_s2 = mid.startswith("S2-")
            if is_s2:
                s2_true += 1
            else:
                s3_true += 1
            rule_text = got.get(mid)
            if not rule_text:
                if len(missed) < 15:
                    missed.append((source_id, mid))
                continue
            union_hits += 1
            if is_s2:
                s2_hit += 1
            else:
                s3_hit += 1
            parts = set(rule_text.split("|"))
            for name in parts:
                if name in hits:
                    hits[name] += 1
            if len(parts) == 1:
                only = next(iter(parts))
                if only in exclusive:
                    exclusive[only] += 1

    cand_counts = []
    for source_id in source1_ids:
        cand_counts.append(len(found.get(source_id, {})))
    count_arr = np.asarray(cand_counts, dtype=np.int64) if cand_counts else np.zeros(1, dtype=np.int64)

    stats = candidates.attrs.get("blocking_stats", {}) if hasattr(candidates, "attrs") else {}
    n_pairs = int(stats.get("n_candidate_pairs", len(candidates)))
    same = int(stats.get("search_space_same_country", 0))
    full = int(stats.get("search_space_full_cartesian", 0))

    def _ratio(space):
        if not space or not n_pairs:
            return None
        return space / n_pairs

    def _safe(numer, denom):
        return (numer / denom) if denom else None

    summary = {
        "n_s1": len(source1_ids),
        "n_s1_with_truth": int(sum(1 for s in source1_ids if truth.get(s))),
        "n_true_links": true_links,
        "n_true_links_found": union_hits,
        "recall_union": _safe(union_hits, true_links),
        "recall_macro_s1": float(np.mean(per_s1_recall)) if per_s1_recall else None,
        "recall_by_rule": {name: _safe(hits[name], true_links) for name in rule_names},
        "exclusive_recall_by_rule": {name: _safe(exclusive[name], true_links) for name in rule_names},
        "recall_s2": _safe(s2_hit, s2_true),
        "recall_s3": _safe(s3_hit, s3_true),
        "n_true_s2": s2_true,
        "n_true_s3": s3_true,
        "n_candidate_pairs": n_pairs,
        "candidates_per_s1": {
            "mean": float(count_arr.mean()),
            "median": float(np.median(count_arr)),
            "p95": float(np.percentile(count_arr, 95)),
            "max": int(count_arr.max()),
            "zeros": int((count_arr == 0).sum()),
        },
        "search_space_same_country": same,
        "search_space_full_cartesian": full,
        "reduction_ratio_vs_same_country": _ratio(same),
        "reduction_ratio_vs_full_cartesian": _ratio(full),
        "missed_examples": missed,
    }
    return summary


def format_blocking_report(summary: dict) -> str:
    """Plain-text rendering of evaluate_blocking()."""
    def pct(value):
        return "n/a" if value is None else f"{value:.2%}"

    def num(value):
        return "n/a" if value is None else f"{value:,.1f}"

    counts = summary["candidates_per_s1"]
    lines = [
        "Blocking candidate-recall",
        f"  Source-1 queried:          {summary['n_s1']:,}",
        f"  Source-1 with a true link: {summary['n_s1_with_truth']:,}",
        f"  True links:                {summary['n_true_links']:,}",
        f"  True links found:          {summary['n_true_links_found']:,}",
        f"  Union recall (micro):      {pct(summary['recall_union'])}",
        f"  Union recall (macro S1):   {pct(summary['recall_macro_s1'])}",
        f"  S2 recall:                 {pct(summary['recall_s2'])}  (n={summary['n_true_s2']:,})",
        f"  S3 recall:                 {pct(summary['recall_s3'])}  (n={summary['n_true_s3']:,})",
        "  Recall by pass (a link can count in several passes):",
    ]
    for name, value in summary["recall_by_rule"].items():
        exclusive = summary["exclusive_recall_by_rule"][name]
        lines.append(f"    {name:18} {pct(value):>8}   only-this-pass {pct(exclusive)}")
    lines.extend([
        f"  Candidate pairs:           {summary['n_candidate_pairs']:,}",
        "  Candidates per Source-1:   "
        f"mean {counts['mean']:.1f}  median {counts['median']:.0f}  "
        f"p95 {counts['p95']:.0f}  max {counts['max']}  "
        f"with-none {counts['zeros']:,}",
        f"  Same-country search space: {summary['search_space_same_country']:,}",
        f"  Full cartesian space:      {summary['search_space_full_cartesian']:,}",
        f"  Reduction vs same-country: {num(summary['reduction_ratio_vs_same_country'])}x",
        f"  Reduction vs full product: {num(summary['reduction_ratio_vs_full_cartesian'])}x",
    ])
    if summary["missed_examples"]:
        lines.append("  Missed true links (sample):")
        for source_id, mid in summary["missed_examples"]:
            lines.append(f"    {source_id} -> {mid}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path runner (project-relative; no machine-specific paths)
# ---------------------------------------------------------------------------

def resolve_dataset_dir(path=None) -> Path:
    """Find the challenge dataset directory.

    An explicit relative path is resolved against the current working
    directory. With no argument, try the locations the repo already uses:
    config.DATA_DIR, then `student_resource/dataset`, then
    `student_resource copy/dataset` next to the project root.
    """
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate

    from .config import DATA_DIR, PROJECT_ROOT
    options = [
        DATA_DIR,
        PROJECT_ROOT / "student_resource" / "dataset",
        PROJECT_ROOT / "student_resource copy" / "dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "student_resource" / "dataset",
        Path.cwd() / "student_resource copy" / "dataset",
    ]
    for option in options:
        if (option / "train" / "train_source1.tsv").is_file():
            return option
    return DATA_DIR


def _iter_raw_tsv(path: Path):
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "entity_id" not in header:
            raise ValueError(f"unexpected header in {path}: {header!r}")
        for line in handle:
            entity_id, _, rest = line.rstrip("\n").partition("\t")
            name, _, rest = rest.partition("\t")
            address, _, country = rest.partition("\t")
            yield entity_id, name, address, country.strip()


def _load_records(path: Path, countries: set | None, cfg: BlockingConfig, id_allow: set | None = None) -> list:
    records = []
    seen = 0
    for entity_id, name, address, country in _iter_raw_tsv(path):
        seen += 1
        if countries is not None and country not in countries:
            continue
        if id_allow is not None and entity_id not in id_allow:
            continue
        records.append(make_record(entity_id, name, address, country, cfg))
        if cfg.verbose and len(records) % 250_000 == 0:
            _log(cfg, f"  loaded {len(records):,} matching rows from {path.name} (scanned {seen:,})")
    _log(cfg, f"  {path.name}: kept {len(records):,} / scanned {seen:,}")
    return records


def _reservoir_queries(path: Path, countries: set | None, max_rows: int | None, seed: int, cfg: BlockingConfig) -> list:
    if max_rows is None:
        return _load_records(path, countries, cfg)
    rng = __import__("random").Random(seed)
    kept = []
    seen = 0
    for row in _iter_raw_tsv(path):
        if countries is not None and row[3] not in countries:
            continue
        seen += 1
        if len(kept) < max_rows:
            kept.append(row)
        else:
            slot = rng.randrange(seen)
            if slot < max_rows:
                kept[slot] = row
        if cfg.verbose and seen % 250_000 == 0:
            _log(cfg, f"  reservoir scanned {seen:,} {path.name}")
    _log(cfg, f"  {path.name}: sampled {len(kept):,} / eligible {seen:,}")
    return [make_record(*row, cfg) for row in kept]


def _load_truth_map(path: Path) -> dict:
    """All ground-truth links. Evaluation only; not used to add candidates."""
    mapping = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "source1_entity_id" not in header:
            raise ValueError(f"unexpected ground-truth header in {path}: {header!r}")
        for line in handle:
            source_id, _, rest = line.rstrip("\n").partition("\t")
            mapping[source_id] = {mid for mid in _parse_id_list(rest) if _is_target_id(mid)}
    return mapping


def _load_truth_for(path: Path, source1_ids: set) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "source1_entity_id" not in header:
            raise ValueError(f"unexpected ground-truth header in {path}: {header!r}")
        for line in handle:
            source_id, _, rest = line.rstrip("\n").partition("\t")
            if source_id in source1_ids:
                rows.append((source_id, rest))
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def _stream_country_index(paths, country: str, cfg: BlockingConfig):
    """Build one country's inverted index without retaining row objects.

    Source files are scanned once. Each row is normalized, inserted, and
    dropped, so peak memory is the index itself rather than index plus a
    second copy of every record.
    """
    buckets = {
        "exact_name": {},
        "exact_core": {},
        "exact_sorted": {},
        "translit_name": {},
        "name_token": {},
        "address_token": {},
        "address_number": {},
        "address_postal": {},
    }
    overflow = defaultdict(lambda: [0])
    kept = 0
    for path in paths:
        seen = 0
        for entity_id, name, address, row_country in _iter_raw_tsv(path):
            seen += 1
            if row_country != country or not _is_target_id(entity_id):
                continue
            rec = make_record(entity_id, name, address, row_country, cfg)
            _accumulate(buckets["exact_name"], rec.basic, rec.eid, cfg.exact_max_fanout, overflow["exact_name"])
            _accumulate(buckets["exact_core"], rec.core, rec.eid, cfg.exact_max_fanout, overflow["exact_core"])
            _accumulate(buckets["exact_sorted"], rec.sorted_name, rec.eid, cfg.exact_max_fanout, overflow["exact_sorted"])
            _accumulate(buckets["translit_name"], rec.trans, rec.eid, cfg.exact_max_fanout, overflow["translit_name"])
            for token in rec.name_toks:
                _accumulate(buckets["name_token"], token, rec.eid, cfg.name_token_max_df, overflow["name_token"])
            for token in rec.addr_toks:
                _accumulate(buckets["address_token"], token, rec.eid, cfg.addr_token_max_df, overflow["address_token"])
            if rec.number_key is not None:
                _accumulate(
                    buckets["address_number"], rec.number_key, rec.eid,
                    cfg.address_number_max_df, overflow["address_number"],
                )
            for postal in rec.postals:
                _accumulate(
                    buckets["address_postal"], postal, rec.eid,
                    cfg.postal_max_df, overflow["address_postal"],
                )
            kept += 1
            if cfg.verbose and kept % 250_000 == 0:
                _log(cfg, f"  indexed {kept:,} {country} rows (scanning {path.name}, line {seen:,})")
        _log(cfg, f"  scanned {path.name} ({seen:,} lines) for {country}")

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
    del buckets, overflow
    return index, kept


def _stream_country_names(paths, country: str, cfg: BlockingConfig):
    """Second pass: transliterated names only, for character TF-IDF.

    Runs after the token index has been released so the two structures
    are not resident together.
    """
    ids = []
    texts = []
    for path in paths:
        for entity_id, name, _address, row_country in _iter_raw_tsv(path):
            if row_country != country or not _is_target_id(entity_id):
                continue
            text = build_name_views(name)["business_name_transliterated"]
            if text:
                ids.append(entity_id)
                texts.append(text)
            if cfg.verbose and len(ids) % 500_000 == 0:
                _log(cfg, f"  tf-idf names collected {len(ids):,}")
    return ids, texts


def _pairs_frame(source_ids, cand_ids, rules, queried_ids, n_index, search_same) -> pd.DataFrame:
    frame = pd.DataFrame({
        "source1_entity_id": source_ids,
        "candidate_entity_id": cand_ids,
        "rules": rules,
    })
    n_s1 = len(queried_ids)
    frame.attrs["blocking_stats"] = {
        "n_s1": n_s1,
        "n_index": int(n_index),
        "n_candidate_pairs": int(len(frame)),
        "search_space_same_country": int(search_same),
        "search_space_full_cartesian": int(n_s1 * n_index),
    }
    frame.attrs["queried_ids"] = queried_ids
    return frame


class _StreamStats:
    """Compact counters. Candidate rows themselves are not retained."""

    def __init__(self, collect_ids: bool):
        self.n_s1 = 0
        self.n_index = 0
        self.n_pairs = 0
        self.search_same = 0
        self.counts = array.array("Q")
        self.collect_ids = collect_ids
        self.queried_ids = [] if collect_ids else None
        self.found_true = {}

    def add_query(self, eid: str, found: dict, truth) -> None:
        self.n_s1 += 1
        self.n_pairs += len(found)
        self.counts.append(len(found))
        if not self.collect_ids:
            return
        self.queried_ids.append(eid)
        if not truth:
            return
        true_ids = truth.get(eid)
        if not true_ids:
            return
        for cand, mask in found.items():
            if cand in true_ids:
                self.found_true[(eid, cand)] = _mask_to_rules(mask)


def _chunk_size(cfg: BlockingConfig) -> int:
    size = int(cfg.query_chunk_size)
    if size < 1:
        raise ValueError(f"query_chunk_size must be >= 1, got {size}")
    return size


def _countries_in_source1(path: Path, countries: set | None) -> list:
    found = set()
    seen = 0
    for _entity_id, _name, _address, country in _iter_raw_tsv(path):
        seen += 1
        if countries is not None and country not in countries:
            continue
        found.add(country)
    return sorted(found), seen


def _iter_country_query_chunks(path: Path, country: str, cfg: BlockingConfig, chunk_size: int):
    """Yield Source-1 records for one country, one chunk at a time."""
    batch = []
    for entity_id, name, address, row_country in _iter_raw_tsv(path):
        if row_country != country:
            continue
        batch.append(make_record(entity_id, name, address, row_country, cfg))
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_record_chunks(records: list, chunk_size: int):
    for start in range(0, len(records), chunk_size):
        yield records[start:start + chunk_size]


class _TsvWriter:
    """Append-only UTF-8 TSV. Rows are written as they are finished."""

    def __init__(self, path, header: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", buffering=1024 * 1024)
        self.handle.write(header + "\n")
        self.rows = 0

    def write(self, text: str, n_rows: int) -> None:
        if not text:
            return
        self.handle.write(text)
        self.rows += n_rows

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _emit_query(eid: str, found: dict, collapsed: _TsvWriter | None, pairs: _TsvWriter | None) -> None:
    """Write one Source-1 entity. `found` is already deduplicated by candidate id."""
    if collapsed is not None:
        cell = ",".join(sorted(found)) if found else ""
        collapsed.write(f"{eid}\t{cell}\n", 1)
    if pairs is not None and found:
        rule_strings = {mask: _mask_to_rules(mask) for mask in set(found.values())}
        lines = "".join(
            f"{eid}\t{cand}\t{rule_strings[mask]}\n" for cand, mask in found.items()
        )
        pairs.write(lines, len(found))


def _clean_found(eid: str, found: dict) -> dict:
    if eid in found:
        del found[eid]
    stale = [cand for cand in found if not _is_target_id(cand)]
    for cand in stale:
        del found[cand]
    return found


def _spill_non_tfidf(directory: Path, ordinal: int, records: list, index: _Index, cfg: BlockingConfig) -> Path:
    payload = []
    for rec in records:
        payload.append((rec.eid, rec.trans, _query_record(rec, index, (), cfg)))
    path = directory / f"chunk_{ordinal:06d}.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del payload
    return path


def _merge_tfidf_and_emit(
    spill_path: Path,
    prepared,
    cfg: BlockingConfig,
    collapsed: _TsvWriter | None,
    pairs: _TsvWriter | None,
    stats: _StreamStats,
    truth,
) -> None:
    with spill_path.open("rb") as handle:
        payload = pickle.load(handle)
    spill_path.unlink()
    neighbours = _search_char_tfidf(
        prepared, [trans for _eid, trans, _found in payload], cfg, log=False,
    )
    for (eid, _trans, found), tfidf_ids in zip(payload, neighbours):
        if tfidf_ids:
            _add(found, tfidf_ids, 1 << 8, budget=None, force=True)
        found = _clean_found(eid, found)
        _emit_query(eid, found, collapsed, pairs)
        stats.add_query(eid, found, truth)
    del payload, neighbours


def _emit_chunk_without_tfidf(
    records: list,
    index: _Index,
    cfg: BlockingConfig,
    collapsed: _TsvWriter | None,
    pairs: _TsvWriter | None,
    stats: _StreamStats,
    truth,
) -> None:
    for rec in records:
        found = _clean_found(rec.eid, _query_record(rec, index, (), cfg))
        _emit_query(rec.eid, found, collapsed, pairs)
        stats.add_query(rec.eid, found, truth)


def _stream_summary(stats: _StreamStats, truth) -> dict:
    """Same report shape as evaluate_blocking(), built from counters."""
    count_arr = np.asarray(stats.counts, dtype=np.int64) if stats.counts else np.zeros(1, dtype=np.int64)
    n_pairs = int(stats.n_pairs)
    same = int(stats.search_same)
    full = int(stats.n_s1 * stats.n_index)

    def _ratio(space):
        if not space or not n_pairs:
            return None
        return space / n_pairs

    def _safe(numer, denom):
        return (numer / denom) if denom else None

    summary = {
        "n_s1": int(stats.n_s1),
        "n_index": int(stats.n_index),
        "n_candidate_pairs": n_pairs,
        "search_space_same_country": same,
        "search_space_full_cartesian": full,
        "candidates_per_s1": {
            "mean": float(count_arr.mean()),
            "median": float(np.median(count_arr)),
            "p95": float(np.percentile(count_arr, 95)),
            "max": int(count_arr.max()),
            "zeros": int((count_arr == 0).sum()) if stats.counts else 0,
        },
        "reduction_ratio_vs_same_country": _ratio(same),
        "reduction_ratio_vs_full_cartesian": _ratio(full),
    }
    if truth is None or stats.queried_ids is None:
        return summary

    rule_names = [name for _, name in _RULES]
    hits = {name: 0 for name in rule_names}
    exclusive = {name: 0 for name in rule_names}
    union_hits = 0
    s2_true = s2_hit = s3_true = s3_hit = 0
    true_links = 0
    missed = []
    per_s1_recall = []
    n_with_truth = 0
    for source_id in stats.queried_ids:
        true_ids = truth.get(source_id, set())
        if not true_ids:
            continue
        n_with_truth += 1
        n_hit = 0
        for mid in true_ids:
            true_links += 1
            is_s2 = mid.startswith("S2-")
            if is_s2:
                s2_true += 1
            else:
                s3_true += 1
            rule_text = stats.found_true.get((source_id, mid))
            if not rule_text:
                if len(missed) < 15:
                    missed.append((source_id, mid))
                continue
            n_hit += 1
            union_hits += 1
            if is_s2:
                s2_hit += 1
            else:
                s3_hit += 1
            parts = set(rule_text.split("|"))
            for name in parts:
                if name in hits:
                    hits[name] += 1
            if len(parts) == 1:
                only = next(iter(parts))
                if only in exclusive:
                    exclusive[only] += 1
        per_s1_recall.append(n_hit / len(true_ids))

    summary.update({
        "n_s1_with_truth": n_with_truth,
        "n_true_links": true_links,
        "n_true_links_found": union_hits,
        "recall_union": _safe(union_hits, true_links),
        "recall_macro_s1": float(np.mean(per_s1_recall)) if per_s1_recall else None,
        "recall_by_rule": {name: _safe(hits[name], true_links) for name in rule_names},
        "exclusive_recall_by_rule": {name: _safe(exclusive[name], true_links) for name in rule_names},
        "recall_s2": _safe(s2_hit, s2_true),
        "recall_s3": _safe(s3_hit, s3_true),
        "n_true_s2": s2_true,
        "n_true_s3": s3_true,
        "missed_examples": missed,
    })
    return summary


def _default_pairs_path(output_path) -> Path:
    return Path(output_path).with_suffix(".pairs.tsv")


def generate_candidates_from_paths(
    source1_path,
    source2_path,
    source3_path,
    config: BlockingConfig | None = None,
    countries=None,
    max_s1: int | None = None,
    seed: int = 42,
    output_path=None,
    pairs_path=None,
    truth=None,
) -> dict:
    """Stream the same blocking passes as generate_candidates(), from TSVs.

    Source-1 is processed in `config.query_chunk_size` chunks. Each chunk
    runs every blocking pass, unions and deduplicates that chunk, then
    writes it. Candidate rows are not kept for the whole file.

    `output_path`, when set, is the challenge candidate_pairs.tsv shape:
    one row per Source-1 id, comma-separated Source-2/3 ids, empty when
    a query has no candidates. `pairs_path` is the matcher pair table
    (source1_entity_id, candidate_entity_id, rules). When `output_path`
    is set and `pairs_path` is omitted, the pair table is written beside
    it as `<stem>.pairs.tsv`.

    `countries` restricts the run to those labels (open set: pass whatever
    strings are in the file, including France). None means every country
    that appears in Source-1.

    `max_s1` reservoir-samples that many Source-1 rows after the country
    filter. The Source-2/3 index for each country is still complete.

    `truth` is optional and is read only to fill recall counters. It does
    not add, drop, or reorder candidates.

    Countries are indexed one at a time. When character TF-IDF is on, the
    token index is released before the name lists are loaded; non-TF-IDF
    hits for the country are spilled chunk by chunk and merged afterwards.
    """
    cfg = config or BlockingConfig()
    chunk_size = _chunk_size(cfg)
    country_set = None if countries is None else {c.strip() for c in countries}
    source1_path = Path(source1_path)
    target_paths = [Path(source2_path), Path(source3_path)]
    if output_path is not None and pairs_path is None:
        pairs_path = _default_pairs_path(output_path)

    collapsed = None
    pairs = None
    stats = _StreamStats(collect_ids=truth is not None)

    try:
        if output_path is not None:
            collapsed = _TsvWriter(output_path, "\t".join(SUBMISSION_COLUMNS))
        if pairs_path is not None:
            pairs = _TsvWriter(pairs_path, "\t".join(PAIR_COLUMNS))
        if max_s1 is None:
            _log(cfg, "scanning Source 1 countries")
            country_list, scanned = _countries_in_source1(source1_path, country_set)
            _log(cfg, f"  Source 1 scan {scanned:,} lines, {len(country_list)} countries")
            sampled = None
        else:
            _log(cfg, "sampling Source 1")
            sampled = _group_countries(
                _reservoir_queries(source1_path, country_set, max_s1, seed, cfg)
            )
            country_list = sorted(sampled)

        for country in country_list:
            if sampled is None:
                query_chunks = _iter_country_query_chunks(source1_path, country, cfg, chunk_size)
                n_query_hint = "streamed"
            else:
                query_chunks = _iter_record_chunks(sampled.pop(country), chunk_size)
                n_query_hint = "sampled"
            _log(cfg, f"[{country}] indexing targets ({n_query_hint} queries, chunk={chunk_size})")
            index, n_index = _stream_country_index(target_paths, country, cfg)
            stats.n_index += n_index
            _log(cfg, f"[{country}] indexed {n_index:,} overflow dropped keys: {index.overflow}")
            n_country_queries = 0

            if cfg.enable_tfidf:
                spill_dir = Path(tempfile.mkdtemp(prefix="blocking_chunk_"))
                spills = []
                try:
                    for ordinal, chunk in enumerate(query_chunks):
                        n_country_queries += len(chunk)
                        spills.append(_spill_non_tfidf(spill_dir, ordinal, chunk, index, cfg))
                        del chunk
                        gc.collect()
                    del index
                    gc.collect()
                    _log(cfg, f"[{country}] collecting names for character tf-idf")
                    name_ids, name_texts = _stream_country_names(target_paths, country, cfg)
                    prepared = _fit_char_tfidf(name_ids, name_texts, cfg)
                    del name_ids, name_texts
                    gc.collect()
                    n_spills = len(spills)
                    for ordinal, spill_path in enumerate(spills, start=1):
                        _log(cfg, f"[{country}] tf-idf query chunk {ordinal}/{n_spills}")
                        _merge_tfidf_and_emit(
                            spill_path, prepared, cfg, collapsed, pairs, stats, truth,
                        )
                        if collapsed is not None:
                            collapsed.flush()
                        if pairs is not None:
                            pairs.flush()
                        gc.collect()
                    del prepared, spills
                finally:
                    if spill_dir.exists():
                        for leftover in spill_dir.glob("chunk_*.pkl"):
                            leftover.unlink(missing_ok=True)
                        try:
                            spill_dir.rmdir()
                        except OSError:
                            pass
            else:
                for chunk in query_chunks:
                    n_country_queries += len(chunk)
                    _emit_chunk_without_tfidf(chunk, index, cfg, collapsed, pairs, stats, truth)
                    del chunk
                    if collapsed is not None:
                        collapsed.flush()
                    if pairs is not None:
                        pairs.flush()
                    gc.collect()
                del index
                gc.collect()

            stats.search_same += n_country_queries * n_index
            _log(cfg, f"[{country}] pairs so far {stats.n_pairs:,}")
            if sampled is not None and country in sampled:
                del sampled[country]
            gc.collect()
    finally:
        if collapsed is not None:
            collapsed.close()
        if pairs is not None:
            pairs.close()

    summary = _stream_summary(stats, truth)
    summary["output_path"] = None if output_path is None else str(output_path)
    summary["pairs_path"] = None if pairs_path is None else str(pairs_path)
    summary["query_chunk_size"] = chunk_size
    return summary


# ---------------------------------------------------------------------------
# Self-check on a tiny synthetic open-set (includes France, not only US/India)
# ---------------------------------------------------------------------------

def _self_check() -> None:
    cfg = BlockingConfig(verbose=False, tfidf_top_k=5, tfidf_min_score=0.05, exact_max_fanout=50)
    source1 = pd.DataFrame([
        ["S1-exact", "Orelee Barbershop", "10 Main Street, Austin, TX", "US"],
        ["S1-core", "Foo Incorporated", "1 Oak Road, Dallas, TX", "US"],
        ["S1-order", "Blue River Cafe", "9 Lake Ave, Madison, WI", "US"],
        ["S1-typo", "Vanguard", "500 Other Road, Boise, ID", "US"],
        ["S1-addr", "Alpha Widgets", "35 Hinsdale Plaza, Hinsdale, IL", "US"],
        ["S1-fr", "École Primaire Sainte", "22 Rue Descartes, Calais 62100", "France"],
        ["S1-postal", "Nom Completement Different", "44000", "France"],
        ["S1-cross", "Foo Incorporated", "1 Oak Road, Dallas, TX", "US"],
        ["S1-single", "Zzqzzz Unique Holdings", "999 Nowhere Lane, Nome, AK", "US"],
    ], columns=["entity_id", "business_name", "business_address", "country"])

    indic = "राम मार्केटिंग"
    indic_record = make_record("S1-indic-probe", indic, "1 Test Road", "India", cfg)
    source1 = pd.concat([source1, pd.DataFrame([{
        "entity_id": "S1-indic",
        "business_name": indic_record.trans,
        "business_address": "12 Mandav Flat, Himatnagar",
        "country": "India",
    }])], ignore_index=True)

    source2 = pd.DataFrame([
        ["S2-exact", "Orelee Barbershop", "10 Main St, Austin, TX", "US"],
        ["S2-core", "Foo Inc", "1 Oak Rd, Dallas, TX", "US"],
        ["S2-order", "Cafe Blue River", "9 Lake Avenue, Madison, WI", "US"],
        ["S2-typo", "Vanguamd", "800 Unrelated Ave, Boise, ID", "US"],
        ["S2-addr", "Beta Holdings", "35 Hinsdale Plaza, Hinsdale, IL", "US"],
        ["S2-fr", "Ecole Primaire Sainte", "22 Rue Descartes, Calais", "France"],
        ["S2-cross", "Foo Incorporated", "1 Oak Road, Lyon", "France"],
        ["S2-indic", indic, "12 Mandav Flat, Himatnagar", "India"],
        ["S2-distractor", "Quantum Noodle Cart", "77 Broadway, New York, NY", "US"],
    ], columns=["entity_id", "business_name", "business_address", "country"])
    source3 = pd.DataFrame([
        ["S3-exact", "orelee's barbershop", "10 Main Street Austin TX", "US"],
        ["S3-postal", "Autre Enseigne", "44000", "France"],
    ], columns=["entity_id", "business_name", "business_address", "country"])

    pairs = generate_candidates(source1, source2, source3, cfg)
    assert set(pairs.columns) >= {"source1_entity_id", "candidate_entity_id", "rules"}
    assert pairs["candidate_entity_id"].str.startswith(("S2-", "S3-")).all()
    assert not pairs["candidate_entity_id"].str.startswith("S1-").any()

    def rules_for(source_id, cand_id):
        hit = pairs[(pairs["source1_entity_id"] == source_id) & (pairs["candidate_entity_id"] == cand_id)]
        assert len(hit) == 1, (source_id, cand_id, pairs[pairs["source1_entity_id"] == source_id])
        return set(hit.iloc[0]["rules"].split("|"))

    assert RULE_EXACT_NAME in rules_for("S1-exact", "S2-exact")
    assert RULE_EXACT_CORE in rules_for("S1-core", "S2-core")
    assert RULE_EXACT_SORTED in rules_for("S1-order", "S2-order")
    assert RULE_CHAR_TFIDF in rules_for("S1-typo", "S2-typo")
    assert RULE_ADDRESS_TOKEN in rules_for("S1-addr", "S2-addr")
    assert RULE_TRANSLIT in rules_for("S1-fr", "S2-fr")
    assert RULE_ADDRESS_POSTAL in rules_for("S1-postal", "S3-postal")
    assert RULE_TRANSLIT in rules_for("S1-indic", "S2-indic")

    cross = pairs[(pairs["source1_entity_id"] == "S1-cross") & (pairs["candidate_entity_id"] == "S2-cross")]
    assert cross.empty, cross

    truth = pd.DataFrame({
        "source1_entity_id": ["S1-exact", "S1-typo", "S1-single"],
        "matched_entity_ids": ["S2-exact,S3-exact", "S2-typo", ""],
    })
    summary = evaluate_blocking(
        pairs, truth, source1_ids=["S1-exact", "S1-typo", "S1-single"],
    )
    assert summary["recall_union"] == 1.0, summary
    assert summary["n_true_links"] == 3
    assert summary["recall_s2"] == 1.0
    assert summary["recall_s3"] == 1.0

    table = to_candidate_pairs_frame(pairs, ["S1-exact", "S1-single", "S1-missing"])
    assert list(table.columns) == SUBMISSION_COLUMNS
    assert list(table["source1_entity_id"]) == ["S1-exact", "S1-single", "S1-missing"]
    exact_ids = set(table.loc[table["source1_entity_id"] == "S1-exact", "candidate_entity_ids"].iloc[0].split(","))
    assert "S2-exact" in exact_ids and "S3-exact" in exact_ids
    assert table.loc[table["source1_entity_id"] == "S1-missing", "candidate_entity_ids"].iloc[0] == ""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Business-entity blocking and candidate recall")
    parser.add_argument("--data-dir", default=None, help="Dataset root containing train/ and test/. "
                        "Default: search project-relative locations.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--countries", default=None, help="Comma-separated country labels (default: all).")
    parser.add_argument("--max-s1", type=int, default=None, help="Reservoir-sample this many Source-1 rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--query-chunk-size", type=int, default=5000,
                        help="Source-1 rows per blocking chunk (default: 5000).")
    parser.add_argument("--output", default=None, help="Optional candidate_pairs.tsv path "
                        "(one row per Source-1 id).")
    parser.add_argument("--pairs-output", default=None,
                        help="Pair-level TSV with rules for the matcher. "
                        "When --output is set and this is omitted, writes <stem>.pairs.tsv.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        _self_check()
        print("self-check passed")
        return

    if args.query_chunk_size < 1:
        parser.error("--query-chunk-size must be >= 1")

    data_dir = resolve_dataset_dir(args.data_dir)
    split_dir = data_dir / args.split
    cfg = BlockingConfig(
        enable_tfidf=not args.skip_tfidf,
        verbose=True,
        query_chunk_size=args.query_chunk_size,
    )
    countries = None if not args.countries else [c.strip() for c in args.countries.split(",") if c.strip()]
    prefix = "train" if args.split == "train" else "test"
    truth = None
    if args.split == "train":
        truth = _load_truth_map(split_dir / f"{prefix}_ground_truth.tsv")
    summary = generate_candidates_from_paths(
        split_dir / f"{prefix}_source1.tsv",
        split_dir / f"{prefix}_source2.tsv",
        split_dir / f"{prefix}_source3.tsv",
        config=cfg,
        countries=countries,
        max_s1=args.max_s1,
        seed=args.seed,
        output_path=args.output,
        pairs_path=args.pairs_output,
        truth=truth,
    )
    print(f"candidate pairs: {summary['n_candidate_pairs']:,}")
    if "recall_union" in summary:
        print(format_blocking_report(summary))
    else:
        counts = summary["candidates_per_s1"]
        print(
            f"  candidates/S1 mean {counts['mean']:.1f}  "
            f"median {counts['median']:.0f}  p95 {counts['p95']:.0f}  "
            f"max {counts['max']}  with-none {counts['zeros']:,}"
        )
        print(f"  same-country search space: {summary['search_space_same_country']:,}")
        print(f"  full cartesian space:      {summary['search_space_full_cartesian']:,}")
    if summary["output_path"]:
        print(f"wrote {summary['output_path']}")
    if summary["pairs_path"]:
        print(f"wrote {summary['pairs_path']}")


if __name__ == "__main__":
    main()

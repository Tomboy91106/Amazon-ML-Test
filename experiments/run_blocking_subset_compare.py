"""Compare blocking budgets on a small Source-1 sample against full S2+S3.

Safety: this script refuses to block more than 5,000 Source-1 rows.
It always passes max_s1 and never writes the candidate-pair files.
The Source-2 and Source-3 indexes stay complete for every country in the sample.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from src.blocking import (
    BlockingConfig,
    _is_target_id,
    _parse_id_list,
    _reservoir_queries,
    format_blocking_report,
    generate_candidates_from_paths,
    resolve_dataset_dir,
)

# Hard ceiling. Do not raise this to the full training Source-1 file.
HARD_CAP = 5_000
DEFAULT_MAX_S1 = 3_000
SEED = 42

# A_current is the previous blocker. B_tight matches BlockingConfig defaults.
# C_mid is optional; pass it with --configs when the tight caps lose recall.
CONFIGS = {
    "A_current": dict(
        name_token_max_df=10_000,
        addr_token_max_df=40_000,
        name_token_budget=20_000,
        addr_token_budget=60_000,
    ),
    "B_tight": dict(
        name_token_max_df=1_000,
        addr_token_max_df=3_000,
        name_token_budget=2_000,
        addr_token_budget=3_000,
    ),
    "C_mid": dict(
        name_token_max_df=3_000,
        addr_token_max_df=8_000,
        name_token_budget=4_000,
        addr_token_budget=6_000,
    ),
}


def _load_truth_for(path: Path, source_ids: set[str]) -> dict:
    mapping = {eid: set() for eid in source_ids}
    found = 0
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "source1_entity_id" not in header:
            raise ValueError(f"unexpected ground-truth header in {path}: {header!r}")
        for line in handle:
            source_id, _, rest = line.rstrip("\n").partition("\t")
            if source_id not in source_ids:
                continue
            mapping[source_id] = {mid for mid in _parse_id_list(rest) if _is_target_id(mid)}
            found += 1
    if found != len(source_ids):
        raise RuntimeError(
            f"ground truth covered {found} of {len(source_ids)} sampled Source-1 ids"
        )
    return mapping


def _sample_profile(records, truth: dict) -> dict:
    countries = Counter(rec.country for rec in records)
    card = Counter()
    links = 0
    for rec in records:
        n = len(truth.get(rec.eid, ()))
        links += n
        if n == 0:
            card["zero"] += 1
        elif n == 1:
            card["singleton"] += 1
        else:
            card["multi"] += 1
    return {
        "n_s1": len(records),
        "countries": dict(countries),
        "match_cardinality": dict(card),
        "n_true_links": links,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_misses(path: Path, missed) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("source1_entity_id\tmatched_entity_id\n")
        for source_id, matched_id in missed:
            handle.write(f"{source_id}\t{matched_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Dataset root containing train/. "
                        "Default: search project-relative locations.")
    parser.add_argument("--max-s1", type=int, default=DEFAULT_MAX_S1)
    parser.add_argument("--configs", default="B_tight,A_current")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        default="experiments/blocking_subset_compare.json",
    )
    args = parser.parse_args()

    if args.max_s1 is None or not 1 <= args.max_s1 <= HARD_CAP:
        raise SystemExit(
            f"refusing to run: --max-s1 must be between 1 and {HARD_CAP} "
            "(this script will not block the full Source-1 training file)"
        )
    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}; choices: {sorted(CONFIGS)}")

    data_dir = resolve_dataset_dir(args.data_dir)
    train = data_dir / "train"
    source1 = train / "train_source1.tsv"
    source2 = train / "train_source2.tsv"
    source3 = train / "train_source3.tsv"
    truth_path = train / "train_ground_truth.tsv"
    for path in (source1, source2, source3, truth_path):
        if not path.is_file():
            raise SystemExit(f"missing dataset file: {path}")

    print(
        f"SAFETY max_s1={args.max_s1} seed={args.seed}. "
        "Blocking only this Source-1 sample. Indexes use full Source-2 and Source-3.",
        flush=True,
    )
    sampled = _reservoir_queries(
        source1, None, args.max_s1, args.seed, BlockingConfig(verbose=False),
    )
    if len(sampled) != args.max_s1:
        raise SystemExit(f"reservoir returned {len(sampled)} rows, expected {args.max_s1}")
    sampled_ids = {rec.eid for rec in sampled}
    if len(sampled_ids) != args.max_s1:
        raise SystemExit("reservoir produced duplicate Source-1 ids")
    truth = _load_truth_for(truth_path, sampled_ids)
    profile = _sample_profile(sampled, truth)
    print("sample", json.dumps(profile, sort_keys=True), flush=True)
    del sampled

    output_path = Path(args.output)
    payload = {
        "max_s1": args.max_s1,
        "seed": args.seed,
        "targets": "full train_source2.tsv + train_source3.tsv",
        "sample": profile,
        "runs": [],
    }
    expected_links = profile["n_true_links"]

    for name in names:
        settings = CONFIGS[name]
        cfg = BlockingConfig(verbose=True, enable_tfidf=True, **settings)
        print(f"\n=== {name} {settings} ===", flush=True)
        started = time.perf_counter()
        summary = generate_candidates_from_paths(
            source1,
            source2,
            source3,
            config=cfg,
            countries=None,
            max_s1=args.max_s1,
            seed=args.seed,
            output_path=None,
            pairs_path=None,
            truth=truth,
        )
        seconds = time.perf_counter() - started
        n_s1 = int(summary["n_s1"])
        if n_s1 != args.max_s1 or n_s1 > HARD_CAP:
            raise SystemExit(f"SAFETY STOP: blocker queried {n_s1} Source-1 rows")
        if int(summary.get("n_true_links", -1)) != expected_links:
            raise SystemExit(
                "sample mismatch: blocker true-link count "
                f"{summary.get('n_true_links')} != reservoir {expected_links}"
            )

        missed = summary.get("missed_true_links", [])
        miss_path = Path("experiments") / f"blocking_subset_missed_{name}.tsv"
        _write_misses(miss_path, missed)
        counts = summary["candidates_per_s1"]
        run = {
            "name": name,
            "config": settings,
            "seconds": round(seconds, 1),
            "n_s1": n_s1,
            "n_index": summary["n_index"],
            "n_candidate_pairs": summary["n_candidate_pairs"],
            "candidates_per_s1": counts,
            "n_true_links": summary["n_true_links"],
            "n_true_links_found": summary["n_true_links_found"],
            "recall_union": summary["recall_union"],
            "recall_s2": summary["recall_s2"],
            "recall_s3": summary["recall_s3"],
            "n_true_s2": summary["n_true_s2"],
            "n_true_s3": summary["n_true_s3"],
            "hits_by_rule": summary["hits_by_rule"],
            "exclusive_hits_by_rule": summary["exclusive_hits_by_rule"],
            "n_missed": len(missed),
            "missed_true_links": missed,
            "missed_path": str(miss_path),
        }
        payload["runs"].append(_jsonable(run))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"{name}: candidates={summary['n_candidate_pairs']:,} "
            f"mean/s1={counts['mean']:.1f} median={counts['median']:.0f} "
            f"p95={counts['p95']:.0f} max={counts['max']} "
            f"recall={summary['n_true_links_found']}/{summary['n_true_links']} "
            f"({summary['recall_union']:.6f}) seconds={seconds:.1f} "
            f"missed={len(missed)}",
            flush=True,
        )
        print(format_blocking_report(summary), flush=True)

    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()

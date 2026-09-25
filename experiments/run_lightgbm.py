"""One LightGBM matcher on the locked baseline split and 40 features.

Does not redraw the split, rewrite the evaluator, or touch baseline artifacts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import (  # noqa: E402
    ground_truth_sets,
    predicted_sets_at_threshold,
    score_entity,
    sweep_thresholds,
)
from src.features import FEATURE_COLUMNS  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "training_pairs_real.parquet"
SPLIT_PATH = ROOT / "experiments" / "grouped_split.json"
RESULTS_PATH = ROOT / "experiments" / "lightgbm_results.csv"
CONFIG_PATH = ROOT / "experiments" / "lightgbm_config.json"

THRESHOLDS = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98,
    0.99, 0.995, 0.999, 0.9995, 0.9999,
)

BASELINE = {
    "model": "logistic_regression",
    "threshold": 0.9995,
    "macro_f05": 0.972330,
    "entity_precision": 0.975417,
    "entity_recall": 0.968625,
}


def match_bucket(n_positive: int) -> str:
    if n_positive == 0:
        return "zero"
    if n_positive == 1:
        return "singleton"
    return "multi"


def bucket_metrics(true_by_s1, predicted_by_s1, buckets, s1_ids) -> dict:
    groups = {
        "zero": [s1 for s1 in s1_ids if buckets[s1] == "zero"],
        "singleton": [s1 for s1 in s1_ids if buckets[s1] == "singleton"],
        "multi": [s1 for s1 in s1_ids if buckets[s1] == "multi"],
        "all": list(s1_ids),
    }
    summary = {}
    for name, ids in groups.items():
        scores = [score_entity(true_by_s1[s1], predicted_by_s1.get(s1, set())) for s1 in ids]
        arr = np.asarray(scores, dtype=float)
        summary[name] = {
            "n": len(ids),
            "macro_f05": float(arr[:, 2].mean()),
            "precision": float(arr[:, 0].mean()),
            "recall": float(arr[:, 1].mean()),
            "zero_prediction_s1": int(sum(len(predicted_by_s1.get(s1, set())) == 0 for s1 in ids)),
        }
    return summary


def probability_quantiles(probabilities: np.ndarray) -> dict:
    levels = (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)
    names = ("min", "p1", "p5", "median", "p95", "p99", "max")
    values = np.quantile(probabilities, levels)
    return {name: float(value) for name, value in zip(names, values)}


def main() -> None:
    started = time.perf_counter()
    split = json.loads(SPLIT_PATH.read_text())
    feature_cols = list(split["feature_columns"])
    if feature_cols != list(FEATURE_COLUMNS):
        raise SystemExit("split feature list does not match FEATURE_COLUMNS")
    if len(feature_cols) != 40:
        raise SystemExit(f"expected 40 features, found {len(feature_cols)}")

    train_ids = split["train_s1_ids"]
    val_ids = split["validation_s1_ids"]
    train_set = set(train_ids)
    val_set = set(val_ids)
    if len(train_ids) != 1600 or len(val_ids) != 400:
        raise SystemExit(f"unexpected split sizes: train {len(train_ids)} val {len(val_ids)}")
    if len(train_set) != 1600 or len(val_set) != 400 or train_set & val_set:
        raise SystemExit("train/validation S1 ids overlap or contain duplicates")

    frame = pd.read_parquet(DATA_PATH)
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise SystemExit(f"training frame is missing feature columns: {missing}")

    train = frame[frame["source1_entity_id"].isin(train_set)]
    val = frame[frame["source1_entity_id"].isin(val_set)]
    if train["source1_entity_id"].nunique() != 1600 or val["source1_entity_id"].nunique() != 400:
        raise SystemExit("saved S1 ids do not match rows in the training parquet")

    x_train = train[feature_cols].to_numpy(dtype=np.float64, copy=True)
    x_val = val[feature_cols].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(x_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(x_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train["is_true_match"].to_numpy()
    y_val = val["is_true_match"].to_numpy()

    train_positives = int(y_train.sum())
    train_negatives = int(len(y_train) - train_positives)
    scale_pos_weight = train_negatives / train_positives

    model = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    probabilities = model.predict_proba(x_val)[:, 1]
    true_by_s1 = ground_truth_sets(val, val_ids)
    results = sweep_thresholds(
        val["source1_entity_id"].to_numpy(),
        val["candidate_entity_id"].to_numpy(),
        probabilities,
        true_by_s1,
        THRESHOLDS,
    )
    RESULTS_PATH.write_text(results.to_csv(index=False))

    best = results.loc[results["macro_f05"].idxmax()]
    best_threshold = float(best["threshold"])
    predicted = predicted_sets_at_threshold(
        val["source1_entity_id"].to_numpy(),
        val["candidate_entity_id"].to_numpy(),
        probabilities,
        best_threshold,
        val_ids,
    )
    positive_counts = val.groupby("source1_entity_id")["is_true_match"].sum().astype(int)
    buckets = {s1: match_bucket(int(positive_counts.get(s1, 0))) for s1 in val_ids}
    buckets_summary = bucket_metrics(true_by_s1, predicted, buckets, val_ids)

    pair_pred = probabilities >= best_threshold
    tp = int((pair_pred & (y_val == 1)).sum())
    fp = int((pair_pred & (y_val == 0)).sum())
    fn = int((~pair_pred & (y_val == 1)).sum())

    pos_q = probability_quantiles(probabilities[y_val == 1])
    neg_q = probability_quantiles(probabilities[y_val == 0])
    delta_f05 = float(best["macro_f05"]) - BASELINE["macro_f05"]
    runtime_seconds = time.perf_counter() - started

    config = {
        "model": "LGBMClassifier",
        "parameters": {
            "objective": "binary",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
        "scaling": None,
        "nan_fill": 0.0,
        "early_stopping": False,
        "split_file": "experiments/grouped_split.json",
        "train_s1_count": 1600,
        "validation_s1_count": 400,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "train_positive_count": train_positives,
        "train_negative_count": train_negatives,
        "validation_positive_count": int(y_val.sum()),
        "validation_negative_count": int(len(y_val) - y_val.sum()),
        "scale_pos_weight": scale_pos_weight,
        "feature_names": feature_cols,
        "best_threshold": best_threshold,
        "best_macro_f05": float(best["macro_f05"]),
        "precision": float(best["entity_precision"]),
        "recall": float(best["entity_recall"]),
        "avg_predicted_matches": float(best["avg_predicted_matches"]),
        "p95_predicted_matches": float(best["p95_predicted_matches"]),
        "n_zero_prediction_s1": int(best["n_zero_prediction_s1"]),
        "frac_zero_prediction_s1": float(best["frac_zero_prediction_s1"]),
        "buckets": buckets_summary,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "positive_probability_quantiles": pos_q,
        "negative_probability_quantiles": neg_q,
        "training_seconds": fit_seconds,
        "runtime_seconds": runtime_seconds,
        "baseline_comparison": {
            **BASELINE,
            "lightgbm_macro_f05": float(best["macro_f05"]),
            "lightgbm_precision": float(best["entity_precision"]),
            "lightgbm_recall": float(best["entity_recall"]),
            "lightgbm_threshold": best_threshold,
            "delta_f05": delta_f05,
        },
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

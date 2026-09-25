"""Logistic-regression baseline on the real grouped candidate pairs.

Reuses src.features.FEATURE_COLUMNS and src.evaluation. Does not refit
blocking, normalization, or the feature definitions. The grouped split
written next to the results is the validation set for later models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import ground_truth_sets, sweep_thresholds  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "training_pairs_real.parquet"
OUT_DIR = ROOT / "experiments"
RESULTS_PATH = OUT_DIR / "baseline_results.csv"
SPLIT_PATH = OUT_DIR / "grouped_split.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLDS = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98,
    0.99, 0.995, 0.999, 0.9995, 0.9999,
)


def match_bucket(n_positive: int) -> str:
    if n_positive == 0:
        return "zero"
    if n_positive == 1:
        return "singleton"
    return "multi"


def grouped_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """80/20 split of S1 ids, stratified by zero / singleton / multi-match."""
    counts = frame.groupby("source1_entity_id", sort=False)["is_true_match"].sum().astype(int)
    groups = counts.rename("n_positive").reset_index()
    groups["bucket"] = groups["n_positive"].map(match_bucket)
    train_groups, val_groups = train_test_split(
        groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=groups["bucket"],
    )
    train_ids = np.sort(train_groups["source1_entity_id"].to_numpy())
    val_ids = np.sort(val_groups["source1_entity_id"].to_numpy())
    return train_ids, val_ids, groups


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    feature_cols = list(FEATURE_COLUMNS)
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise SystemExit(f"training frame is missing feature columns: {missing}")

    train_ids, val_ids, groups = grouped_split(frame)
    train_mask = frame["source1_entity_id"].isin(set(train_ids))
    val_mask = frame["source1_entity_id"].isin(set(val_ids))
    train = frame.loc[train_mask]
    val = frame.loc[val_mask]

    # Similarity features are NaN when either side has no text. Filling with
    # 0 is the safe input for logistic regression; the columns themselves
    # are unchanged on disk.
    x_train = train[feature_cols].to_numpy(dtype=np.float64, copy=True)
    x_val = val[feature_cols].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(x_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(x_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train["is_true_match"].to_numpy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_val)[:, 1]

    true_by_s1 = ground_truth_sets(val, val_ids)
    results = sweep_thresholds(
        val["source1_entity_id"].to_numpy(),
        val["candidate_entity_id"].to_numpy(),
        probabilities,
        true_by_s1,
        THRESHOLDS,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    def bucket_counts(ids: np.ndarray) -> dict[str, int]:
        selected = groups[groups["source1_entity_id"].isin(set(ids))]
        counts = selected["bucket"].value_counts().to_dict()
        return {name: int(counts.get(name, 0)) for name in ("zero", "singleton", "multi")}

    split_payload = {
        "dataset": "data/processed/training_pairs_real.parquet",
        "split": "grouped_by_source1_entity_id",
        "method": "sklearn.model_selection.train_test_split",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": "match_bucket from positive count per S1: 0=zero, 1=singleton, >1=multi",
        "feature_columns": feature_cols,
        "train_s1_count": int(len(train_ids)),
        "validation_s1_count": int(len(val_ids)),
        "train_buckets": bucket_counts(train_ids),
        "validation_buckets": bucket_counts(val_ids),
        "train_s1_ids": train_ids.tolist(),
        "validation_s1_ids": val_ids.tolist(),
        "note": "Reuse validation_s1_ids exactly. Do not re-draw the split.",
    }
    SPLIT_PATH.write_text(json.dumps(split_payload, indent=2) + "\n")

    best = results.loc[results["macro_f05"].idxmax()]
    summary = {
        "train_s1": int(len(train_ids)),
        "validation_s1": int(len(val_ids)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "train_positives": int(y_train.sum()),
        "validation_positives": int(val["is_true_match"].sum()),
        "train_buckets": bucket_counts(train_ids),
        "validation_buckets": bucket_counts(val_ids),
        "n_iter": int(np.max(model.n_iter_)),
        "converged": bool(np.max(model.n_iter_) < model.max_iter),
        "best_threshold": float(best["threshold"]),
        "best_macro_f05": float(best["macro_f05"]),
        "best_entity_precision": float(best["entity_precision"]),
        "best_entity_recall": float(best["entity_recall"]),
        "best_avg_predicted": float(best["avg_predicted_matches"]),
        "best_p95_predicted": float(best["p95_predicted_matches"]),
        "best_zero_prediction_s1": int(best["n_zero_prediction_s1"]),
    }
    print(json.dumps(summary, indent=2))
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

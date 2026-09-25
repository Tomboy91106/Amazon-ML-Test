"""Entity-level F0.5 for the Business Entity Resolution challenge.

Scores are computed per Source-1 entity, then averaged. This is the
competition metric: set precision and recall on matched candidate IDs,
combined as F_beta with beta = 0.5. Pair-level F0.5 is not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA = 0.5
BETA_SQ = BETA * BETA
F_NUMERATOR = 1.0 + BETA_SQ  # 1.25


def f05(precision: float, recall: float) -> float:
    """F0.5 from precision and recall. Both zero yields 0."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return F_NUMERATOR * precision * recall / (BETA_SQ * precision + recall)


def score_entity(true_ids, predicted_ids) -> tuple[float, float, float]:
    """Precision, recall, and F0.5 for one Source-1 entity.

    Empty-set rules:
    - true empty and predicted empty -> 1, 1, 1
    - either side empty and the other not -> 0, 0, 0
    """
    true = set(true_ids)
    predicted = set(predicted_ids)
    if not true and not predicted:
        return 1.0, 1.0, 1.0
    if not true or not predicted:
        return 0.0, 0.0, 0.0
    overlap = len(true & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(true)
    return precision, recall, f05(precision, recall)


def ground_truth_sets(
    frame: pd.DataFrame,
    s1_ids,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    label_col: str = "is_true_match",
) -> dict[str, set[str]]:
    """Positive candidate IDs for each S1. S1s with no positives map to empty."""
    positives = frame.loc[frame[label_col] == 1, [s1_col, cand_col]]
    grouped = positives.groupby(s1_col, sort=False)[cand_col].agg(lambda values: set(values))
    return {s1_id: grouped.get(s1_id, set()) for s1_id in s1_ids}


def predicted_sets_at_threshold(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    entity_ids,
) -> dict[str, set[str]]:
    """Candidate IDs with probability >= threshold, keyed by S1. Others stay empty."""
    keep = probabilities >= threshold
    predicted: dict[str, set[str]] = {s1_id: set() for s1_id in entity_ids}
    if not np.any(keep):
        return predicted
    selected = pd.DataFrame(
        {
            "source1_entity_id": s1_ids[keep],
            "candidate_entity_id": candidate_ids[keep],
        }
    )
    for s1_id, group in selected.groupby("source1_entity_id", sort=False)["candidate_entity_id"]:
        predicted[s1_id] = set(group)
    return predicted


def evaluate_entities(true_by_s1: dict[str, set[str]], predicted_by_s1: dict[str, set[str]]) -> dict:
    """Macro entity-level precision, recall, F0.5, and prediction-size stats."""
    precisions = []
    recalls = []
    f_scores = []
    sizes = []
    for s1_id, true_ids in true_by_s1.items():
        predicted_ids = predicted_by_s1.get(s1_id, set())
        precision, recall, score = score_entity(true_ids, predicted_ids)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(score)
        sizes.append(len(predicted_ids))
    sizes_arr = np.asarray(sizes, dtype=float)
    n_entities = len(sizes_arr)
    n_zero = int(np.sum(sizes_arr == 0))
    return {
        "n_entities": n_entities,
        "macro_f05": float(np.mean(f_scores)) if n_entities else float("nan"),
        "entity_precision": float(np.mean(precisions)) if n_entities else float("nan"),
        "entity_recall": float(np.mean(recalls)) if n_entities else float("nan"),
        "avg_predicted_matches": float(np.mean(sizes_arr)) if n_entities else float("nan"),
        "p95_predicted_matches": float(np.percentile(sizes_arr, 95)) if n_entities else float("nan"),
        "n_zero_prediction_s1": n_zero,
        "frac_zero_prediction_s1": float(n_zero / n_entities) if n_entities else float("nan"),
    }


def sweep_thresholds(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    probabilities: np.ndarray,
    true_by_s1: dict[str, set[str]],
    thresholds,
) -> pd.DataFrame:
    """Entity-level metrics at each probability threshold."""
    rows = []
    entity_ids = list(true_by_s1)
    for threshold in thresholds:
        predicted = predicted_sets_at_threshold(
            s1_ids, candidate_ids, probabilities, float(threshold), entity_ids
        )
        metrics = evaluate_entities(true_by_s1, predicted)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    columns = [
        "threshold",
        "macro_f05",
        "entity_precision",
        "entity_recall",
        "avg_predicted_matches",
        "p95_predicted_matches",
        "n_zero_prediction_s1",
        "frac_zero_prediction_s1",
        "n_entities",
    ]
    return pd.DataFrame(rows)[columns]


def _self_check() -> None:
    # README example: predicted {A, B, C}, truth {A, C}.
    precision, recall, score = score_entity({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(precision - 2 / 3) < 1e-12
    assert recall == 1.0
    assert abs(score - (1.25 * (2 / 3) * 1.0) / (0.25 * (2 / 3) + 1.0)) < 1e-12
    assert score_entity(set(), set()) == (1.0, 1.0, 1.0)
    assert score_entity(set(), {"S2-1"}) == (0.0, 0.0, 0.0)
    assert score_entity({"S2-1"}, set()) == (0.0, 0.0, 0.0)
    assert score_entity({"S2-1"}, {"S2-2"})[2] == 0.0
    macro = evaluate_entities({"a": set(), "b": {"S2-1"}}, {"a": set(), "b": set()})
    assert macro["macro_f05"] == 0.5
    assert macro["n_zero_prediction_s1"] == 2


if __name__ == "__main__":
    _self_check()
    print("evaluation self-check passed")

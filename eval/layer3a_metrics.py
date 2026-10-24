"""Checkpoint-level metrics for defensible Layer 3a reports.

Patch rows are aggregated by ``base_checkpoint_id`` before confidence intervals are
computed, preventing a large tensor from masquerading as many independent models.
"""

from __future__ import annotations

import math
import random
from typing import Iterable


def binary_metrics(rows: Iterable[dict], threshold: float = 0.5) -> dict[str, float | int]:
    rows = list(rows)
    tp = fp = tn = fn = 0
    for row in rows:
        truth = int(row["label"])
        predicted = int(float(row["score"]) >= threshold)
        if truth and predicted:
            tp += 1
        elif not truth and predicted:
            fp += 1
        elif not truth and not predicted:
            tn += 1
        else:
            fn += 1
    positives = tp + fn
    negatives = tn + fp
    total = len(rows)
    tpr = tp / positives if positives else 0.0
    fpr = fp / negatives if negatives else 0.0
    return {
        "n": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": 1.0 - tpr if positives else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (tpr + (tn / negatives if negatives else 0.0)) / 2.0,
    }


def group_mean_metrics(rows: Iterable[dict], threshold: float = 0.5) -> dict[str, float | int]:
    """Average per-checkpoint metrics so each checkpoint has equal weight."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["base_checkpoint_id"]), []).append(row)
    values = [binary_metrics(group, threshold) for group in grouped.values()]
    if not values:
        return {"groups": 0}
    keys = ("tpr", "fpr", "fnr", "precision", "accuracy", "balanced_accuracy")
    return {"groups": len(values), **{key: sum(float(value[key]) for value in values) / len(values) for key in keys}}


def bootstrap_group_ci(rows: Iterable[dict], threshold: float = 0.5, iterations: int = 2000, seed: int = 42) -> dict[str, tuple[float, float]]:
    """Bootstrap checkpoint groups, not patches, and return percentile 95% CIs."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["base_checkpoint_id"]), []).append(row)
    groups = list(grouped.values())
    if not groups:
        return {}
    rng = random.Random(seed)
    keys = ("tpr", "fpr", "fnr", "balanced_accuracy")
    samples = {key: [] for key in keys}
    for _ in range(iterations):
        resampled = [groups[rng.randrange(len(groups))] for _ in groups]
        summary = group_mean_metrics([row for group in resampled for row in group], threshold)
        for key in keys:
            samples[key].append(float(summary[key]))
    result = {}
    for key, values in samples.items():
        values.sort()
        low = values[max(0, math.floor(0.025 * len(values)))]
        high = values[min(len(values) - 1, math.floor(0.975 * len(values)))]
        result[key] = (low, high)
    return result

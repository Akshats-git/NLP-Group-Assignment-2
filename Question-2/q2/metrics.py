"""Q2 metrics: accuracy, macro / per-class F1, MAE in stars, and how far off the errors are.

Labels are 0-4, so the difference between two labels equals the difference in stars and
MAE needs no shift.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import NUM_CLASSES


def classification_metrics(labels, preds) -> dict:
    """Scalars first (so common.metrics.summarize_runs can average them over seeds), then lists.

    off_by_one / off_by_two_plus are shares of all reviews; adjacent_share_of_errors is the
    share of mistakes that are only one star off; extreme_confusions counts 1 <-> 5 star swaps.
    """
    labels, preds = np.asarray(labels), np.asarray(preds)
    distance = np.abs(labels - preds)
    errors = int((distance > 0).sum())
    classes = list(range(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=classes, zero_division=0)),
        "mae": float(distance.mean()),
        "off_by_one": float((distance == 1).mean()),
        "off_by_two_plus": float((distance >= 2).mean()),
        "adjacent_share_of_errors": float((distance == 1).sum() / errors) if errors else 0.0,
        "extreme_confusions": int((distance == NUM_CLASSES - 1).sum()),
        "per_class_f1": f1_score(labels, preds, average=None, labels=classes, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds, labels=classes).tolist(),
    }

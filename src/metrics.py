"""Classification metrics and paired uncertainty estimates."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import CLASS_NAMES


def classification_metrics(targets, predictions):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro")),
        "weighted_f1": float(
            f1_score(targets, predictions, labels=labels, average="weighted")
        ),
        "per_class_f1": {
            name: float(score)
            for name, score in zip(
                CLASS_NAMES,
                f1_score(targets, predictions, labels=labels, average=None),
            )
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "class_names": CLASS_NAMES,
    }


def paired_bootstrap_delta(targets, baseline, candidate, seed=42, samples=2000):
    """Sentence-level paired bootstrap interval for macro-F1 improvement."""
    targets = np.asarray(targets)
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        indices = rng.integers(0, len(targets), size=len(targets))
        base = f1_score(targets[indices], baseline[indices], average="macro")
        cand = f1_score(targets[indices], candidate[indices], average="macro")
        differences.append(cand - base)
    low, high = np.percentile(differences, [2.5, 97.5])
    observed = (
        f1_score(targets, candidate, average="macro")
        - f1_score(targets, baseline, average="macro")
    )
    return {
        "delta_macro_f1": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": int(samples),
    }

"""Shared numerical evaluation and training-only input conditioning helpers."""

import numpy as np

CLASS_NAMES = ("Normal", "Warning", "Critical")


def conditioning_scales(training_features):
    values = np.asarray(training_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("Training conditioning requires a nonempty (N, 6) matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("Training features must be finite")
    return np.maximum(np.max(np.abs(values), axis=0), 1.0).astype(np.float32)


def fold_first_kernel(kernel, scales):
    kernel, scales = np.asarray(kernel, dtype=np.float32), np.asarray(scales, dtype=np.float32)
    if kernel.ndim != 2 or kernel.shape[0] != 6 or scales.shape != (6,):
        raise ValueError("Expected a six-input kernel and six conditioning scales")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Conditioning scales must be finite and positive")
    return kernel / scales[:, None]


def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) == 0 or
            probabilities.shape != (len(labels), 3)):
        raise ValueError("Expected integer labels and an (N, 3) probability matrix")
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("Evaluation must include all three classes")
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-6) or
            np.any(probabilities > 1 + 1e-6) or
            not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities")
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    support = matrix.sum(axis=1)
    recall = np.diag(matrix) / support
    correct = int(np.trace(matrix))
    return {
        "sample_count": int(len(labels)), "correct_count": correct,
        "accuracy_percent": float(100 * correct / len(labels)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_convention": "rows=true, columns=predicted",
        "class_names": list(CLASS_NAMES), "class_support": support.tolist(),
        "recall_percent": (100 * recall).tolist(),
        "critical_support": int(support[2]),
        "critical_recall_percent": float(100 * recall[2]),
        "accuracy_requirement_met": bool(correct / len(labels) > 0.88),
        "critical_recall_requirement_met": bool(recall[2] > 0.95),
    }

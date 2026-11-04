"""Verify a frozen dataset and its fixed-statistics provenance before training."""

import csv
import json
from pathlib import Path

import numpy as np

from data_pipeline.normalization import FrozenNormalizer, file_sha256
from data_pipeline.preprocessing import FEATURE_NAMES, RECIPE

ARRAY_NAMES = ("X_raw", "X_normalized", "y", "train_indices",
               "validation_indices", "purged_indices")


def validate_arrays(arrays):
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Dataset array names do not match the expected format")
    raw, normalized, labels = arrays["X_raw"], arrays["X_normalized"], arrays["y"]
    if raw.ndim != 2 or raw.shape[1] != 6 or normalized.shape != raw.shape:
        raise ValueError("Dataset must contain matching raw and normalized (N, 6) arrays")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(normalized)):
        raise ValueError("Dataset features must be finite")
    count = len(raw)
    if labels.shape != (count,) or labels.dtype.kind not in "iu" or set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("Dataset needs integer scenario labels 0, 1 and 2")
    partitions = []
    for key in ("train_indices", "validation_indices", "purged_indices"):
        indices = arrays[key]
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"Invalid index array: {key}")
        if len(indices) and (indices.min() < 0 or indices.max() >= count):
            raise ValueError(f"Index outside the dataset: {key}")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError(f"Duplicate index in {key}")
        partitions.append(set(indices.tolist()))
    if (partitions[0] & partitions[1] or partitions[0] & partitions[2] or
            partitions[1] & partitions[2] or set.union(*partitions) != set(range(count))):
        raise ValueError("Train, validation and purged indices must partition the dataset")
    for key in ("train_indices", "validation_indices"):
        if set(labels[arrays[key]].tolist()) != {0, 1, 2}:
            raise ValueError(f"All three classes must be represented in {key}")
    if len(arrays["train_indices"]) < 200:
        raise ValueError("Fewer than 200 training windows remain for INT8 calibration")


def validate_temporal_split(rows, train_indices, validation_indices):
    for run_id in {row["source_run_id"] for row in rows}:
        training = [rows[int(index)] for index in train_indices
                    if rows[int(index)]["source_run_id"] == run_id]
        validation = [rows[int(index)] for index in validation_indices
                      if rows[int(index)]["source_run_id"] == run_id]
        if not training or not validation:
            raise ValueError("Every source run needs both training and validation windows")
        last_training_raw = max(float(row["raw_support_end_unix"]) for row in training)
        first_validation_raw = min(float(row["raw_support_start_unix"]) for row in validation)
        if last_training_raw >= first_validation_raw:
            raise ValueError("Training and validation raw-sample support overlaps")


def load_dataset(output_dir=Path("training/dataset"), stats_path=None):
    folder = Path(output_dir)
    meta = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
    if meta.get("feature_names") != list(FEATURE_NAMES) or meta.get("recipe") != RECIPE:
        raise ValueError("Dataset feature order or preprocessing recipe differs from the code")
    archive = folder / "logibridge_dataset.npz"
    csv_path = folder / "dataset_windows.csv"
    if file_sha256(archive) != meta["dataset_sha256"] or file_sha256(csv_path) != meta["windows_csv_sha256"]:
        raise ValueError("Dataset files differ from their recorded hashes")
    with np.load(archive, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    validate_arrays(arrays)
    normalizer = FrozenNormalizer.load(stats_path or meta["normalization_stats_file"])
    if normalizer.stats_sha256 != meta["normalization_stats_sha256"]:
        raise ValueError("Dataset was normalized with different frozen statistics")
    expected = normalizer.transform(arrays["X_raw"])
    if not np.allclose(arrays["X_normalized"], expected, rtol=1e-6, atol=1e-6):
        raise ValueError("Normalized features do not match the frozen statistics")
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(arrays["y"]):
        raise ValueError("Window metadata and dataset row counts differ")
    validate_temporal_split(rows, arrays["train_indices"], arrays["validation_indices"])
    return arrays, meta

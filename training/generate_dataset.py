"""Prepare D1 scenario labels and a purged chronological 20% validation split."""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_pipeline.extract_features import extract_capture
from data_pipeline.normalization import (
    DEFAULT_STATS, FrozenNormalizer, file_sha256, metadata_path,
)
from data_pipeline.preprocessing import FEATURE_NAMES, RECIPE
from training.dataset_io import load_dataset, validate_arrays, validate_temporal_split

SCENARIOS = (
    ("none", 0, "Normal", 1200),
    ("temp_drift", 1, "Warning", 900),
    ("combined", 2, "Critical", 900),
)
LABEL_NOTE = (
    "Scenario labels follow the D1 table. The unbounded temp_drift scenario "
    "exceeds the physical Critical temperature threshold during much of its run. "
    "Scenario labels are not a substitute for operational temperature severity."
)


def validate_training_capture(manifest, mode, duration, normalizer):
    capture_name = manifest.get("capture_file")
    if not isinstance(capture_name, str) or Path(capture_name).name != capture_name:
        raise ValueError("Capture filename must stay inside its manifest directory")
    if manifest.get("purpose") != "training" or manifest.get("anomaly") != mode:
        raise ValueError(f"Need purpose=training and anomaly={mode}")
    requested, elapsed = float(manifest["duration_requested_s"]), float(manifest["elapsed_s"])
    if not math.isfinite(requested) or abs(requested - duration) > 1e-6:
        raise ValueError(f"{mode} requires a requested duration of {duration} seconds")
    if not math.isfinite(elapsed) or elapsed < duration - 0.1:
        raise ValueError(f"{mode} did not run for the required duration")
    if manifest.get("stop_reason") != "duration" or manifest.get("error"):
        raise ValueError(f"{mode} capture did not finish normally")
    if any(manifest["missed_deadlines"].values()):
        raise ValueError(f"{mode} capture has missed deadlines")
    if (manifest["counts"]["temperature"] != duration or
            manifest["counts"]["vibration_rms"] != duration // 2):
        raise ValueError(f"{mode} has unexpected sensor counts")
    if manifest["run_id"] == normalizer.metadata["source_run_id"]:
        raise ValueError("Normalization and training must use different source runs")
    seed = int(manifest["seed"])
    if seed < 0 or seed == int(normalizer.metadata["source_seed"]):
        raise ValueError("Use a different non-negative training seed from normalization")


def select_captures(data_dir, normalizer):
    candidates = {mode: [] for mode, _, _, _ in SCENARIOS}
    for path in Path(data_dir).glob("sensor_*.manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("purpose") == "training" and manifest.get("anomaly") in candidates:
            candidates[manifest["anomaly"]].append(path)
    selected = []
    for mode, _, _, duration in SCENARIOS:
        if not candidates[mode]:
            raise ValueError(f"Missing training capture for {mode}: collect {duration} real seconds")
        path = max(candidates[mode], key=lambda item: item.stat().st_mtime_ns)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_training_capture(manifest, mode, duration, normalizer)
        selected.append((path, manifest))
    seeds = [int(manifest["seed"]) for _, manifest in selected]
    if len(set(seeds)) != 3:
        raise ValueError("Use distinct seeds for the three training scenarios")
    if len({manifest["truck_id"] for _, manifest in selected}) != 1:
        raise ValueError("Use the same demonstration truck for all D1 captures")
    return selected


def validation_counts(sizes, fraction=0.2):
    """Allocate ceil(20% of all windows) proportionally, with deterministic ties."""
    if not 0 < fraction < 0.5 or any(size < 2 for size in sizes):
        raise ValueError("Need at least two windows per class and a sensible validation fraction")
    total = sum(sizes)
    target = math.ceil(total * fraction)
    quotas = [target * size / total for size in sizes]
    result = [max(1, math.floor(quota)) for quota in quotas]
    if sum(result) > target:
        raise ValueError("Too few total windows for a three-class validation split")
    order = sorted(range(len(sizes)), key=lambda index: (-(quotas[index] - math.floor(quotas[index])), index))
    for index in order[:target - sum(result)]:
        result[index] += 1
    if any(count >= size for count, size in zip(result, sizes)):
        raise ValueError("Validation allocation leaves an empty training class")
    return result


def chronological_split(groups, fraction=0.2):
    counts = validation_counts([len(group) for group in groups], fraction)
    training, validation, purged = [], [], []
    offset = 0
    for rows, count in zip(groups, counts):
        first_validation = len(rows) - count
        validation.extend(range(offset + first_validation, offset + len(rows)))
        earliest_raw = min(row["raw_support_start_unix"] for row in rows[first_validation:])
        for local_index, row in enumerate(rows[:first_validation]):
            destination = training if row["raw_support_end_unix"] < earliest_raw else purged
            destination.append(offset + local_index)
        offset += len(rows)
    return tuple(np.asarray(indices, dtype=np.int64)
                 for indices in (training, validation, purged))


def build_dataset(data_dir=Path("data/raw"), output_dir=Path("training/dataset"), stats_path=DEFAULT_STATS):
    data_dir, output_dir, stats_path = Path(data_dir), Path(output_dir), Path(stats_path)
    normalizer = FrozenNormalizer.load(stats_path)
    stats_before = (file_sha256(stats_path), file_sha256(metadata_path(stats_path)))
    selected = select_captures(data_dir, normalizer)
    lineage = [{"mode": manifest["anomaly"], "manifest": str(path),
                "manifest_sha256": file_sha256(path),
                "capture_sha256": file_sha256(path.parent / manifest["capture_file"])}
               for path, manifest in selected]
    manifest_output = output_dir / "dataset_manifest.json"
    if manifest_output.exists():
        arrays, previous = load_dataset(output_dir, stats_path)
        if previous["sources"] != lineage:
            raise ValueError("The frozen dataset belongs to different captures; "
                             "use a new --output-dir for a deliberate new version")
        print("Using the existing verified dataset; validation indices are unchanged")
        return arrays, previous
    if any((output_dir / name).exists() for name in ("logibridge_dataset.npz", "dataset_windows.csv")):
        raise ValueError("Incomplete dataset output exists; inspect it before regenerating")

    groups, vectors, labels, intervals = [], [], [], []
    for (path, expected), (mode, label, name, duration) in zip(selected, SCENARIOS):
        manifest, windows, origin, capture, pipeline = extract_capture(path)
        expected_windows = (duration - 40) // 10 + 1
        if len(windows) != expected_windows:
            raise ValueError(f"{mode}: expected {expected_windows} complete windows, got {len(windows)}")
        intervals.append((origin, origin + float(manifest["elapsed_s"])))
        rows = []
        for window in windows:
            rows.append({
                "source_run_id": manifest["run_id"],
                "source_anomaly": mode, "scenario_label": label, "class_name": name,
                "window_index": window.window_index,
                "window_start_unix": window.window_start_unix,
                "window_end_unix": window.window_end_unix,
                "raw_support_start_unix": window.raw_support_start_unix,
                "raw_support_end_unix": window.raw_support_end_unix,
            })
            vectors.append(window.features)
            labels.append(label)
        groups.append(rows)
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError("Training captures overlap in wall-clock time; run one simulator at a time")

    rows = [row for group in groups for row in group]
    train, validation, purged = chronological_split(groups)
    raw = np.asarray(vectors, dtype=np.float32)
    normalized = normalizer.transform(raw)
    arrays = {"X_raw": raw, "X_normalized": normalized,
              "y": np.asarray(labels, dtype=np.int64), "train_indices": train,
              "validation_indices": validation, "purged_indices": purged}
    validate_arrays(arrays)
    validate_temporal_split(rows, train, validation)
    if stats_before != (file_sha256(stats_path), file_sha256(metadata_path(stats_path))):
        raise ValueError("Frozen normalization files changed while building the dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    roles = {int(index): role for role, indices in (
        ("train", train), ("validation", validation), ("purged", purged)) for index in indices}
    csv_path = output_dir / "dataset_windows.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=(*rows[0], "split", *FEATURE_NAMES))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({**row, "split": roles[index], **dict(zip(FEATURE_NAMES, raw[index]))})
    archive = output_dir / "logibridge_dataset.npz"
    with archive.open("xb") as output:
        np.savez_compressed(output, **arrays)
    per_class = []
    for mode, label, name, _ in SCENARIOS:
        per_class.append({
            "mode": mode, "label": label, "name": name,
            "total": int(np.count_nonzero(arrays["y"] == label)),
            "training": int(np.count_nonzero(arrays["y"][train] == label)),
            "validation": int(np.count_nonzero(arrays["y"][validation] == label)),
            "purged": int(np.count_nonzero(arrays["y"][purged] == label)),
        })
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES), "recipe": RECIPE,
        "label_note": LABEL_NOTE, "sources": lineage,
        "normalization_stats_file": str(stats_path),
        "normalization_stats_sha256": normalizer.stats_sha256,
        "normalization_source_run_id": normalizer.metadata["source_run_id"],
        "normalization_source_seed": normalizer.metadata["source_seed"],
        "total_windows": len(rows), "training_windows": len(train),
        "validation_windows": len(validation), "purged_windows": len(purged),
        "requested_validation_fraction": 0.2,
        "actual_validation_fraction_before_purge": len(validation) / len(rows),
        "split_method": "chronological tail per scenario; purge overlapping raw-sample support",
        "per_class": per_class,
        "dataset_sha256": file_sha256(archive), "windows_csv_sha256": file_sha256(csv_path),
        "calibration_eligible_training_windows": len(train),
    }
    with manifest_output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
        output.write("\n")
    return load_dataset(output_dir, stats_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/dataset"))
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args(argv)
    try:
        arrays, meta = build_dataset(args.data_dir, args.output_dir, args.stats)
        print("Feature order:", ", ".join(FEATURE_NAMES))
        print("Total feature shape:", arrays["X_raw"].shape)
        print("Training / validation / purged:", meta["training_windows"],
              meta["validation_windows"], meta["purged_windows"])
        for item in meta["per_class"]:
            print(f"{item['name']}: total={item['total']}, train={item['training']}, "
                  f"validation={item['validation']}, purged={item['purged']}")
        print("Validation fraction before purging:", round(meta["actual_validation_fraction_before_purge"], 6))
        print("Eligible training-only INT8 calibration samples:", meta["calibration_eligible_training_windows"])
        print("Frozen normalization SHA256:", meta["normalization_stats_sha256"])
        print("Dataset SHA256:", meta["dataset_sha256"])
        print("Dataset folder:", args.output_dir)
        print("Label interpretation:", LABEL_NOTE)
        print("Model training and accuracy evaluation: next step")
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        print(f"Dataset preparation stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

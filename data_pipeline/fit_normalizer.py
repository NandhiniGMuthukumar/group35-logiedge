"""Fit once from a separate ten-minute clean capture and save fixed statistics."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_pipeline.extract_features import save_features
from data_pipeline.normalization import (
    DEFAULT_STATS, STD_FLOOR, FrozenNormalizer, file_sha256, metadata_path,
)
from data_pipeline.preprocessing import FEATURE_NAMES, FEATURE_UNITS, RECIPE


def validate_source(manifest):
    if manifest.get("purpose") != "normalization" or manifest.get("anomaly") != "none":
        raise ValueError("Use purpose=normalization and anomaly=none, not smoke or training data")
    requested = float(manifest["duration_requested_s"])
    elapsed = float(manifest["elapsed_s"])
    if not math.isfinite(requested) or abs(requested - 600.0) > 1e-6:
        raise ValueError("Normalization requires a separately requested 600-second capture")
    if not math.isfinite(elapsed) or elapsed < 599.9:
        raise ValueError("The normalization capture did not last ten minutes")
    if manifest.get("stop_reason") != "duration" or manifest.get("error"):
        raise ValueError("The ten-minute capture must finish normally")
    if any(manifest["missed_deadlines"].values()):
        raise ValueError("Resolve missed sensor deadlines before fitting statistics")
    if (manifest["counts"]["temperature"] != 600 or
            manifest["counts"]["vibration_rms"] != 300):
        raise ValueError("Expected 600 temperature and 300 vibration readings")


def latest_normalization(data_dir):
    candidates = []
    for path in Path(data_dir).glob("sensor_*.manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("purpose") == "normalization":
            candidates.append(path)
    if not candidates:
        raise ValueError("No normalization capture found; let the 600-second run finish")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def save_new_stats(path, matrix, meta):
    path = Path(path)
    sidecar = metadata_path(path)
    if path.suffix != ".npy":
        raise ValueError("Statistics output filename must end in .npy")
    if path.exists() or sidecar.exists():
        raise ValueError("Statistics or metadata already exist; refusing to overwrite them")
    path.parent.mkdir(parents=True, exist_ok=True)
    made_stats = made_meta = False
    try:
        with path.open("xb") as output:
            made_stats = True
            np.save(output, matrix, allow_pickle=False)
        meta["stats_sha256"] = file_sha256(path)
        with sidecar.open("x", encoding="utf-8") as output:
            made_meta = True
            json.dump(meta, output, indent=2, allow_nan=False)
            output.write("\n")
    except Exception:
        if made_meta:
            sidecar.unlink(missing_ok=True)
        if made_stats:
            path.unlink(missing_ok=True)
        raise


def fit_normalizer(manifest_path, stats_path=DEFAULT_STATS, processed_dir=Path("data/processed")):
    manifest_path, stats_path = Path(manifest_path), Path(stats_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source(manifest)
    capture_name = manifest["capture_file"]
    if Path(capture_name).name != capture_name:
        raise ValueError("Manifest capture_file must be a filename")
    capture_path = manifest_path.parent / capture_name
    manifest_digest = file_sha256(manifest_path)
    capture_digest = file_sha256(capture_path)

    if stats_path.exists() or metadata_path(stats_path).exists():
        existing = FrozenNormalizer.load(stats_path)
        if (existing.metadata["source_manifest_sha256"] != manifest_digest or
                existing.metadata["source_capture_sha256"] != capture_digest):
            raise ValueError("Existing frozen statistics belong to a different capture; "
                             "use a new output filename for a deliberately versioned fit")
        print("Using the existing statistics for this same capture; no refit performed")
        return existing

    _, array, feature_meta = save_features(manifest_path, Path(processed_dir))
    if array.shape != (57, 6):
        raise ValueError(f"Expected 57 complete clean windows; observed {array.shape}")
    values = array.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Baseline feature values must be finite")
    mean = values.mean(axis=0)
    raw_std = values.std(axis=0, ddof=0)
    std = np.maximum(raw_std, STD_FLOOR)
    feature_path = Path(processed_dir) / f"{capture_path.stem}.features.npy"
    meta = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats_rows": ["mean", "std"],
        "feature_names": list(FEATURE_NAMES),
        "feature_units": list(FEATURE_UNITS),
        "recipe": RECIPE,
        "std_ddof": 0,
        "std_floor": STD_FLOOR,
        "raw_std": raw_std.tolist(),
        "floored_features": [name for name, value in zip(FEATURE_NAMES, raw_std)
                             if value < STD_FLOOR],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_digest,
        "source_capture": str(capture_path),
        "source_capture_sha256": capture_digest,
        "source_run_id": manifest["run_id"],
        "source_seed": manifest["seed"],
        "source_purpose": manifest["purpose"],
        "source_anomaly": manifest["anomaly"],
        "source_duration_s": manifest["elapsed_s"],
        "source_feature_npy": str(feature_path),
        "source_feature_npy_sha256": file_sha256(feature_path),
        "window_count": int(array.shape[0]),
        "feature_input_dtype": "float32",
        "statistics_dtype": "float64",
    }
    save_new_stats(stats_path, np.stack((mean, std)), meta)
    return FrozenNormalizer.load(stats_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--latest-normalization", action="store_true")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args(argv)
    try:
        manifest_path = latest_normalization(args.data_dir) if args.latest_normalization else args.manifest
        fitted = fit_normalizer(manifest_path, args.output, args.processed_dir)
        print(f"\nStatistics: {args.output} | shape: (2, 6)")
        print(f"Baseline windows: {fitted.metadata['window_count']} | source seed: {fitted.metadata['source_seed']}")
        print("Feature order:", ", ".join(FEATURE_NAMES))
        print("Mean:", [round(float(value), 8) for value in fitted.mean])
        print("Std:", [round(float(value), 8) for value in fitted.std])
        print("Floored features:", fitted.metadata["floored_features"])
        print("Statistics SHA256:", fitted.stats_sha256)
        print("Metadata:", metadata_path(args.output))
        print("Runtime behaviour: load and transform only; no fitting from live readings")
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        print(f"Normalization fit stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

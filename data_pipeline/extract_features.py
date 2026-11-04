"""Extract causal, unnormalised features from genuine simulator captures."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_pipeline.preprocessing import FEATURE_NAMES, FEATURE_UNITS, RECIPE, FeaturePipeline
from data_pipeline.simulator import MODES, STREAMS

CSV_METADATA = (
    "source_run_id", "source_anomaly", "source_purpose", "window_index",
    "window_start_s", "window_end_s", "raw_support_start_s",
    "temperature_samples", "vibration_samples", "door_state_at_end",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_capture(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("error") or manifest.get("stop_reason") not in ("duration", "interrupted"):
        raise ValueError(f"Capture did not finish cleanly: {manifest_path.name}")
    if any(manifest["missed_deadlines"].values()):
        raise ValueError("Capture has missed deadlines; inspect it before using it")
    capture_name = manifest["capture_file"]
    if not isinstance(capture_name, str) or Path(capture_name).name != capture_name:
        raise ValueError("Manifest capture_file must be a filename, not a path")
    capture_path = manifest_path.parent / capture_name
    counts = Counter({stream: 0 for stream in STREAMS})
    windows = []
    with capture_path.open(encoding="utf-8") as source:
        records = (json.loads(line) for line in source if line.strip())
        first = next(records, None)
        if first is None:
            raise ValueError("Capture contains no sensor readings")
        origin = float(first["timestamp_unix"]) - float(first["elapsed_s"])
        pipeline = FeaturePipeline(origin, strict=True)
        for record in itertools.chain([first], records):
            if (record["run_id"] != manifest["run_id"] or
                    record["anomaly"] != manifest["anomaly"] or
                    record["truck_id"] != manifest["truck_id"]):
                raise ValueError("Sensor metadata does not match its run manifest")
            elapsed = float(record["elapsed_s"])
            if (not math.isfinite(elapsed) or elapsed < 0 or
                    abs(float(record["timestamp_unix"]) - origin - elapsed) > 0.001):
                raise ValueError("Sensor elapsed time does not match its timestamp")
            windows.extend(pipeline.add_record(record))
            counts[record["stream"]] += 1
    if dict(counts) != manifest["counts"]:
        raise ValueError("Saved record counts disagree with the simulator manifest")
    if pipeline.duplicates_ignored:
        raise ValueError("Duplicate messages in the saved capture; inspect its source")
    elapsed_total = float(manifest["elapsed_s"])
    if not math.isfinite(elapsed_total) or elapsed_total < 0:
        raise ValueError("Invalid capture duration")
    windows.extend(pipeline.finish(origin + elapsed_total))
    if not windows:
        raise ValueError("No complete filtered window: the first normally closes at 40 seconds")
    return manifest, windows, origin, capture_path, pipeline


def csv_row(manifest, window, origin):
    result = {
        "source_run_id": manifest["run_id"],
        "source_anomaly": manifest["anomaly"],
        "source_purpose": manifest["purpose"],
        "window_index": window.window_index,
        "window_start_s": window.window_start_unix - origin,
        "window_end_s": window.window_end_unix - origin,
        "raw_support_start_s": window.raw_support_start_unix - origin,
        "temperature_samples": window.temperature_samples,
        "vibration_samples": window.vibration_samples,
        "door_state_at_end": window.door_state_at_end,
    }
    result.update(zip(FEATURE_NAMES, window.features))
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=(*CSV_METADATA, *FEATURE_NAMES))
        writer.writeheader()
        writer.writerows(rows)


def save_features(manifest_path, output_dir):
    manifest, windows, origin, capture, pipeline = extract_capture(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = capture.stem
    array = np.asarray([window.features for window in windows], dtype=np.float32)
    rows = [csv_row(manifest, window, origin) for window in windows]
    write_csv(output_dir / f"{stem}.features.csv", rows)
    np.save(output_dir / f"{stem}.features.npy", array, allow_pickle=False)
    with (output_dir / f"{stem}.features.jsonl").open("w", encoding="utf-8") as output:
        for window in windows:
            output.write(json.dumps(window.as_record(), allow_nan=False) + "\n")
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(Path(manifest_path)),
        "source_capture": str(capture),
        "source_capture_sha256": sha256(capture),
        "source_run_id": manifest["run_id"],
        "source_anomaly": manifest["anomaly"],
        "source_purpose": manifest["purpose"],
        "source_seed": manifest["seed"],
        "source_duration_s": manifest["elapsed_s"],
        "origin_unix": origin,
        "feature_names": list(FEATURE_NAMES),
        "feature_units": list(FEATURE_UNITS),
        "recipe": RECIPE,
        "array_shape": list(array.shape),
        "normalised": False,
        "rejected_windows": pipeline.rejected_windows,
    }
    (output_dir / f"{stem}.features.meta.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    first, last = windows[0], windows[-1]
    print(f"\nMode: {manifest['anomaly']} | purpose: {manifest['purpose']} | shape: {array.shape}")
    print(f"First window: {first.window_start_unix-origin:.0f}-{first.window_end_unix-origin:.0f}s"
          f" | last window ends: {last.window_end_unix-origin:.0f}s")
    print(f"First window samples: temperature={first.temperature_samples}, "
          f"vibration={first.vibration_samples}; door state={first.door_state_at_end}")
    print("First feature vector:", [round(value, 6) for value in first.features])
    print(f"Saved: {output_dir / (stem + '.features.csv')}")
    return rows, array, metadata


def select_smoke_tests(data_dir):
    candidates = {mode: [] for mode in MODES}
    for path in data_dir.glob("sensor_*.manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("purpose") == "smoke_test" and
                manifest.get("anomaly") in candidates and
                manifest.get("stop_reason") == "duration" and not manifest.get("error")):
            candidates[manifest["anomaly"]].append(path)
    missing = [mode for mode, paths in candidates.items() if not paths]
    if missing:
        raise ValueError("Missing completed smoke captures: " + ", ".join(missing))
    return [max(candidates[mode], key=lambda path: path.stat().st_mtime_ns) for mode in MODES]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--smoke-tests", action="store_true")
    selection.add_argument("--latest", action="store_true")
    selection.add_argument("--manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)
    try:
        if args.smoke_tests:
            paths = select_smoke_tests(args.data_dir)
        elif args.latest:
            candidates = list(args.data_dir.glob("sensor_*.manifest.json"))
            if not candidates:
                raise ValueError(f"No simulator manifests in {args.data_dir}")
            paths = [max(candidates, key=lambda path: path.stat().st_mtime_ns)]
        else:
            paths = [args.manifest]
        print("Feature order:", ", ".join(FEATURE_NAMES))
        print("Causal five-sample filters | window 30s | step 10s | full filter warm-up")
        combined_rows = []
        for path in paths:
            rows, _, _ = save_features(path, args.output_dir)
            combined_rows.extend(rows)
        if args.smoke_tests:
            write_csv(args.output_dir / "smoke_features.csv", combined_rows)
            print(f"\nSmoke feature table: {len(combined_rows)} rows, 6 numerical features per row")
            print(f"Saved: {args.output_dir / 'smoke_features.csv'}")
        print("Saved raw feature vectors; fixed normalization is applied separately for model inputs")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(f"Feature extraction stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect an actual sensor capture and its matching run manifest."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from data_pipeline.simulator import STREAMS


def inspect(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest_path.parent / manifest["capture_file"]
    if not capture.is_file():
        raise ValueError(f"Capture file missing: {capture}")
    records = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    grouped = {stream: [] for stream in STREAMS}
    seen = set()
    for record in records:
        stream = record["stream"]
        if stream not in grouped:
            raise ValueError(f"Unknown sensor stream: {stream}")
        if record["message_id"] in seen:
            raise ValueError("Duplicate message_id in capture")
        seen.add(record["message_id"])
        if (record["run_id"] != manifest["run_id"] or
                record["anomaly"] != manifest["anomaly"] or
                record["truck_id"] != manifest["truck_id"]):
            raise ValueError("Record metadata does not match the manifest")
        if record["seq"] != len(grouped[stream]):
            raise ValueError(f"Sequence gap in {stream}")
        if not math.isfinite(record["timestamp_unix"]):
            raise ValueError("Invalid timestamp")
        if grouped[stream] and record["timestamp_unix"] <= grouped[stream][-1]["timestamp_unix"]:
            raise ValueError(f"Non-increasing timestamps in {stream}")
        if stream == "door_event":
            if record["value"] not in ("OPEN", "CLOSE"):
                raise ValueError("Invalid door event")
        elif (not isinstance(record["value"], (int, float)) or
              not math.isfinite(record["value"])):
            raise ValueError(f"Invalid numeric value in {stream}")
        grouped[stream].append(record)
    observed = {stream: len(rows) for stream, rows in grouped.items()}
    if observed != manifest["counts"]:
        raise ValueError("Saved record counts do not match acknowledged counts")
    print(f"Manifest: {manifest_path}")
    print(f"Anomaly: {manifest['anomaly']} | seed: {manifest['seed']} | purpose: {manifest['purpose']}")
    print(f"Stop reason: {manifest['stop_reason']} | elapsed: {manifest['elapsed_s']:.2f}s")
    print(f"Counts: {observed}")
    print(f"Missed deadlines: {manifest['missed_deadlines']}")
    for stream in ("temperature", "vibration_rms"):
        rows = grouped[stream]
        values = [row["value"] for row in rows]
        if not values:
            print(f"{stream}: no readings")
            continue
        intervals = [right["timestamp_unix"] - left["timestamp_unix"]
                     for left, right in zip(rows, rows[1:])]
        cadence = f"{statistics.median(intervals):.3f}s" if intervals else "not enough samples"
        print(f"{stream}: mean={statistics.mean(values):.4f}, "
              f"std={statistics.pstdev(values):.4f}, median interval={cadence}")
    doors = grouped["door_event"]
    print(f"Door states: {dict(Counter(row['value'] for row in doors))}")
    for stream in STREAMS:
        if grouped[stream]:
            print(f"Example {stream}: {json.dumps(grouped[stream][0])}")
    if manifest["error"]:
        raise ValueError(f"Capture ended with an error: {manifest['error']}")
    return manifest, grouped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--latest", action="store_true")
    selection.add_argument("--manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    try:
        manifest = args.manifest
        if args.latest:
            files = list(args.data_dir.glob("sensor_*.manifest.json"))
            if not files:
                raise ValueError(f"No sensor manifests found under {args.data_dir}")
            manifest = max(files, key=lambda path: path.stat().st_mtime_ns)
        inspect(manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Capture inspection failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

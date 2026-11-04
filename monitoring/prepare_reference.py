"""Run real project-model inference on independent clean and diagnostic data.

Sensor timestamps are nominal simulation time; no real-time measurements or
MQTT monitoring claims are made by this offline preparation command.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from data_pipeline.preprocessing import FEATURE_NAMES, RECIPE, FeaturePipeline
from monitoring.psi import (DIAGNOSTIC_SCORE, PRIMARY_SCORE, RollingConfidencePSI,
                            canonical, confidence_scores, file_sha256, load_reference,
                            make_reference, require, write_json)
from monitoring.scenario import nominal_records

REFERENCE_SECONDS = 3035
PILOT_PHASES = ((0, "none"), (1100, "combined"), (1400, "none"))
PILOT_SECONDS = 2505


def outputs(model, duration, seed, phases, raw_output=None, origin=None, run_id=None):
    origin = time.time() - duration if origin is None else origin
    run_id = run_id or uuid.uuid4().hex
    pipeline = FeaturePipeline(origin)
    for record in nominal_records(duration, seed, phases, origin, run_id):
        if raw_output is not None:
            raw_output.write(canonical(record) + "\n")
        for window in pipeline.add_record(record):
            prediction = model.predict(window.features)
            normality, maximum = confidence_scores(prediction)
            event = {"schema_version": 1, "kind": "inference", "event_id": f"offline:{run_id}:{window.window_index}",
                     "run_id": run_id, "truck_id": window.truck_id,
                     "model_variant": model.variant, "model_sha256": model.sha256,
                     "window_start_unix": window.window_start_unix, "window_end_unix": window.window_end_unix,
                     "window_end_logical_s": window.window_end_unix - origin,
                     "processing_deadline_logical_s": record["elapsed_s"],
                     "raw_features": list(window.features), **prediction,
                     "normality_confidence": normality, "confidence_max": maximum}
            yield event
    require(not pipeline.rejected_windows, "The nominal-timestamp replay rejected feature windows")


def save_csv(path, events):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_end_logical_s", *FEATURE_NAMES, "p_normal", "p_warning", "p_critical", "model_class"])
        for event in events:
            writer.writerow([event["window_end_logical_s"], *event["raw_features"],
                             *event["probabilities"], event["model_class"]])


def reference(model, args, folder):
    path = args.output
    if path.exists() and not args.rebuild:
        value = load_reference(path)
        require(value["model_sha256"] == model.sha256 and
                value["normalization_stats_sha256"] == model.normalizer.stats_sha256 and
                value["source"]["seed"] == args.seed,
                "Existing reference belongs to different inputs; inspect it before an explicit --rebuild")
        for artifact in value["source"]["artifacts"]:
            file = Path(artifact["path"])
            require(file.resolve().is_relative_to(Path.cwd().resolve()), "Reference artifact is outside the project")
            require(file.is_file() and file_sha256(file) == artifact["sha256"], "Reference artifact missing or modified")
        print("Reusing the verified frozen 300-window reference", flush=True)
        return value
    print("Generating 300 clean windows at nominal timestamps; no real-time waiting", flush=True)
    raw_path = folder / "reference_raw.jsonl"
    with raw_path.open("x", encoding="utf-8") as raw:
        events = list(outputs(model, REFERENCE_SECONDS, args.seed, ((0, "none"),), raw))
    require(len(events) == 300, "Expected exactly 300 complete causal reference windows")
    inference_path, csv_path = folder / "reference_inferences.jsonl", folder / "reference_features.csv"
    inference_path.write_text("".join(canonical(event) + "\n" for event in events))
    save_csv(csv_path, events)
    value = make_reference(events, created_utc=datetime.now(timezone.utc).isoformat(),
                           model_variant=model.variant, model_sha256=model.sha256,
                           normalization_stats_sha256=model.normalizer.stats_sha256,
                           normalization_metadata_sha256=file_sha256(args.stats.with_suffix(".meta.json")),
                           truck_id="TRUCK_001",
                           feature_names=list(FEATURE_NAMES), recipe=RECIPE,
                           clean_scenario_predicted_class_counts=dict(Counter(str(event["model_class"]) for event in events)),
                           source={"execution_mode": "offline_nominal_timestamp_replay",
                                   "truth": "Normal scenario; independently seeded from training and normalisation",
                                   "seed": args.seed, "logical_sensor_duration_s": REFERENCE_SECONDS,
                                   "window_count": 300, "first_window_end_s": events[0]["window_end_logical_s"],
                                   "last_window_end_s": events[-1]["window_end_logical_s"],
                                   "artifacts": [{"path": str(file), "sha256": file_sha256(file)}
                                                 for file in (raw_path, inference_path, csv_path)]})
    write_json(folder / "reference_dist.json", value)
    write_json(path, value)
    return value


def diagnostic(model, ref, seed, folder):
    monitor = RollingConfidencePSI(ref, start_time=0)
    evaluations, events = [], []
    # Feed actual model probabilities into one unreset rolling monitor. Tick on
    # logical sensor deadlines; these are not observed wall-clock timings.
    origin = time.time() - PILOT_SECONDS
    next_event = iter(outputs(model, PILOT_SECONDS, seed, PILOT_PHASES, origin=origin))
    current = next(next_event, None)
    for elapsed in range(PILOT_SECONDS):
        while current is not None and current["processing_deadline_logical_s"] <= elapsed:
            monitor.observe(current)
            events.append(current)
            current = next(next_event, None)
        evaluation = monitor.tick(elapsed)
        if evaluation is not None:
            evaluations.append(evaluation)
    require(current is None, "Pilot left unconsumed model outputs")
    baseline = [row for row in evaluations if row["ready"] and row["elapsed_s"] < 1100]
    drift = [row for row in evaluations if row["ready"] and 1100 <= row["elapsed_s"] < 1400 and row["psi"] > .25]
    recovery = [row for row in evaluations if row["ready"] and row["elapsed_s"] >= 1400 and row["psi"] < .10]
    first_drift, first_recovery = (drift[0] if drift else None), (recovery[0] if recovery else None)
    primary = {"baseline_psi": baseline[-1]["psi"] if baseline else None,
               "first_drift_crossing_logical_elapsed_s": first_drift["elapsed_s"] if first_drift else None,
               "injection_to_crossing_logical_s": first_drift["elapsed_s"] - 1100 if first_drift else None,
               "first_recovery_logical_elapsed_s": first_recovery["elapsed_s"] if first_recovery else None,
               "clean_restore_to_recovery_logical_s": first_recovery["elapsed_s"] - 1400 if first_recovery else None,
               "recovery_psi": first_recovery["psi"] if first_recovery else None}
    maximum = max((row["max_confidence_psi"] for row in evaluations if row["ready"]), default=None)
    passed = bool(baseline and baseline[-1]["psi"] < .10 and first_drift and
                  first_drift["elapsed_s"] - 1100 <= 300 and first_recovery)
    result = {"execution_mode": "offline_nominal_timestamp_replay", "live_requirement_evidence": False,
              "reference_sha256": ref["reference_sha256"], "model_sha256": model.sha256,
              "seed": seed, "logical_sensor_duration_s": PILOT_SECONDS,
              "phase_plan_logical_seconds": list(PILOT_PHASES), "rolling_window": 100, "evaluation_interval_s": 60,
              "rolling_window_reset_on_phase_changes": False, "primary_score": PRIMARY_SCORE,
              "primary_check": primary, "max_class_confidence_peak_psi": maximum,
              "max_class_confidence_crossed_025": maximum is not None and maximum > .25,
              "offline_algorithm_check_passed": passed,
              "inference_count": len(events), "evaluations": evaluations}
    (folder / "pilot_inferences.jsonl").write_text("".join(canonical(event) + "\n" for event in events))
    write_json(folder / "offline_pilot_results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("deployment/runtime_assets/models/M1_FP32.tflite"))
    parser.add_argument("--stats", type=Path, default=Path("data_pipeline/training_stats.npy"))
    parser.add_argument("--registry", type=Path, default=Path("deployment/runtime_assets/runtime_registry.json"))
    parser.add_argument("--output", type=Path, default=Path("monitoring/reference_dist.json"))
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--pilot-seed", type=int, default=7002)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    from inference.runtime import LocalModel
    frozen = (args.model, args.stats, args.stats.with_suffix(".meta.json"), args.registry)
    before = {str(path): file_sha256(path) for path in frozen}
    model = LocalModel(args.model, args.stats, args.registry)
    folder = Path("monitoring/reference_runs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    ref = reference(model, args, folder)
    ref_bytes_before = file_sha256(args.output)
    pilot = diagnostic(model, ref, args.pilot_seed, folder)
    require(before == {str(path): file_sha256(path) for path in frozen}, "A protected runtime file changed")
    require(file_sha256(args.output) == ref_bytes_before, "The frozen reference changed during the pilot")
    write_json(Path("monitoring/reference_pointer.json"), {"reference": str(args.output), "latest_check": str(folder)})
    print("Reference count:", ref["sample_count"])
    print("Primary confidence:", PRIMARY_SCORE)
    print("P(Normal) reference bins:", ref["primary_distribution"]["counts"])
    print("Maximum-confidence reference bins:", ref["max_confidence_distribution"]["counts"])
    print("Reference clean-scenario predicted class counts:", ref["clean_scenario_predicted_class_counts"])
    print("Model / frozen statistics unchanged: True")
    print("Reference unchanged during the diagnostic: True")
    print("Reference SHA256:", ref["reference_sha256"])
    print("OFFLINE diagnostic using logical seconds; not live SLA evidence")
    print("Baseline PSI:", pilot["primary_check"]["baseline_psi"])
    print("Injection-to-crossing logical seconds:", pilot["primary_check"]["injection_to_crossing_logical_s"])
    print("Recovery PSI:", pilot["primary_check"]["recovery_psi"])
    print("Restore-to-recovery logical seconds:", pilot["primary_check"]["clean_restore_to_recovery_logical_s"])
    print("Maximum-class-confidence peak PSI:", pilot["max_class_confidence_peak_psi"])
    print("Offline algorithm check passed:", pilot["offline_algorithm_check_passed"])
    print("Reference:", args.output)
    print("Results:", folder / "offline_pilot_results.json")
    return 0 if pilot["offline_algorithm_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

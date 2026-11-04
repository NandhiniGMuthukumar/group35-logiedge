#!/usr/bin/env python3
"""Readable display of live status or explicitly labelled saved evidence."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def pointed(pointer, key):
    value = read(ROOT / pointer)[key]
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Evidence pointer is outside this project")
    return path


def run(section, original=False):
    if section == "status":
        process = subprocess.run(
            ["docker", "compose", "exec", "-T", "inference", "python", "-m", "inference.inspect_state"],
            cwd=ROOT, text=True, capture_output=True, timeout=25, check=True,
        )
        value = json.loads(process.stdout)
        service = value["service"]
        print("LIVE STATUS SNAPSHOT")
        print("Current run:", service.get("current_run_id"))
        print("Model:", service["variant"], "| ready:", service["model_ready"])
        print("MQTT connected:", service["mqtt_connected"])
        print("Current-run prediction ready:", service["current_run_prediction_ready"])
        print("Records:", service["records"], "| feature windows:", service["windows"])
        print("Operational state:", ("Normal", "Warning", "Critical")[service["operational_severity"]])
        print("Pending uploads:", value["pending_uplink"])
        print("Acknowledged uploads:", value["acknowledged_uplink"])
        remote = value["operations_centre"]
        if remote.get("available") is False:
            print("Operations centre: unavailable; local processing and queuing continue")
        else:
            print("Remote unique events:", remote["unique_received_events"])
        print("Historical UTC-derived alert durations are omitted from this compact display.")
        return
    if section == "benchmark":
        folder = pointed("optimisation/results/benchmark_pointer.json", "run_dir")
        value = read(folder / "benchmark_results.json")
        print("SAVED LAPTOP BENCHMARK:", folder.relative_to(ROOT))
        print("Variant                  mean_ms   p95_ms    KB      accuracy%  proxy_mJ")
        for row in value["rows"]:
            print(f"{row['variant']:25s} {row['mean_latency_ms']:.6f}  {row['p95_latency_ms']:.6f}  "
                  f"{row['model_KB_decimal']:.3f}  {row['accuracy_percent']:8.2f}  {row['energy_proxy_mJ']:.6f}")
        print("10 excluded warm-ups and 200 timed invokes per model.")
        print("Energy is a CPU/TDP proxy. These are saved measurements, not a new timing run.")
        return
    if section == "psi":
        folder = ROOT / "reports/psi_live_20261104T044934Z_L9uSpi" if original else pointed("reports/psi_live_pointer.json", "folder")
        result = read(folder / "live_results.json")
        value = read(folder / "audited_summary.json")
        if not value.get("audit_passed") or not result.get("passed"):
            raise ValueError("The selected live PSI run has not completed and passed its audit")
        print("SAVED AUDITED LIVE PSI:", folder.relative_to(ROOT))
        print("Run:", value["run_id"])
        print("Primary score: P(Normal). Rolling outputs: 100. Evaluation interval: 60 s.")
        print("Baseline PSI:", value["baseline_psi"])
        print("First drift PSI:", round(value["first_drift_psi"], 6))
        print("Injection to alert:", round(value["injection_to_alert_seconds"], 3), "s")
        print("Recovery PSI:", value["recovery_psi"])
        print("Clean restoration to recovery:", round(value["clean_restore_to_recovery_seconds"], 3), "s")
        print("Maximum-class-confidence peak PSI:", value["maximum_class_confidence_peak_psi"])
        print("Audited model outputs:", value["model_inferences"])
        print("Audit passed:", value["audit_passed"])
        return
    if section == "ota":
        target = pointed("reports/ota_pointer.json", "results")
        value = read(target)
        budget = value["transfer_budget"]
        print("SAVED MEASURED OTA:", target.relative_to(ROOT))
        print("Changed filesystem layers:", value["changed_filesystem_layers"])
        print("Earlier layers unchanged:", value["all_earlier_filesystem_layers_identical"])
        switch = value["model_path_switch_without_rebuild"]
        print("MODEL_PATH loaded without rebuild:", switch["variant"], "| input/output:", switch["input_dtype"], "/", switch["output_dtype"])
        print("Cold target payload per truck:", budget["cold_target_image_payload_bytes_per_truck"], "bytes")
        print("Cached update per truck:", budget["cached_update_payload_bytes_per_truck"], "bytes")
        print("85-truck cached payload:", budget["fleet_cached_update_payload_mb"], "decimal MB")
        print("Payload saving:", round(budget["bandwidth_saved_percent"], 6), "%")
        print("Assumes the previous image is cached. Protocol/cellular overhead is excluded.")
        return
    if section == "deployment":
        files = sorted((ROOT / "reports").glob("ansible_*/ansible_results.json"))
        if not files:
            raise ValueError("No completed two-run Ansible result exists")
        target = files[-1]
        value = read(target)
        print("SAVED ANSIBLE RESULT:", target.relative_to(ROOT))
        for name in ("first_run", "second_run"):
            row = value[name]
            print(name, "ok=", row["ok"], "changed=", row["changed"], "failed=", row["failed"], "unreachable=", row["unreachable"])
        print("Exact task count:", value["exact_task_count"])
        print("Second-run changed=0:", value["second_run_changed_zero"])
        print("Container ID unchanged:", value["container_id_unchanged_between_runs"])
        print("Start time unchanged:", value["container_started_at_unchanged_between_runs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("status", "benchmark", "psi", "ota", "deployment"))
    parser.add_argument("--original", action="store_true", help="Show the original audited 4 November PSI run")
    args = parser.parse_args()
    try:
        run(args.section, args.original)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit("Cannot display evidence: " + str(error))

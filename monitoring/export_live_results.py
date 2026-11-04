"""Audit the recorded output probabilities and plot actual completed live PSI."""

import csv
import json
from pathlib import Path

from monitoring.live_control import current
from monitoring.psi import RollingConfidencePSI, load_reference, require, write_json


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(folder):
    result = json.loads((folder / "live_results.json").read_text())
    require(result.get("passed") is True and result.get("status") == "completed",
            "The live experiment is not complete and passing; inspect its status first")
    clock = result.get('clock') or {}
    require(clock.get('guard_version') == 2 and clock.get('clock_review_required') is False and
            clock.get('last_interruption') is None, 'Clock evidence has not passed the corrected continuity checks')
    reference = load_reference(folder / "frozen_reference_dist.json")
    require(result["reference_sha256"] == reference["reference_sha256"], "Saved reference/result identity mismatch")
    events = json_lines(folder / "inferences.jsonl")
    evaluations = json_lines(folder / "psi_values.jsonl")
    require(len(events) == result["total_inferences"], "Saved inference count differs from final status")
    require(all(event["run_id"] == result["run_id"] for event in events), "Mixed run IDs in the evidence")
    require(all(b["monitor_elapsed_s"] >= a["monitor_elapsed_s"] for a, b in zip(events, events[1:])),
            "Observer timestamps moved backwards")
    replay = RollingConfidencePSI(reference, 0)
    index = 0
    for expected in evaluations:
        while index < len(events) and events[index]["monitor_elapsed_s"] <= expected["elapsed_s"]:
            replay.observe(events[index])
            index += 1
        actual = replay.tick(expected["elapsed_s"])
        require(actual is not None and actual["ready"] == expected["ready"], "Recorded evaluation cadence differs")
        require(actual["sample_count"] == expected["sample_count"], "Recorded rolling-window size differs")
        if expected["ready"]:
            require(abs(actual["psi"] - expected["psi"]) < 1e-10 and
                    abs(actual["max_confidence_psi"] - expected["max_confidence_psi"]) < 1e-10,
                    "Recorded PSI cannot be reproduced from the actual probabilities")
    ready = [row for row in evaluations if row["ready"]]
    require(ready, "No complete live PSI distributions were recorded")
    baseline = [row for row in ready if row["phase"] == "baseline_warmup" and row["psi"] < .10]
    changes = result["phase_changes"]
    require([row["mode"] for row in changes] == ["none", "combined", "none"], "Expected one continuous three-phase experiment")
    drift = [row for row in ready if row["phase"] == "combined_fault" and row["psi"] > .25]
    recovered = [row for row in ready if row["phase"] == "clean_recovery" and row["psi"] < .10]
    require(baseline and drift and recovered, "Required measured baseline/drift/recovery values are missing")
    elapsed = drift[0]["elapsed_s"] - changes[1]["elapsed_s"]
    require(0 <= elapsed <= 300, "Recorded live injection-to-alert exceeds five minutes")
    require(abs(elapsed - result["injection_to_alert_seconds"]) < 1e-8, "SLA summary differs from measured timestamps")
    require(result["reference_file_unchanged"] is True, "Reference was not frozen during the experiment")
    summary = {"audit_passed": True, "source": "actual saved MQTT model probabilities",
               "clock": clock,
               "run_id": result["run_id"], "reference_sha256": reference["reference_sha256"],
               "reproduced_full_psi_evaluations": len(ready), "model_inferences": len(events),
               "baseline_psi": baseline[-1]["psi"], "first_drift_psi": drift[0]["psi"],
               "injection_to_alert_seconds": elapsed, "recovery_psi": recovered[0]["psi"],
               "clean_restore_to_recovery_seconds": recovered[0]["elapsed_s"] - changes[2]["elapsed_s"],
               "maximum_class_confidence_peak_psi": max(row["max_confidence_psi"] for row in ready)}
    return result, evaluations, summary


def main():
    _, folder = current()
    result, evaluations, summary = audit(folder)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ready = [row for row in evaluations if row["ready"]]
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    minutes = [row["elapsed_s"] / 60 for row in ready]
    axis.plot(minutes, [row["psi"] for row in ready], color="#135D9D", marker="o", markersize=3, label="P(Normal) PSI")
    axis.plot(minutes, [row["max_confidence_psi"] for row in ready], color="#777777", linestyle="--", label="Maximum-class-confidence PSI")
    axis.axhline(.25, color="#B74334", linewidth=1, linestyle=":", label="Drift threshold 0.25")
    axis.axhline(.10, color="#36834A", linewidth=1, linestyle=":", label="Recovery threshold 0.10")
    axis.axvspan(0, minutes[0], color="#EEEEEE", label="Collecting the first 100 live outputs")
    for row, colour in zip(result["phase_changes"][1:], ("#B74334", "#36834A")):
        axis.axvline(row["elapsed_s"] / 60, color=colour, alpha=.75)
        axis.text(row["elapsed_s"] / 60, 1.02, row["mode"], transform=axis.get_xaxis_transform(),
                  color=colour, ha="center", fontsize=9)
    axis.set(xlabel="Elapsed real time (minutes)", ylabel="Population Stability Index",
             title="LogiEdge: live MQTT output-confidence drift and recovery",
             xlim=(0, result["elapsed_s"] / 60))
    axis.grid(alpha=.15)
    axis.legend(fontsize=8, loc="best")
    fig.savefig(folder / "psi_live_timeline.png", dpi=180)
    plt.close(fig)
    with (folder / "psi_live_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_real_seconds", "phase", "sample_count", "psi_p_normal", "psi_max_confidence"])
        for row in evaluations:
            writer.writerow([row["elapsed_s"], row["phase"], row["sample_count"], row["psi"], row["max_confidence_psi"]])
    write_json(folder / "audited_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Chart:", folder / "psi_live_timeline.png")
    print("Table:", folder / "psi_live_values.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

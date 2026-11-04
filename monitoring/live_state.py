"""Real-time E1 phase coordination, independent of MQTT and file operations."""

import math

from monitoring.psi import RollingConfidencePSI, require


class LiveRun:
    def __init__(self, reference, run_id, start_mono, start_unix):
        self.reference = reference
        self.run_id = run_id
        self.start_mono, self.start_unix = start_mono, start_unix
        self.monitor = RollingConfidencePSI(reference, start_mono)
        self.phase, self.mode = "baseline_warmup", "none"
        self.pending_mode = None
        self.pending_source = None
        self.transitions = []
        self.evaluations = []
        self.baseline = self.first_drift = self.recovery = None
        self.injection_mono = self.restore_mono = None
        self.last_inference_mono = None
        self.latest_model_label = None
        self.clipped_feature_values = 0
        self.ignored_other_run_events = 0
        self.status, self.error = "running", None

    def observe(self, event, now):
        if (event.get("kind") != "inference" or event.get("run_id") != self.run_id or
                event.get("truck_id") != self.reference.get("truck_id", "TRUCK_001")):
            self.ignored_other_run_events += 1
            return False
        require(event.get("model_variant") == self.reference["model_variant"], "Live model variant differs from reference")
        label = event.get("model_class")
        require(type(label) is int and 0 <= label <= 2, "Invalid inference class")
        if not self.monitor.observe(event):
            return False
        self.last_inference_mono = now
        self.latest_model_label = ("Normal", "Warning", "Critical")[label]
        self.clipped_feature_values += int(event.get("clipped_feature_count", 0))
        return True

    def request_mode(self, mode, source):
        require(mode in ("none", "combined"), "E1 uses only none and combined")
        if mode == self.mode and self.pending_mode is None:
            return False
        require(self.pending_mode is None, "A mode change is already pending")
        if mode == "combined":
            require(self.phase == "baseline_warmup" and self.baseline is not None,
                    "Wait for a complete stable live baseline before injecting combined")
        else:
            require(self.phase == "combined_fault" and self.first_drift is not None,
                    "Wait for an observed drift alert before restoring clean mode")
        self.pending_mode, self.pending_source = mode, source
        return True

    def automatic_request(self, now):
        if self.pending_mode is not None:
            return None
        if self.phase == "baseline_warmup" and self.baseline is not None:
            return "combined"
        if (self.phase == "combined_fault" and self.first_drift is not None and
                now - self.injection_mono >= 300):
            return "none"
        return None

    def sample_mode(self, now, unix):
        """Apply a requested mode at the first affected numeric sensor sample."""
        transition = None
        if self.pending_mode is not None or not self.transitions:
            if self.pending_mode is not None:
                self.mode = self.pending_mode
            if self.mode == "combined":
                self.phase, self.injection_mono = "combined_fault", now
            elif self.phase == "combined_fault":
                self.phase, self.restore_mono = "clean_recovery", now
            transition = {"run_id": self.run_id, "mode": self.mode, "phase": self.phase,
                          "elapsed_s": now - self.start_mono, "source_timestamp_unix": unix,
                          "source": self.pending_source or "initial_none",
                          "time_origin": "first numeric sample in this mode; monotonic elapsed clock"}
            self.transitions.append(transition)
            self.pending_mode = self.pending_source = None
        return self.mode, transition

    def evaluate(self, now, unix):
        value = self.monitor.tick(now)
        if value is None:
            return None
        value.update(run_id=self.run_id, phase=self.phase, evaluated_unix=unix,
                     execution_mode="live_wall_clock_mqtt")
        self.evaluations.append(value)
        if value["ready"]:
            if self.phase == "baseline_warmup" and value["psi"] < .10:
                self.baseline = dict(value)
            if self.phase == "combined_fault" and value["psi"] > .25 and self.first_drift is None:
                self.first_drift = {**value, "injection_to_alert_seconds": now - self.injection_mono}
                require(self.first_drift["injection_to_alert_seconds"] <= 300,
                        "Drift crossed 0.25 after the required five-minute limit")
            if self.phase == "clean_recovery" and value["psi"] < .10 and self.first_drift is not None:
                self.recovery = {**value, "clean_restore_to_recovery_seconds": now - self.restore_mono}
                self.status, self.phase = "completed", "completed"
        return value

    def check_deadlines(self, now):
        elapsed = now - self.start_mono
        require(math.isfinite(elapsed) and elapsed >= 0, "Invalid live elapsed time")
        if self.last_inference_mono is None:
            require(elapsed <= 100, "No matching live inference arrived within 100 seconds")
        else:
            require(now - self.last_inference_mono <= 90, "Matching MQTT inference stream stalled for over 90 seconds")
        if self.phase == "baseline_warmup":
            require(elapsed <= 1320, "A stable full live baseline was not obtained within 22 minutes")
        if self.phase == "combined_fault" and self.first_drift is None:
            require(now - self.injection_mono <= 300, "No PSI drift alert within five minutes of injection")
        if self.phase == "clean_recovery":
            require(now - self.restore_mono <= 1260, "PSI did not recover within the 21-minute experiment guard")

    def summary(self, now):
        latest = self.evaluations[-1] if self.evaluations else None
        peak = max((row["max_confidence_psi"] for row in self.evaluations if row["ready"]), default=None)
        return {"status": self.status, "phase": self.phase, "error": self.error,
                "run_id": self.run_id, "elapsed_s": round(now - self.start_mono, 3),
                "execution_mode": "live_wall_clock_mqtt", "sample_count": len(self.monitor.samples),
                "total_inferences": self.monitor.total_inferences, "latest_model_label": self.latest_model_label,
                "latest_psi": latest["psi"] if latest else None,
                "latest_max_confidence_psi": latest["max_confidence_psi"] if latest else None,
                "baseline_psi": self.baseline["psi"] if self.baseline else None,
                "injection_to_alert_seconds": self.first_drift["injection_to_alert_seconds"] if self.first_drift else None,
                "recovery_psi": self.recovery["psi"] if self.recovery else None,
                "clean_restore_to_recovery_seconds": self.recovery["clean_restore_to_recovery_seconds"] if self.recovery else None,
                "max_class_confidence_peak_psi": peak,
                "reference_sha256": self.reference["reference_sha256"],
                "model_sha256": self.reference["model_sha256"],
                "normalization_stats_sha256": self.reference["normalization_stats_sha256"],
                "rolling_window": 100, "evaluation_interval_seconds": 60,
                "rolling_window_reset_on_mode_changes": False,
                "clipped_feature_values": self.clipped_feature_values,
                "ignored_other_run_events": self.ignored_other_run_events,
                "criteria": {"stable_full_live_baseline": self.baseline is not None,
                             "drift_above_025_within_300_seconds": bool(self.first_drift and self.first_drift["injection_to_alert_seconds"] <= 300),
                             "recovered_below_010": bool(self.recovery and self.recovery["psi"] < .10)},
                "phase_changes": self.transitions}

"""Live C1 MQTT signals -> existing D2 service -> actual output-confidence PSI.

One run ID, real 1 Hz / 0.5 Hz deadlines, fixed last-100/60-second monitoring.
No accelerated sensor timestamps, offline predictions or synthetic PSI values.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import queue
import shutil
import signal
import threading
import time
import uuid

from data_pipeline.simulator import MqttPublisher, SAMPLE_PERIODS, event_schedule, utc_text
from monitoring.live_control import write_control
from monitoring.live_clock import LiveClockGuard, boot_time
from monitoring.live_state import LiveRun
from monitoring.psi import PRIMARY_SCORE, canonical, file_sha256, load_reference, require, write_json
from monitoring.scenario import SwitchableSignals


def same_file(path, expected):
    try:
        return bool(expected and file_sha256(path) == expected)
    except OSError:
        return False


def validate_identity(reference, identity):
    require(identity.get("model_ready") is True and identity.get("mqtt_connected") is True,
            "The existing inference service must be model-ready and MQTT-connected")
    require(0 <= identity["health_age_seconds"] < 15, "Inference health snapshot is stale")
    require(identity["model_sha256"] == reference["model_sha256"], "Running model differs from the frozen PSI reference")
    require(identity["model_variant"] == reference["model_variant"], "Running model variant differs from reference")
    require(identity["normalization_stats_sha256"] == reference["normalization_stats_sha256"], "Running normalisation differs from reference")
    require(identity["normalization_metadata_sha256"] == reference["normalization_metadata_sha256"], "Running normalisation metadata differs from reference")


class InferenceSubscriber:
    """Wait for SUBACK before producing sensors; ACK only after log processing."""
    def __init__(self, host, port, topic, client_id):
        import paho.mqtt.client as mqtt
        self.mqtt = mqtt
        self.ready = threading.Event()
        self.pending = queue.Queue(maxsize=1024)
        self.error = None
        self.loop_started = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                                  clean_session=True, protocol=mqtt.MQTTv311, manual_ack=True)

        def connected(client, userdata, flags, reason_code, properties):
            if reason_code.is_failure:
                self.error = "PSI MQTT connection rejected: " + str(reason_code)
                return
            code, _ = client.subscribe(topic, qos=1)
            if code != mqtt.MQTT_ERR_SUCCESS:
                self.error = "PSI MQTT subscribe could not be queued"

        def subscribed(client, userdata, mid, reason_codes, properties):
            if any(code.is_failure for code in reason_codes):
                self.error = "PSI MQTT subscription rejected"
            else:
                self.ready.set()

        def received(client, userdata, message):
            try:
                require(len(message.payload) <= 32768, "Oversized inference message")
                self.pending.put_nowait(message)
            except (ValueError, queue.Full) as error:
                self.error = "PSI receive queue failed: " + str(error)

        self.client.on_connect = connected
        self.client.on_subscribe = subscribed
        self.client.on_message = received
        self.host, self.port = host, port

    def start(self):
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()
        self.loop_started = True
        require(self.ready.wait(10) and self.error is None, self.error or "No PSI SUBACK within 10 seconds")

    def close(self):
        if self.loop_started:
            self.client.disconnect()
            self.client.loop_stop()


def drain_inferences(subscriber, state, output, now, unix):
    require(subscriber.error is None, subscriber.error or "MQTT observer failure")
    accepted = 0
    while True:
        try:
            message = subscriber.pending.get_nowait()
        except queue.Empty:
            break
        event = json.loads(message.payload, parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
        require(isinstance(event, dict), "Inference MQTT payload must be an object")
        if state.observe(event, now):
            output.write(canonical({**event, "monitor_received_unix": unix,
                                    "monitor_elapsed_s": now - state.start_mono}) + "\n")
            output.flush()
            accepted += 1
        code = subscriber.client.ack(message.mid, message.qos)
        require(code == subscriber.mqtt.MQTT_ERR_SUCCESS, "Failed to acknowledge the monitored inference")
        subscriber.pending.task_done()
    return accepted


def run(args):
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "live_status.json", {"status": "starting", "phase": "startup"})
    state = None
    clock_guard = None
    original_reference_sha = None
    sensor_counts = dict.fromkeys(("temperature", "vibration_rms", "door_event"), 0)
    missed = dict.fromkeys(sensor_counts, 0)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    subscriber = publisher = None
    error = None
    start_mono = time.monotonic()
    try:
        require(args.anomaly == "none", "The measured live experiment must begin in none mode")
        ref = load_reference(args.reference)
        identity = json.loads(args.runtime_identity.read_text())
        validate_identity(ref, identity)
        original_reference_sha = file_sha256(args.reference)
        shutil.copyfile(args.reference, folder / "frozen_reference_dist.json")
        run_id = uuid.uuid4().hex
        topic = f"logibridge/trucks/{args.truck_id}/inference"
        subscriber = InferenceSubscriber(args.host, args.port, topic, "psi-sub-" + run_id[:16])
        subscriber.start()
        publisher = MqttPublisher(args.host, args.port, "psi-sensors-" + run_id[:16])
        publisher.start()
        start_mono, start_unix = time.monotonic(), time.time()
        clock_guard = LiveClockGuard(start_mono, start_unix, boot_time())
        state = LiveRun(ref, run_id, start_mono, start_unix)
        signals = SwitchableSignals("none", args.seed)
        control_path = folder / "control.json"
        initial = write_control(control_path, "none", "initial_none")
        seen_control = initial["request_id"]
        schedule = iter(event_schedule(0))
        next_sample = next(schedule)
        next_status = start_mono
        print("LIVE wall-clock MQTT experiment; confidence = " + PRIMARY_SCORE, flush=True)
        print("Reference SHA256: " + ref["reference_sha256"], flush=True)
        print("Run ID: " + run_id + " | sensor seed: " + str(args.seed), flush=True)
        print("Automatic sequence: full clean baseline -> --anomaly combined for 300 s -> --anomaly none -> PSI recovery", flush=True)
        print("The rolling last-100 buffer and frozen reference are retained across both changes.", flush=True)
        print("Clock guard v2: monotonic intervals; UTC adjustments logged; suspend/process gaps checked separately.", flush=True)
        with ExitStack() as stack:
            raw = stack.enter_context((folder / "raw_records.jsonl").open("x", encoding="utf-8"))
            inferences = stack.enter_context((folder / "inferences.jsonl").open("x", encoding="utf-8"))
            evaluations = stack.enter_context((folder / "psi_values.jsonl").open("x", encoding="utf-8"))
            changes = stack.enter_context((folder / "phase_changes.jsonl").open("x", encoding="utf-8"))
            controls = stack.enter_context((folder / "control_requests.jsonl").open("x", encoding="utf-8"))
            clocks = stack.enter_context((folder / "clock_events.jsonl").open("x", encoding="utf-8"))
            while not stop.is_set() and state.status == "running":
                now, unix = time.monotonic(), time.time()
                clock_event = clock_guard.check(now, unix, boot_time())
                if clock_event is not None:
                    clocks.write(canonical(clock_event) + "\n")
                    clocks.flush()
                    print(f"[CLOCK ADJUSTMENT] UTC-minus-elapsed={clock_event['utc_minus_elapsed_seconds']:.3f}s; "
                          f"review_required={clock_event['clock_review_required']}; monotonic sampling continues", flush=True)
                drain_inferences(subscriber, state, inferences, now, unix)
                evaluation = state.evaluate(now, unix)
                if evaluation is not None:
                    require(same_file(args.reference, original_reference_sha), "Frozen reference was modified during monitoring")
                    evaluations.write(canonical(evaluation) + "\n")
                    evaluations.flush()
                    if evaluation["ready"]:
                        print(f"[LOGIBRIDGE PSI] PSI={evaluation['psi']:.3f} max_class_PSI={evaluation['max_confidence_psi']:.3f} "
                              f"n=100 phase={evaluation['phase']} elapsed={evaluation['elapsed_s']:.1f}s", flush=True)
                        if evaluation["drift_alert"]:
                            print(f"[LOGIBRIDGE DRIFT ALERT] PSI={evaluation['psi']:.3f}", flush=True)
                        if state.status == "completed":
                            print(f"[LOGIBRIDGE DRIFT RECOVERY] PSI={evaluation['psi']:.3f}", flush=True)
                    else:
                        print(f"[LOGIBRIDGE PSI WARMUP] n={evaluation['sample_count']}/100 elapsed={evaluation['elapsed_s']:.1f}s", flush=True)
                if state.status == "completed":
                    break
                state.check_deadlines(now)
                control = json.loads(control_path.read_text())
                if control["request_id"] != seen_control:
                    seen_control = control["request_id"]
                    try:
                        state.request_mode(control["anomaly"], control["source"])
                        controls.write(canonical({**control, "accepted": True, "elapsed_s": now - start_mono}) + "\n")
                        print("[CONTROL REQUEST] --anomaly " + control["anomaly"] + " source=" + control["source"], flush=True)
                    except ValueError as rejected:
                        controls.write(canonical({**control, "accepted": False, "reason": str(rejected)}) + "\n")
                        print("[CONTROL REJECTED] " + str(rejected), flush=True)
                    controls.flush()
                automatic = state.automatic_request(now)
                if automatic is not None:
                    control = write_control(control_path, automatic, "automatic_coordinator")
                    seen_control = control["request_id"]
                    state.request_mode(automatic, control["source"])
                    controls.write(canonical({**control, "accepted": True, "elapsed_s": now - start_mono}) + "\n")
                    controls.flush()
                    print("[CONTROL REQUEST] --anomaly " + automatic + " source=automatic_coordinator", flush=True)
                if now >= next_status:
                    write_json(folder / "live_status.json", {**state.summary(now), "sensor_counts": sensor_counts,
                               "missed_deadlines": missed, "updated_unix": unix, "clock": clock_guard.summary()})
                    next_status = now + 2
                deadline, stream, door = next_sample
                elapsed = now - start_mono
                if elapsed < deadline:
                    stop.wait(min(.25, deadline - elapsed))
                    continue
                if elapsed - deadline >= SAMPLE_PERIODS.get(stream, 1.0):
                    missed[stream] += 1
                    print(f"[MISSED SENSOR DEADLINE] {stream} scheduled={deadline:.1f}s elapsed={elapsed:.3f}s", flush=True)
                    next_sample = next(schedule)
                    continue
                source_unix = start_unix + elapsed
                if stream != "door_event":
                    mode, transition = state.sample_mode(now, source_unix)
                    signals.set_anomaly(mode)
                    if transition is not None:
                        changes.write(canonical(transition) + "\n")
                        changes.flush()
                        print(f"[PHASE APPLIED] --anomaly {mode} elapsed={elapsed:.3f}s", flush=True)
                value = door
                if stream == "temperature":
                    value = round(signals.temperature(), 4)
                elif stream == "vibration_rms":
                    value = round(signals.vibration(), 4)
                record = {"schema_version": 1, "message_id": f"{run_id}:{stream}:{sensor_counts[stream]}",
                          "run_id": run_id, "truck_id": args.truck_id, "stream": stream,
                          "seq": sensor_counts[stream], "timestamp": utc_text(source_unix),
                          "timestamp_unix": source_unix, "elapsed_s": round(elapsed, 6),
                          "value": value, "anomaly": signals.anomaly}
                if stream != "door_event":
                    record["unit"] = "degC" if stream == "temperature" else "g"
                payload = canonical(record)
                publisher.publish(f"logibridge/trucks/{args.truck_id}/{stream}", payload)
                raw.write(payload + "\n")
                raw.flush()
                sensor_counts[stream] += 1
                next_sample = next(schedule)
        if stop.is_set() and state.status != "completed":
            state.status, state.error = "interrupted", "Experiment received a stop signal"
    except Exception as failure:
        error = str(failure)
        if state is not None:
            state.status, state.error = "failed", error
        print("[LIVE PSI FAILED] " + error, flush=True)
    finally:
        for connection in (publisher, subscriber):
            if connection is not None:
                try:
                    connection.close()
                except Exception as closing_error:
                    error = error or str(closing_error)
                    if state is not None:
                        state.status, state.error = "failed", error
        now = time.monotonic()
        result = state.summary(now) if state is not None else {"status": "failed", "phase": "startup", "error": error}
        reference_unchanged = same_file(args.reference, original_reference_sha)
        result.update(reference_file_unchanged=reference_unchanged,
                      sensor_counts=sensor_counts, missed_deadlines=missed,
                      completed_unix=time.time(), seed=args.seed,
                      clock=clock_guard.summary() if clock_guard is not None else None,
                      score_definition=PRIMARY_SCORE,
                      live_requirement_evidence=bool(state and state.monitor.total_inferences > 0))
        if result["status"] == "completed" and not reference_unchanged:
            result.update(status="failed", error="Reference changed before completion")
        if result["status"] == "completed" and clock_guard.review_required:
            result.update(status="failed", error="A large abrupt UTC change requires clock-evidence review; the run is not automatically passing")
        result["passed"] = bool(result["status"] == "completed" and reference_unchanged and all(result["criteria"].values()))
        write_json(folder / "live_status.json", result)
        write_json(folder / "live_results.json", result)
        print("Live experiment status: " + result["status"], flush=True)
        print("Live E1 criteria passed: " + str(result["passed"]), flush=True)
        print("Results: " + str(folder / "live_results.json"), flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("monitoring/reference_dist.json"))
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument("--seed", type=int, default=7003)
    parser.add_argument("--anomaly", choices=("none",), default="none")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

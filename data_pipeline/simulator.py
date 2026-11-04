"""Real-time cold-chain sensor simulator; Component C1 of LogiEdge."""

import argparse
import json
import math
import random
import re
import secrets
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODES = ("none", "temp_drift", "vibration", "combined")
STREAMS = ("temperature", "vibration_rms", "door_event")
SAMPLE_PERIODS = {"temperature": 1.0, "vibration_rms": 2.0}


class SensorSignals:
    """Separate random generators keep the two sensor streams independent."""

    def __init__(self, anomaly, seed):
        if anomaly not in MODES:
            raise ValueError("Unknown anomaly mode")
        self.anomaly = anomaly
        self.temperature_rng = random.Random(seed)
        self.vibration_rng = random.Random(seed + 1)
        self.temperature_readings = 0

    def temperature(self):
        self.temperature_readings += 1
        noise = self.temperature_rng.gauss(4.0, 0.3)
        drift = 0.0
        if self.anomaly in ("temp_drift", "combined"):
            # The brief specifies +0.08 degrees C for each temperature reading.
            drift = 0.08 * self.temperature_readings
        return noise + drift

    def vibration(self):
        if self.anomaly in ("vibration", "combined"):
            return self.vibration_rng.gauss(1.2, 0.15)
        return self.vibration_rng.gauss(0.45, 0.05)


def event_schedule(duration, door_period=30.0, door_open_for=5.0):
    """Yield real-time deadlines, stream names, and discrete door states.

    Duration 0 means continuous operation. Door behaviour is a declared demo
    assumption: initially CLOSE, OPEN at 30 s, CLOSE at 35 s, and repeat.
    """
    deadlines = dict.fromkeys(STREAMS, 0.0)
    door_state = "CLOSE"
    while True:
        deadline = min(deadlines.values())
        if duration > 0 and deadline >= duration:
            return
        for stream in STREAMS:
            if deadlines[stream] != deadline:
                continue
            if stream == "door_event":
                if deadline == 0:
                    door_state = "CLOSE"
                    deadlines[stream] = door_period
                elif door_state == "CLOSE":
                    door_state = "OPEN"
                    deadlines[stream] += door_open_for
                else:
                    door_state = "CLOSE"
                    deadlines[stream] += door_period - door_open_for
                yield deadline, stream, door_state
            else:
                deadlines[stream] += SAMPLE_PERIODS[stream]
                yield deadline, stream, None


def utc_text(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


class MqttPublisher:
    """Wait for MQTT QoS 1 PUBACK before counting a publication as sent."""

    def __init__(self, host, port, client_id):
        # Import here so help, offline signal tests, and capture inspection
        # remain usable without importing a network dependency.
        import paho.mqtt.client as mqtt

        self.mqtt = mqtt
        self.connected = threading.Event()
        self.loop_started = False
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.host = host
        self.port = port

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.connected.set()
        else:
            print(f"MQTT connection rejected: {reason_code}", file=sys.stderr)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()

    def start(self):
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()
        self.loop_started = True
        if not self.connected.wait(timeout=10):
            raise RuntimeError("No successful MQTT CONNACK within 10 seconds")

    def publish(self, topic, payload):
        result = self.client.publish(topic, payload, qos=1, retain=False)
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed: {result.rc}")
        result.wait_for_publish(timeout=5)
        if not result.is_published():
            raise RuntimeError("No MQTT PUBACK within 5 seconds; stopping capture")

    def close(self):
        if self.loop_started:
            self.client.disconnect()
            self.client.loop_stop()
            self.loop_started = False


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly", choices=MODES, default="none")
    parser.add_argument("--duration", type=float, default=0,
                        help="Real seconds to run; 0 runs until Ctrl+C")
    parser.add_argument("--seed", type=int,
                        help="Optional reproducible seed; otherwise generated and recorded")
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--purpose",
                        choices=("smoke_test", "normalization", "training", "reference", "live"),
                        default="smoke_test")
    parser.add_argument("--door-period", type=float, default=30.0)
    parser.add_argument("--door-open-for", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every acknowledged sensor record")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative")
    if not (math.isfinite(args.door_period) and
            math.isfinite(args.door_open_for) and
            0 < args.door_open_for < args.door_period):
        parser.error("Require 0 < --door-open-for < --door-period, both finite")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.truck_id):
        parser.error("--truck-id must contain 1-64 letters, digits, underscores or hyphens")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def run(args):
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    run_id = uuid.uuid4().hex
    sensor = SensorSignals(args.anomaly, seed)
    stop = threading.Event()
    stop_reason = {"value": "duration" if args.duration else "interrupted"}

    def request_stop(signum, frame):
        stop_reason["value"] = "interrupted"
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"sensor_{args.anomaly}_{stamp}_{run_id[:8]}"
    capture_path = args.output_dir / f"{stem}.jsonl"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    publisher = MqttPublisher(args.host, args.port, f"sim-{run_id[:16]}")
    counts = dict.fromkeys(STREAMS, 0)
    skipped = dict.fromkeys(STREAMS, 0)
    started = None
    epoch_start = None
    error = None
    try:
        publisher.start()
        started, epoch_start = time.monotonic(), time.time()
        print(f"Truck: {args.truck_id} | anomaly: {args.anomaly} | seed: {seed}", flush=True)
        print(f"Broker: {args.host}:{args.port} | QoS: 1 | purpose: {args.purpose}", flush=True)
        print(f"Capture: {capture_path}", flush=True)
        next_progress = 10.0
        with capture_path.open("x", encoding="utf-8") as output:
            for deadline, stream, door_value in event_schedule(
                    args.duration, args.door_period, args.door_open_for):
                if stop.wait(max(0.0, started + deadline - time.monotonic())):
                    break
                elapsed = time.monotonic() - started
                if args.duration and elapsed >= args.duration:
                    break
                # Do not create a burst of backdated samples after a long stall.
                lateness_limit = SAMPLE_PERIODS.get(stream, 1.0)
                if elapsed - deadline >= lateness_limit:
                    skipped[stream] += 1
                    print(f"Missed {stream} deadline at {deadline:.1f}s", file=sys.stderr)
                    continue
                value = door_value
                if stream == "temperature":
                    value = round(sensor.temperature(), 4)
                elif stream == "vibration_rms":
                    value = round(sensor.vibration(), 4)
                epoch = epoch_start + elapsed
                record = {
                    "schema_version": 1,
                    "message_id": f"{run_id}:{stream}:{counts[stream]}",
                    "run_id": run_id,
                    "truck_id": args.truck_id,
                    "stream": stream,
                    "seq": counts[stream],
                    "timestamp": utc_text(epoch),
                    "timestamp_unix": epoch,
                    "elapsed_s": round(elapsed, 6),
                    "value": value,
                    "anomaly": args.anomaly,
                }
                if stream != "door_event":
                    record["unit"] = "degC" if stream == "temperature" else "g"
                payload = json.dumps(record, separators=(",", ":"), allow_nan=False)
                topic = f"logibridge/trucks/{args.truck_id}/{stream}"
                publisher.publish(topic, payload)
                output.write(payload + "\n")
                output.flush()
                counts[stream] += 1
                if args.verbose:
                    print(f"{topic} {payload}", flush=True)
                if elapsed >= next_progress:
                    print(f"Elapsed {elapsed:.1f}s | acknowledged: {counts}", flush=True)
                    next_progress += 10.0
            # Honour the full requested duration even if the last sample is earlier.
            if args.duration and not stop.is_set():
                stop.wait(max(0.0, started + args.duration - time.monotonic()))
    except (OSError, RuntimeError, ValueError) as exc:
        error = str(exc)
        stop_reason["value"] = "error"
        print(f"Simulator stopped: {error}", file=sys.stderr, flush=True)
    finally:
        elapsed = time.monotonic() - started if started is not None else 0.0
        publisher.close()
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "truck_id": args.truck_id,
            "anomaly": args.anomaly,
            "purpose": args.purpose,
            "seed": seed,
            "duration_requested_s": args.duration,
            "elapsed_s": round(elapsed, 6),
            "started_at": utc_text(epoch_start) if epoch_start is not None else None,
            "capture_file": capture_path.name,
            "counts": counts,
            "missed_deadlines": skipped,
            "qos": 1,
            "retained_sensor_messages": False,
            "door_period_s": args.door_period,
            "door_open_for_s": args.door_open_for,
            "stop_reason": stop_reason["value"],
            "error": error,
            "python_version": sys.version.split()[0],
        }
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"Finished: {stop_reason['value']} | elapsed: {elapsed:.1f}s", flush=True)
        print(f"Acknowledged messages: {counts}", flush=True)
        print(f"Missed deadlines: {skipped}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
    return 1 if error else 0


def main(argv=None):
    args = arguments(argv)
    try:
        return run(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run with the provided Docker image.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

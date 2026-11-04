"""C1 signals with continuous RNG state and controllable anomaly transitions."""

import uuid

from data_pipeline.simulator import MODES, SensorSignals, event_schedule, utc_text


class SwitchableSignals:
    def __init__(self, anomaly, seed):
        self.signal = SensorSignals(anomaly, seed)

    @property
    def anomaly(self):
        return self.signal.anomaly

    def set_anomaly(self, anomaly):
        if anomaly not in MODES:
            raise ValueError("Unknown anomaly mode")
        active = ("temp_drift", "combined")
        if anomaly in active and self.signal.anomaly not in active:
            # Start +0.08 C per reading when the fault is activated, rather
            # than applying the count of earlier clean readings as a jump.
            self.signal.temperature_readings = 0
        self.signal.anomaly = anomaly

    def temperature(self):
        return self.signal.temperature()

    def vibration(self):
        return self.signal.vibration()


def nominal_records(duration, seed, phases=((0, "none"),), origin=0.0, run_id=None,
                    truck_id="TRUCK_001"):
    """Offline generation at nominal C1 timestamps; never a real-time SLA test."""
    if not phases or phases[0][0] != 0 or any(mode not in MODES for _, mode in phases):
        raise ValueError("Phase plan must start at zero and use known modes")
    if any(b[0] <= a[0] for a, b in zip(phases, phases[1:])):
        raise ValueError("Phase times must increase")
    run_id = run_id or uuid.uuid4().hex
    signal = SwitchableSignals(phases[0][1], seed)
    phase_index = 0
    counts = {stream: 0 for stream in ("temperature", "vibration_rms", "door_event")}
    for elapsed, stream, door in event_schedule(duration):
        while phase_index + 1 < len(phases) and elapsed >= phases[phase_index + 1][0]:
            phase_index += 1
            signal.set_anomaly(phases[phase_index][1])
        value = door
        if stream == "temperature":
            value = round(signal.temperature(), 4)
        elif stream == "vibration_rms":
            value = round(signal.vibration(), 4)
        timestamp = origin + elapsed
        record = {"schema_version": 1, "message_id": f"{run_id}:{stream}:{counts[stream]}",
                  "run_id": run_id, "truck_id": truck_id, "stream": stream, "seq": counts[stream],
                  "timestamp_unix": timestamp, "timestamp": utc_text(timestamp),
                  "elapsed_s": elapsed, "value": value, "anomaly": signal.anomaly}
        if stream != "door_event":
            record["unit"] = "degC" if stream == "temperature" else "g"
        counts[stream] += 1
        yield record

"""Causal filtering and six-feature fusion shared by training and inference."""

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

FEATURE_NAMES = (
    "temperature_mean",
    "temperature_std",
    "temperature_rate_c_per_min",
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis",
)
FEATURE_UNITS = ("degC", "degC", "degC/min", "g", "g", "dimensionless")
RECIPE = {
    "version": "c2-v1",
    "moving_average_samples": 5,
    "filter_warmup": "discard until five real samples per stream",
    "window_seconds": 30,
    "step_seconds": 10,
    "window_interval": "start inclusive, end exclusive",
    "temperature_std": "population, ddof=0",
    "temperature_rate": "OLS slope against actual timestamps, multiplied by 60",
    "vibration_rms": "sqrt(mean(smoothed vibration_rms squared))",
    "vibration_peak": "max(abs(smoothed vibration_rms))",
    "vibration_kurtosis": "Fisher excess, population central moments",
    "constant_vibration_kurtosis": 0.0,
}
PERIODS = {"temperature": 1.0, "vibration_rms": 2.0}
MIN_COUNTS = {"temperature": 30, "vibration_rms": 15}


class WindowDataError(ValueError):
    """A window does not contain enough fresh sensor data."""


@dataclass(frozen=True)
class FilteredPoint:
    timestamp: float
    value: float
    raw_support_start: float
    run_id: str


class MovingAverage:
    """A trailing average, emitted only after five actual samples exist."""

    def __init__(self, length=5):
        if length < 1:
            raise ValueError("Moving-average length must be positive")
        self.length = length
        self.samples = deque(maxlen=length)

    def add(self, timestamp, value, run_id=""):
        if not math.isfinite(timestamp) or not math.isfinite(value):
            raise ValueError("Filter inputs must be finite")
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("Filter timestamps must increase")
        self.samples.append((float(timestamp), float(value)))
        if len(self.samples) < self.length:
            return None
        average = math.fsum(point[1] for point in self.samples) / self.length
        return FilteredPoint(timestamp, average, self.samples[0][0], run_id)


def temperature_features(points):
    values = np.asarray([point.value for point in points], dtype=np.float64)
    # Subtract a local origin before fitting, avoiding large Unix-time offsets.
    times = np.asarray([point.timestamp - points[0].timestamp
                        for point in points], dtype=np.float64)
    centred_times = times - times.mean()
    denominator = float(np.dot(centred_times, centred_times))
    if denominator <= 0:
        raise WindowDataError("Temperature timestamps have no spread")
    mean = float(values.mean())
    slope_per_second = float(np.dot(centred_times, values - mean) / denominator)
    return mean, float(values.std(ddof=0)), slope_per_second * 60.0


def excess_kurtosis(values):
    """Fisher excess kurtosis; constant inputs use a documented finite sentinel."""
    values = np.asarray(values, dtype=np.float64)
    centred = values - values.mean()
    second_moment = float(np.mean(centred ** 2))
    if second_moment <= 1e-12:
        return 0.0, True
    fourth_moment = float(np.mean(centred ** 4))
    return fourth_moment / (second_moment ** 2) - 3.0, False


def vibration_features(points):
    values = np.asarray([point.value for point in points], dtype=np.float64)
    kurtosis, constant = excess_kurtosis(values)
    features = (
        float(np.sqrt(np.mean(values ** 2))),
        float(np.max(np.abs(values))),
        float(kurtosis),
    )
    return features, constant


@dataclass(frozen=True)
class FeatureWindow:
    window_index: int
    truck_id: str
    window_start_unix: float
    window_end_unix: float
    raw_support_start_unix: float
    raw_support_end_unix: float
    temperature_samples: int
    vibration_samples: int
    features: tuple
    door_state_at_end: str
    door_events: tuple
    source_run_ids: tuple
    constant_vibration: bool

    def as_record(self):
        result = asdict(self)
        result["feature_names"] = list(FEATURE_NAMES)
        return result


class FeaturePipeline:
    """Independent filters followed by 30-second, 10-second-step feature fusion.

    Add raw messages in timestamp order within each stream. Normal operation
    closes windows when both numeric streams have passed the end boundary.
    finish() can close the last complete window using a completed capture's
    known end time. Filters continue across windows and source run IDs.
    """

    def __init__(self, origin_unix, strict=True):
        if not math.isfinite(origin_unix):
            raise ValueError("A finite pipeline time origin is required")
        self.origin = float(origin_unix)
        self.strict = strict
        self.filters = {name: MovingAverage(5) for name in PERIODS}
        self.points = {name: deque(maxlen=512) for name in PERIODS}
        self.first_valid = {}
        self.latest = {}
        self.last_timestamp = {}
        self.truck_id = None
        self.next_end_offset = None
        self.window_index = 0
        self.seen_ids = set()
        self.id_history = deque()
        self.duplicates_ignored = 0
        self.rejected_windows = []
        self.doors = deque(maxlen=512)
        self.door_state_before_buffer = "UNKNOWN"

    def add_record(self, record):
        stream = record["stream"]
        if stream not in (*PERIODS, "door_event"):
            raise ValueError(f"Unknown sensor stream: {stream}")
        message_id = record["message_id"]
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("Each reading needs a nonempty message_id")
        if message_id in self.seen_ids:
            self.duplicates_ignored += 1
            return []
        timestamp = float(record["timestamp_unix"])
        if not math.isfinite(timestamp) or timestamp < self.origin - 1e-5:
            raise ValueError("Invalid sensor timestamp")
        if stream in self.last_timestamp and timestamp <= self.last_timestamp[stream]:
            raise ValueError(f"Out-of-order timestamp in {stream}")
        truck_id = record["truck_id"]
        if not isinstance(truck_id, str) or not truck_id:
            raise ValueError("Missing truck_id")
        if self.truck_id is not None and truck_id != self.truck_id:
            raise ValueError("Use a separate feature pipeline for each truck")
        value = record["value"]
        if stream == "door_event":
            if value not in ("OPEN", "CLOSE"):
                raise ValueError("Door value must be OPEN or CLOSE")
            self.doors.append((timestamp, value))
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric value in {stream}")
            point = self.filters[stream].add(timestamp, float(value), record["run_id"])
            self.latest[stream] = timestamp
            if point is not None:
                self.points[stream].append(point)
                self.first_valid.setdefault(stream, point.timestamp)
        self.truck_id = truck_id
        self.last_timestamp[stream] = timestamp
        self.seen_ids.add(message_id)
        self.id_history.append(message_id)
        if len(self.id_history) > 4096:
            self.seen_ids.remove(self.id_history.popleft())
        if self.next_end_offset is None and len(self.first_valid) == 2:
            earliest_start = max(self.first_valid.values()) - self.origin
            self.next_end_offset = int(math.ceil((earliest_start + 30.0) / 10.0)) * 10
        if self.next_end_offset is None:
            return []
        return self._emit_until(min(self.latest.values()))

    def finish(self, end_unix):
        """Use only after the caller knows the capture/window time has elapsed."""
        if not math.isfinite(end_unix):
            raise ValueError("A finite completion timestamp is required")
        return self._emit_until(float(end_unix))

    def _emit_until(self, limit):
        emitted = []
        while (self.next_end_offset is not None and
               self.origin + self.next_end_offset <= limit):
            end = self.origin + self.next_end_offset
            index = self.window_index
            try:
                emitted.append(self._make_window(end - 30.0, end, index))
            except WindowDataError as exc:
                self.rejected_windows.append({
                    "window_end_unix": end, "reason": str(exc),
                })
                if self.strict:
                    raise
            finally:
                self.window_index += 1
                self.next_end_offset += 10
                self._trim(self.origin + self.next_end_offset - 30.0)
        return emitted

    def _make_window(self, start, end, index):
        selected = {}
        for name, buffer in self.points.items():
            rows = [point for point in buffer if start <= point.timestamp < end]
            if len(rows) < MIN_COUNTS[name]:
                raise WindowDataError(
                    f"Window {start-self.origin:.0f}-{end-self.origin:.0f}s: "
                    f"{name} has {len(rows)} readings; need {MIN_COUNTS[name]}"
                )
            period = PERIODS[name]
            times = np.asarray([point.timestamp for point in rows])
            if (times[0] > start + period + 0.25 or
                    times[-1] < end - period - 0.25 or
                    np.any(np.diff(times) > period + 0.5)):
                raise WindowDataError(f"Stale data or a sampling gap in {name}")
            selected[name] = rows
        temp = temperature_features(selected["temperature"])
        vibration, constant = vibration_features(selected["vibration_rms"])
        features = tuple(float(value) for value in (*temp, *vibration))
        if len(features) != 6 or not np.all(np.isfinite(features)):
            raise WindowDataError("Expected exactly six finite feature values")
        door_history = [(timestamp, state) for timestamp, state in self.doors
                        if timestamp < end]
        door_state = door_history[-1][1] if door_history else self.door_state_before_buffer
        events = tuple({"timestamp_unix": timestamp, "value": state}
                       for timestamp, state in door_history if timestamp >= start)
        all_points = selected["temperature"] + selected["vibration_rms"]
        return FeatureWindow(
            index, self.truck_id, start, end,
            min(point.raw_support_start for point in all_points),
            max(point.timestamp for point in all_points),
            len(selected["temperature"]), len(selected["vibration_rms"]),
            features, door_state, events,
            tuple(sorted({point.run_id for point in all_points})),
            constant,
        )

    def _trim(self, earliest_needed):
        for buffer in self.points.values():
            while buffer and buffer[0].timestamp < earliest_needed:
                buffer.popleft()
        while self.doors and self.doors[0][0] < earliest_needed:
            _, self.door_state_before_buffer = self.doors.popleft()

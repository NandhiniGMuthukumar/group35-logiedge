"""Four-bin confidence PSI. Only prediction probabilities enter the statistic."""

from bisect import bisect_right
from collections import deque
import hashlib
import json
import math
from pathlib import Path

BINS = (0.0, 0.25, 0.50, 0.75, 1.0)
EPSILON = 1e-6
PRIMARY_SCORE = "normality_confidence=P(Normal)"
DIAGNOSTIC_SCORE = "max_class_confidence=max(P(Normal),P(Warning),P(Critical))"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bin_index(value):
    require(not isinstance(value, bool), "A confidence must be a number, not a Boolean")
    value = float(value)
    require(math.isfinite(value) and 0.0 <= value <= 1.0, "Confidence must be finite and within [0,1]")
    # The rightmost bin includes exactly 1.0; FP32 softmax can produce it.
    return min(3, bisect_right(BINS, value) - 1)


def counts(scores):
    result = [0, 0, 0, 0]
    for value in scores:
        result[bin_index(value)] += 1
    require(sum(result) > 0, "An empty confidence distribution is invalid")
    return result


def proportions(frequencies, epsilon=EPSILON):
    require(len(frequencies) == 4 and all(type(v) is int and v >= 0 for v in frequencies),
            "Expected four non-negative integer bin counts")
    total = sum(frequencies)
    require(total > 0 and 0 < epsilon < 0.25, "Invalid distribution or smoothing epsilon")
    floored = [max(value / total, epsilon) for value in frequencies]
    denominator = math.fsum(floored)
    return [value / denominator for value in floored]


def psi(reference_counts, current_counts, epsilon=EPSILON):
    expected, actual = proportions(reference_counts, epsilon), proportions(current_counts, epsilon)
    value = math.fsum((q - p) * math.log(q / p) for p, q in zip(expected, actual))
    require(math.isfinite(value), "Non-finite PSI")
    return max(0.0, value)


def confidence_scores(event):
    values = event["probabilities"]
    require(isinstance(values, (list, tuple)) and len(values) == 3, "Expected three class probabilities")
    for value in values:
        bin_index(value)
    values = [float(value) for value in values]
    # Full INT8 output can differ slightly from one after dequantisation.
    require(abs(math.fsum(values) - 1.0) <= 0.02, "Output probabilities do not approximately sum to one")
    normality, maximum = values[0], max(values)
    for key, expected in (("normality_confidence", normality), ("confidence_max", maximum)):
        if key in event:
            require(math.isclose(float(event[key]), expected, abs_tol=1e-6), "Inconsistent confidence field: " + key)
    return normality, maximum


def distribution(scores):
    frequencies = counts(scores)
    total = sum(frequencies)
    return {"counts": frequencies, "proportions": [v / total for v in frequencies],
            "smoothed_proportions": proportions(frequencies)}


def make_reference(events, **metadata):
    require(len(events) == 300, "Reference requires exactly 300 clean-window inferences")
    require(len({event["event_id"] for event in events}) == 300, "Reference windows must have distinct event IDs")
    require(all(event["model_sha256"] == metadata["model_sha256"] for event in events),
            "Reference events use different model identities")
    pairs = [confidence_scores(event) for event in events]
    value = {"schema_version": 1, "sample_count": 300, "bin_edges": list(BINS),
             "last_bin_right_inclusive": True,
             "smoothing": {"epsilon": EPSILON, "method": "floor proportions, then renormalise"},
             "primary_score": PRIMARY_SCORE, "diagnostic_score": DIAGNOSTIC_SCORE,
             "primary_distribution": distribution([pair[0] for pair in pairs]),
             "max_confidence_distribution": distribution([pair[1] for pair in pairs]),
             **metadata}
    value["reference_sha256"] = digest(value)
    validate_reference(value)
    return value


def validate_reference(value):
    require(value.get("schema_version") == 1 and value.get("sample_count") == 300,
            "Unsupported reference version/count")
    require(value.get("bin_edges") == list(BINS) and value.get("last_bin_right_inclusive") is True,
            "Reference bins differ from the fixed four-bin recipe")
    require(value.get("primary_score") == PRIMARY_SCORE and value.get("diagnostic_score") == DIAGNOSTIC_SCORE,
            "Reference confidence definition differs")
    require(value.get("smoothing") == {"epsilon": EPSILON, "method": "floor proportions, then renormalise"},
            "Reference smoothing differs")
    unsigned = {key: item for key, item in value.items() if key != "reference_sha256"}
    require(value.get("reference_sha256") == digest(unsigned), "Reference content hash differs")
    for key in ("primary_distribution", "max_confidence_distribution"):
        part = value[key]
        frequencies = part["counts"]
        require(sum(frequencies) == 300, "Reference bin counts do not sum to 300")
        expected = proportions(frequencies)
        require(all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(expected, part["smoothed_proportions"])) and
                len(part["smoothed_proportions"]) == 4, "Saved proportions differ from bin counts")
        require(len(part["proportions"]) == 4 and all(math.isclose(a / 300, b, abs_tol=1e-12)
                for a, b in zip(frequencies, part["proportions"])), "Saved raw proportions differ from counts")
    for key in ("model_sha256", "normalization_stats_sha256"):
        require(isinstance(value.get(key), str) and len(value[key]) == 64, "Missing model/statistics identity")
    return value


def load_reference(path):
    return validate_reference(json.loads(Path(path).read_text(encoding="utf-8")))


class RollingConfidencePSI:
    """Last 100 distinct inferences; fixed 60-second cadence; no phase resets.

    The caller supplies a monotonic clock for live use. Offline checks supply
    explicitly labelled logical seconds. Anomaly labels are not read here.
    """

    def __init__(self, reference, start_time):
        validate_reference(reference)
        # Keep a private JSON copy so callers cannot mutate the frozen baseline.
        self.reference = json.loads(canonical(reference))
        self.samples = deque(maxlen=100)
        self.seen = set()
        self.id_history = deque()
        self.total_inferences = 0
        self.start_time = float(start_time)
        require(math.isfinite(self.start_time), "A finite clock origin is required")
        self.next_tick = self.start_time + 60.0
        self.last_tick_time = self.start_time

    def observe(self, event):
        event_id = event["event_id"]
        require(isinstance(event_id, str) and event_id, "Missing inference event ID")
        require(event["model_sha256"] == self.reference["model_sha256"], "Model/reference mismatch")
        if event_id in self.seen:
            return False
        normality, maximum = confidence_scores(event)
        self.seen.add(event_id)
        self.id_history.append(event_id)
        if len(self.id_history) > 4096:
            self.seen.remove(self.id_history.popleft())
        self.samples.append((normality, maximum))
        self.total_inferences += 1
        return True

    def tick(self, now):
        now = float(now)
        require(math.isfinite(now) and now >= self.last_tick_time, "Monitoring clock moved backwards")
        self.last_tick_time = now
        if now < self.next_tick:
            return None
        # A delayed process evaluates current data once, without inventing
        # intermediate measurements for missed deadlines.
        self.next_tick = self.start_time + (math.floor((now - self.start_time) / 60) + 1) * 60
        value = {"elapsed_s": now - self.start_time, "sample_count": len(self.samples),
                 "total_inferences": self.total_inferences, "reference_sha256": self.reference["reference_sha256"],
                 "ready": len(self.samples) == 100}
        if not value["ready"]:
            return {**value, "psi": None, "max_confidence_psi": None, "drift_alert": False}
        primary = counts([sample[0] for sample in self.samples])
        maximum = counts([sample[1] for sample in self.samples])
        primary_psi = psi(self.reference["primary_distribution"]["counts"], primary)
        maximum_psi = psi(self.reference["max_confidence_distribution"]["counts"], maximum)
        return {**value, "primary_counts": primary, "max_confidence_counts": maximum,
                "psi": primary_psi, "max_confidence_psi": maximum_psi,
                "drift_alert": primary_psi > 0.25, "below_recovery_threshold": primary_psi < 0.10}

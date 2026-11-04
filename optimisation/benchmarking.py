"""Timing, CPU/TDP proxy and Pareto calculations without fitted parameters."""

import math
import time

import numpy as np

from training.modeling import get_tf
from training.quantization import dequantize_int8, quantize_int8

WARMUPS = 10
TIMED_RUNS = 200
BACKEND = "TFLite optimized built-in CPU kernels; default delegates disabled; one thread"


def latency_summary(values_ms, expected_count=TIMED_RUNS):
    values = np.asarray(values_ms, dtype=np.float64)
    if (values.shape != (expected_count,) or not np.all(np.isfinite(values)) or np.any(values <= 0)):
        raise ValueError(f"Need exactly {expected_count} finite positive timing samples")
    return {
        "count": len(values), "mean_ms": float(values.mean()),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "median_ms": float(np.median(values)), "min_ms": float(values.min()),
        "max_ms": float(values.max()), "std_ms": float(values.std(ddof=1)),
        "p95_method": "numpy.percentile(method='linear', q=95)",
    }


def energy_proxy(tdp_w, process_cpu_percent, host_logical_cpus, mean_latency_ms):
    if (not all(math.isfinite(v) and v > 0 for v in (tdp_w, process_cpu_percent, mean_latency_ms)) or
            not isinstance(host_logical_cpus, int) or isinstance(host_logical_cpus, bool) or host_logical_cpus < 1):
        raise ValueError("Energy proxy needs positive TDP, CPU%, latency and a positive integer host CPU count")
    fraction = process_cpu_percent / (100 * host_logical_cpus)
    if fraction > 1:
        raise ValueError("Process CPU% exceeds the supplied whole-host CPU capacity; verify logical CPU count")
    power_w = tdp_w * fraction
    return {"normalized_host_cpu_fraction": fraction, "estimated_power_w": power_w,
            "energy_proxy_mJ": power_w * mean_latency_ms}


def pareto_names(rows, objectives):
    """Exact dominance: no worse on every declared axis, better on at least one."""
    if not objectives or any(direction not in ("min", "max") for _, direction in objectives):
        raise ValueError("Declare at least one min/max objective")
    names = [row["variant"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Variant names must be unique")
    points = np.asarray([[row[key] * (1 if direction == "min" else -1)
                          for key, direction in objectives] for row in rows], dtype=np.float64)
    if not rows:
        return []
    if not np.all(np.isfinite(points)):
        raise ValueError("Pareto objectives must be finite")
    return [names[i] for i, point in enumerate(points)
            if not any(np.all(other <= point) and np.any(other < point)
                       for j, other in enumerate(points) if j != i)]


class LiteRunner:
    def __init__(self, name, content, features, expected_dtype):
        self.name = name
        tf = get_tf()
        self.interpreter = tf.lite.Interpreter(
            model_content=content, num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.interpreter.allocate_tensors()
        inputs, outputs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one model input and output")
        self.input_info, self.output_info = inputs[0], outputs[0]
        if (self.input_info["dtype"] != expected_dtype or self.output_info["dtype"] != expected_dtype or
                list(self.input_info["shape"]) != [1,6] or list(self.output_info["shape"]) != [1,3]):
            raise ValueError(f"{name}: unexpected input/output dtype or shape")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("Runner requires nonempty finite six-feature rows")
        self.count = len(values)
        self.is_int8 = expected_dtype == np.int8
        self.inputs = quantize_int8(values, *self.input_info["quantization"]) if self.is_int8 else values.copy()
        self.inputs = np.ascontiguousarray(self.inputs)

    def set_input(self, row):
        self.interpreter.set_tensor(self.input_info["index"], self.inputs[row:row+1])

    def invoke(self):
        self.interpreter.invoke()

    def get_output(self):
        return self.interpreter.get_tensor(self.output_info["index"])[0]

    def probabilities(self, output):
        return dequantize_int8(output, *self.output_info["quantization"]) if self.is_int8 else output


def collect_latency(runners, clock=time.perf_counter_ns):
    if not runners or len({r.name for r in runners}) != len(runners) or len({r.count for r in runners}) != 1:
        raise ValueError("Need uniquely named runners using the same validation row count")
    for runner in runners:
        for index in range(WARMUPS):
            runner.set_input(index % runner.count)
            runner.invoke()
            runner.get_output()
    rows = []
    # Rotate order each round, giving all variants the same validation row and
    # balancing which one runs first. Every variant gets exactly 200 timers.
    for index in range(TIMED_RUNS):
        offset = index % len(runners)
        ordered = runners[offset:] + runners[:offset]
        for order, runner in enumerate(ordered):
            row = index % runner.count
            api_start = clock()
            runner.set_input(row)
            start = clock()
            runner.invoke()
            stop = clock()
            runner.get_output()
            api_stop = clock()
            rows.append({"variant": runner.name, "measured_call": index+1, "order_in_round": order,
                         "validation_row": row, "invoke_ms": (stop-start)/1e6,
                         "api_roundtrip_ms": (api_stop-api_start)/1e6})
    return rows


def evaluate_runner(runner):
    probabilities = []
    for index in range(runner.count):
        runner.set_input(index)
        runner.invoke()
        probabilities.append(runner.probabilities(runner.get_output()))
    return np.asarray(probabilities, dtype=np.float32)


def cpu_probe(runner, seconds=5.0, process=None, clock=time.perf_counter):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("CPU probe duration must be finite and positive")
    if process is None:
        import psutil
        process = psutil.Process()
    process.cpu_percent(interval=None)  # Prime/discard the first sample.
    before = process.cpu_times()
    start = clock()
    count = 0
    while True:
        for _ in range(200):
            runner.set_input(count % runner.count)
            runner.invoke()
            runner.get_output()
            count += 1
        elapsed = clock()-start
        if elapsed >= seconds:
            break
    cpu_percent = float(process.cpu_percent(interval=None))
    after = process.cpu_times()
    if not math.isfinite(cpu_percent) or cpu_percent <= 0:
        raise ValueError("CPU probe returned no usable process CPU%; verify sampling before reporting energy")
    return {"probe_requested_seconds": seconds, "probe_elapsed_seconds": elapsed,
            "probe_inference_calls": count, "process_cpu_percent": cpu_percent,
            "process_user_cpu_seconds": float(after.user-before.user),
            "process_system_cpu_seconds": float(after.system-before.system),
            "scope": "separate sustained set_tensor/invoke/get_tensor loop, with Python overhead; not the 200 latency samples"}

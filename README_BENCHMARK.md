# LogiEdge — F2 five-metric benchmark and Pareto analysis

M1, M2 and M3 passed their accuracy checks on the fixed 59-window validation
set. M3's actual file is 3216 bytes, compared with M2's 3552 and M1's 5416.
This package measures latency and the required CPU/TDP energy proxy on the
team's laptop. It does not assume that the smallest file is fastest.

## First collect the Windows CPU information

Open ordinary Windows PowerShell. Administrator mode is not required:

```powershell
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
```

Share that output. The exact processor model lets us verify an appropriate
manufacturer power figure and label its use as a TDP estimate. The Windows
logical CPU count is needed to normalize Process.cpu_percent. A laptop model
name alone does not establish its installed CPU/power configuration.

This update deliberately supplies no guessed host power profile. After CPU
verification, create hardware/benchmark_host.json with these fields:

- cpu_name: exact reported host processor name.
- host_logical_cpus: Windows logical processor count, as an integer.
- tdp_estimate_w: explicit positive power estimate in watts.
- tdp_source: manufacturer URL or a clearly identified assumption/source.
- tdp_basis: for example, manufacturer processor base power used as a proxy.

We will provide the filled profile after checking your CPU output. Do not
substitute a processor's maximum turbo power without documenting that choice.

## Install and verify in Ubuntu

Save LogiEdge_Benchmark.zip into C:\Users\Nandhini G\Downloads. Keep Docker
Desktop running and run one command at a time:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Benchmark.zip" -d .
docker compose run --rm --entrypoint python trainer -m unittest discover -s tests -p 'test_benchmarking.py' -v
```

Expected: ten new tests passing without skips. They test exact warm-up/run
counts, order rotation, timing arithmetic, CPU normalization, energy units,
Pareto tradeoffs, host-profile validation, actual FP32/INT8 runtimes and an
isolated complete benchmark that writes charts/results with a mocked CPU probe.
A separate target-container test samples the actual current process. Test
fixtures and their power assumptions never become project results.

No image rebuild is required: the existing trainer image contains TensorFlow,
NumPy, psutil and Matplotlib; the project bind mount supplies this code.

An optional read-only readiness check can run before the host profile exists:

```bash
bash deployment/run_benchmark.sh --check-only
```

It verifies the saved model files, hashes, validation split, INT8 graphs and
compact M3 structure. It prints the guest CPU information without creating
performance results. Share any failure rather than continuing past it.

## Run after the host profile is filled and verified

Plug in the charger and close heavy background applications. Keep Docker
Desktop running. Use the same laptop and power mode for the complete session.

```bash
bash deployment/run_benchmark.sh
```

The default uses a separate five-second CPU probe per model, so allow for
those probes as well as startup and chart generation. Each session writes a
new optimisation/results/benchmark_TIMESTAMP_ID folder. Earlier sessions,
models, dataset and normalization statistics are preserved. Do not choose the
best-looking session while discarding unfavorable measurements.

## Declared F2 measurement method

All three .tflite files run through TensorFlow 2.14.1's TFLite interpreter, one
thread each, with optimized built-in CPU kernels and default delegates disabled
using BUILTIN_WITHOUT_DEFAULT_DELEGATES. This setting prevents differences in
automatic default-delegate assignment between float and integer graphs. It is
a declared CPU backend; it does not measure XNNPACK, Hailo, Jetson or MCU speed.

Preprocessing and float/INT8 input encoding happen before timing. Each model
receives the same normalized validation inputs, encoded using that model's
actual input scale/zero point where required.

1. Allocate each interpreter outside the timing region.
2. Run exactly ten untimed warm-up calls per model.
3. Run exactly 200 timed calls per model. Each round uses the same validation
   row for all variants and rotates which variant goes first. Cycle through all
   59 validation rows, in their saved order.
4. Primary latency times Interpreter.invoke only, using perf_counter_ns.
   set_tensor/get_tensor, input quantization, filtering and window extraction
   are outside that interval. No timer-overhead subtraction is applied.
5. Mean is the arithmetic average of the 200 samples. p95 uses NumPy's linear
   percentile method. Save every sample, including slow outliers.
6. Record set_tensor+invoke+get_tensor round-trip time as an additional
   diagnostic. It includes timing instrumentation and pre-encoded inputs; it
   is not full sensor-to-alert latency.
7. Evaluate all 59 validation rows after timing with the same interpreters.
   Record accuracy, the confusion matrix and Critical recall/support again.
8. Read actual model bytes; decimal KB means bytes/1000, not KiB.

These implement the explicit F2 counts/metrics in the project brief. The exact
course Lab 2 handout has not been supplied. If it specifies a different timing
boundary or backend, align the final method with that handout and document the
change before final submission.

## CPU/TDP energy proxy

The 200 invoke calls may last much less than a CPU-time accounting interval.
The benchmark therefore uses a separate sustained loop in the same process,
with the same model/interpreter, thread count and validation-input cycle. The
first psutil.Process.cpu_percent(None) reading is discarded. The next reading
covers at least five seconds by default. Save the actual duration/call count
and user/system CPU-time deltas.

Process.cpu_percent uses 100% for one fully occupied logical CPU and can exceed
100% for multiple occupied CPUs. Normalize it by the verified Windows host
logical processor count:

    host_CPU_fraction = process_CPU_percent / (100 * host_logical_CPUs)
    estimated_power_W = TDP_estimate_W * host_CPU_fraction
    energy_proxy_mJ = estimated_power_W * mean_invoke_latency_ms

The units are intentional: watts times milliseconds equals millijoules.
This is an estimated CPU-energy proxy, not a power-meter measurement. The
sustained loop includes Python/I/O overhead. Equal-capacity logical CPUs,
linear power scaling and use of TDP as an operating-power proxy are assumptions.
Hybrid cores, SMT, idle power, boost, DVFS and WSL/Docker virtualization limit
the physical accuracy. Report those limits with the figure.

The longer probe does not add samples to the 200-call latency statistics.
Raw_latency_samples.csv always has exactly 600 data rows: 200 for each model.
Warm-up, validation and sustained-probe call counts have separate roles.

## Pareto objectives and eligibility

The chart minimizes two axes: mean invoke latency and decimal file size.
The feasible frontier includes only models whose newly measured accuracy is
strictly >88% and Critical recall strictly >95%. A model dominates another if
it is no worse on both axes and strictly better on at least one. Equal points
remain nondominated.

The chart annotates accuracy and Critical recall, while p95 and energy appear
in the metric table. A separate five-metric frontier in pareto.json minimizes
mean latency, p95, size and energy proxy, and maximizes accuracy. This prevents
mistaking a 2D plot for a five-dimensional frontier. No significance margin is
used; tiny timing differences must be interpreted cautiously.

The package records candidates, rather than automatically selecting the
smallest or fastest model as the final deployment recommendation. F3 must
also consider the full 90-second sensor-to-alert SLA and chosen-device memory
and runtime constraints. Laptop kernel timing alone cannot establish those.

## Outputs

- benchmark_table.csv: all five primary metrics for three variants, plus
  Critical recall/support and eligibility.
- raw_latency_samples.csv: all 600 timing samples, order and dataset indices.
- benchmark_results.json: timing boundaries, actual results, CPU probes,
  provenance, validation-index hash and unchanged-source checks.
- validation_predictions.csv: newly measured predictions/probabilities.
- environment.json: package/runtime versions, host profile, guest CPU details,
  affinity, backend and timer resolution. Host logical count and guest-visible
  logical count are recorded separately.
- pareto.json: the declared 2D and five-metric frontier calculations.
- benchmark_metrics.png/.svg and pareto_latency_size.png/.svg: actual measured
  charts for the report and PPT.
- F2_Benchmark_Record.md: generated table and method/energy interpretation.

optimisation/results/benchmark_pointer.json identifies the latest complete
session. Share the printed table and frontier summary after the first run.
Keep CSV/JSON files and the charts for the final submission.

## Primary references

- https://psutil.readthedocs.io/stable/index.html
- https://www.tensorflow.org/api_docs/python/tf/lite/experimental/OpResolverType
- https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
- https://developers.google.com/edge/litert/models/measurement

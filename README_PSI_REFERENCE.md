# LogiEdge — E1 confidence reference and offline diagnostic

This incremental package prepares E1's frozen 300-window reference and the
shared four-bin/rolling-window calculation. The subsequent live MQTT drift
monitor and mid-run demonstration remain a separate execution checkpoint.

## What the confidence score means

The monitored score is the model's probability that conditions are Normal:
`normality_confidence = P(Normal) = probabilities[0]`.

The model also outputs `confidence_max`, the maximum probability across the
three classes. A model can confidently predict both Normal and Critical, so
maximum confidence may remain in the same bin across a fault. This package
records both distributions and both PSI values. It does not rename maximum
confidence, alter the model probabilities, or assume a desired result.

Using class-specific confidence is an explicit interpretation of the brief's
"model output confidence score". State this definition in the report and demo.
If the instructor specifically requires maximum-class confidence, retain its
measured result and resolve that interpretation; do not claim that a different
score is maximum confidence or fabricate a crossing.

## Run inside Ubuntu

Save `LogiEdge_PSI_Reference.zip` in Windows Downloads. Keep the existing Docker
services running and execute one command at a time:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_PSI_Reference.zip" -d .
```

Run the focused checks using the already-built trainer image:

```bash
docker compose run --rm --no-deps --entrypoint python trainer \
  -m unittest discover -s tests -p 'test_psi_reference.py' -v
```

Expected: 13 tests, OK. Tests use a clearly labelled predictor double for
plumbing; they do not constitute actual M1 performance measurements.

Build the real reference and run the real-model offline diagnostic:

```bash
bash deployment/prepare_psi_reference.sh
```

No model training, normalisation fitting, MQTT publication or active-container
replacement occurs. The command uses your existing registered M1 FP32 model,
the same LocalModel runtime as live inference, and the frozen training_stats.npy.
The model/statistics/registry hashes are checked before and after.

Send the printed reference-bin counts and diagnostic summary. The actual
reference and diagnostic outcomes are not prefilled or guaranteed.

## How 300 clean windows are produced

- Independent clean seed 7001; the original SensorSignals and event_schedule
  supply the 1 Hz temperature, 0.5 Hz vibration RMS and door events.
- The unchanged C2 FeaturePipeline performs the actual causal five-sample
  filters, 30-second windows and 10-second steps.
- Nominal sensor timestamps span 3035 logical seconds. The 300 window ends
  run from 40 through 3030 seconds. Every window contains at least 30 actual
  temperature and 15 actual vibration samples.
- The windows overlap by design. The capture is independently seeded from
  training/normalisation; the overlapping windows are not statistically
  independent samples.
- This is offline synthetic-data preparation. It does not wait 50 minutes of
  wall time or claim a live MQTT/SLA measurement.
- Actual raw records, features and model probabilities are saved, with hashes,
  beneath monitoring/reference_runs. The frozen target is
  monitoring/reference_dist.json; reruns reuse a verified matching reference.

The reference file records predicted-class counts on the clean scenario. A
clean input scenario does not guarantee that every model prediction is Normal;
the counts are observations, not forced labels.

## Four-bin PSI recipe

| Bin | Score interval |
|---|---|
| 1 | 0 <= score < 0.25 |
| 2 | 0.25 <= score < 0.50 |
| 3 | 0.50 <= score < 0.75 |
| 4 | 0.75 <= score <= 1.0 |

The final bin includes exactly 1.0, which floating-point softmax can produce.
Both reference and current distributions use the same boundary convention.

PSI = sum((current_share - reference_share) x
ln(current_share / reference_share)). For empty bins, floor proportions at
1e-6, then renormalise. This numerical rule is fixed in the reference and used
unchanged for monitoring. It prevents log(0); bins are not adapted to force a
threshold crossing.

RollingConfidencePSI retains the last 100 distinct model inferences, evaluates
on a 60-second cadence, rejects model/reference mismatches, and never resets
the rolling window on phase changes. Anomaly metadata is not a statistical
input. The monitored probabilities themselves determine PSI.

## Offline diagnostic before the longer live experiment

An independently seeded continuous sequence (7002) supplies:

| Logical sensor time | Mode |
|---|---|
| 0–1100 seconds | none |
| 1100–1400 seconds | combined |
| 1400–2505 seconds | none |

The sequence keeps one run ID and continuous random-number/filter state. When
temperature drift is activated, +0.08 C per reading starts at the first fault
reading; earlier clean readings do not create an artificial temperature jump.

The trained model is actually invoked for every extracted window. The same
rolling PSI implementation processes those probabilities without resetting.
The summary reports baseline PSI, the first crossing above 0.25, recovery below
0.10, and maximum-class-confidence PSI. All elapsed times in this diagnostic
are explicitly logical simulation seconds, not live detection measurements.

The diagnostic must pass before we proceed to the live monitor. If it fails,
the command saves its actual outputs and exits unsuccessfully. Inspect those
outputs rather than changing thresholds, bins, scores or the reference live.

## Remaining E1 evidence

This preparation does not complete E1. The next stage must subscribe to actual
MQTT inference results, print PSI every 60 wall-clock seconds after warm-up,
activate combined mode mid-run, show the required alert string, and recover
below 0.10 on the unreset last-100 rolling window.

At one inference every 10 seconds, filling and then replacing 100 live outputs
takes about 17 minutes each, plus pipeline/timer delays. Plan roughly 40 minutes
for the complete live clean/fault/recovery sequence. The 15–20-minute final
video can use clearly labelled chronological excerpts/time jumps, retaining
the complete original logs; offline diagnostic time must not be presented as
live wall-clock time.

## Author-side validation and sources

Thirteen focused tests and Bash syntax validation passed in the authoring
environment. The tests cover analytic PSI arithmetic, all bin boundaries,
finite empty-bin handling, reference hashes/reuse, duplicates, 60-second timing,
300 shared-pipeline windows, phase transitions and rolling-window recovery.
The author's tests use a predictor fixture; actual trained-model reference and
diagnostic results are pending your laptop execution.

PSI formula checked against the FATE project's documentation:
https://fate.readthedocs.io/en/develop-1.10.0/federatedml_component/psi/

The four bins, rolling-window size, interval and alert thresholds come from
the mini-project brief. The confidence definition, endpoint inclusion and
empty-bin rule above are explicit implementation choices.

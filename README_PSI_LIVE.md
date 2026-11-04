# LogiEdge — live E1 MQTT confidence drift and recovery

Prerequisites: the existing MQTT/inference/operations services are running, and
the 300-window PSI reference/offline diagnostic have passed. This package adds
new files; it preserves the reference, shared preprocessing, trained models,
inference image and Docker named volumes.

## What will happen

An additional Docker container generates the original C1 signals at real
1 Hz/0.5 Hz deadlines and subscribes to the existing inference result topic.
Your already-running D2 inference service performs every prediction.

One run ID, one continuous sensor random-number/filter history and one last-100
rolling monitor cover all phases. No confidence scores are prefilled from the
reference or offline diagnostic.

| Phase | Automatic behaviour | Approximate real time |
|---|---|---|
| baseline_warmup | Gather 100 live outputs and observe PSI below 0.10 | About 18 minutes |
| combined_fault | Apply --anomaly combined; require PSI > 0.25 within 300 s; hold fault for 300 s | 5 minutes |
| clean_recovery | Apply --anomaly none; keep the unreset last-100 buffer until PSI < 0.10 | About 17–19 minutes |

Allow roughly 40–42 minutes overall. Keep the laptop plugged in, awake and
Docker Desktop running. Display-off is different from sleep. Suspend-clock gaps, long processing gaps, stalled inference or failure to
meet a criterion stop the run with failed/interrupted evidence. Ordinary UTC
adjustments are logged separately while monotonic sampling continues. A very
large abrupt UTC jump is flagged for review and cannot automatically pass.

## Start from Ubuntu

Save LogiEdge_PSI_Live.zip in Windows Downloads. Run one command at a time:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_PSI_Live.zip" -d .
```

Run the combined tests:

```bash
docker compose run --rm --no-deps --entrypoint python trainer \
  -m unittest discover -s tests -p 'test_psi*.py' -v
```

Expected after the clock fix: 34 tests, OK. They include the earlier reference tests, phase/cadence
checks, frozen-reference validation, output-log auditing, and the actual pinned
Paho client talking to an isolated MQTT wire fixture on an ephemeral port.
Fixture tests are not actual project model/SLA results. Stop on a failure.

Start the real experiment:

```bash
bash deployment/start_psi_live.sh
```

The command checks the currently loaded model/statistics against the reference,
starts a separate named container, waits for MQTT subscription readiness and
returns. The experiment keeps running in Docker. A second active PSI run is
blocked. Keep the existing regular sensor simulator stopped during this run.

The first status should say running/baseline_warmup. Sample count starts near
zero; the first inference usually arrives around 40 seconds. The script aborts
if no matching inference arrives within 100 seconds.

## Watch the run

```bash
bash deployment/psi_live_status.sh
```

This prints phase, elapsed real seconds, sample count, total inferences, latest
model label, PSI values and the measured criteria. A null PSI during initial
warm-up is expected; it is not filled with reference values.

To watch the actual terminal output continuously:

```bash
bash deployment/psi_live_logs.sh
```

Ctrl+C exits this log viewer; the separate experiment container continues.
After warm-up, logs include a PSI reading every 60 real seconds, for example
the following *formats*, not prefilled numerical results:

- [LOGIBRIDGE PSI] PSI=... max_class_PSI=... n=100 phase=...
- [CONTROL REQUEST] --anomaly combined source=automatic_coordinator
- [PHASE APPLIED] --anomaly combined elapsed=...
- [LOGIBRIDGE DRIFT ALERT] PSI=...
- [CONTROL REQUEST] --anomaly none source=automatic_coordinator
- [LOGIBRIDGE DRIFT RECOVERY] PSI=...

Mode requests and their actual first affected sensor-sample timestamps are
recorded separately. Injection-to-alert time is measured from that sensor
sample to a real PSI evaluation with a monotonic clock.

## Complete and audit the evidence

When status says completed and all three criteria are true:

```bash
bash deployment/finalize_psi_live.sh
bash deployment/runtime_status.sh
```

The finalizer independently recomputes the saved PSI values from the actual
recorded model probabilities, verifies the live three-phase timing, writes an
audited summary/CSV and renders psi_live_timeline.png for the report/PPT.
It refuses incomplete, failed or internally inconsistent evidence.

The unique folder printed at startup, also in reports/psi_live_pointer.json,
contains raw acknowledged sensor records, accepted inference messages, PSI
evaluations, phase/control changes, the frozen reference copy, runtime identity,
live_status.json and live_results.json. Docker also retains the container's
complete terminal log; the log-viewer wrapper finds its current name.

Send the final audited summary and runtime status for review. Successful
startup alone does not complete E1.

## Confidence definition and interpretation

The primary score remains P(Normal), explicitly defined in the previous
package. Maximum-class confidence is retained as a diagnostic, using its own
frozen reference distribution. The four fixed bins, 1e-6 empty-bin rule,
last-100 samples and 60-second interval are unchanged.

The team's offline diagnostic observed maximum-confidence PSI = 0.0 while the
model's predicted operating class changed. Report that observation honestly.
Do not describe the primary P(Normal) statistic as maximum-class-confidence PSI.

Anomaly labels and control metadata coordinate the test and measure elapsed
times; they do not enter the statistical score, model features or model
classification. PSI is recalculated only from actual prediction probabilities.

The model may return to Normal well before PSI does: the model uses its next
30-second feature window, while PSI deliberately remembers the last 100 model
outputs. PSI is a distribution-change monitor, distinct from the immediate
temperature alarm and the earlier sub-90-second model/local-alert evidence.
It does not, by itself, establish a drop in classification accuracy or concept
drift. The injected operating-regime change produces input/output-distribution
shift; live labels would be needed to diagnose changed prediction relationships.

## Controls and interruption

The default measured run handles both mode changes automatically. For a later
manual demonstration the same explicit CLI requests are available:

```bash
bash deployment/set_psi_anomaly.sh --anomaly combined
bash deployment/set_psi_anomaly.sh --anomaly none
```

Premature injection/restoration is rejected. Mode requests never reset the
monitor. Avoid manual changes during the first automatic measurement.

If you must cancel:

```bash
bash deployment/stop_psi_live.sh
```

Evidence is retained and marked interrupted. If interrupted during combined
mode, the existing inference service can still retain Critical severity until
fresh clean sensor readings qualify recovery. Resume with a new experiment
only after reviewing the interrupted state; it does not resume a partially
filled rolling buffer or claim a completed live test.

## Video and validation limits

Capture the real terminal output or retain the full original recording. In the
15–20-minute final video, use clearly labelled chronological excerpts/time
jumps to show baseline, injection, alert and recovery. Keep the complete logs.
The earlier offline 40-second/1060-second logical timings must not be presented
as measurements from this live run.

Author-side validation after the clock fix: 34 combined tests passed, including the actual
paho-mqtt 2.1.0 SUBACK/manual-PUBACK protocol test; Bash syntax checks passed
for all six wrappers. State-machine and probability fixtures are labelled test
data. The authoring environment has no Docker engine, so actual Compose/WSL,
trained-model streaming and the complete real-time E1 experiment remain pending
the team's laptop execution.

Primary API sources checked:
https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html
https://docs.docker.com/reference/cli/docker/compose/run/

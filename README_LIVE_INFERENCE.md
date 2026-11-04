# LogiEdge — Docker live inference and durable offline alerts

This update adds the D2 live service, a persistent local alert/outbox database,
and an operations-centre simulator for the later offline/recovery test. The
existing MQTT/simulator/trainer service definitions are preserved.

CPU-demo default: M1 FP32. In the team's measured CPU session, M1 had the lowest
mean/p95 latency and energy proxy. M3 was smallest, at 3216 bytes. M1's 5416-byte
file adds only 2200 bytes, so the default prioritizes the observed CPU latency
and energy proxy. This does not claim Raspberry Pi or Hailo measurements.

## First checkpoint: one clean live inference

Keep Docker Desktop running. Save LogiEdge_Live_Inference.zip into
C:\Users\Nandhini G\Downloads. Run one command at a time in Ubuntu:

```bash
cd ~/logibridge
cp compose.yaml compose.before-inference.yaml
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Live_Inference.zip" -d .
docker compose config -q
docker compose run --rm --entrypoint python trainer -m unittest discover -s tests -p 'test_live_inference.py' -v
```

Expected: 17 tests passing without skips. They include actual FP32/INT8 parity
with the benchmark backend, shared-feature-pipeline parity, physical temperature
escalation, persistent queues, lost-response retry, duplicate protection, a real
loopback HTTP delivery test and static asset preparation. Test-only synthetic
fixtures are not project models or performance results.

Stage verified copies of the team's existing files:

```bash
bash deployment/prepare_inference.sh --default-variant M1_FP32
```

Expected: runtime assets prepared and originals unchanged. This command verifies
the dataset, normalization, model hashes, INT8 graphs and M3 structure, then
copies model/statistics files into deployment/runtime_assets. It does not
retrain or fit any statistics. The immutable registry lists all three variants.

Build and start the services:

```bash
docker compose build inference operations
docker compose up -d --wait --wait-timeout 120 mqtt operations inference
docker compose ps
```

The first inference-image build installs TensorFlow and can take several
minutes. Expected: mqtt, operations and inference are healthy. The service
loads the staged M1 model and frozen stats before becoming ready. Health means
the local model is loaded and MQTT is connected; operations-centre availability
is independent of inference health.

If any step fails, stop there and share its error and relevant logs:

```bash
docker compose logs --tail 80 inference operations
```

Open two Ubuntu terminals/tabs. In Terminal A, subscribe before the simulator
starts:

```bash
cd ~/logibridge
docker compose exec mqtt mosquitto_sub -h 127.0.0.1 -t logibridge/trucks/TRUCK_001/inference -q 1 -C 1 -W 120 -v
```

Terminal A waits for one actual MQTT result. In Terminal B:

```bash
cd ~/logibridge
bash deployment/run_simulator.sh --anomaly none --duration 45 --seed 6001 --purpose live
```

The first complete filtered window normally appears around 40 seconds into
the capture. Its interval is 10–40 seconds, with 30 temperature and 15 vibration
samples. The MQTT message should show six features, model_variant M1_FP32 and
the model's actual prediction/probabilities. Door state is context, not a
seventh numerical feature. A clean run is expected to classify Normal; verify
the actual message rather than fabricating that result.

After the capture finishes:

```bash
bash deployment/runtime_status.sh
docker compose logs --tail 40 inference
```

Share the MQTT message, service status and final log lines at this checkpoint.
The initial 45-second capture produces only one complete live model window;
it is an integration check, not another accuracy evaluation.

## What runs locally

- Simulator publishes temperature, vibration RMS and door events to the local
  Mosquitto broker. MQTT uses the broker's shared loopback namespace.
- A network callback enqueues readings; one processing worker owns the TFLite
  interpreter. Blocking publication acknowledgements happen outside callbacks.
- The unchanged C2 FeaturePipeline performs independent causal five-sample
  filters and six-feature 30-second windows every ten seconds.
- FrozenNormalizer loads the independently fitted training_stats.npy and checks
  its metadata/hash. No runtime fitting is possible in this class.
- The model uses the same optimized built-in TFLite CPU backend as F2, one
  interpreter thread, with default delegates disabled.
- Each inference is committed to SQLite and published at QoS 1 to the required
  logibridge/trucks/{truck_id}/inference topic.
- Local alerts are committed and printed immediately; an independent worker
  retries their delivery to the operations simulator. Heartbeats are eligible
  every 60 seconds when inference windows are available. Raw sensor streams and
  every model result are not continuously uploaded.

Incoming QoS-1 sensor messages use Paho manual acknowledgement after processing
and SQLite writes. On a worker failure, a persistent broker session can replay
unacknowledged messages. The in-memory feature buffers warm up after process
restart; the alert/outbox database and last Critical model latch survive.

## Model labels and physical safety are explicit

D1's temp_drift scenario is labelled Warning throughout its 15-minute capture,
even after the unbounded simulated temperature passes the physical Critical
threshold. The model is evaluated against those declared scenario labels.

The live service therefore reports the learned model class and an explicit
operational safety severity separately. The documented setpoint is 4 C:

- deviation up to 1 C: physical Normal;
- deviation greater than 1 C through 3 C: physical Warning;
- deviation greater than 3 C: immediate physical Critical.

Operational severity is the maximum of the current temperature-safety level and
the latest model class. Its alert states why it was raised. A raw-temperature
alarm does not claim a neural-network detection. During warm-up, alert messages
say WARMING_UP instead of inventing a model prediction. A previous model Critical
state stays latched through restart/new-run warm-up until a fresh clean model
window qualifies recovery.

The sensor payload's anomaly field is not an input to preprocessing, model
classification or safety decisions. It is retained only as controlled-run
evaluation metadata in the local run-start record.

## Next checkpoint: demonstrate offline alerts and recovery

After the first clean integration check, the following sequence is ready for
execution. It simulates an unavailable operations/cellular path by stopping
the operations-centre container. Local MQTT and inference keep running.

```bash
docker compose stop operations
bash deployment/run_simulator.sh --anomaly combined --duration 90 --seed 6002 --purpose live
bash deployment/runtime_status.sh --sla-latest-combined
docker compose logs --tail 60 inference
```

Verify actual results: local Critical alerts, continued inference, pending
uplink rows and operations unavailable. The SLA summary separately reports
first local Critical and first model Critical times from this controlled
combined run's start. Both are observations, not prefilled results. Missing
Critical observations stay null/false. The 90-second criterion must be checked.

Then demonstrate that pending data survives an inference-process restart:

```bash
docker compose restart inference
bash deployment/runtime_status.sh
```

The service needs a few seconds to reload the model. Pending outbox rows and
local alerts must remain. Restarting does not clear databases or acknowledge
undelivered records. It does reset feature warm-up and in-process counters.

Restore the operations-centre simulator:

```bash
docker compose start operations
```

Allow up to about 15 seconds for startup/retry, then:

```bash
bash deployment/runtime_status.sh --sla-latest-combined
```

Expected: pending_uplink reaches zero, acknowledged_uplink increases, and the
operations centre contains the original event IDs. Delivery is at least once;
the backend stores an event ID only once. A lost HTTP response causes a retry,
not event loss. The backend acknowledges only after its database commit.

Run a clean capture to qualify model/temperature recovery:

```bash
bash deployment/run_simulator.sh --anomaly none --duration 45 --seed 6003 --purpose live
bash deployment/runtime_status.sh
```

Review the actual fresh model window and severity-0 recovery alert. These
separate captures demonstrate local inference/offline delivery. E1's required
mid-run drift injection and PSI recovery are separate upcoming work.

## Durable state and inspection

Named Docker volumes edge_state and operations_state store SQLite databases.
WAL transactions use synchronous=FULL. Events and their outbox entries are
inserted in the same local transaction. An event stays pending until the
matching server acknowledgement arrives. FIFO retries back off up to ten
seconds. Completed records remain as audit history.

The status command runs as the inference container's own user, reads the local
database and queries the operations-centre simulator. Operations unavailability
does not make the local read fail. Preserve the named volumes for the demo;
volume deletion would discard their state.

The operations simulator's optional host endpoint is bound to loopback at
http://localhost:18080/stats. It represents the operations backend for this
private demonstration. Production cellular transport would need authenticated
TLS and a production backend; this HTTP simulator is not that deployment.

## MODEL_PATH and cache preparation

All three verified files are staged under deployment/runtime_assets/models and
mounted read-only at /models. A later switch requires no rebuild:

```bash
MODEL_PATH=/models/M3_STRUCTURED_INT8.tflite docker compose up -d --no-deps --force-recreate inference
```

Startup prints the actual variant and SHA256. The model registry and statistics
are checked before inference. Restore the measured CPU default with:

```bash
MODEL_PATH=/app/model.tflite docker compose up -d --no-deps --force-recreate inference
```

The inference Dockerfile uses python:3.11-slim. Pip layers, source and frozen
statistics precede the final COPY of model.tflite. The registry contains all
three known variants and stays byte-identical when staging a different embedded
model. This prepares the layer-cache demonstration; actual cached-layer output,
compressed transfer bytes and the 85-truck saving still need measurement.

## Verification scope and remaining requirements

Authoring tests exercise the actual pinned TensorFlow runtime, SQLite durable
delivery and loopback HTTP backend. Docker is unavailable in the authoring
environment, so image build, live MQTT wiring and the team's live classifications
must be verified on the Windows/WSL/Docker laptop using the commands above.

The supplied image/dependencies target the demonstrated Linux amd64 CPU setup.
The proposed Pi requires an ARM64 runtime build and device measurements. No
Hailo accelerator performance or complete truck qualification is claimed.

F1 and F2 results are recorded. D2 integration/offline tests, measured OTA cache
evidence, E1 PSI, exactly seven idempotent Ansible tasks, final hardware/SLA
recommendation, report, PPT and four-member video remain on the completion list.

Primary references:
- https://eclipse.dev/paho/clients/python/docs/client.html
- https://docs.python.org/3.11/library/sqlite3.html
- https://www.sqlite.org/isolation.html
- https://docs.docker.com/reference/cli/docker/compose/up/

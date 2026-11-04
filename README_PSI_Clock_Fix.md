# LogiEdge — correction for the E1 clock guard

Apply this incremental patch after the original PSI Reference and PSI Live
packages. It fixes the monitor's treatment of adjustable UTC without changing
the model, normalisation, reference, four bins, rolling count, interval or PSI
thresholds.

## What the failed run actually established

The team's run received eight Normal inferences and recorded zero missed
sensor deadlines before stopping at 116.508 elapsed seconds. Its reference
remained unchanged. It had not collected 100 outputs, injected a fault or
completed E1.

The original guard rejected a greater-than-five-second difference between
UTC and monotonic elapsed time. That condition can arise from a UTC adjustment
as well as suspension, so the original error wording did not establish which
event occurred. The output alone cannot identify the cause conclusively.

## Corrected timing policy

- Keep time.monotonic for sampling deadlines, PSI intervals and elapsed SLA
  measurements. UTC continues to be recorded as event metadata.
- Log UTC/elapsed differences to clock_events.jsonl. A small forward/backward
  UTC correction does not abort or reset the experiment.
- On Linux, compare CLOCK_BOOTTIME with monotonic elapsed time to detect a
  suspend-clock gap. Record that capability explicitly.
- Reject a processing-loop gap over ten seconds or a suspend-clock gap over
  five seconds; record the measured gap and its type.
- A sudden UTC offset jump of at least 60 seconds is ambiguous on a VM. Retain
  its evidence, flag clock_review_required, and do not automatically mark the
  completed experiment passing.
- Keep the frozen reference/model checks and the existing inference-stream,
  baseline, five-minute fault-detection and recovery watchdogs.

This is not a larger allowed UTC-difference threshold replacing the old one.
The clock domains now serve separate purposes. Small UTC corrections are
diagnostic events; evidence of actual sampling/suspend gaps is checked
separately. Guest clocks cannot conclusively identify every possible host/VM
interruption, which is why capabilities and ambiguous large jumps are recorded.

The patch also sets Docker's process-scoped COMPOSE_IGNORE_ORPHANS setting in
the live wrappers. It suppresses the warning about preserved experiment
containers and does not remove any containers or logs. Do not use
--remove-orphans to resolve this monitor-code issue.

## Apply in Ubuntu

Save LogiEdge_PSI_Clock_Fix.zip into Windows Downloads.

Preserve the failed container's console output:

```bash
cd ~/logibridge
docker logs --timestamps logibridge-psi_live_20261103t121227z_jstnwb \
  > reports/psi_live_20261103T121227Z_JsTnwb/container.log
```

Extract the patch and run the combined checks:

```bash
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_PSI_Clock_Fix.zip" -d .
docker compose run --rm --no-deps --entrypoint python trainer \
  -m unittest discover -s tests -p 'test_psi*.py' -v
```

Expected: 34 tests, OK. Stop and share any error.

Start a fresh measurement using the existing frozen reference:

```bash
bash deployment/start_psi_live.sh
```

The start command creates a new run ID/evidence folder. The old eight samples
are not reused as a live baseline; the failed evidence remains preserved.
No Docker/Ubuntu reboot or model/reference regeneration is needed for this
code correction.

Keep the laptop awake. After about three minutes:

```bash
bash deployment/psi_live_status.sh
```

Share this status before waiting for the entire run. Expect running status,
increasing inferences, clock.guard_version=2 and clock_review_required=false.
The clock object now reports any UTC notices, maximum processing/suspend gaps,
and the last measured interruption if one occurs. A CLOCK ADJUSTMENT notice
alone is not a model failure. If status is failed, preserve/share its error
and clock fields rather than restarting repeatedly.

Ansible deployment remains paused until the new live experiment completes and
its final audit passes.

## Verification

All 34 combined tests passed in the authoring environment, including pinned
Paho protocol checks, the prior PSI/pipeline tests and six additional clock
regressions. Forward/backward UTC steps and a correction at logical second
116 are tested with continuous sampling; process pauses, backwards elapsed
time, suspend-clock gaps, large ambiguous jumps and capability fallback are
tested separately.

These tests do not substitute for the team's full live measurement. The
target Docker/WSL clock behaviour and eventual live PSI results still need
to be observed. The final audit now checks version-2 clock evidence too.

Primary clock and Compose documentation:
https://docs.python.org/3.11/library/time.html
https://docs.kernel.org/core-api/timekeeping.html
https://docs.docker.com/compose/how-tos/environment-variables/envvars/

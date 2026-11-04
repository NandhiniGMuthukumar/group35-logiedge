# LogiEdge — D1 dataset collection and preparation

This update prepares the required scenario dataset using the verified simulator, preprocessing engine and frozen normalizer. It does not train a model yet.

The team's fixed normalization was confirmed from 57 independent clean windows, seed 1001. The user demonstrated successful load/transform checks and unchanged files. The later correct-versus-shifted model accuracy comparison remains pending.

## Required captures: 50 real minutes in total

Run one simulator at a time, keeping the laptop awake and Docker Desktop running. Every capture must finish with stop_reason=duration and zero missed deadlines.

| D1 scenario label | Mode | Duration | Seed | Expected temperature / vibration counts |
| --- | --- | --- | ---: | --- |
| 0 Normal | none | 1200 seconds / 20 minutes | 2001 | 1200 / 600 |
| 1 Warning | temp_drift | 900 seconds / 15 minutes | 3001 | 900 / 450 |
| 2 Critical | combined | 900 seconds / 15 minutes | 4001 | 900 / 450 |

These seeds are distinct from normalization seed 1001 and the smoke-test seeds.

If the 20-minute Normal capture was already started from the chat update, let it finish and use that completed capture. Do not repeat it unnecessarily.

### Normal

```bash
cd ~/logibridge
bash deployment/run_simulator.sh --anomaly none --duration 1200 --seed 2001 --purpose training
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest
```

### Temperature drift

```bash
bash deployment/run_simulator.sh --anomaly temp_drift --duration 900 --seed 3001 --purpose training
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest
```

### Combined

```bash
bash deployment/run_simulator.sh --anomaly combined --duration 900 --seed 4001 --purpose training
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest
```

Save the final inspection summaries, not every progress line. Raw JSONL and manifests are already saved under data/raw.

## Important interpretation of D1 labels

The brief assigns the whole 15-minute temp_drift scenario to Warning. However, its prescribed +0.08 degrees C per second crosses a +3-degree deviation after roughly 38 seconds. Consequently much of that scenario physically exceeds the temperature Critical threshold.

The dataset records scenario_label using D1's prescribed mapping. This mapping must be described honestly in the report. Operational temperature severity and any safety override in the later live service must remain separately identified; the scenario label must not be presented as the real physical safety state for every reading. No drift cap or undisclosed relabelling is applied.

Raw capture and feature metadata preserve the original anomaly modes.

## Install this update

Save LogiEdge_Dataset_Preparation.zip to C:\Users\Nandhini G\Downloads. After collection, run in Ubuntu:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Dataset_Preparation.zip" -d .
docker compose run --rm --entrypoint python simulator -m unittest discover -s tests -v
```

Expected: 34 tests pass. The code uses the existing image and project bind mount; there are no new dependencies or Compose changes.

This update also changes an old preprocessing CLI status message to explain that it saves raw feature vectors and normalization is applied separately. It no longer reports normalization as pending after statistics already exist. Filtering, formulas and feature order are unchanged.

## Build the dataset

Only after all three completed captures are available:

```bash
bash deployment/prepare_dataset.sh
```

The builder selects the newest purpose=training capture for each required mode. It verifies durations, completion, counts, deadlines, one truck ID, distinct source seeds, separation from normalization, and non-overlapping wall-clock capture times.

It extracts exactly the six existing features and uses FrozenNormalizer.load(...).transform(...). It does not refit means or standard deviations.

Expected window counts with full filter warm-up:

| Class | All windows | Training | Validation | Purged overlap |
| --- | ---: | ---: | ---: | ---: |
| Normal | 117 | 90 | 24 | 3 |
| Warning | 87 | 66 | 18 | 3 |
| Critical | 87 | 67 | 17 | 3 |
| Total | 291 | 223 | 59 | 9 |

A minimum of 200 training-only windows is enforced so full INT8 representative calibration is feasible later.

## Why validation has 59 rows

20% of 291 is 58.2; the global validation count is rounded up to 59, as in a conventional fractional hold-out. The count is allocated proportionally across the three scenarios using deterministic remainder allocation, giving 24, 18 and 17.

Within each run, validation uses the final windows chronologically. Training windows whose raw input support overlaps validation are purged. Because a moving average depends on earlier raw samples, checking only nominal window boundaries would be insufficient.

The observed validation fraction before purging is 59/291, approximately 20.27%. Nine overlapping training candidates are discarded after choosing validation. The remaining 223 training rows are not independently mixed with validation.

All model variants, normalization experiments and benchmarks must reuse these exact saved indices. The Critical validation support is 17 windows; report that count beside the later recall.

## Saved artifacts on the laptop

Under training/dataset:

- logibridge_dataset.npz — numeric arrays only:
  - X_raw: (291, 6), float32;
  - X_normalized: (291, 6), float32;
  - y: (291,), integer D1 scenario labels;
  - train_indices: (223,);
  - validation_indices: (59,);
  - purged_indices: (9,).
- dataset_windows.csv — source run IDs, scenario labels, window/raw-support timestamps, split roles, and the six feature values.
- dataset_manifest.json — source hashes, normalization hash, feature order, recipe, class counts, split method, and dataset hashes.

CSV metadata are not model features.

The dataset loader verifies file hashes, feature order, numerical validity, split partitions, raw-support separation, and consistency with the same frozen statistics.

Repeating preparation for the same sources reuses the existing dataset and validation indices. Different sources cannot overwrite a frozen dataset; use a deliberate new --output-dir for a new dataset version.

## Verification

All 34 authoring tests passed (Python 3.12.14 / NumPy 2.3.5). Run them inside the target Python 3.11 / NumPy 1.26.4 container as shown above.

The eight new tests cover global 20% allocation, raw-history purging, source roles/durations/independence, numeric dataset round-trip with fixed statistics, reuse without regeneration, partition and 200-sample checks, overlapping-capture rejection, and corrupted dataset detection.

Synthetic fixtures exist only in tests and temporary test directories. This archive contains no measured training dataset, trained weights, or fabricated accuracy values.

## Current requirement checkpoint

C1 sensor generation: verified for all four modes.
C2 features and fixed normalization: verified on the laptop, with model-accuracy comparison pending.
D1: collect the specified training captures, then verify this dataset builder.
Next: train the 6-32-16-3 MLP, evaluate the fixed held-out set, and require accuracy strictly above 88% before continuing to conversion.

A/B report analysis, D2 inference/OTA, model variants/benchmarks, PSI, Ansible, and final deliverables remain separate required work.

## Official references

- https://numpy.org/doc/1.26/reference/generated/numpy.savez_compressed.html
- https://numpy.org/doc/1.26/reference/generated/numpy.load.html


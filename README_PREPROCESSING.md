# LogiEdge — C2 filtering and six-feature extraction

This is an update to the working C1 simulator project. Install it into /home/nandhini/logibridge after all four C1 smoke modes have passed. It adds the shared preprocessing engine and capture extraction, plus NumPy in simulator image version 0.2.0. It does not yet fit normalization statistics or train a model.

## What is implemented

- Independent causal five-sample moving averages for temperature and vibration.
- Full filter warm-up: the first four readings of each stream do not emit a filtered value.
- Thirty-second sliding windows, stepped every ten seconds.
- Exactly six numerical features: three temperature features, then three vibration features.
- Door events retained as context and excluded from the feature vector.
- CSV, numeric NPY, detailed JSONL windows, and provenance metadata under data/processed.
- A shared FeaturePipeline class for both batch extraction and later live inference.
- QoS duplicate suppression, per-truck separation, and rejection of incomplete/stale windows.

## Installation in Ubuntu

Download LogiEdge_Preprocessing.zip to C:\Users\Nandhini G\Downloads. Run one command at a time:

```bash
cd ~/logibridge
cp compose.yaml compose.before-preprocessing.yaml
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Preprocessing.zip" -d .
docker compose config -q
docker compose build simulator
```

The updated Compose file selects logibridge-simulator:0.2.0. Its MQTT service settings and named volume match the previous configuration. The existing simulator source and raw captures are not included in this update archive.

The image pins paho-mqtt 2.1.0 and NumPy 1.26.4. Python remains the assignment's required 3.11-slim base.

## Validate in the actual target image

```bash
docker compose run --rm --entrypoint python simulator -m unittest discover -s tests -v
```

Expected: 18 tests pass (the previous 8 plus 10 preprocessing tests).

The authoring checks ran on Python 3.12.14 and NumPy 2.3.5. The provided target-container test is therefore an explicit compatibility check on Python 3.11 and NumPy 1.26.4; the implementation uses APIs documented in NumPy 1.26.

## First feature extraction

```bash
bash deployment/run_preprocessing.sh --smoke-tests
```

This selects the newest completed purpose=smoke_test capture for each mode: none, temp_drift, vibration, combined. Existing raw data are read; no new sensor capture is started.

For each existing 45-second capture, expect:
- shape (1, 6);
- a complete window covering seconds 10 through 40, start included and end excluded;
- 30 smoothed temperature values and 15 smoothed vibration values;
- finite features in the fixed order shown below.

Across the four captures, expect four rows in data/processed/smoke_features.csv. These rows are smoke-test evidence, not a training dataset.

## Why the first window ends at 40 seconds

At 1 Hz, the first full five-sample temperature average becomes available around second 4. At 0.5 Hz, the first full vibration average becomes available around second 8. A full 30-second window after that warm-up is aligned to the next 10-second boundary. Thus the first accepted window is [10, 40), followed by [20, 50), [30, 60), and so on.

There is no zero padding or use of future readings. Filters continue across overlapping windows; they are not restarted for each window. A later ten-minute completed capture will normally produce 57 windows, ending at seconds 40, 50, ..., 600. This is consistent with the brief's approximate window counts.

## Fixed feature definitions

| Index | Feature | Definition |
| --- | --- | --- |
| 0 | temperature_mean | Mean of the smoothed temperature values |
| 1 | temperature_std | Population standard deviation, ddof=0 |
| 2 | temperature_rate_c_per_min | Least-squares slope against actual timestamps, multiplied by 60 |
| 3 | vibration_rms | Square root of the mean squared smoothed vibration_rms values |
| 4 | vibration_peak | Maximum absolute smoothed vibration_rms value |
| 5 | vibration_kurtosis | Fourth central moment / variance squared, minus 3: Fisher excess kurtosis |

For a constant vibration window, kurtosis is mathematically undefined. This implementation records a constant_vibration flag and uses the documented finite sentinel 0 for that feature. This convention must stay consistent in training and inference.

For nonconstant values, the moment convention matches scipy.stats.kurtosis(fisher=True, bias=True). SciPy is not required at runtime because the short formula is implemented directly with NumPy.

The simulator publishes vibration RMS summaries at 0.5 Hz. This feature extractor aggregates those summaries; it does not reconstruct an unobserved 500 Hz waveform.

## Output files

For each source capture stem, four derived files are written:
- .features.csv: human-readable metadata plus the six named feature columns.
- .features.npy: numeric float32 array with shape (number_of_windows, 6), saved without pickled objects.
- .features.jsonl: window boundaries, raw-data support, counts, door context and feature vectors.
- .features.meta.json: source hashes, feature order, units, processing recipe, and source purpose.

Only FEATURE_NAMES should be selected as model inputs. Do not pass CSV metadata, run IDs, anomaly names, purpose tags, or door state into the classifier.

The normalised flag in metadata is false. Frozen normalization follows after a separate clean ten-minute capture; no statistics are fitted from these smoke tests.

## Temporal support and validation leakage

A feature window also depends on earlier raw samples used by its causal filter. For example, the vibration average at second 10 includes samples from seconds 2, 4, 6, 8, and 10. Therefore the raw support extends before the nominal 30-second feature window.

The output records raw_support_start_unix and raw_support_end_unix. Use these when enforcing a guard interval between training and validation windows. Adjacent overlapping windows must not be randomly split as if independent.

## Single-capture commands for later use

```bash
bash deployment/run_preprocessing.sh --latest
bash deployment/run_preprocessing.sh --manifest data/raw/EXACT_CAPTURE.manifest.json
```

The extractor checks capture metadata and acknowledged record counts. It refuses missed-deadline captures and windows with insufficient or stale readings. It does not fill absent sensor readings with zeros or invent replacement measurements.

## Implementation walkthrough

- MovingAverage holds the last five real values for one sensor.
- temperature_features and vibration_features compute separate three-value vectors.
- FeaturePipeline concatenates them and keeps door context outside the six values.
- extract_features replays genuine saved records through the same engine and saves their results.
- run_preprocessing.sh uses the Ubuntu UID/GID so generated files belong to the user.

## Verification completed during authoring

The 10 new tests cover full filter warm-up; an analytic 4.8 degrees C/min ramp and its full six-feature vector; window width/step/counts/support; causality; duplicate suppression; incomplete-window rejection; exclusion of door/anomaly metadata from features; kurtosis conventions; per-truck/out-of-order validation; and consistency between incremental processing and saved capture extraction. All 18 combined tests passed.

Synthetic fixtures exist only inside tests and temporary test directories. They are not included as project measurements or in data/raw. The actual target-container tests and the team's real-capture extraction still need to run on the demonstration laptop.

## Requirement checkpoint

C1: all four modes have been demonstrated by the user with correct counts and zero missed deadlines.
C2: filtering and six-feature extraction are ready for target-machine validation. Independent ten-minute clean statistics, runtime normalization, and the 3-sigma normalization/accuracy experiment remain pending.
C3: feature-level concatenation is implemented; its report justification remains to be written by the team.
D1 onward: training, TFLite variants, inference image, offline alert delivery, benchmarks, PSI, and Ansible remain pending.

The required 20/15/15-minute training scenario captures are separate future steps.

## Official references

- https://numpy.org/doc/1.26/reference/generated/numpy.std.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
- https://numpy.org/doc/1.26/reference/generated/numpy.save.html


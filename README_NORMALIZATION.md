# LogiEdge — fixed normalization from an independent clean capture

This update follows the user's successful C2 test: 18 tests passed on the Windows/Ubuntu Docker environment and all four 45-second captures produced shape (1, 6) feature arrays. It adds a fit-once command, a runtime load/transform class, a calibration sanity check, and eight tests.

No training_stats.npy is supplied in this archive. The actual statistics must come from the team's own ten-minute clean capture.

## Step 1: collect once

In Ubuntu at ~/logibridge, keep Docker Desktop running and the laptop awake:

```bash
bash deployment/run_simulator.sh --anomaly none --duration 600 --seed 1001 --purpose normalization
```

If this capture was already started from the chat update, let that run finish. Do not start a second simulator for the same truck concurrently.

Expected for an uninterrupted run: 600 temperature readings, 300 vibration readings, zero missed deadlines, and stop_reason=duration. With the existing 30-second door period and 5-second opening, the 600-second capture has 39 door events: the initial CLOSE, 19 OPEN, and 19 subsequent CLOSE events. The OPEN event at exactly 600 seconds is outside the capture.

This is independent normalization data. The later 20/15/15-minute training captures will use purpose=training and different seeds. The four short purpose=smoke_test runs are not used to fit the scaler.

## Step 2: install these source files

Save LogiEdge_Normalization.zip to C:\Users\Nandhini G\Downloads. After the capture finishes, run:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Normalization.zip" -d .
```

The new modules use the existing simulator:0.2.0 image's NumPy dependency and the project bind mount. This update contains no Compose, Dockerfile, simulator, or preprocessing-engine changes.

## Step 3: inspect the real capture

```bash
docker compose run --rm --entrypoint python simulator -m data_pipeline.inspect_capture --latest
```

Confirm anomaly=none, purpose=normalization, elapsed about 600 seconds, temperature=600, vibration_rms=300, and no missed deadlines. Stop if the capture reports an error or missed deadlines.

## Step 4: run the target-container checks

```bash
docker compose run --rm --entrypoint python simulator -m unittest discover -s tests -v
```

Expected: 26 tests pass. The original 18 are included, plus 8 normalization tests.

## Step 5: fit once and save

```bash
bash deployment/fit_normalizer.sh --latest-normalization
```

This:
1. selects the latest capture tagged purpose=normalization;
2. checks its clean mode, requested 600 seconds, completion, record counts, and deadlines;
3. applies the already verified C2 feature pipeline;
4. produces the expected 57 complete six-feature windows;
5. computes each feature's mean and population standard deviation in float64;
6. saves data_pipeline/training_stats.npy, with shape (2, 6);
7. saves data_pipeline/training_stats.meta.json, containing source hashes, seed, run ID, feature order, preprocessing recipe, and statistics hash.

Rows of training_stats.npy are: row 0 = mean, row 1 = std, in FEATURE_NAMES order. Standard deviations smaller than 1e-8 are floored to 1e-8 and the affected features are recorded.

Re-running the command for the exact same capture loads the existing statistics without refitting. A different capture cannot overwrite existing frozen statistics. Deliberate future versions can use a different --output filename.

## Step 6: load and check

```bash
bash deployment/check_normalizer.sh
```

Expected:
- baseline feature shape (57, 6);
- normalized calibration means close to 0;
- normalized calibration standard deviations close to 1 for non-floored features;
- statistics and metadata unchanged after transformations;
- a successful three-sigma mean-shift arithmetic check.

These are calibration and implementation sanity checks on the fitting data. They are not held-out model accuracy results. If a feature was constant and required flooring, its normalized calibration standard deviation can be 0 instead of 1; the checker accounts for that explicitly.

## Runtime formula and API

For feature j:

z_j = (x_j - mean_j) / std_j

The runtime interface is:

```python
from data_pipeline.normalization import FrozenNormalizer

normalizer = FrozenNormalizer.load("data_pipeline/training_stats.npy")
normalized_features = normalizer.transform(six_value_feature_vector)
```

A vector returns shape (6,); a batch returns shape (N, 6). Results are float32 for model input. The saved statistics remain float64. The class has no data-fitting method.

The loader verifies feature order, preprocessing recipe, shape, numeric validity, and the statistics hash. Both statistics files must be kept together when preparing the later inference image.

## Required three-sigma experiment: still pending model training

The implementation defines shifted statistics as mean_shifted = mean + 3*std, with the same saved std. Applying them makes the normalized vector equal to the correct vector minus 3. No saved file is modified.

The arithmetic check is preparation only. The mandatory experiment must still run the trained model on the same held-out validation examples using correct versus shifted statistics, compare actual accuracies, and report the measured difference. Neither a 0/1 calibration check nor a three-sigma arithmetic check replaces that accuracy experiment.

## Verification

All 26 authoring tests passed on Python 3.12.14 / NumPy 2.3.5. Run them on the existing Python 3.11 / NumPy 1.26.4 container as shown above before fitting real data.

The eight new tests check vector/batch standardization, constant-feature flooring and numeric serialization, rejection of unsuitable source captures, repeat-use without refitting, prevention of accidental overwrite from a different capture, feature-order/recipe mismatches, hash-corruption detection, and the experimental shift/invalid inputs.

Calibration fixtures in tests are synthetic and stored only in temporary test directories. They are not project measurements. This archive contains source and tests, not measured statistics or calibration data.

## Current requirement checkpoint

- C1: all four anomaly modes verified on the team laptop.
- C2 filtering/features: 18 target tests and four real smoke feature vectors verified.
- C2 fixed normalization: this update is ready for execution on independent ten-minute data.
- C2 accuracy comparison: pending trained model and held-out validation.
- C3 code: separate three-feature sensor vectors concatenated into six values; report justification pending.
- D1 onward: training scenarios, model training/variants, inference, offline alert synchronization, benchmarks, PSI, and Ansible remain pending.

## Official references

- https://numpy.org/doc/1.26/reference/generated/numpy.std.html
- https://numpy.org/doc/1.26/reference/generated/numpy.save.html
- https://numpy.org/doc/1.26/reference/generated/numpy.load.html


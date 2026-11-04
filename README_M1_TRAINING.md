# LogiEdge — train M1 and measure the normalization experiment

This update adds the pinned ML container, the recommended 6-32-16-3 MLP, held-out evaluation, and gated FP32 TFLite conversion.

The user's prepared dataset was verified as 291 six-feature windows: 223 training, 59 validation, 9 purged. Normalization statistics remain independent and frozen.

## Install in Ubuntu

Save LogiEdge_M1_Training.zip to C:\Users\Nandhini G\Downloads. Run one command at a time:

```bash
cd ~/logibridge
cp compose.yaml compose.before-ml.yaml
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_M1_Training.zip" -d .
docker compose config -q
docker compose build trainer
```

The first ML build downloads TensorFlow and its dependencies, so it may take several minutes. Existing sensor/broker service settings are preserved. The new trainer service reads the project bind mount and does not need the MQTT broker for training.

## Pinned stack

- Python 3.11-slim
- tensorflow-cpu 2.14.1
- keras 2.14.0
- tensorflow-model-optimization 0.8.0
- NumPy 1.26.4
- Paho MQTT 2.1.0
- psutil 6.1.1
- Matplotlib 3.8.4

TFMOT 0.8.0's official release notes state that it uses Keras 2 and was tested against TensorFlow 2.14.1. The environment check verifies the chosen package versions and layer-wrapper compatibility. No M3 pruning is performed in this phase.

CPU execution is selected, with two TensorFlow intra-op threads and one inter-op thread. The image installs the needed C++/OpenMP runtime libraries and checks pip dependency consistency.

## Check the target environment first

```bash
docker compose run --rm --entrypoint python trainer -m training.check_ml_environment
```

Expected: the pinned versions, architecture 6-32-16-3, 803 parameters, no visible GPU devices, and a successful Keras/pruning API check.

Then run the tests in this ML image:

```bash
docker compose run --rm --entrypoint python trainer -m unittest discover -s tests -v
```

Expected: 43 tests, all passing with no skips. Do not use the smaller simulator image for this check: it has no TensorFlow dependency.

Authoring verification: 39 tests passed and the four TensorFlow-specific tests were skipped because the authoring runtime has neither TensorFlow nor Docker. Python compilation, shell syntax and Compose parsing were checked. The target tests deliberately cover model architecture, conditioning-fold equivalence and save/reload, FP32 TFLite conversion/equivalence, and a small synthetic training operation before project training starts.

## Train once

After environment and tests pass:

```bash
bash deployment/run_training.sh
```

The default uses seed 2026, Adam learning rate 0.003, batch size 16 and at most 400 epochs. It prints progress every 25 epochs. Early stopping and weight selection monitor training loss only. Held-out validation data are not passed to model.fit or used for checkpoint selection.

Each attempt gets a separate timestamped directory under training/models. Existing runs are not overwritten.

## Architecture

Inputs: six normalized features in the existing fixed order.
Hidden layer 1: Dense 32, ReLU.
Hidden layer 2: Dense 16, ReLU.
Output: Dense 3, softmax.

Parameters:
6*32 + 32 + 32*16 + 16 + 16*3 + 3 = 803.

Class-weight balancing uses training counts only. No validation rows or purged rows are used in fitting.

## Why training uses an internal scale, and why it is folded away

The required normal-only statistics make fault values potentially hundreds of normal standard deviations from the baseline. To keep optimization numerically well conditioned, the trainer computes one positive max-absolute scale per feature using the 223 training rows only, with a minimum scale of 1.

It trains the MLP on normalized_input / scale. Afterward it divides each input row of the first Dense kernel by that same scale. Algebraically:

normalized_input @ (kernel / scale) = (normalized_input / scale) @ kernel

The exported MLP therefore accepts the original six normalized values directly. It has the same 803 parameters and no extra scaling layer. An equivalence check compares predictions before and after folding. The saved normalization mean/std files are never refitted.

The conditioned training model and its six scaling factors are saved for reproducible future pruning/fine-tuning. The deployed model uses the folded weights.

## Accuracy gate and FP32 conversion

The trainer evaluates the frozen 59-row validation set after training.

- Keras validation accuracy must be strictly greater than 88% before conversion starts.
- If the gate fails, the run's model, history and metrics are retained for diagnosis and conversion is not started.
- If the gate passes, an unquantized FP32 TFLite model is created and evaluated on the same validation rows.
- The FP32 TFLite result must also exceed 88% before baseline_pointer.json is updated.

Critical recall and support are printed separately. The recommendation requirement is strictly greater than 95%. With 17 Critical validation examples, a model must correctly classify all 17 to exceed that threshold. Always report the support count and the dataset's scenario-label limitation.

## Actual three-sigma normalization experiment

The trainer runs the same held-out raw features with:
1. the correct fixed means/stds;
2. means shifted by +3*std, with std unchanged.

It evaluates the trained model in both cases. For a passing FP32 model, the experiment is repeated using the actual TFLite interpreter and written to normalization_experiment.json.

The recorded accuracy difference is shifted accuracy minus correct accuracy, in percentage points. A measured difference of zero is possible and must be reported honestly. No degradation is invented.

The validation-index hash proves that both conditions used the same rows. The dataset/statistics hashes are checked again after training.

## Files saved for each run

- model.keras — ordinary six-input MLP accepting the existing normalized features.
- conditioned_model.keras — equivalent training-coordinate model.
- conditioning_scale.npy — training-only scales for reproducing fine-tuning.
- training_history.json — measured training loss/accuracy history.
- environment.json and pip_freeze.txt — runtime versions and settings.
- metrics.json — actual held-out metrics, confusion matrix, support counts and provenance.
- M1_FP32.tflite — created only after the Keras accuracy gate passes.
- normalization_experiment.json — actual TFLite correct/shifted comparison.
- validation_predictions.csv — true/predicted labels and probabilities for both conditions.

training/models/baseline_pointer.json identifies the latest baseline that passed both accuracy gates. Future conversion tools must verify its dataset and normalization hashes.

This update contains source code and tests, not trained weights or fabricated accuracy results.

## If training fails or accuracy is too low

Stop at the first failed environment/test/training step and share the full error or final metrics. Keep the saved attempt folder and the same dataset/validation indices. Changes to the architecture or training procedure should be documented and evaluated on that same declared split; the split must not be changed to improve a score.

## Dataset label scope

Evaluation uses the D1 scenario mapping: none=Normal, temp_drift=Warning, combined=Critical. The physical temperature mismatch in the temp_drift scenario remains documented in dataset metadata. The live operational temperature safety rule is separate work for the inference/alert service.

## Arithmetic and benchmark scope

The MLP has 803 parameters and about 1504 Dense-layer multiply/add FLOPs per inference, excluding activation overhead. The brief's B2 45-MFLOP/18-MB Roofline calculation is a separate assigned hypothetical case.

This phase measures accuracy and model size. It does not yet measure the required 10-warm-up/200-run mean/p95 latency or energy proxy; those measurements will use one consistent runtime for all three model variants.

## Current requirement checkpoint

D1 dataset preparation: verified by user.
D1 training: execute this package and require >88% held-out accuracy.
C2 normalization experiment: actual model comparison is produced by a passing run.
M1: FP32 baseline is prepared only after the accuracy gate.
M2/M3 INT8/pruning: pending.
D2 inference, local/offline alerts and OTA cache demonstration: pending.
PSI, Ansible, hardware/architecture report, final report/PPT/video: remaining required work.

## Primary references

- https://github.com/tensorflow/model-optimization/blob/master/RELEASE.md
- https://pypi.org/project/tensorflow-cpu/2.14.1/
- https://pypi.org/project/keras/2.14.0/
- https://www.tensorflow.org/lite/models/convert/convert_models


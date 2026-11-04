# LogiEdge — M2 full INT8 post-training quantization

M1 passed on the team's laptop: 100% validation accuracy, 17/17 Critical
examples correct, FP32 model 5416 bytes. The three-sigma shifted-statistics
experiment measured 0.00 percentage points accuracy change.

This incremental update converts the already trained M1 model to M2. It uses
all 223 eligible training windows as representative calibration samples and
evaluates the same 59 validation windows. It does not retrain M1 or refit the
normalization statistics. No new captures are needed.

## Install in Ubuntu

Save LogiEdge_M2_INT8.zip into C:\Users\Nandhini G\Downloads. Keep Docker Desktop
running and run these commands one at a time in Ubuntu:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_M2_INT8.zip" -d .
docker compose run --rm --entrypoint python trainer -m unittest discover -s tests -p 'test_quantization.py' -v
```

Expected: twelve new tests pass with no skips in the trainer image. The three
TensorFlow checks actually convert a small synthetic model, inspect its stored
tensor types and compare its INT8 execution to Keras. The synthetic test data
are isolated test fixtures; actual project calibration uses only saved D1
training rows.

The existing trainer image already has the pinned dependencies. The project
bind mount supplies this source update, so an image rebuild is not required.
The simulator image does not contain TensorFlow; use trainer for these checks.

After those tests pass:

```bash
bash deployment/run_m2.sh
```

Stop at the first failure and share the error. After a successful run, share
the summary with calibration count, integer graph check, accuracy, Critical
recall/support, confusion matrix, model sizes and input clipping.

## Actual conversion

The converter uses Optimize.DEFAULT plus a representative generator containing
every training row. It enables only TFLITE_BUILTINS_INT8 and explicitly sets
both input and output to signed INT8. This disallows floating-point fallback.
Conversion records which unique calibration offsets were actually consumed.

An independent FlatBuffer inspection checks all stored tensor types, both I/O
types/shapes, and the three FullyConnected plus Softmax operations. INT32 biases
and integer shape tensors are permitted. FLOAT32/FLOAT16 tensors, custom ops,
and dequantization/float fallback are rejected. The schema parser is bundled
with the pinned TensorFlow 2.14.1 image.

## Calibration isolation and baseline identity

- Use all training indices, in original order: 223 rows in the current dataset.
- Never use any of the 59 validation or nine purged rows for calibration.
- Save the actual dataset indices, class counts, value ranges and hashes.
- Verify the baseline pointer, dataset hash, statistics hash, validation-index
  hash and saved M1 FP32 model hash.
- Re-evaluate M1 on the same validation set and check its recorded confusion
  matrix. Compare saved Keras and FP32 predictions over training plus validation
  inputs to establish that the Keras source corresponds to the measured M1.
- Hash the dataset, statistics and baseline artifacts before/after conversion.

No validation values are given to the representative generator. Baseline
identity checks use validation only to evaluate already frozen models.

## Correct INT8 input and output handling

The model still logically accepts the same six C2-normalized features. At its
integer boundary, inputs use:

    q = clip(round(x / input_scale + input_zero_point), -128, 127)

Rounding is nearest with ties to even. Clipping happens before the signed INT8
cast to prevent integer wraparound. Dequantization uses:

    probability = (q_output - output_zero_point) * output_scale

Softmax rounding can make the three dequantized values sum slightly above or
below one. Predictions use their unmodified argmax, which equals argmax of the
integer output codes. Stored probabilities are not renormalized. The metric
helper uses one-hot encoding of that argmax only to reuse the common confusion
matrix/count implementation; it does not alter inference probabilities.

## Clipping is measured

Temporal validation tails of a monotonically drifting signal may exceed the
input range calibrated from earlier training windows. The runtime audit reports
the representable input range, counts per feature, rows with clipped inputs and
total clipped values. The validation CSV still records every row's prediction.

Do not add validation data to calibration to hide clipping. Review its actual
effect on accuracy and recall. A passing accuracy result on these captures
does not establish robustness to larger faults or new real-world conditions.

## Outputs

Each run creates a new training/models/ptq_TIMESTAMP_ID folder:

- M2_INT8.tflite — real converted model, not a placeholder.
- metrics.json — held-out accuracy, confusion matrix, Critical recall/support,
  exact bytes/decimal KB, FP32 comparison, clipping and provenance.
- calibration_indices.npy and calibration_manifest.json — actual training-only
  representative rows, hashes and converter consumption.
- integer_audit.json — stored tensor dtypes, scales, zero points and operations.
- validation_predictions.csv — actual FP32 and INT8 predictions plus raw
  dequantized probabilities and integer output codes.
- normalization_experiment.json — optional additional correct/shifted comparison
  using M2 on the same validation rows.
- environment.json — runtime package versions.

If M2 accuracy is strictly greater than 88%, training/models/m2_pointer.json is
updated to this passing run. A failed run's files remain for diagnosis; an older
passing pointer, if present, is preserved. Critical recall is a separate check:
the eventual recommended variant must strictly exceed 95%. With 17 Critical
validation windows that requires 17 correct predictions.

File-size reduction is measured from actual files. A tiny MLP may not obtain a
fourfold total file reduction because metadata and tensor bookkeeping occupy
part of the file. No latency or energy advantage is assumed here.

## Verification and remaining work

The authoring environment lacks TensorFlow and Docker. Nine new numerical/provenance
checks can run there; the three new conversion checks must pass in the team's
existing trainer container before the project conversion runs. Actual M2
accuracy, size and speed are unknown until executed on the laptop.

M3 must use genuine structured removal of hidden units with a PolynomialDecay
schedule followed by full INT8 PTQ. Default scalar-weight sparsity alone does
not meet structured pruning. M3 follows review of the M2 measurements.

After M3, benchmark all variants on one machine/runtime with the required ten
warm-ups and 200 measured calls, then build the Pareto comparison. Offline
inference/alerts, durable sync queue, model-layer OTA evidence, PSI, seven-task
idempotent Ansible deployment, report, slides and recorded demo remain in scope.

## Primary references

- https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant
- https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization
- https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec

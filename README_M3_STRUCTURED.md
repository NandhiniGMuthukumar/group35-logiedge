# LogiEdge — M3 whole-unit structured pruning followed by INT8 PTQ

This incremental update implements the final F1 variant, using the already
verified M1 model and the existing frozen dataset/statistics. M1 and M2 both
achieved 100% accuracy and 17/17 Critical recall on the same 59 validation
windows. M2 is 3552 bytes versus M1's 5416 bytes, a measured 34.42% reduction.
M2's 35 clipped input values remain documented; no validation rows are added to
calibration to hide clipping.

## Run in Ubuntu

Download LogiEdge_M3_Structured.zip into C:\Users\Nandhini G\Downloads. Keep
Docker Desktop running. Execute one command at a time:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_M3_Structured.zip" -d .
docker compose run --rm --entrypoint python trainer -m unittest discover -s tests -p 'test_structured_pruning.py' -v
```

Expected: ten new tests pass with no skips. These test the actual TFMOT wrapper,
scheduled unit masks, post-ReLU zero outputs, dense compaction, save/reload and
full INT8 conversion. The numerical checks also test bias removal, exact
dimensions, deterministic magnitude ties and M2 provenance checks.

After tests pass:

```bash
bash deployment/run_m3.sh
```

The existing trainer image already contains the pinned dependencies. The bind
mount supplies this source update; no image rebuild or new capture is needed.
Training progress prints every 20 epochs. Stop at the first error and share it.
Share the final summary for evaluation before proceeding to benchmarking.

Authoring verification: the complete 65-test regression suite passed with no
skips in an isolated Linux Python 3.11.17 environment using the pinned
TensorFlow 2.14.1 / Keras 2.14.0 / TFMOT 0.8.0 stack. This includes real pruning
and INT8 conversion of synthetic test fixtures. The team's M3 weights and
performance remain to be measured on the laptop.

## What structured pruning means for this MLP

The source model has 32 and 16 hidden units. For an MLP, the removable structure
is one hidden output unit/channel. Its incoming weights, bias contribution and
outgoing weights must be removed together. This implementation uses that
structure; the exported graph is a smaller ordinary dense MLP.

The 35% target requires integer rounding:

| Layer | Original units | Retained | Removed | Actual removal |
| --- | ---: | ---: | ---: | ---: |
| First hidden | 32 | 21 | 11 | 34.375% |
| Second hidden | 16 | 10 | 6 | 37.500% |
| Combined hidden | 48 | 31 | 17 | 35.4167% |

35% of 48 is 16.8 units, so exactly 35% is impossible with whole units. Report
the nominal 35% target and the actual 17/48 removal, rather than claiming an
exact 35.000%. The compact architecture is 6-21-10-3, with 400 parameters versus
803 originally. The 50.19% parameter reduction differs from the percentage of
hidden units removed because adjacent weight matrices shrink together.

## Actual TFMOT PolynomialDecay algorithm

Each hidden layer is a custom PrunableLayer based on Dense. Each output has a
non-trainable score that controls a binary gate after ReLU. A zero score turns
off the entire unit, including any nonzero bias contribution. Dense kernels are
trained normally for active units.

Importance is the pretrained incoming group L2 magnitude, including the bias,
in M1's conditioned training coordinates. Magnitudes are encoded as unique
positive ranks; this preserves the order and breaks exact ties by unit index.
Rank scores are fixed apart from pruning; importance is not recomputed during
fine-tuning. No activation or validation examples are used to rank units.

The custom get_prunable_weights returns only the per-unit score vector.
TFMOT prune_low_magnitude and its UpdatePruningStep callback apply the actual
PolynomialDecay schedule to that vector. Since each scalar controls an entire
hidden output, its mask is a whole-unit mask. Merely pruning 35% of individual
kernel scalars would not produce this smaller dense structure.

Default schedule:

- initial_sparsity=0.0, final_sparsity=0.35, power=3, frequency=1.
- 100 pruning epochs, batch size 16, Adam learning rate 0.0005, seed 2027.
- With 223 training rows there are 14 batches per epoch.
- begin_step=70 and end_step=1120, followed by the remaining 20 pruning epochs.
- Pruning is not stopped early; the final masks must retain exactly 21 and 10
  units and must have reached the schedule end.

The pruning history records actual mask counts and optimizer/pruning steps at
every epoch. The audit saves the raw magnitude ranks, final masks and kept-unit
indices. Wrapped weights preserve the TFMOT mask/step state as additional
evidence; that weights file is not presented as a resumable optimizer checkpoint.

## Physical compaction and equivalence

After strip_pruning removes wrappers, the model still has whole-unit gates.
Compaction physically slices the dense arrays:

1. First kernel columns and bias entries for retained first-layer units.
2. Second kernel rows for retained first-layer units, and columns/bias entries
   for retained second-layer units.
3. Output kernel rows for retained second-layer units; keep all three classes.

Because inactive units produce zero after ReLU, this preserves the gated
network's function. The implementation compares predictions on all training
inputs before and after compaction and refuses a mismatch beyond numerical
tolerance. It checks the 400-parameter compact model and later verifies that
the INT8 file contains Dense weight shapes [21,6], [10,21] and [3,10].

The compact model is then fine-tuned for up to 80 epochs on training rows only.
Checkpoint selection and early stopping monitor training loss only. Validation
rows are not passed to either fit call. Input conditioning uses the original
M1 scales; these are folded back into the first kernel for export, preserving
the six C2-normalized inputs and the frozen normalization files.

## Conversion, calibration and evaluation

The compact Keras model must exceed 88% accuracy on the frozen validation set
before INT8 conversion begins. Full signed INT8 PTQ uses all 223 training rows.
The integer audit checks I/O, tensor types, operations and compacted dimensions.

The final INT8 model is evaluated on the same 59 validation rows. It must also
exceed 88% before m3_pointer.json is updated. Critical recall/support is a
separate check; the eventual recommended model must exceed 95%. For the current
17 Critical windows, this means all 17 correct predictions.

M1 and M2 are re-evaluated on the same split to verify their recorded confusion
matrices. Dataset, statistics, M1 and M2 file hashes must remain unchanged.
Every run gets its own output folder; earlier results are preserved. A failed
run is retained for diagnosis and does not replace an older passing pointer.

## Saved evidence

Output folder: training/models/structured_TIMESTAMP_ID/

- structured_pruning_audit.json: schedule, actual masks, unit indices, widths,
  parameter counts and compaction equivalence.
- pruning_history.json: per-epoch loss, optimizer/pruning steps and kept units.
- wrapped_pruning.weights.h5: actual wrapper masks, scores and pruning steps.
- pruned_gated_model.keras: stripped model retaining binary gates, for audit.
- compacted_before_finetune.keras: ordinary dense model at compaction time.
- conditioned_model.keras and conditioning_scale.npy: reproducible training
  coordinates after compact fine-tuning.
- model.keras: folded compact model accepting the original normalized inputs.
- fine_tuning_history.json and environment.json: training and runtime evidence.
- M3_STRUCTURED_INT8.tflite: actual final model, created after the Keras gate.
- integer_audit.json and calibration_manifest.json/calibration_indices.npy:
  integer-only, compact structure and training-only calibration evidence.
- validation_predictions.csv: actual predictions from M1, M2 and M3.
- variant_accuracy_size.csv: measured accuracy, Critical recall and file sizes.
- metrics.json: complete measured results, source hashes and accuracy gates.

Loading the audit-only custom gated model requires importing
training.structured_pruning and calling gated_dense_class() first. The deployed
M3 TFLite model and ordinary model.keras contain no custom gates or wrappers.

## Requirement checkpoint

F1 M1 and M2 passed on the team's laptop. M3 performance is unknown until this
package is executed. File-size, accuracy and Critical recall will be measured;
latency/energy benefits are not assumed from the unit or parameter counts.

Next: benchmark all three variants with ten warm-ups and 200 measured runs on
the same machine/runtime, then produce Pareto analysis and a deployment
recommendation. Local offline inference/alerts and durable sync, deployment/OTA
evidence, PSI, the seven-task idempotent Ansible playbook, final report, PPT and
four-member video remain required work.

## Primary references

- https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide
- https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PrunableLayer
- https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PolynomialDecay
- https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant

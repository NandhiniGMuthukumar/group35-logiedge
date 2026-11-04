# Ten assessed lifecycle activities

We map the Module 4 lifecycle to ten explicit activities below. Its slide 4 groups Collect, Label and Augment inside Data; we separate collection from labelling and augmentation to make the ten assessed activities explicit, while preserving the remaining course labels [3].

1 Data collection. We capture independent Normal, temperature-drift and combined runs for 20, 15 and 15 minutes. The resulting 291 feature windows retain capture provenance.

2 Labelling and augmentation. Scenario manifests assign the prescribed classes. Seeded noise supplies variation; we do not claim additional augmentation or relabel using model predictions.

3 Train. TensorFlow trains a CPU-based 6-32-16-3 ReLU MLP with 803 parameters. Training-only conditioning is folded into its first kernel without changing the frozen normaliser.

4 Validate. Chronological per-run holdouts and purging separate overlapping raw support: 223 training, 59 validation and nine discarded windows. Validation data never enter calibration.

5 Optimise. We compare FP32, full INT8 and structured-pruned INT8 variants. PolynomialDecay gates entire hidden units before physical compaction and fine-tuning.

6 Convert. TFLite exports preserve the six-input, three-output interface. Integer audits verify signed INT8 input/output and absence of floating-point fallback.

7 Deploy. Docker packages preprocessing, inference and result publication. Ansible copies the selected model and matching reference, then manages the pinned image.

8 Infer. One CPU thread executes the verified model. Local classification and temperature rules operate independently of backend availability.

9 Monitor. The live observer evaluates output-confidence PSI and records drift, recovery and clock continuity. Saved reference and model identities remain fixed.

10 Update. A proposed canary rollout promotes verified model/reference pairs and retains rollback assets. We demonstrated model-layer distribution; autonomous fleet rollout remains proposed.

References: see [project references](../docs/references.md).

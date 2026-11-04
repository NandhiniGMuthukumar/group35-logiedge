[README.md](https://github.com/user-attachments/files/34393280/README.md)

# Group 35 LogiEdge

LogiEdge is a Linux amd64 software prototype for the course cold-chain truck scenario. A local simulator publishes temperature, vibration RMS and door events. Mosquitto, shared preprocessing, a TFLite classifier and physical temperature rules support local detection. SQLite retains events and retries selected alerts/heartbeats to a separate operations service. The pilot and full-fleet hardware figures are projections, not actual installed trucks.

## Team

| Member | Student ID |
|---|---|
| NANDHINI G | 2025AFO5070 |
| GANESAN V | 2025ae05074 |
| VINAYAK SHARMA | 2025ae05018 |
| AHAMED IMTHIAS J | 2025ae05985 |

## Verified results

The dataset has 291 six-feature windows: 223 training, 59 validation and nine purged. M1/M2/M3 sizes are 5416/3552/3216 bytes. All achieved 59/59 synthetic validation accuracy and 17/17 Critical recall. M1 was fastest on the measured laptop; M3 was smallest. See the original dated metrics and explicit limits before interpreting these point results.

The live P(Normal) PSI run 2291a861 recorded 242 actual model outputs, detected drift in 60.00 seconds and recovered to zero after 1079.00 seconds of clean restoration. Its audit reproduced all 24 full-window evaluations. The seven-task Ansible run changed five resources initially and zero on the unchanged repeat; container ID/start time stayed unchanged. A fresh post-deployment capture produced Normal and matching local/remote delivery counts of 72 with none pending.

## Existing verified laptop

Open Docker Desktop and wait for Engine running. In Ubuntu:

```bash
cd ~/logibridge
bash deployment/start_demo_services.sh
bash deployment/run_simulator.sh \
  --anomaly none --duration 45 --seed 8001 --purpose live
docker compose logs --tail 12 inference
python3 demo/show_evidence.py status
```

The startup helper retains the Ansible override and named volumes. Use the guide before stopping services, starting PSI or performing deployment during a recording.

## Reproduction on another Linux amd64 machine

The project uses Docker Compose and Python 3.11 ML containers. Ansible preparation uses native Ubuntu 24.04 Python 3.12 in an isolated environment. Enable Docker Desktop WSL integration when using Windows. Build service images with the supplied Dockerfiles, start MQTT/operations, then start inference. The staged checked model/statistics/registry are included in the working project.

```bash
docker compose build simulator trainer operations inference
docker compose up -d --wait --wait-timeout 120 mqtt operations
docker compose up -d --no-deps --wait --wait-timeout 120 inference
```

The private registry and pushed OTA tags must exist before the Ansible preparation step. `deployment/run_ota_demo.sh` creates the local registry and performs the measured-cache workflow; inspect its output and preserve the first comparison. The original laptop already has this registry and its images. A new clone does not include Docker image caches or named-volume databases. Full data-generation and training instructions are in the phase READMEs. Reproduction requires network access to obtain pinned packages/images. No ARM64/Pi/Hailo runtime result is claimed.

## Reading order

- `demo/35_LogiEdge_Demo_Guide.pdf` gives startup, commands, timing, scripts and viva preparation.
- `demo/35_LogiEdge_Presentation.pptx` is the editable 18-slide deck.
- `reports/35_LogiEdge_Report_Review.docx` is the editable report for team review and final link insertion.
- `docs/requirements_and_evidence.md` maps each task to its implementation/evidence.
- `scenario_architecture/constraint_analysis.md` and `hardware/hardware_justification.md` contain the calculations.
- `data_pipeline/mqtt_architecture.md` explains topics, features and transport policy.
- `deployment/ota_strategy.md` distinguishes the assigned 280 KB cycle from the measured image-layer transfer.

## Measured evidence

The selected benchmark is indexed by `optimisation/results/benchmark_pointer.json`. The original model-layer comparison is indexed by `reports/ota_pointer.json`. The audited PSI experiment is in `reports/psi_live_20261104T044934Z_L9uSpi`. The initial two-run deployment is in `reports/ansible_20261104T062754Z_C85o1J`. Publication figures use those saved measurements; their dated originals remain unchanged. The compact display labels saved measurements explicitly and uses a live Docker query only for its `status` command.

## Interpretation limits

- Validation is synthetic and small. Operational label consistency and field data require further work.
- Maximum-class-confidence PSI stayed zero. The primary monitored score is P(Normal), stated explicitly.
- Source and receiver UTC mappings diverged during clock corrections. Negative historical alert-duration fields are invalid timing evidence. The live PSI audit uses monotonic elapsed intervals.
- Energy is a CPU-utilisation/TDP proxy. File bytes do not equal peak runtime memory.
- Current HTTP transport is a private local demonstration. Authentication, encryption, production rollout security and target profiling remain proposed.

## Submission

Use one private repository with verified instructor access, one 15-20 minute group video and one final PDF. Record actual working actions and label saved evidence. Insert the actual repository and video URLs into the report before exporting `35_LogiEdge_Final.pdf`. Put the video URL in `demo/demo_video_link.txt`. The review report is not the final LMS copy until the group has revised the analysis, inserted the real links and completed the required recording.

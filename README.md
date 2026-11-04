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

The startup helper retains the Ansible override and named volumes. Do not restart services or deploy while a live PSI experiment is active.

## Reproduction on another Linux amd64 machine

The project uses Docker Compose and Python 3.11 ML containers. Ansible preparation uses native Ubuntu 24.04 Python 3.12 in an isolated environment. Enable Docker Desktop WSL integration when using Windows. Build service images with the supplied Dockerfiles, start MQTT/operations, then start inference. The staged checked model/statistics/registry are included in the working project.

```bash
docker compose build simulator trainer operations inference
docker compose up -d --wait --wait-timeout 120 mqtt operations
docker compose up -d --no-deps --wait --wait-timeout 120 inference
```

The private registry and pushed OTA tags must exist before the Ansible preparation step. `deployment/run_ota_demo.sh` creates the local registry and performs the measured-cache workflow; inspect its output and preserve the first comparison. The original laptop already has this registry and its images. A new clone does not include Docker image caches or named-volume databases. Full data-generation and training instructions are in the phase READMEs. Reproduction requires network access to obtain pinned packages/images. No ARM64/Pi/Hailo runtime result is claimed.

## Project documentation

- [Architecture analysis](scenario_architecture/constraint_analysis.md)
- [System architecture](scenario_architecture/system_architecture_diagram.png)
- [MQTT topic diagram](scenario_architecture/mqtt_topic_tree_diagram.png)
- [Hardware justification](hardware/hardware_justification.md)
- [MQTT and feature pipeline](data_pipeline/mqtt_architecture.md)
- [Lifecycle mapping](docs/ten_stage_mapping.md)
- [Deployment recommendation](optimisation/deployment_recommendation.md)
- [OTA strategy](deployment/ota_strategy.md)
- [Presentation](demo/35_LogiEdge_Presentation.pptx)
- [References](docs/references.md)

Technical README files at the repository root describe the individual implementation phases and reproduction steps. Practice guides and editable report drafts are retained outside this repository export.

## Measured evidence

The selected benchmark is indexed by `optimisation/results/benchmark_pointer.json`. The original model-layer comparison is indexed by `reports/ota_pointer.json`. The audited PSI experiment is in `reports/psi_live_20261104T044934Z_L9uSpi`. The initial two-run deployment is in `reports/ansible_20261104T062754Z_C85o1J`. Publication figures use those saved measurements; their dated originals remain unchanged. The compact display labels saved measurements explicitly and uses a live Docker query only for its `status` command.

## Interpretation limits

- Validation is synthetic and small. Operational label consistency and field data require further work.
- Maximum-class-confidence PSI stayed zero. The primary monitored score is P(Normal), stated explicitly.
- Source and receiver UTC mappings diverged during clock corrections. Negative historical alert-duration fields are invalid timing evidence. The live PSI audit uses monotonic elapsed intervals.
- Energy is a CPU-utilisation/TDP proxy. File bytes do not equal peak runtime memory.
- Current HTTP transport is a private local demonstration. Authentication, encryption, production rollout security and target profiling remain proposed.

## Submission artefacts

This repository snapshot contains source code, model artefacts and saved software-pilot evidence from the supplied 4 November 2026 export. A clone does not include Docker images, image caches, named-volume databases or the original physical laptop.

The group adds its reviewed final report as `reports/35_LogiEdge_Final.pdf` and its actual accessible demonstration URL as `demo/demo_video_link.txt` after recording. These two final artefacts are not present in the supplied source archives. The presentation is included under `demo/`.

The final report contains the actual private repository URL and video URL. The instructor must have collaborator access. One member submits the common repository URL, the 15-20 minute video link and the final report PDF for Group 35.

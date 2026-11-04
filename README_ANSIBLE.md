# LogiEdge — E2 seven-task deployment and two-run idempotency

This package prepares the final implementation component. Preparation may run
while the live PSI baseline collects. Actual deployment must wait until the
live PSI experiment is completed and passing; the deployment wrapper enforces
that requirement before invoking the playbook.

## Prepare inside Ubuntu

Download LogiEdge_Ansible.zip into Windows Downloads, then:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_Ansible.zip" -d .
bash deployment/prepare_ansible.sh
```

If prompted, enter the Ubuntu sudo password locally. The bootstrap installs
python3-venv only if needed and creates a separate Ansible controller environment
under the user's local application-data directory. It pins ansible-core 2.20.11
and uses Ubuntu 24.04's Python 3.12; the existing Python 3.11 ML containers are
unaffected.

Preparation verifies the registered model/reference hashes, pins the pushed
local-registry image digest, stages input copies, checks playbook syntax, lists
the seven tasks and runs nine focused tests. Expected: syntax OK, exactly seven
tasks, nine tests OK. No /opt directory is created and no inference service is
stopped/replaced by this preparation command.

## Model choice

The default is the measured CPU-demo selection, M1 FP32, with the existing M1
300-window reference. D2 already demonstrated the separate M1-to-M3 image-layer
update. E2 demonstrates deploying the selected configuration from a private
registry to a new /opt/logibridge installation and rerunning it unchanged.

The deployed inference service loads /ota/model.tflite through MODEL_PATH.
The matching reference is staged at /ota/reference_dist.json and exposed through
REFERENCE_PATH for the separate PSI monitoring client; the seventh task checks
its seal/model identity. The inference worker and PSI monitoring client remain
separate components.

The helper also supports M3 when supplied with a separately generated matching
M3 reference. A reference with a different model/statistics hash is rejected.
Do not change the model or reference during the current live PSI measurement.

## Exactly seven Ansible tasks

| Task | Operation | How an unchanged repeat behaves |
|---|---|---|
| 1 | Create /opt/logibridge | Existing correct directory stays unchanged |
| 2 | Copy model.tflite | Matching bytes/permissions stay unchanged |
| 3 | Copy reference_dist.json | Matching bytes/permissions stay unchanged |
| 4 | Stop inference when an update is needed | A matching running container is left running |
| 5 | Pull the pinned image from localhost:15000 | An unchanged image ID reports no change |
| 6 | Start inference with the desired environment/volumes | Matching container stays running with the same ID/start time |
| 7 | Wait 15 seconds and verify running/healthy/artifact identities | Read-only verification reports no configuration change |

There are no hidden fact-gathering, pre/post, handler or role tasks. The file
and copy modules manage the first three resources. The remaining command
tasks use a tested Python helper to operate the existing Docker CLI and inspect
actual before/after state; no community collection or Docker Python SDK is
required. Stop/pull/start do not use a blanket changed_when:false. Only the
read-only verification task does.

The conditional stop resolves the apparent conflict between "stop the running
container" and "second run changed=0": stop only when new files or a different
desired deployment require a reload. An unchanged deployment must not be
stopped on the repeat run.

## After live PSI completes

First retain and audit the live evidence:

```bash
bash deployment/psi_live_status.sh
bash deployment/finalize_psi_live.sh
```

Proceed when the experiment is completed/passing and the audit passes. Then:

```bash
bash deployment/run_ansible_demo.sh
```

The wrapper calls the same playbook twice, with the same configuration/source
files between runs. It may request the Ubuntu sudo password for each run;
enter it locally, never paste it into chat. The first run's changed count
depends on current machine state. The second must report:

```text
localhost : ok=7 changed=0 unreachable=0 failed=0
```

The script also checks that both runs have zero failures/unreachable hosts and
that the second run preserves the exact container ID and StartedAt timestamp.
This independently detects a hidden restart even if someone incorrectly
suppresses a changed flag.

It waits 15 real seconds in task 7 of both runs. The verification confirms the
actually loaded model hash, mounted reference hash/seal, frozen normalisation,
current model/MQTT health and desired Docker configuration. The resulting
reports are audit outputs, not changes to managed application configuration.

Evidence is saved in a unique reports/ansible_... folder: both complete logs,
both container inspections, both verification outputs, controller package
versions, deployment plan and ansible_results.json. Send the two PLAY RECAP
lines and the final JSON summary for review. Both actual runs must appear in
the team's demo video.

## How deployment preserves the existing project

The target remains the Compose inference service named logibridge-inference.
An override file selects the pinned registry image, adds the read-only /opt
directory mount and sets MODEL_PATH/REFERENCE_PATH. Existing MQTT namespace,
operations URL, health check, non-root user and edge_state named volume remain
part of the base Compose configuration.

The helper uses Compose to recreate/start that service, preserving its labels
so existing runtime_status.sh/logging commands continue to find it. It does
not delete volumes, retrain models or fit normalisation/reference statistics.
Docker Desktop may expose an internal WSL relay in Mounts.Source; checks use
the mount's destination/type/permissions and verify actual mounted file hashes.

For future manual restarts after this deployment, retain its override:

```bash
docker compose -f compose.yaml -f deployment/ansible/runtime.override.json \
  up -d --no-deps --wait inference
```

Running a base-only Compose up can restore the base embedded-model settings;
the ordinary status/log/exec commands still work because service labels are
preserved.

After verifying both unchanged runs, a fresh functional check is:

```bash
bash deployment/run_simulator.sh --anomaly none --duration 45 --seed 8001 --purpose live
bash deployment/runtime_status.sh
```

Expect an actual fresh Normal prediction using the selected variant; verify
the output. Container recreation resets feature warm-up, while the SQLite
alert/outbox volume persists.

## Validation limits

Nine focused tests passed in the authoring environment. The real pinned
Ansible 2.20.11 playbook was also executed twice against a stateful Docker CLI
fixture, with actual filesystem copies and both 15-second waits. That fixture
reported seven OK/five changed on its first run and seven OK/zero changed on
its second, with stable second-run container identity.

Those fixture observations are not the team's Docker/WSL deployment evidence.
Actual native Ubuntu sudo, Docker Desktop Compose, image pulling and the real
two-run result remain pending the team's laptop execution. No laptop changed
count is prefilled or guaranteed.

Primary Ansible documentation checked:
https://docs.ansible.com/projects/ansible/latest/collections/ansible/builtin/command_module.html
https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_error_handling.html
https://docs.ansible.com/projects/ansible/latest/playbook_guide/playbooks_privilege_escalation.html

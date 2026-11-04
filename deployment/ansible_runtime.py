"""Seven-task Ansible support using the laptop's existing Docker CLI.

Mutation results come from observed image/container states. No changed flag is
suppressed for stop/pull/start; verification is explicitly read-only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitoring.psi import canonical, file_sha256, load_reference, require, write_json


class Docker:
    def run(self, *args, missing_ok=False, timeout=180):
        result = subprocess.run(['docker', *args], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, cwd=ROOT)
        if result.returncode:
            absent = re.search(r'No such (?:image|object|container)', result.stderr, re.I)
            if missing_ok and absent:
                return None
            raise RuntimeError('Docker command failed: ' + result.stderr.strip())
        return result.stdout

    def container(self, name):
        text = self.run('inspect', '--type', 'container', name, missing_ok=True)
        return json.loads(text)[0] if text is not None else None

    def image(self, name):
        text = self.run('image', 'inspect', name, missing_ok=True)
        return json.loads(text)[0] if text is not None else None


def ensure_live_finished(root, docker):
    pointer = json.loads((root / 'reports/psi_live_pointer.json').read_text())
    folder = root / pointer['folder']
    require(folder.resolve().is_relative_to((root / 'reports').resolve()), 'Live evidence is outside reports')
    status = json.loads((folder / 'live_status.json').read_text())
    require(status.get('status') == 'completed' and status.get('passed') is True,
            'Complete and verify the live PSI experiment before deploying inference')
    require(not docker.run('ps', '--filter', 'label=logibridge.component=psi-live', '--format', '{{.Names}}').strip(),
            'The live PSI container is still running; wait for it to finish')


def write_if_different(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(data)
        temporary.chmod(0o644)
        temporary.replace(path)


def prepare(args, docker):
    ref = load_reference(args.reference)
    registry = json.loads((ROOT / 'deployment/runtime_assets/runtime_registry.json').read_text())
    candidates = [m for m in registry['models'] if m['variant'] == args.variant]
    require(len(candidates) == 1, 'Selected variant is not uniquely registered')
    model = ROOT / 'deployment/runtime_assets/models' / (args.variant + '.tflite')
    require(file_sha256(model) == candidates[0]['sha256'] == ref['model_sha256'],
            'Selected model and PSI reference must have the same model hash')
    require(ref['model_variant'] == args.variant and
            ref['normalization_stats_sha256'] == registry['normalization_stats_sha256'] and
            ref['normalization_metadata_sha256'] == registry['normalization_metadata_sha256'],
            'Reference/variant/frozen statistics mismatch')
    tag = 'ota-m1' if args.variant == 'M1_FP32' else 'ota-m3'
    repository = 'localhost:15000/logibridge-inference'
    image = docker.image(repository + ':' + tag)
    require(image is not None, 'The OTA registry image is missing; retain the successful D2 images')
    digests = [value for value in (image.get('RepoDigests') or []) if value.startswith(repository + '@sha256:')]
    require(len(set(digests)) == 1, 'Expected one pushed digest for the selected local image')
    image_ref = digests[0]
    source = ROOT / 'deployment/ansible/inputs'
    model_copy, reference_copy = source / 'model.tflite', source / 'reference_dist.json'
    write_if_different(model_copy, model.read_bytes())
    write_if_different(reference_copy, args.reference.read_bytes())
    desired = {'image_reference': image_ref, 'model_sha256': file_sha256(model_copy),
               'reference_sha256': ref['reference_sha256'], 'reference_file_sha256': file_sha256(reference_copy),
               'directory': str(args.directory), 'truck_id': 'TRUCK_001',
               'model_path': '/ota/model.tflite', 'reference_path': '/ota/reference_dist.json'}
    fingerprint = hashlib.sha256(canonical(desired).encode()).hexdigest()
    environment = {'MODEL_PATH': '/ota/model.tflite', 'REFERENCE_PATH': '/ota/reference_dist.json',
                   'TRUCK_ID': 'TRUCK_001', 'MQTT_HOST': 'localhost', 'MQTT_PORT': '1883',
                   'OPERATIONS_URL': 'http://operations:8080/events'}
    override = ROOT / 'deployment/ansible/runtime.override.json'
    override_value = {'services': {'inference': {'image': image_ref, 'environment': environment,
                      'labels': {'logibridge.deployment.sha256': fingerprint},
                      'volumes': [{'type': 'bind', 'source': str(args.directory), 'target': '/ota', 'read_only': True}]}}}
    write_if_different(override, (canonical(override_value) + '\n').encode())
    plan = {'schema_version': 1, 'project_dir': str(ROOT), **desired,
            'model_variant': args.variant, 'deployment_sha256': fingerprint,
            'normalization_stats_sha256': ref['normalization_stats_sha256'],
            'normalization_metadata_sha256': ref['normalization_metadata_sha256'],
            'model_source': str(model_copy), 'reference_source': str(reference_copy),
            'expected_environment': environment, 'container_name': 'logibridge-inference',
            'mqtt_container_name': 'logibridge-mqtt', 'state_volume': 'logibridge_edge_state',
            'compose_files': [str(ROOT / 'compose.yaml'), str(override)]}
    write_if_different(args.plan, (canonical(plan) + '\n').encode())
    variables = {'deployment_directory': str(args.directory), 'deployment_owner_uid': os.getuid(),
                 'deployment_owner_gid': os.getgid(), 'deployment_model_source': str(model_copy),
                 'deployment_reference_source': str(reference_copy),
                 'deployment_helper': str(ROOT / 'deployment/ansible_runtime.py'),
                 'deployment_plan': str(args.plan.resolve()), 'ansible_python_interpreter': sys.executable,
                 'deployment_require_sudo': True}
    write_json(ROOT / 'deployment/ansible/variables.json', variables)
    print(json.dumps({'prepared': True, 'variant': args.variant, 'model_sha256': plan['model_sha256'],
                      'reference_sha256': plan['reference_sha256'], 'image_reference': image_ref,
                      'tasks': 7, 'deployment_started': False}, indent=2))
    return plan


def load_plan(path):
    plan = json.loads(path.read_text())
    require(plan['schema_version'] == 1, 'Unsupported deployment plan')
    require(file_sha256(plan['model_source']) == plan['model_sha256'], 'Prepared model source changed')
    require(file_sha256(plan['reference_source']) == plan['reference_file_sha256'], 'Prepared reference source changed')
    return plan


def owned(container):
    labels = container['Config'].get('Labels') or {}
    require(labels.get('com.docker.compose.project') == 'logibridge' and
            labels.get('com.docker.compose.service') == 'inference',
            'The inference container name is occupied by an unrelated service')


def matches(plan, container, docker):
    if container is None:
        return False
    owned(container)
    config, host = container['Config'], container['HostConfig']
    labels = config.get('Labels') or {}
    environment = dict(value.split('=', 1) for value in config.get('Env', []) if '=' in value)
    mqtt, image = docker.container(plan['mqtt_container_name']), docker.image(plan['image_reference'])
    if mqtt is None or image is None:
        return False
    mounts = {item['Destination']: item for item in container.get('Mounts', [])}
    ota, state = mounts.get('/ota', {}), mounts.get('/state', {})
    # Docker Desktop can expose an internal WSL relay as Mounts.Source. Check
    # destination/type/permissions here and verify actual mounted bytes later.
    return bool(labels.get('logibridge.deployment.sha256') == plan['deployment_sha256'] and
                container['Image'] == image['Id'] and
                all(environment.get(key) == value for key, value in plan['expected_environment'].items()) and
                host.get('NetworkMode') in ('container:' + mqtt['Id'], 'container:' + plan['mqtt_container_name']) and
                host.get('ReadonlyRootfs') is True and
                ota.get('Type') == 'bind' and ota.get('RW') is False and
                state.get('Name') == plan['state_volume'] and state.get('RW') is True)


def deployment_action(action, plan, docker, assets_changed=False):
    root = Path(plan['project_dir'])
    ensure_live_finished(root, docker)
    container = docker.container(plan['container_name'])
    if container is not None:
        owned(container)
    if action == 'stop':
        needs_update = assets_changed or not matches(plan, container, docker)
        running = bool(container and container['State']['Running'])
        if running and needs_update:
            docker.run('stop', '--time', '15', plan['container_name'])
            require(not docker.container(plan['container_name'])['State']['Running'], 'Inference did not stop')
        return {'changed': running and needs_update, 'needed_update': needs_update}
    if action == 'pull':
        before = docker.image(plan['image_reference'])
        docker.run('pull', plan['image_reference'])
        after = docker.image(plan['image_reference'])
        require(after is not None, 'Pinned image was not available after pull')
        return {'changed': before is None or before['Id'] != after['Id'], 'image_id': after['Id']}
    require(file_sha256(root / plan['directory'] / 'model.tflite') == plan['model_sha256'], 'Deployed model bytes differ')
    require(file_sha256(root / plan['directory'] / 'reference_dist.json') == plan['reference_file_sha256'], 'Deployed reference bytes differ')
    if action == 'start':
        running = bool(container and container['State']['Running'])
        if running and matches(plan, container, docker):
            return {'changed': False, 'container_id': container['Id'], 'reason': 'Already running with the desired configuration'}
        arguments = ['compose', '--project-directory', str(root)]
        for file in plan['compose_files']:
            arguments.extend(['-f', file])
        arguments.extend(['up', '-d', '--no-deps', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '120', 'inference'])
        docker.run(*arguments)
        after = docker.container(plan['container_name'])
        require(after and after['State']['Running'] and matches(plan, after, docker), 'Started container differs from the deployment plan')
        return {'changed': not running or container is None or container['Id'] != after['Id'], 'container_id': after['Id']}
    require(action == 'verify', 'Unknown deployment action')
    time.sleep(15)
    after = docker.container(plan['container_name'])
    require(after and after['State']['Running'] and after['State'].get('Health', {}).get('Status') == 'healthy',
            'Inference is not running and healthy after 15 seconds')
    require(matches(plan, after, docker), 'Running container configuration differs from plan')
    script = '''import hashlib,json,os,time
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
h=json.loads(Path('/state/health.json').read_text())
r=json.loads(Path(os.environ['REFERENCE_PATH']).read_text())
unsigned={k:v for k,v in r.items() if k!='reference_sha256'}
seal=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
print(json.dumps({'model_sha256':sha(os.environ['MODEL_PATH']),'loaded_model_sha256':h['model_sha256'],
'normalization_stats_sha256':sha(os.environ['STATS_PATH']),
'normalization_metadata_sha256':sha(Path(os.environ['STATS_PATH']).with_suffix('.meta.json')),
'reference_sha256':seal,'reference_seal_valid':seal==r['reference_sha256'],
'reference_model_sha256':r['model_sha256'],'reference_samples':r['sample_count'],
'model_ready':h['model_ready'],'mqtt_connected':h['mqtt_connected'],'health_age':time.time()-h['updated_unix']}))'''
    identity = json.loads(docker.run('exec', plan['container_name'], 'python', '-c', script))
    require(identity['model_sha256'] == identity['loaded_model_sha256'] == plan['model_sha256'], 'Loaded model verification failed')
    require(identity['reference_seal_valid'] and identity['reference_sha256'] == plan['reference_sha256'] and
            identity['reference_model_sha256'] == plan['model_sha256'] and identity['reference_samples'] == 300,
            'Mounted reference verification failed')
    require(identity['normalization_stats_sha256'] == plan['normalization_stats_sha256'] and
            identity['normalization_metadata_sha256'] == plan['normalization_metadata_sha256'], 'Runtime statistics verification failed')
    require(identity['model_ready'] and identity['mqtt_connected'] and 0 <= identity['health_age'] < 15, 'Runtime health verification failed')
    result = {'changed': False, 'waited_seconds': 15, 'verified': True, 'container_id': after['Id'], **identity}
    write_json(root / 'reports/ansible_last_verification.json', result)
    return result


def summarize(first, second, first_container, second_container, output):
    def recap(path):
        candidates = [line for line in path.read_text().splitlines() if re.match(r'^localhost\s*:', line)]
        require(len(candidates) == 1, 'Expected one localhost PLAY RECAP')
        value = {key: int(number) for key, number in re.findall(r'([a-z_]+)=(\d+)', candidates[0])}
        require(value.get('failed') == 0 and value.get('unreachable') == 0, 'Ansible reported failed/unreachable tasks')
        return value
    a, b = recap(first), recap(second)
    one, two = json.loads(first_container.read_text())[0], json.loads(second_container.read_text())[0]
    require(b['changed'] == 0 and b['ok'] == 7, 'Second run must show all seven tasks OK and changed=0')
    require(one['Id'] == two['Id'] and one['State']['StartedAt'] == two['State']['StartedAt'],
            'The second run recreated or restarted inference')
    result = {'first_run': a, 'second_run': b, 'exact_task_count': 7,
              'second_run_changed_zero': True, 'container_id_unchanged_between_runs': True,
              'container_started_at_unchanged_between_runs': True, 'container_id': two['Id']}
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--variant', choices=('M1_FP32', 'M3_STRUCTURED_INT8'), default='M1_FP32')
    prepare_parser.add_argument('--reference', type=Path, default=ROOT/'monitoring/reference_dist.json')
    prepare_parser.add_argument('--directory', type=Path, default=Path('/opt/logibridge'))
    prepare_parser.add_argument('--plan', type=Path, default=ROOT/'deployment/ansible/plan.json')
    for name in ('guard', 'stop', 'pull', 'start', 'verify'):
        command = sub.add_parser(name)
        command.add_argument('--plan', type=Path, default=ROOT/'deployment/ansible/plan.json')
        if name == 'stop':
            command.add_argument('--assets-changed', choices=('yes', 'no'), default='no')
    summary = sub.add_parser('summarize')
    for name in ('first', 'second', 'first-container', 'second-container', 'output'):
        summary.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    docker = Docker()
    if args.action == 'prepare':
        prepare(args, docker)
    elif args.action == 'summarize':
        summarize(args.first, args.second, args.first_container, args.second_container, args.output)
    elif args.action == 'guard':
        load_plan(args.plan)
        ensure_live_finished(ROOT, docker)
        print('The completed live PSI measurement permits deployment')
    else:
        result = deployment_action(args.action, load_plan(args.plan), docker, getattr(args, 'assets_changed', 'no') == 'yes')
        print(canonical(result))


if __name__ == '__main__':
    main()

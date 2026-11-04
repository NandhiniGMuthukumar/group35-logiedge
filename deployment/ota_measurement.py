"""Measure a model-only image update against a private Registry V2 endpoint.

The registry's descriptor sizes are distribution bytes. Docker image .Size is
recorded separately and is never used as a bandwidth measurement.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import time
import urllib.request


M1 = "M1_FP32"
M3 = "M3_STRUCTURED_INT8"
ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def input_snapshot(root):
    names = ["inference/Dockerfile", "data_pipeline/__init__.py",
             "data_pipeline/preprocessing.py", "data_pipeline/normalization.py",
             "deployment/runtime_assets/training_stats.npy",
             "deployment/runtime_assets/training_stats.meta.json",
             "deployment/runtime_assets/runtime_registry.json"]
    if (root / ".dockerignore").is_file():
        names.append(".dockerignore")
    for folder in (root / "inference", root / "deployment/runtime_assets/models"):
        names.extend(str(p.relative_to(root)) for p in folder.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    result = {}
    for name in sorted(set(names)):
        path = root / name
        require(path.is_file(), "Missing required input: " + name)
        result[name] = {"sha256": sha256(path.read_bytes()),
                        "mode": stat.S_IMODE(path.stat().st_mode)}
    return result


def preflight(root):
    dockerfile = root / "inference/Dockerfile"
    instructions = [line.strip() for line in dockerfile.read_text().replace("\\\n", " ").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    require(instructions[0] == "FROM python:3.11-slim", "Unexpected inference base image")
    require(instructions[-1].startswith("COPY ") and
            "deployment/runtime_assets/model.tflite" in instructions[-1],
            "The staged model must be the final Dockerfile instruction")
    require(any("pip install" in line for line in instructions[:-1]), "Missing earlier pip installation")
    registry = read_json(root / "deployment/runtime_assets/runtime_registry.json")
    models = {}
    for variant in (M1, M3):
        entries = [entry for entry in registry["models"] if entry["variant"] == variant]
        require(len(entries) == 1, "Missing/ambiguous registered model: " + variant)
        path = root / "deployment/runtime_assets/models" / (variant + ".tflite")
        content = path.read_bytes()
        require(sha256(content) == entries[0]["sha256"] and len(content) == entries[0]["bytes"],
                "Staged model does not match its immutable registry: " + variant)
        models[variant] = {"sha256": sha256(content), "bytes": len(content)}
    require(models[M1]["sha256"] != models[M3]["sha256"], "Demo models must differ")
    original = (root / "deployment/runtime_assets/model.tflite").read_bytes()
    return {"schema_version": 1, "models": models, "protected_inputs": input_snapshot(root),
            "original_staged_model_sha256": sha256(original)}


def check_inputs(root, before):
    require(input_snapshot(root) == before["protected_inputs"],
            "A protected input changed. Inspect the files before claiming a model-only update")


class Registry:
    def __init__(self, url, repository="logibridge-inference"):
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        require(parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost"),
                "Use the loopback endpoint inside the private registry container namespace")
        require(re.fullmatch(r"[a-z0-9][a-z0-9/_-]*", repository), "Invalid repository")
        self.base = url.rstrip("/")
        self.repository = repository
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.head_cache = {}

    def request(self, path, method="GET"):
        request = urllib.request.Request(self.base + path, method=method, headers={"Accept": ACCEPT})
        with self.opener.open(request, timeout=5) as response:
            body = b"" if method == "HEAD" else response.read(4 * 1024 * 1024 + 1)
            require(len(body) <= 4 * 1024 * 1024, "Registry metadata exceeds the size limit")
            return body, response.headers

    @staticmethod
    def descriptor(value):
        digest, size = value["digest"], value["size"]
        require(isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest),
                "Unsupported content digest")
        require(type(size) is int and size > 0, "Invalid registry descriptor size")
        return digest, size

    def head_blob(self, descriptor):
        digest, size = self.descriptor(descriptor)
        if digest not in self.head_cache:
            _, headers = self.request(f"/v2/{self.repository}/blobs/{digest}", "HEAD")
            self.head_cache[digest] = int(headers.get("Content-Length", "-1"))
        require(self.head_cache[digest] == size, "Blob Content-Length differs from its manifest")

    def image(self, tag, inspect):
        """Resolve one platform, including the actual index/manifest/config bytes."""
        objects, manifests = {}, []

        def add(digest, size, kind):
            require(digest not in objects or objects[digest]["bytes"] == size, "Inconsistent digest size")
            objects[digest] = {"bytes": size, "kind": kind}

        def manifest(reference, expected=None):
            body, headers = self.request(f"/v2/{self.repository}/manifests/{reference}")
            digest = "sha256:" + sha256(body)
            advertised = headers.get("Docker-Content-Digest")
            require(not advertised or advertised == digest, "Manifest digest mismatch")
            if expected:
                require(self.descriptor(expected) == (digest, len(body)), "Index child descriptor mismatch")
            add(digest, len(body), "manifest")
            value = json.loads(body)
            require(value.get("schemaVersion") == 2, "Registry schema v2 is required")
            manifests.append({"digest": digest, "content": value})
            return value

        value = manifest(tag)
        if "manifests" in value:
            platform = {"os": inspect["Os"], "architecture": inspect["Architecture"]}
            matches = [entry for entry in value["manifests"]
                       if all(entry.get("platform", {}).get(key) == item for key, item in platform.items())]
            require(len(matches) == 1, "Expected exactly one matching runtime platform")
            value = manifest(matches[0]["digest"], matches[0])
        require("layers" in value and "config" in value, "Expected an image manifest")
        config_descriptor = value["config"]
        self.head_blob(config_descriptor)
        config_digest, config_size = self.descriptor(config_descriptor)
        config_bytes, _ = self.request(f"/v2/{self.repository}/blobs/{config_digest}")
        require(len(config_bytes) == config_size and "sha256:" + sha256(config_bytes) == config_digest,
                "Image configuration digest/size mismatch")
        config = json.loads(config_bytes)
        require(config["rootfs"]["diff_ids"] == inspect["RootFS"]["Layers"],
                "Registry image does not match the locally inspected image")
        require(config["os"] == inspect["Os"] and config["architecture"] == inspect["Architecture"],
                "Registry platform differs from the local image")
        add(config_digest, config_size, "config")
        layers = []
        for descriptor in value["layers"]:
            self.head_blob(descriptor)
            digest, size = self.descriptor(descriptor)
            media = descriptor.get("mediaType", "")
            require(media.endswith((".gzip", "+gzip", "+zstd")), "Expected a compressed registry layer")
            layers.append({"digest": digest, "bytes": size, "media_type": media})
            add(digest, size, "layer")
        require(len(layers) == len(inspect["RootFS"]["Layers"]), "Registry/rootfs layer count mismatch")
        return {"objects": objects, "layers": layers, "manifests": manifests,
                "config_digest": config_digest}


def load_inspect(path):
    values = read_json(path)
    require(isinstance(values, list) and len(values) == 1, "Expected one docker image inspect result")
    return values[0]


def check_layers(before, after):
    a, b = before["RootFS"]["Layers"], after["RootFS"]["Layers"]
    require(len(a) == len(b) and len(a) > 1, "Image layer counts differ or are insufficient")
    changed = [index for index, pair in enumerate(zip(a, b)) if pair[0] != pair[1]]
    require(changed == [len(a) - 1], "Expected only the final model filesystem layer to change")
    return len(a)


def cache_steps(text):
    steps = {}
    for line in text.splitlines():
        match = re.match(r"^#(\d+) \[[^\]]+\] ((?:RUN|COPY|WORKDIR)\b.*)$", line)
        if match:
            steps[match[1]] = {"instruction": match[2], "cached": False}
        cached = re.match(r"^#(\d+) CACHED\b", line)
        if cached and cached[1] in steps:
            steps[cached[1]]["cached"] = True
    models = [value for value in steps.values() if "deployment/runtime_assets/model.tflite" in value["instruction"]]
    require(len(models) == 1, "Could not identify the final model COPY in the plain BuildKit log")
    others = [value for value in steps.values() if value is not models[0]]
    require(len(others) >= 5 and all(value["cached"] for value in others),
            "Not all earlier build steps were cached. Inspect the updated build log")
    return {"all_non_model_build_steps_cached": True,
            "model_copy_cached_from_an_earlier_build": models[0]["cached"],
            "steps": list(steps.values())}


def transfer_budget(before, after, trucks=85, price_per_mb=0.10):
    require(type(trucks) is int and trucks > 0, "Invalid fleet size")
    require(price_per_mb >= 0, "Invalid price")
    a, b = before["layers"], after["layers"]
    require(len(a) == len(b) and a[:-1] == b[:-1] and a[-1]["digest"] != b[-1]["digest"],
            "Compressed registry layers must also differ only at the final model layer")
    old, new = before["objects"], after["objects"]
    cold = sum(value["bytes"] for value in new.values())
    changed = {digest: value for digest, value in new.items() if digest not in old}
    delta = sum(value["bytes"] for value in changed.values())
    saved = cold - delta
    return {"cold_target_image_payload_bytes_per_truck": cold,
            "cached_update_payload_bytes_per_truck": delta,
            "bandwidth_saved_bytes_per_truck": saved,
            "bandwidth_saved_percent": 100.0 * saved / cold,
            "new_model_compressed_layer_bytes": b[-1]["bytes"],
            "new_config_and_manifest_bytes": sum(value["bytes"] for value in changed.values() if value["kind"] != "layer"),
            "new_objects": changed, "pilot_trucks": trucks, "rupees_per_decimal_mb": price_per_mb,
            "fleet_cold_payload_mb": trucks * cold / 1e6,
            "fleet_cached_update_payload_mb": trucks * delta / 1e6,
            "fleet_saved_payload_mb": trucks * saved / 1e6,
            "fleet_cold_cost_rupees": trucks * cold / 1e6 * price_per_mb,
            "fleet_cached_update_cost_rupees": trucks * delta / 1e6 * price_per_mb,
            "fleet_cost_saved_rupees": trucks * saved / 1e6 * price_per_mb}


def analyze(root, folder, url):
    before = read_json(folder / "preflight.json")
    check_inputs(root, before)
    require(sha256((root / "deployment/runtime_assets/model.tflite").read_bytes()) ==
            before["original_staged_model_sha256"], "Original staged model was not restored")
    old = load_inspect(folder / "m1_image_inspect.json")
    new = load_inspect(folder / "m3_image_inspect.json")
    count = check_layers(old, new)
    for filename, variant in (("m1_embedded.sha256", M1), ("m3_embedded.sha256", M3)):
        require((folder / filename).read_text().strip() == before["models"][variant]["sha256"],
                "An image embeds the wrong model: " + variant)
    switch = read_json(folder / "model_path_switch.json")
    require(switch.get("variant") == M3 and switch.get("sha256") == before["models"][M3]["sha256"] and
            switch.get("input_dtype") == "int8" and switch.get("output_dtype") == "int8",
            "MODEL_PATH switch did not load the registered INT8 runtime model")
    cache = cache_steps((folder / "build_m3.log").read_text())
    registry = Registry(url)
    old_graph = registry.image("ota-m1", old)
    new_graph = registry.image("ota-m3", new)
    budget = transfer_budget(old_graph, new_graph)
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "filesystem_layers_per_image": count, "changed_filesystem_layers": 1,
              "changed_layer_index_zero_based": count - 1, "all_earlier_filesystem_layers_identical": True,
              "cache_evidence": cache, "model_path_switch_without_rebuild": switch,
              "models": before["models"], "original_staged_model_restored": True,
              "protected_inputs_unchanged": True,
              "local_image_size_bytes_not_transfer_bytes": {M1: old.get("Size"), M3: new.get("Size")},
              "registry_content": {M1: old_graph, M3: new_graph}, "transfer_budget": budget,
              "assumptions": [
                  "One Linux platform; provenance/SBOM attachments disabled for this controlled demonstration.",
                  "Cold case: no target-image objects cached; warm case: all previous M1 image objects retained.",
                  "Both cases count unique compressed layers plus configuration and manifest/index payloads.",
                  "HTTP headers, registry handshakes, authentication, TLS, retries and cellular billing overhead excluded.",
                  "85-truck results are arithmetic projections from this laptop's measured registry objects.",
                  "The registry is a loopback-only local demonstration; this is not an authenticated production OTA service."]}
    write_json(folder / "ota_results.json", result)
    write_json(root / "reports/ota_pointer.json", {"results": str((folder / "ota_results.json").relative_to(root))})
    print("Changed filesystem layers: 1 (final model COPY)")
    print("All earlier build steps cached: True")
    print("Model COPY reused from an earlier run:", cache["model_copy_cached_from_an_earlier_build"])
    print("MODEL_PATH switched M1 image to registered M3 INT8 without rebuilding: True")
    print("Original staged model restored; protected inputs unchanged: True")
    print("Compressed model layer bytes:", budget["new_model_compressed_layer_bytes"])
    print("Cold image / cached update payload bytes per truck:",
          budget["cold_target_image_payload_bytes_per_truck"], budget["cached_update_payload_bytes_per_truck"])
    print("Bandwidth saving: {:.6f}%".format(budget["bandwidth_saved_percent"]))
    print("85-truck cold / cached update MB: {:.6f} / {:.6f}".format(
        budget["fleet_cold_payload_mb"], budget["fleet_cached_update_payload_mb"]))
    print("85-truck bandwidth saved MB: {:.6f}".format(budget["fleet_saved_payload_mb"]))
    print("85-truck cost saved at Rs 0.10/MB: {:.6f}".format(budget["fleet_cost_saved_rupees"]))
    print("Payload accounting excludes protocol/cellular overhead; fleet totals are projections.")
    print("Results:", (folder / "ota_results.json").relative_to(root))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("preflight")
    first.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check-inputs")
    check.add_argument("--before", type=Path, required=True)
    wait = commands.add_parser("wait-registry")
    wait.add_argument("--url", default="http://127.0.0.1:5000")
    wait.add_argument("--timeout", type=float, default=45)
    measure = commands.add_parser("analyze")
    measure.add_argument("--folder", type=Path, required=True)
    measure.add_argument("--url", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "preflight":
        value = preflight(root)
        write_json(args.output, value)
        print("Registered models checked:", M1, M3)
        print("Model file bytes:", value["models"][M1]["bytes"], value["models"][M3]["bytes"])
        print("Protected runtime inputs recorded:", len(value["protected_inputs"]))
    elif args.command == "check-inputs":
        check_inputs(root, read_json(args.before))
        print("Protected runtime inputs unchanged: True")
    elif args.command == "wait-registry":
        client = Registry(args.url)
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                client.request("/v2/")
                print("Private Registry V2 endpoint ready")
                break
            except (OSError, ValueError) as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Local registry did not become ready") from error
                time.sleep(1)
    else:
        analyze(root, args.folder.resolve(), args.url)


if __name__ == "__main__":
    main()

# LogiEdge — D2 model-layer cache and measured OTA payload

This incremental package adds a measurement tool, its tests, and a checkpoint
record. Extract it in the existing `~/logibridge` project. It uses your staged
M1 and M3 models and existing inference Dockerfile. No new model training is
required.

## Why this demonstration matters

An inference image contains Linux, Python, TensorFlow, application code,
normalisation statistics, and a model. Retransmitting all of those files for a
small model update wastes a truck's cellular allowance. Docker identifies
unchanged layers by content. Placing the model last lets an updated image reuse
the earlier layers.

The comparison is deliberately controlled: build M1, replace only the staged
`model.tflite` with the registered M3 model, and build again. Only the final
filesystem layer should change. The new image also needs new configuration and
manifest metadata; the bandwidth calculation includes those payloads.

## Run on the Windows laptop, inside Ubuntu

Save `LogiEdge_OTA_Layer_Cache.zip` in Windows Downloads. Keep Docker Desktop
running, and execute one command at a time:

```bash
cd ~/logibridge
unzip -o "/mnt/c/Users/Nandhini G/Downloads/LogiEdge_OTA_Layer_Cache.zip" -d .
```

Run the focused tests using the Python image already available on your laptop:

```bash
docker run --rm --network none \
  --mount "type=bind,source=$PWD,target=/workspace" \
  --workdir /workspace -e PYTHONDONTWRITEBYTECODE=1 \
  python:3.11-slim \
  python -m unittest discover -s tests -p 'test_ota_measurement.py' -v
```

Expected: 11 tests pass. The tests use synthetic registry fixtures; they do not
create project performance results. If a test fails, stop and share the output.

Run the actual Docker demonstration:

```bash
bash deployment/run_ota_demo.sh
```

Allow the first run to finish. If the earlier build cache is available, the M1
build should reuse it. If it is missing, rebuilding the runtime may take longer.
The controlled second build should cache every earlier RUN/COPY/WORKDIR step.

The script:

1. Verifies registered M1/M3 hashes and records protected runtime inputs.
2. Starts a private Registry V2 server using `registry:3`, bound only to
   `127.0.0.1:15000`, with a separate named data volume.
3. Builds `localhost:15000/logibridge-inference:ota-m1`.
4. Changes only the staged model and builds the `ota-m3` tag.
5. Restores the exact original staged model, including on a build failure.
6. Loads registered M3 INT8 through `MODEL_PATH` inside the existing M1 image,
   without rebuilding that image.
7. Pushes both demonstration tags to the local registry and measures their
   content-addressed distribution objects.

The active MQTT, inference and operations containers keep their current images
and durable state. Demonstration image tags are separate from the Compose
inference tag. Keep port 15000 free for the local registry.

## What success looks like

The summary reports:

- exactly one changed filesystem layer, at the final model COPY;
- all earlier build steps cached;
- M3 selected through MODEL_PATH in the already-built M1 image;
- original staged model restored and protected inputs unchanged;
- actual compressed model-layer bytes, complete cached-update payload, and
  bandwidth/cost projections for 85 trucks.

Actual byte counts are not prefilled. Send the printed summary and the last
part of `build_m3.log` for verification. The unique evidence folder is printed
at startup; its files include `ota_results.json`, both build/push logs, image
inspection results, and `model_path_switch.json`. The latest successful result
is identified by `reports/ota_pointer.json`.

On a repeat run, the M3 model layer may itself be cached from the first run.
The summary explicitly reports this. Preserve the first successful build log
for the video; do not describe a repeat cache hit as a fresh rebuild.

## Bandwidth calculation

Use decimal MB (1 MB = 1,000,000 bytes) and the assignment's Rs 0.10/MB tariff.

- Cold transfer: sum of unique compressed layer, configuration and
  manifest/index payload sizes for the target M3 image.
- Cached update: target-image objects whose content digests are absent from
  the previous M1 image.
- Saving per truck: cold payload minus cached-update payload.
- Saving for 85 trucks: per-truck saved bytes x 85 / 1,000,000 MB.
- Avoided cost: saved fleet MB x Rs 0.10.

The registry's declared blob sizes are checked against HTTP HEAD Content-Length.
Configuration/manifest contents are checked against their digests. Registry
configuration rootfs identifiers must match the locally inspected images.
`docker image inspect .Size` is recorded separately; it is not used as network
transfer size.

These are content-payload projections for trucks retaining the complete old
image cache. They exclude HTTP headers, authentication/TLS, retries, and
cellular billing overhead. No claim is made that 85 physical trucks were
updated. E3's assigned hypothetical 280 KB model/6-week update calculation is
a separate scenario from the measured 5416-byte and 3216-byte project models.

## Evidence to show in the video

Show the final-model COPY position in `inference/Dockerfile`, the second build's
CACHED lines, its final model COPY, the one-layer comparison, the MODEL_PATH
selection result, and the measured fleet saving. Explain the cold/warm cache
assumption. This addresses D2; PSI monitoring and the exactly-seven-task
Ansible deployment remain separate next steps.

## Sources checked

- Docker build-cache invalidation:
  https://docs.docker.com/build/cache/invalidation/
- Official Distribution registry deployment:
  https://distribution.github.io/distribution/about/deploying/
- Registry V2 image manifest/layer descriptors:
  https://distribution.github.io/distribution/spec/manifest-v2-2/
- OCI Distribution specification:
  https://github.com/opencontainers/distribution-spec/blob/main/spec.md

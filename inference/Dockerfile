FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2 \
    TF_ENABLE_ONEDNN_OPTS=0 KERAS_HOME=/tmp/logibridge_keras \
    MODEL_PATH=/app/model.tflite STATS_PATH=/app/data_pipeline/training_stats.npy \
    MODEL_REGISTRY=/app/runtime_registry.json EDGE_DB=/state/edge.sqlite
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY inference/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && python -m pip check
RUN mkdir -p /state && chown 10001:10001 /state
COPY --chown=10001:10001 data_pipeline/__init__.py data_pipeline/preprocessing.py data_pipeline/normalization.py /app/data_pipeline/
COPY --chown=10001:10001 inference/ /app/inference/
COPY --chown=10001:10001 deployment/runtime_assets/training_stats.npy deployment/runtime_assets/training_stats.meta.json /app/data_pipeline/
COPY --chown=10001:10001 deployment/runtime_assets/runtime_registry.json /app/runtime_registry.json
USER 10001:10001
ENTRYPOINT ["python", "-m", "inference.service"]
# Keep the model as the last filesystem layer for the later OTA cache demo.
COPY --chown=10001:10001 deployment/runtime_assets/model.tflite .

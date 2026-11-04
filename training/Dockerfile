FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=2 TF_ENABLE_ONEDNN_OPTS=0 \
    KERAS_HOME=/tmp/logibridge_keras MPLCONFIGDIR=/tmp/logibridge_matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY training/requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-ml.txt \
    && python -m pip check
COPY data_pipeline/ /workspace/data_pipeline/
COPY training/ /workspace/training/
ENTRYPOINT ["python", "-m", "training.train_model"]

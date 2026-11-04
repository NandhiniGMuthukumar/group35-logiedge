FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
COPY data_pipeline/requirements-simulator.txt /tmp/requirements-simulator.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-simulator.txt
COPY data_pipeline/ /workspace/data_pipeline/
ENTRYPOINT ["python", "-m", "data_pipeline.simulator"]

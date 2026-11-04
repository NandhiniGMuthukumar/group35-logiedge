#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec docker compose run --rm --user "$(id -u):$(id -g)" \
  --entrypoint python simulator -m data_pipeline.check_normalizer "$@"

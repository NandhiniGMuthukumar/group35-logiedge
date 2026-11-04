#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
    --entrypoint python trainer -m monitoring.export_live_results

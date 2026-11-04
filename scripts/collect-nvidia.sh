#!/usr/bin/env bash
set -euo pipefail
# Local telemetry only; stop with Ctrl-C. No UUIDs, serials or credentials collected.
target="${1:-runs/nvidia-telemetry.csv}"
mkdir -p "$(dirname "$target")"
exec nvidia-smi --query-gpu=timestamp,name,driver_version,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv --loop=1 > "$target"

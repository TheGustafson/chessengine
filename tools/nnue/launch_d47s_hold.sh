#!/usr/bin/env bash
# Launch the D47S floor-hold leg (resume sb240 -> 280 at constant floor LR).
# Run when the base run's unit exits cleanly with superbatch 240 done.
set -euo pipefail
cd "$(dirname "$0")/../.."
grep -q "superbatch 240 done" /tmp/d47s_v15_train.log || { echo "base run not complete"; exit 1; }
systemd-run --user --unit=archon-d47s-hold --working-directory="$PWD/jax" \
  -p MemoryHigh=5G -p MemoryMax=7G -p MemorySwapMax=0 --collect \
  -E PYTHONUNBUFFERED=1 -E ARCHON_JAX_ALLOW_GPU=1 -E ARCHON_JAX_FT_IMPL=pallas \
  /bin/sh -c 'exec uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 python tools/pump_train.py ../tools/nnue/runs/v15_hold.json 240 >> /tmp/d47s_v15_train.log 2>&1'
echo "hold leg launched (sb241-280 at floor LR); same log, same tb shim."

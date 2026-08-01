#!/usr/bin/env bash
# D49 continuation on top of the cycle-3 checkpoint (owner, 2026-08-01).
#   warm from sb1630, LR 1.2492e-5 (5x the base decay schedule's value there),
#   decaying 0.996/sb for 500 sb -> sb2130, ending 1.6839e-6.
# Self-restarting: relaunches on any death, resuming from the newest checkpoint.
set -uo pipefail
R=/home/nick/projects/chessengine; D=$R/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd "$R/jax"
while :; do
  RESUME=$(ls "$D"/d49-1280-sb*-ema-raw.bin 2>/dev/null | sed -E 's/.*-sb([0-9]+)-ema-raw\.bin/\1/' | sort -n | tail -1)
  RESUME=${RESUME:-1630}
  [ "$RESUME" -ge 2130 ] && { echo "[x500] reached sb${RESUME} -- done $(date -Is)" >> "$LOG"; break; }
  echo "[x500] launching from sb${RESUME} $(date -Is)" >> "$LOG"
  PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
    uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \
    python tools/pump_train.py ../tools/nnue/runs/d49_x500_1280.json "$RESUME" >> "$LOG" 2>&1
  rc=$?
  echo "[x500] trainer exited rc=$rc $(date -Is)" >> "$LOG"
  sleep 30
done

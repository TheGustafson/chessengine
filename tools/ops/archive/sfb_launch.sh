#!/usr/bin/env bash
# THE OWNER'S SF-TEACHER PROBE -- OWNED BY THE COORDINATOR, not a subagent.
# Killed by a GLOBAL OOM at 15:43:42 (3.8 GB anon + 927 MB shmem) while a
# simplification-execution agent was building and d49xml was matching. Not a
# defect in the probe -- external pressure. This loop brings it back.
set -uo pipefail
R=/home/nick/projects/chessengine; D=$R/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd "/home/nick/projects/chessengine-mainwt/jax"
while :; do
  RESUME=$(ls "$D"/d49sfb-sb*-ema-raw.bin 2>/dev/null | sed -E 's/.*-sb([0-9]+)-ema-raw\.bin/\1/' | sort -n | tail -1)
  RESUME=${RESUME:-1630}
  [ "$RESUME" -ge 1642 ] && { echo "[sfb] probe complete at sb${RESUME} $(date -Is)" >> "$LOG"; break; }
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  if [ "$((A/1024))" -lt 3000 ]; then
      echo "[sfb] holding: avail $((A/1024))MB < 3000 $(date -Is)" >> "$LOG"; sleep 60; continue; fi
  echo "[sfb] launching from sb${RESUME} (alpha 0.4) $(date -Is)" >> "$LOG"
  PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
    uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \
    python tools/pump_train.py ../tools/nnue/runs/d49_sfblend_probe.json "$RESUME" >> "$LOG" 2>&1
  echo "[sfb] exited rc=$? $(date -Is)" >> "$LOG"
  sleep 120   # GPU memory release after a kill takes longer than 30s (measured 15:49)
done

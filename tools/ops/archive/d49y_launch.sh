#!/bin/bash
# D49 CYCLICAL CONTINUATION -> sb1700.
# v2: resume point is DISCOVERED, not hardcoded. v1 passed a literal 700, so a
# watchdog restart at sb1000 would have thrown away 300 superbatches (~11 h GPU).
cd /home/nick/projects/chessengine/jax || exit 1
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
while systemctl --user is-active d49x.service >/dev/null 2>&1; do sleep 30; done
for i in $(seq 1 240); do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$free_mb" ] && [ "$free_mb" -gt 5500 ] && break
  sleep 30
done
# highest sbN having BOTH raw and ema-raw -- that pair is what pump_train loads
RESUME=700
for f in "$D"/d49-1280-sb*-raw.bin; do
  case "$f" in *ema-raw.bin) continue;; esac
  n=$(basename "$f" | sed -E 's/.*-sb([0-9]+)-raw\.bin/\1/')
  [ -r "$D/d49-1280-sb${n}-ema-raw.bin" ] || continue
  [ "$n" -gt "$RESUME" ] && RESUME=$n
done
echo "[d49y] GPU free (${free_mb}MiB); RESUMING FROM sb${RESUME} (discovered) $(date -Is)" >> "$LOG"
PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
  uv run --no-project --with numpy==2.5.1 --with 'jax[cuda12]==0.11.0' --with optax==0.2.8 \
  python tools/pump_train.py ../tools/nnue/runs/d49z_decay_1280.json "$RESUME" >> "$LOG" 2>&1
echo "[d49y] EXIT $? $(date -Is)" >> "$LOG"

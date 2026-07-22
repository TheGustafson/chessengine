#!/usr/bin/env bash
# Keep only the 5 newest raw checkpoints (each ~420MB); quants all stay.
set -u
cd /home/nick/projects/chessengine/data/nnue/experiments/d47c-1280-240sb/ckpt 2>/dev/null || exit 0
while true; do
  ls -t *-sb*-raw.bin 2>/dev/null | tail -n +6 | xargs -r rm -f
  sleep 300
done

#!/usr/bin/env bash
# Give the anchor A/B priority over the recurring matcher, then hand back.
#
# ROOT CAUSE this fixes: d49xml is a long-running LOOP service, so
# `systemctl is-active d49xml` is ALWAYS true between screens. Any waiter that
# blocks on "d49xml inactive" starves forever -- which is what happened to the
# 535 arm this morning and to d49ab2 now.
set -uo pipefail
LOG=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log
# 1. wait for the CURRENT screen to finish (no fastchess), not for the service
for _ in $(seq 1 60); do
  [ "$(pgrep -c fastchess 2>/dev/null || echo 0)" -eq 0 ] && break
  sleep 30
done
systemctl --user stop d49xml 2>/dev/null
echo "[abpri] matcher stood down; A/B has the box $(date -Is)" >> "$LOG"
# 2. wait for the A/B to finish
while systemctl --user is-active --quiet d49ab2 2>/dev/null; do sleep 60; done
echo "[abpri] A/B finished; restoring matcher $(date -Is)" >> "$LOG"
# 3. hand back
cd /home/nick/projects/chessengine
systemd-run --user --unit=d49xml --same-dir \
  --property=MemoryMax=3G --property=MemorySwapMax=0 \
  /usr/bin/bash /home/nick/.local/share/archon-d49/export_match_loop.sh >/dev/null 2>&1
echo "[abpri] matcher restored $(date -Is)" >> "$LOG"

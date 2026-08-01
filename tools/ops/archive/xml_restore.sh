#!/usr/bin/env bash
# Restore the matcher once study 71's BUILD is done and memory is back.
# Stood down at 14:02 when MemAvailable dipped to 1861 MB (below the 2500 floor)
# with the trainer + 2 rustc + a match all live.
set -uo pipefail
LOG=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log
for _ in $(seq 1 480); do
  b=$(pgrep -c -f 'rustc|cargo' 2>/dev/null || echo 0)
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  [ "$b" -eq 0 ] && [ "$((A/1024))" -ge 4000 ] && break
  sleep 60
done
cd /home/nick/projects/chessengine
systemd-run --user --unit=d49xml --same-dir --property=MemoryMax=3G --property=MemorySwapMax=0 \
  /usr/bin/bash /home/nick/.local/share/archon-d49/export_match_loop.sh >/dev/null 2>&1
echo "[xmlrestore] matcher restored $(date -Is)" >> "$LOG"

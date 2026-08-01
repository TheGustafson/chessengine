#!/usr/bin/env bash
# Re-arm the per-export matcher, but ONLY once the priority jobs are done.
# v2: also waits for d49retest. v1 waited only on d49len and so restored the
# matcher at 08:29 while the tunables re-test was mid-PGO-build.
set -uo pipefail
LOG=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log
for i in $(seq 1 720); do
  busy=0
  for u in d49len d49retest; do
    systemctl --user is-active --quiet "$u" 2>/dev/null && busy=1
  done
  [ "$busy" -eq 0 ] && break
  sleep 60
done
for _ in $(seq 1 30); do
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  [ "$((A/1024))" -ge 3500 ] && break; sleep 60
done
cd /home/nick/projects/chessengine
systemd-run --user --unit=d49xml --same-dir \
  --property=MemoryMax=3G --property=MemorySwapMax=0 \
  /usr/bin/bash /home/nick/.local/share/archon-d49/export_match_loop.sh >/dev/null 2>&1
echo "[rearm] d49xml restored after priority jobs $(date -Is)" >> "$LOG"

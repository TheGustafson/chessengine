#!/usr/bin/env bash
# perf study 61 — block until a panel file has `runs` records, then stop.
# A background wait rather than a foreground `sleep`: the standing rule is never
# to hold a turn hostage on a long job.
#   bash tools/perf61/waitpanel61.sh <panel.txt> <runs> [max-seconds]
set -euo pipefail
cd "$(dirname "$0")/../.."
P=${1:?panel}
WANT=${2:?runs}
MAX=${3:-1800}
t=0
while [ "$t" -lt "$MAX" ]; do
  n=$(grep -c '^### round=' "$P" 2>/dev/null || true)
  n=${n:-0}
  if [ "$n" -ge "$WANT" ]; then
    echo "[waitpanel61] $P complete: $n runs"
    exit 0
  fi
  sleep 15
  t=$((t + 15))
done
echo "[waitpanel61] TIMEOUT after ${MAX}s: $P has ${n:-0}/$WANT" >&2
exit 1

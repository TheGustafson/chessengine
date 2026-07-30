#!/usr/bin/env bash
# Block until a panel log has produced `n` result blocks, then print it.
#   bash waitfor22.sh <logfile> <n-blocks>
set -euo pipefail
LOG=${1:?log}
WANT=${2:-1}
while true; do
    n=$(grep -c 'nodes=' "$LOG" 2>/dev/null) || n=0
    [ "${n:-0}" -ge "$WANT" ] && break
    sleep 20
done
cat "$LOG"

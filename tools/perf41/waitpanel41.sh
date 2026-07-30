#!/usr/bin/env bash
# perf study 41 — block until a panel has produced N rows, or until it stalls.
#
#   bash tools/perf41/waitpanel41.sh <panel.txt> <rows> [stall-polls]
#
# Both exit paths report, so silence never has to be interpreted: DONE at the
# row target, STALLED if the file stops growing for `stall-polls` minutes.
#
# It deliberately does NOT ask "is a panel process alive". The obvious spelling
# of that question, `pgrep -f panel40.sh`, also matches the waiter's own
# subshell and the pattern it typed, and a first cut of this file reported GONE
# on a panel that was running perfectly well. Progress of the artifact is the
# honest liveness signal, and it needs no pattern at all.
set -euo pipefail
cd "$(dirname "$0")/../.."
F=${1:?panel}; WANT=${2:?rows}; STALL=${3:-8}
last=-1; quiet=0
while :; do
  have=$(wc -l < "$F" 2>/dev/null || echo 0)
  if [ "$have" -ge "$WANT" ]; then echo "DONE $have/$WANT rows in $F"; exit 0; fi
  if [ "$have" -eq "$last" ]; then
    quiet=$((quiet + 1))
    if [ "$quiet" -ge "$STALL" ]; then
      echo "STALLED at $have/$WANT rows in $F after ${STALL}m with no growth"
      exit 1
    fi
  else
    quiet=0
  fi
  last=$have
  sleep 60
done

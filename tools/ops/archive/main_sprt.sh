#!/usr/bin/env bash
# main (studies 70+71 + simplification tier0/0b) vs CANONICAL engine, ultra-bullet.
# Both PGO per the standing rule. 2s+0.1 is a REAL clock, so unlike the fixed-node
# screens this one CAN see study 71's +3.99 % instructions/node -- that real-TC
# confirmation was the outstanding debt on its +11.00.
set -uo pipefail
cd /home/nick/projects/chessengine
LOG=/home/nick/.local/share/archon-d49/main_sprt.log
CAND=engines/bin/archon-main-6b59a8cc
BASE=engines/canonical/archon
for _ in $(seq 1 2); do
  [ "$(systemctl --user is-active pgobuild)" != "active" ] && break
  sleep 30
done
[ -x "$CAND" ] || { echo "[sprt] candidate missing: $CAND" >> "$LOG"; exit 1; }
echo "[sprt] $(basename $CAND) vs $(basename $(readlink -f $BASE)) @ 2+0.1  $(date -Is)" >> "$LOG"
TC=2+0.1 CONCURRENCY=3 bash tools/sprt.sh "$CAND" "$BASE" >> "$LOG" 2>&1
echo "[sprt] exit=$? $(date -Is)" >> "$LOG"

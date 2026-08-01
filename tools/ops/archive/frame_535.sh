#!/usr/bin/env bash
# PRE-REGISTERED prediction test (v2). Prediction written 2026-08-01 01:05,
# BEFORE any result, and unchanged here:
#
#   Under eca43a97 tunables, D49 sb1300 vs canonical @ 25k fixed nodes:
#     scale  756 (slope 0.477) = +3.24  +/- 10.65   n=1500
#     scale 1068 (slope 0.672) = -8.57  +/- 10.57   n=1500
#     scale 1585 (slope 0.995) = -18.08 +/- 10.79   n=1500
#   Monotonic, OPPOSITE to the frame theory. PREDICTION: scale 535
#   (slope 0.339) beats +3.24. If it turns DOWN, 756 is an interior optimum
#   and the monotonic reading is falsified.
#
# v1 exited silently: its wait loop checked the three units SEQUENTIALLY and
# raced d49frame re-arming d49xml in the same second. This version polls all
# units together, repeatedly, and logs every state it sees.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
DBG=/home/nick/.local/share/archon-d49/frame535.debug.log
cd /home/nick/projects/chessengine
echo "[frame535] up $(date -Is)" | tee -a "$DBG" >> "$LOG"

busy() { for u in d49frame d49xm d49xml d49rerun; do
           systemctl --user is-active --quiet "$u" 2>/dev/null && { echo "$u"; return 0; }
         done; return 1; }

for i in $(seq 1 480); do            # up to 8 h
  b=$(busy) || break
  [ $((i % 10)) -eq 1 ] && echo "[frame535] waiting on $b $(date -Is)" >> "$DBG"
  sleep 60
done

for i in $(seq 1 30); do
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  [ "$((A/1024))" -ge 3500 ] && break
  echo "[frame535] mem hold $((A/1024))MB" >> "$DBG"; sleep 60
done

CAND=data/nnue/tmp/verdict/d49-sb1300-535.cenn
OUT=/home/nick/.local/share/archon-d49/frame-sb1300-535.out
[ -s "$CAND" ] || { echo "[frame535] MISSING $CAND -- abort" | tee -a "$DBG" >> "$LOG"; exit 1; }
echo "[frame535] scale 535 vs canonical, 1500 games @ 25k $(date -Is)" | tee -a "$DBG" >> "$LOG"
ENGINE=./target-sota/release/chessengine CONC=3 bash tools/screen.sh \
    "$CAND" engines/canonical/net.cenn 1500 25000 > "$OUT" 2>&1
rc=$?
printf "[frame535] rc=%s RESULT: " "$rc" >> "$LOG"
grep -E '^Elo:' "$OUT" | tail -1 >> "$LOG"
grep -E '^Games:' "$OUT" | tail -1 >> "$LOG"
echo "[frame535] done $(date -Is)" | tee -a "$DBG" >> "$LOG"

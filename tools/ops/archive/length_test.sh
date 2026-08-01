#!/usr/bin/env bash
# THE UNDERTRAINING TEST, 2026-08-01. Recipe held FIXED, only length differs.
#   CAND = D49 sb1500   REF = D49 sb500
# sb500 is within 20 sb of canonical's own checkpoint (d48x sb520), so this is
# also a like-for-like length comparison against the incumbent's training budget.
# Both packed at scale 1068; stdev 603.0 vs 596.7 (canonical 610) => frame-fair.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd /home/nick/projects/chessengine
for _ in $(seq 1 60); do
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo); [ "$((A/1024))" -ge 3000 ] && break; sleep 60
done
OUT=/home/nick/.local/share/archon-d49/lentest.out
echo "[len] sb1500 vs sb500, 1500 games @ 25k $(date -Is)" >> "$LOG"
ENGINE=./target-sota/release/chessengine CONC=2 bash tools/screen.sh \
    data/nnue/tmp/verdict/d49-len-sb1500.cenn \
    data/nnue/tmp/verdict/d49-len-sb500.cenn 1500 25000 > "$OUT" 2>&1
printf "[len] RESULT: " >> "$LOG"; grep -E '^Elo:' "$OUT" | tail -1 >> "$LOG"
grep -E '^Games:' "$OUT" | tail -1 >> "$LOG"
echo "[len] done $(date -Is)" >> "$LOG"

#!/usr/bin/env bash
# FRAME EXPERIMENT (2026-08-01). What eval spread should we want?
#
# Our razoring/futility/RFP/ProbCut margins are Stockfish eca43a97 internal
# constants transcribed verbatim. They only prune what they were tuned to prune
# if our eval distribution matches SF's internal frame. Measured slope vs that
# frame (tools/nnue/net_scale_check.sh, n=503):
#
#   D49 sb1300 @ scale  756 -> 0.477   (constants 2.10x too large)  <- today
#   canonical  @ scale  600 -> 0.674   (constants 1.48x too large)
#   D49 sb1300 @ scale 1068 -> 0.672   (canonical's frame)
#   D49 sb1300 @ scale 1585 -> 0.995   (PARITY -- the theoretical target)
#
# PERFECTLY CONTROLLED: all three arms are the SAME PAYLOAD, byte-identical
# from offset 64. Only the 4-byte header scale differs. No training, no
# recompile. Any Elo difference is the frame and nothing else.
#
# Common opponent = canonical, so the 756 arm already running (xm-sb1300.out)
# is the third point and needs no rerun.
#
# PRIOR: the one previous move toward parity (0.538 -> 0.674, header-only) was
# +14.78 +/- 5.45 Elo over 5880 games with -12.4% nodes. That is evidence of
# DIRECTION, not of magnitude at 1.0 -- nobody has ever tested parity, and the
# SPSA params co-adapted to the wrong frame. Games arbitrate.
set -uo pipefail
SHARE=/home/nick/.local/share/archon-d49
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd /home/nick/projects/chessengine

# Never two matches at once: memory, and fixed-node results stay clean anyway.
while systemctl --user is-active --quiet d49xm 2>/dev/null; do sleep 60; done
echo "[frame] predecessor match done; starting frame experiment $(date -Is)" >> "$LOG"

BIN=./target-sota/release/chessengine
for S in 1585 1068; do
  CAND=data/nnue/tmp/verdict/d49-sb1300-${S}.cenn
  [ -s "$CAND" ] || { echo "[frame] missing $CAND -- skip" >> "$LOG"; continue; }

  for _ in $(seq 1 30); do
    A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    [ "$((A/1024))" -ge 3500 ] && break
    sleep 60
  done

  echo "[frame] scale ${S} vs canonical, 1500 games @ 25k $(date -Is)" >> "$LOG"
  OUT=$SHARE/frame-sb1300-${S}.out
  ENGINE=$BIN CONC=3 bash tools/screen.sh "$CAND" engines/canonical/net.cenn \
      1500 25000 > "$OUT" 2>&1
  printf "[frame] scale %s RESULT: " "$S" >> "$LOG"
  grep -E '^Elo:' "$OUT" | tail -1 >> "$LOG"
  grep -E '^Games:' "$OUT" | tail -1 >> "$LOG"
  echo "[frame] scale ${S} done $(date -Is)" >> "$LOG"
done

echo "[frame] experiment complete $(date -Is)" >> "$LOG"
# Hand the box back to the recurring per-export matcher.
systemd-run --user --unit=d49xml --same-dir \
  --property=MemoryMax=3G --property=MemorySwapMax=0 \
  /usr/bin/bash "$SHARE/export_match_loop.sh" >/dev/null 2>&1
echo "[frame] recurring matcher rearmed $(date -Is)" >> "$LOG"

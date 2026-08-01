#!/usr/bin/env bash
# FRAME RE-RUN, post-tunable-transcription.
#
#   bash frame_rerun.sh <engine-binary> [games] [tag]
#
# WHY: the frame arms run on 2026-08-01 (756 / 1068 / 1585 vs canonical) were
# measured against the eca43a97 tunable set. The transcription refresh moves
# ~many search constants to ebcea3ef values. A pruning margin only means what it
# means RELATIVE to the eval distribution it is compared against, so changing the
# margins can move the optimal frame. Every frame conclusion from the old set
# expires when the new set lands -- hence this re-run, not a reuse.
#
# NOT armed automatically and deliberately so: the transcription must be
# VERIFIED + gated + merged by a human-in-the-loop first (the standing perf loop
# is verify-myself -> CHECK_NNUE_GATE -> build PGO -> merge -> then test).
# Launch this only after that.
#
# The engine binary is an ARGUMENT because both sides of the screen must use the
# SAME binary: the frame comparison is between NETS, with tunables held fixed.
set -uo pipefail
BIN="${1:?usage: frame_rerun.sh <engine-binary> [games] [tag]}"
GAMES="${2:-1500}"
TAG="${3:-newtunables}"
SHARE=/home/nick/.local/share/archon-d49
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd /home/nick/projects/chessengine

[ -x "$BIN" ] || { echo "[rerun] not executable: $BIN" | tee -a "$LOG"; exit 2; }

# Never two matches at once (memory, and the box is shared with a GPU trainer).
for u in d49xm d49frame d49xml; do
  while systemctl --user is-active --quiet "$u" 2>/dev/null; do sleep 60; done
done

echo "[rerun/$TAG] start $(date -Is) bin=$BIN games=$GAMES" >> "$LOG"
"$BIN" --version >> "$LOG" 2>&1 || true

for S in 1585 1068 756 535; do
  CAND=data/nnue/tmp/verdict/d49-sb1300-${S}.cenn
  [ -s "$CAND" ] || { echo "[rerun/$TAG] missing $CAND -- skip" >> "$LOG"; continue; }

  for _ in $(seq 1 30); do
    A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    [ "$((A/1024))" -ge 3500 ] && break
    echo "[rerun/$TAG] holding: avail $((A/1024))MB < 3500MB" >> "$LOG"
    sleep 60
  done

  OUT=$SHARE/rerun-${TAG}-sb1300-${S}.out
  echo "[rerun/$TAG] scale ${S} vs canonical, ${GAMES} games @ 25k $(date -Is)" >> "$LOG"
  ENGINE=$BIN CONC=3 bash tools/screen.sh "$CAND" engines/canonical/net.cenn \
      "$GAMES" 25000 > "$OUT" 2>&1
  printf "[rerun/%s] scale %s RESULT: " "$TAG" "$S" >> "$LOG"
  grep -E '^Elo:' "$OUT" | tail -1 >> "$LOG"
  grep -E '^Games:' "$OUT" | tail -1 >> "$LOG"
done
echo "[rerun/$TAG] complete $(date -Is)" >> "$LOG"

# Hand the box back to the per-export matcher, which this run displaced.
systemd-run --user --unit=d49xml --same-dir \
  --property=MemoryMax=3G --property=MemorySwapMax=0 \
  /usr/bin/bash /home/nick/.local/share/archon-d49/export_match_loop.sh >/dev/null 2>&1
echo "[rerun/$TAG] recurring matcher rearmed $(date -Is)" >> "$LOG"

#!/usr/bin/env bash
# RECURRING export matcher. The predecessor (export_match.sh) fired ONCE and
# exited, so every export after the first passed unscreened -- it had to be
# re-armed by hand and twice was not. This loops for the life of the run.
#
# Per export: pack the EMA checkpoint, print its eval stdev (the frame canary),
# then screen 1500 games at fixed 25k nodes vs the canonical net.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
SHARE=/home/nick/.local/share/archon-d49
cd /home/nick/projects/chessengine

newest() { ls -t "$D"/*-ema-raw.bin 2>/dev/null | head -1 | grep -oE 'sb[0-9]+' | tr -d 'sb'; }

# Don't collide with a hand-armed one-shot that may still be screening.
while systemctl --user is-active --quiet d49xm 2>/dev/null; do sleep 60; done

LAST=$(newest); LAST=${LAST:-0}
echo "[xml] recurring matcher up, baseline sb${LAST} $(date -Is)" >> "$LOG"

while :; do
  N=$(newest); N=${N:-0}
  if [ "$N" -le "$LAST" ]; then sleep 60; continue; fi

  # MEMORY GATE. Swap has sat at 100% for hours and the trainer was oom-killed
  # once on 2026-07-31; packing allocates ~1-2 GB on top of a 433 MB read. Wait
  # for real headroom rather than racing the export's own peak.
  for _ in $(seq 1 30); do
    A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    [ "$((A/1024))" -ge 3500 ] && break
    echo "[xml] holding: avail $((A/1024))MB < 3500MB $(date -Is)" >> "$LOG"
    sleep 60
  done

  SB=$N
  echo "[xml] export sb${SB} -- packing and screening $(date -Is)" >> "$LOG"
  OUT=data/nnue/tmp/verdict; mkdir -p "$OUT"
  CK=$D/d49-1280-sb${SB}-ema-raw.bin

  if ! uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
        "$CK" "$OUT/q${SB}.bin" --scale 256 --run-spec tools/nnue/runs/d49_1280.json >/dev/null 2>&1; then
    echo "[xml] sb${SB} requant FAILED -- skipping" >> "$LOG"; LAST=$SB; continue
  fi
  if ! D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
       D40_THREAT_QUANT=256 D40_NETWORK_SCALE=1068 \
       uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
         "$OUT/q${SB}.bin" "$OUT/d49-sb${SB}-1068.cenn" d40 >/dev/null 2>&1; then
    echo "[xml] sb${SB} netpack FAILED -- skipping" >> "$LOG"; LAST=$SB; continue
  fi

  BIN=./target-sota/release/chessengine
  CENN=$(readlink -f "$OUT/d49-sb${SB}-1068.cenn")
  printf "[xml] sb%s stdev=" "$SB" >> "$LOG"
  $BIN nnue-eval "$CENN" < tools/nnue/calib/wl50.fens 2>/dev/null \
    | grep -oE '^-?[0-9]+' \
    | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}' if v else 'NA')" >> "$LOG"

  # CONC=3 (~830 MB of nets). Fixed nodes, so concurrency changes only wall
  # time, never the result. Output on ext4: /tmp is tmpfs and swap is full.
  XM=$SHARE/xml-sb${SB}.out
  ENGINE=$BIN CONC=3 bash tools/screen.sh "$OUT/d49-sb${SB}-1068.cenn" \
      engines/canonical/net.cenn 1500 25000 > "$XM" 2>&1
  grep -A5 'Results of' "$XM" | tail -5 >> "$LOG"
  echo "[xml] sb${SB} done $(date -Is)" >> "$LOG"

  rm -f "$OUT/q${SB}.bin"          # 433 MB intermediate; disk is 85% full
  LAST=$SB
done

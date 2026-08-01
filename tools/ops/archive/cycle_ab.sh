#!/usr/bin/env bash
# THE DECISIVE A/B for the jump->anneal experiment.
#
# The per-export screens measure each checkpoint against CANONICAL, so comparing
# a cycle to the anchor adds their bars: cycle 1 read +9.27 +/- 11.04 against a
# +3.47 +/- 11.22 anchor = +5.80 +/- 15.7, which cannot clear zero.
#
# This screens the cycle-3 checkpoint DIRECTLY against the sb1540 anchor: ONE
# match, ONE bar (+/-10 at 1500 games). Same design that produced the
# undertraining number +30.31 +/- 11.0 cleanly.
set -uo pipefail
R=/home/nick/projects/chessengine
D=$R/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
OUT=$R/data/nnue/tmp/verdict
cd "$R"

# wait for all three cycles to finish
while systemctl --user is-active --quiet d49lrc 2>/dev/null; do sleep 60; done
CK=$D/d49lrc3-sb1630-ema-raw.bin
[ -s "$CK" ] || { echo "[ab] cycle 3 checkpoint missing -- abort" >> "$LOG"; exit 1; }

echo "[ab] packing cycle-3 sb1630 at scale 1068 $(date -Is)" >> "$LOG"
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CK" "$OUT/ab1630.bin" --scale 256 --run-spec tools/nnue/runs/d49_1280.json >/dev/null 2>&1 || exit 1
D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
D40_THREAT_QUANT=256 D40_NETWORK_SCALE=1068 \
  uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
    "$OUT/ab1630.bin" "$OUT/d49-cyc3-sb1630.cenn" d40 >/dev/null 2>&1 || exit 1

# don't run two matches at once
while systemctl --user is-active --quiet d49xml 2>/dev/null; do sleep 60; done
for _ in $(seq 1 30); do
  A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo); [ "$((A/1024))" -ge 3000 ] && break; sleep 60
done

echo "[ab] CYCLE-3 vs ANCHOR sb1540, 1500 games @ 25k $(date -Is)" >> "$LOG"
O=/home/nick/.local/share/archon-d49/cycle_ab.out
ENGINE=./target-sota/release/chessengine CONC=3 bash tools/screen.sh \
    "$OUT/d49-cyc3-sb1630.cenn" "$OUT/d49-sb1540-1068.cenn" 1500 25000 > "$O" 2>&1
printf "[ab] RESULT (3 cycles vs anchor): " >> "$LOG"
grep -E '^Elo:' "$O" | tail -1 >> "$LOG"; grep -E '^Games:' "$O" | tail -1 >> "$LOG"
echo "[ab] done $(date -Is)" >> "$LOG"

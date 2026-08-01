#!/usr/bin/env bash
# The anchor A/B, re-run after d49c3 was cgroup-OOM-killed at MemoryMax=512M.
# That cap suited the original orchestrator (which only supervised); this script
# also PACKS (python, ~1.5-2 GB), so it needs real headroom. 3G here.
set -uo pipefail
R=/home/nick/projects/chessengine; D=$R/data/nnue/experiments/d49-1280
LOG=$D/cyc.log; OUT=$R/data/nnue/tmp/verdict
cd "$R"
CK=$D/d49lrc3-sb1630-ema-raw.bin
[ -s "$CK" ] || { echo "[ab] cycle-3 checkpoint missing -- abort" >> "$LOG"; exit 1; }
echo "[ab] packing cycle-3 sb1630 $(date -Is)" >> "$LOG"
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CK" "$OUT/ab1630.bin" --scale 256 --run-spec tools/nnue/runs/d49_1280.json >/dev/null 2>&1 || { echo "[ab] requant FAILED" >> "$LOG"; exit 1; }
D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
D40_THREAT_QUANT=256 D40_NETWORK_SCALE=1068 \
  uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
    "$OUT/ab1630.bin" "$OUT/d49-cyc3-sb1630.cenn" d40 >/dev/null 2>&1 || { echo "[ab] netpack FAILED" >> "$LOG"; exit 1; }
printf "[ab] cycle-3 stdev=" >> "$LOG"
./target-sota/release/chessengine nnue-eval "$(readlink -f "$OUT/d49-cyc3-sb1630.cenn")" < tools/nnue/calib/wl50.fens 2>/dev/null \
  | grep -oE '^-?[0-9]+' | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}' if v else 'NA')" >> "$LOG"
while systemctl --user is-active --quiet d49xml 2>/dev/null; do sleep 60; done
for _ in $(seq 1 30); do A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo); [ "$((A/1024))" -ge 3000 ] && break; sleep 60; done
echo "[ab] A/B: 3 cycles (sb1630) vs ANCHOR sb1540, 1500 games @ 25k $(date -Is)" >> "$LOG"
O=/home/nick/.local/share/archon-d49/cycle_ab.out
ENGINE=./target-sota/release/chessengine CONC=3 bash tools/screen.sh \
    "$OUT/d49-cyc3-sb1630.cenn" "$OUT/d49-sb1540-1068.cenn" 1500 25000 > "$O" 2>&1
printf "[ab] RESULT (3 cycles vs anchor): " >> "$LOG"
grep -E '^Elo:' "$O" | tail -1 >> "$LOG"; grep -E '^Games:' "$O" | tail -1 >> "$LOG"
echo "[ab] done $(date -Is)" >> "$LOG"

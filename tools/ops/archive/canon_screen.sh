#!/usr/bin/env bash
# Distilled D49 vs CANONICAL (D48X) -- the promotion question, early read.
# Plain D49 @756 measured -1.39 +/- 10.43 vs canonical, so this is like-for-like.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log; cd /home/nick/projects/chessengine
SB=$(ls -t "$D"/d49sfb-sb*-ema-raw.bin | head -1 | grep -oE 'sb[0-9]+' | tr -d 'sb')
OUT=data/nnue/tmp/canon; mkdir -p "$OUT"
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$D/d49sfb-sb${SB}-ema-raw.bin" "$OUT/q${SB}.bin" --scale 256 \
    --run-spec tools/nnue/runs/d49_sfblend_500.json >/dev/null 2>&1 || { echo "[cs] requant FAIL" >>"$LOG"; exit 1; }
D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
D40_THREAT_QUANT=256 D40_NETWORK_SCALE=756 \
  uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
    "$OUT/q${SB}.bin" "$OUT/sfb-sb${SB}-756.cenn" d40 >/dev/null 2>&1 || { echo "[cs] pack FAIL" >>"$LOG"; exit 1; }
echo "[cs] DISTILLED sb${SB} vs CANONICAL, 1500 games @25k $(date -Is)" >> "$LOG"
CS=/home/nick/.local/share/archon-d49/cs-sb${SB}.out
ENGINE=./target-sota/release/chessengine CONC=3 \
  bash tools/screen.sh "$OUT/sfb-sb${SB}-756.cenn" engines/canonical/net.cenn 1500 25000 > "$CS" 2>&1
{ echo "[cs] === DISTILLED sb${SB} vs CANONICAL ==="; grep -A5 'Results of' "$CS" | tail -5; } >> "$LOG"

#!/usr/bin/env bash
# Fires ONCE on the next D49 checkpoint export: pack it and screen vs canonical
# at equal nodes. Durable location (ext4) so it survives a tmpfs wipe.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd /home/nick/projects/chessengine
newest() { ls -t "$D"/*-ema-raw.bin 2>/dev/null | head -1 | grep -oE 'sb[0-9]+' | tr -d 'sb'; }
START=$(newest); START=${START:-0}
echo "[xm] waiting for an export newer than sb${START} $(date -Is)" >> "$LOG"
for _ in $(seq 1 240); do            # up to ~2 h
  N=$(newest); N=${N:-0}
  [ "$N" -gt "$START" ] && break
  sleep 30
done
SB=$(newest)
[ -z "$SB" ] && { echo "[xm] no export found; giving up" >> "$LOG"; exit 1; }
echo "[xm] export sb${SB} detected -- packing and screening $(date -Is)" >> "$LOG"
OUT=data/nnue/tmp/verdict; mkdir -p "$OUT"
CK=$D/d49-1280-sb${SB}-ema-raw.bin
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CK" "$OUT/q${SB}.bin" --scale 256 --run-spec tools/nnue/runs/d49_1280.json >/dev/null 2>&1
D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
D40_THREAT_QUANT=256 D40_NETWORK_SCALE=756 \
  uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
    "$OUT/q${SB}.bin" "$OUT/d49-sb${SB}-756.cenn" d40 >/dev/null 2>&1
BIN=./target-sota/release/chessengine
printf "[xm] sb%s stdev=" "$SB" >> "$LOG"
$BIN nnue-eval "$(readlink -f "$OUT/d49-sb${SB}-756.cenn")" < tools/nnue/calib/wl50.fens 2>/dev/null \
  | grep -oE '^-?[0-9]+' | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')" >> "$LOG"
# Output lands on ext4, NOT the session scratchpad: /tmp is tmpfs (RAM) and swap
# is already full, and a session reset wipes the scratchpad mid-match.
# CONC=3 (~830 MB of nets) not 5: the trainer was oom-killed once tonight and a
# study is building concurrently. Fixed nodes, so a lower conc costs only time.
XM=/home/nick/.local/share/archon-d49/xm-sb${SB}.out
ENGINE=$BIN CONC=3 bash tools/screen.sh "$OUT/d49-sb${SB}-756.cenn" engines/canonical/net.cenn 1500 25000 \
  > "$XM" 2>&1
grep -A5 'Results of' "$XM" | tail -5 >> "$LOG"
echo "[xm] done $(date -Is)" >> "$LOG"

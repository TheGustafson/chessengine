#!/usr/bin/env bash
# Cycle 3, relaunched so checkpoints land ONLY at the BOTTOM of the cycle
# (owner, 2026-08-01). Measured reason: a mid-cycle checkpoint taken 10 sb after
# a 5x LR jump was 13 Elo weaker than the anneal-end one (sb1580 -3.94 vs
# sb1570 +9.27, 1500 games each), and 7 Elo below the anchor it started from.
#
# export_every=1000000 + superbatches=1630 => the ONLY export is at sb1630.
# This script also does the work the orchestrator would have: save the protected
# checkpoint, then run the decisive A/B against the sb1540 anchor.
set -uo pipefail
R=/home/nick/projects/chessengine
D=$R/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
OUT=$R/data/nnue/tmp/verdict
cd "$R"
newest() { ls -t "$D"/d49-1280-sb*-ema-raw.bin 2>/dev/null | head -1 | grep -oE 'sb[0-9]+' | tr -d 'sb'; }

echo "[c3] relaunching cycle 3 from sb1600, exports at the BOTTOM only $(date -Is)" >> "$LOG"
systemd-run --user --unit=d49lrc3 --working-directory=/home/nick/projects/chessengine/jax \
  --property=MemoryMax=9G --property=MemorySwapMax=0 \
  /usr/bin/env PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
    uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \
    python tools/pump_train.py ../tools/nnue/runs/d49_lrcycle3_1280.json 1600 >/dev/null 2>&1
sleep 45
if ! systemctl --user is-active --quiet d49lrc3; then
  echo "[c3] FAILED TO START -- aborting" >> "$LOG"
  journalctl --user -u d49lrc3 --no-pager 2>/dev/null | tail -5 >> "$LOG"; exit 1
fi

for _ in $(seq 1 480); do
  if ! systemctl --user is-active --quiet d49lrc3; then
    m=$(newest); [ -z "$m" ] || [ "$m" -lt 1630 ] && { echo "[c3] DIED at sb${m:-?} before 1630 -- aborting" >> "$LOG"; exit 1; }
    break
  fi
  m=$(newest); [ -n "$m" ] && [ "$m" -ge 1630 ] && break
  sleep 30
done
systemctl --user stop d49lrc3 2>/dev/null; sleep 8
for suf in ema-raw ema-quant raw quant; do
  s="$D/d49-1280-sb1630-${suf}.bin"; [ -f "$s" ] && cp -n "$s" "$D/d49lrc3-sb1630-${suf}.bin" 2>/dev/null
done
echo "[c3] cycle 3 done; checkpoint saved ($(ls -1 "$D"/d49lrc3-sb1630-*.bin 2>/dev/null | wc -l) files) $(date -Is)" >> "$LOG"

# ---- the decisive A/B: 3 cycles vs the anchor, ONE bar ----------------------
CK=$D/d49lrc3-sb1630-ema-raw.bin
[ -s "$CK" ] || { echo "[c3] no cycle-3 checkpoint -- no A/B" >> "$LOG"; exit 1; }
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CK" "$OUT/ab1630.bin" --scale 256 --run-spec tools/nnue/runs/d49_1280.json >/dev/null 2>&1
D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
D40_THREAT_QUANT=256 D40_NETWORK_SCALE=1068 \
  uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
    "$OUT/ab1630.bin" "$OUT/d49-cyc3-sb1630.cenn" d40 >/dev/null 2>&1
while systemctl --user is-active --quiet d49xml 2>/dev/null; do sleep 60; done
for _ in $(seq 1 30); do A=$(awk '/^MemAvailable/{print $2}' /proc/meminfo); [ "$((A/1024))" -ge 3000 ] && break; sleep 60; done
echo "[c3] A/B: cycle-3 sb1630 vs ANCHOR sb1540, 1500 games @ 25k $(date -Is)" >> "$LOG"
O=/home/nick/.local/share/archon-d49/cycle_ab.out
ENGINE=./target-sota/release/chessengine CONC=3 bash tools/screen.sh \
    "$OUT/d49-cyc3-sb1630.cenn" "$OUT/d49-sb1540-1068.cenn" 1500 25000 > "$O" 2>&1
printf "[c3] A/B RESULT (3 cycles vs anchor): " >> "$LOG"
grep -E '^Elo:' "$O" | tail -1 >> "$LOG"; grep -E '^Games:' "$O" | tail -1 >> "$LOG"
echo "[c3] ALL DONE $(date -Is)" >> "$LOG"

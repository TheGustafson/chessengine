#!/usr/bin/env bash
# ELO FOR THE TEACHER BLEND.
#   CAND = d49sfb sb1642  (12 sb of 60/40 SF-teacher blend)
#   REF  = d49sfb sb1630-EMA -- the EXACT checkpoint the probe warmed from.
# Same warm start, same pack path, same scale => the delta IS the teacher.
# Comparing to canonical instead would confound the blend with 100+ sb of drift.
set -uo pipefail
D=/home/nick/projects/chessengine/data/nnue/experiments/d49-1280
LOG=$D/cyc.log
cd /home/nick/projects/chessengine
CAND_SB=1642; REF_SB=1630

# Predicate is CHECKPOINT EXISTENCE, never `is-active`: sfb_launch.sh is a loop
# service, so is-active is true forever and has starved three prior arms.
echo "[tv] waiting for sb${CAND_SB} $(date -Is)" >> "$LOG"
for _ in $(seq 1 180); do
  [ -s "$D/d49sfb-sb${CAND_SB}-ema-raw.bin" ] && break
  sleep 30
done
[ -s "$D/d49sfb-sb${CAND_SB}-ema-raw.bin" ] || { echo "[tv] sb${CAND_SB} never appeared" >> "$LOG"; exit 1; }
sleep 60   # let the trainer finish its final write + release the GPU

# SERIALIZE vs the simplification agent's build unit. Both of us wake on the
# trainer exiting; 6G of rustc + 4G of match does NOT fit the ~9G that frees up,
# and this box has OOM-killed three times today. Builds are ~30 min, this match
# is hours -- so the short job goes first. Safe predicate here because
# verify_after_trainer.sh is a ONESHOT script, not a loop service.
for _ in $(seq 1 120); do
  st=$(systemctl --user is-active simplify-verify 2>/dev/null)
  [ "$st" = "active" ] || [ "$st" = "activating" ] || break
  sleep 60
done
echo "[tv] simplify-verify=$(systemctl --user is-active simplify-verify 2>/dev/null); starting pack $(date -Is)" >> "$LOG"

OUT=data/nnue/tmp/teacher; mkdir -p "$OUT"
pack() {  # $1=sb  -> prints .cenn path
  local sb=$1
  uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
      "$D/d49sfb-sb${sb}-ema-raw.bin" "$OUT/q${sb}.bin" --scale 256 \
      --run-spec tools/nnue/runs/d49_sfblend_probe.json >/dev/null 2>&1 || return 1
  D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
  D40_THREAT_QUANT=256 D40_NETWORK_SCALE=756 \
    uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
      "$OUT/q${sb}.bin" "$OUT/sfb-sb${sb}-756.cenn" d40 >/dev/null 2>&1 || return 1
  echo "$OUT/sfb-sb${sb}-756.cenn"
}
CAND=$(pack $CAND_SB) || { echo "[tv] pack cand FAILED" >> "$LOG"; exit 1; }
REF=$(pack $REF_SB)   || { echo "[tv] pack ref FAILED"  >> "$LOG"; exit 1; }

BIN=./target-sota/release/chessengine
for sb in $CAND_SB $REF_SB; do
  printf "[tv] sb%s stdev=" "$sb" >> "$LOG"
  $BIN nnue-eval "$(readlink -f "$OUT/sfb-sb${sb}-756.cenn")" < tools/nnue/calib/wl50.fens 2>/dev/null \
    | grep -oE '^-?[0-9]+' \
    | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')" >> "$LOG"
done
# A big stdev split between the two would mean the blend moved the FRAME, not the
# knowledge -- that confounds the match, so it gets read BEFORE the games.

echo "[tv] TEACHER sb${CAND_SB} vs WARM-POINT sb${REF_SB}, 1500 games @25k $(date -Is)" >> "$LOG"
TV=/home/nick/.local/share/archon-d49/tv-sb${CAND_SB}.out
ENGINE=$BIN CONC=3 bash tools/screen.sh "$CAND" "$REF" 1500 25000 > "$TV" 2>&1
{ echo "[tv] === TEACHER BLEND VERDICT (sb${CAND_SB} vs sb${REF_SB} warm point) ==="
  grep -A5 'Results of' "$TV" | tail -5; echo "[tv] done $(date -Is)"; } >> "$LOG"

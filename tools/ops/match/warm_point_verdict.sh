#!/usr/bin/env bash
# tools/ops/match/warm_point_verdict.sh — screen a checkpoint against the exact
# checkpoint it warmed from.
#
# PURPOSE
#   Isolates what a CHANGE bought from what ordinary training bought. Screening
#   a continuation against `canonical` conflates the two; screening it against
#   its own warm point does not, because both arms share every superbatch before
#   the warm.
#
# USAGE
#   systemd-run --user --unit=d49tv -p MemoryMax=4G -p MemorySwapMax=0 \
#       /path/to/tools/ops/match/warm_point_verdict.sh
#
#   Environment:
#     CAND_SB / REF_SB   superbatches to compare (REF_SB = the warm point)
#     CKPT_PREFIX        default d49sfb        SCALE default 756
#     GAMES 1500   NODES 25000   CONC 3
#     RUN_SPEC           default d49_sfblend_500.json
#
# WHAT IT CANNOT TELL YOU
#   The candidate has trained for (CAND_SB - REF_SB) superbatches that the
#   reference has not, so ORDINARY PROGRESS is inside the margin. To attribute
#   the result to the change itself you still need a matched control: the same
#   span of training from the same warm with the change disabled. Price the
#   ordinary-progress term before believing the attribution — at ~0.03 Elo/sb,
#   12 superbatches is worth ~0.4 Elo, so a +17 result is not explained by it.
#
# WHY THE WARM POINT MUST BE THE *EMA* CHECKPOINT
#   Warm from `-ema-raw` and compare against `-ema-raw`. Mixing raw and EMA
#   silently compares two different starting points and the result means
#   nothing. A previous run warmed from raw while the reference was EMA; the
#   raw file was later reaped and the comparison became unreconstructable.
#   (`ckpt_reaper.sh` never deletes symlinks for exactly this reason.)

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

EXP_DIR=${EXP_DIR:-$ARCHON_ROOT/data/nnue/experiments/d49-1280}
CKPT_PREFIX=${CKPT_PREFIX:-d49sfb}
SCALE=${SCALE:-756}; GAMES=${GAMES:-1500}; NODES=${NODES:-25000}; CONC=${CONC:-3}
RUN_SPEC=${RUN_SPEC:-d49_sfblend_500.json}
OPS_LOG="$EXP_DIR/cyc.log"; export OPS_LOG

CAND_SB=${CAND_SB:-$(ls "$EXP_DIR/$CKPT_PREFIX"-sb*-ema-raw.bin 2>/dev/null \
    | sed -E 's/.*-sb([0-9]+)-ema-raw\.bin/\1/' | sort -n | tail -1)}
REF_SB=${REF_SB:?REF_SB (the warm point superbatch) must be set}
[ -n "$CAND_SB" ] || { log "FATAL: no candidate checkpoint found"; exit 2; }

OUT="$ARCHON_ROOT/data/nnue/tmp/warmpoint"; mkdir -p "$OUT"
BIN=./target-sota/release/chessengine

pack() {
    local sb=$1 ck="$EXP_DIR/$CKPT_PREFIX-sb${1}-ema-raw.bin"
    [ -s "$ck" ] || { log "FATAL: missing checkpoint sb$sb ($ck)"; return 1; }
    uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
        "$ck" "$OUT/q${sb}.bin" --scale 256 --run-spec "tools/nnue/runs/$RUN_SPEC" >/dev/null 2>&1 || return 1
    D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
    D40_THREAT_QUANT=256 D40_NETWORK_SCALE="$SCALE" \
      uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
        "$OUT/q${sb}.bin" "$OUT/sb${sb}-${SCALE}.cenn" d40 >/dev/null 2>&1 || return 1
    echo "$OUT/sb${sb}-${SCALE}.cenn"
}

CAND=$(pack "$CAND_SB") || exit 3
REF=$(pack "$REF_SB")   || exit 3

# Frame check BEFORE the games: a large stdev split means the arms differ in
# scale as well as knowledge, which confounds the Elo.
for sb in "$CAND_SB" "$REF_SB"; do
    sd=$("$BIN" nnue-eval "$(readlink -f "$OUT/sb${sb}-${SCALE}.cenn")" \
         < tools/nnue/calib/wl50.fens 2>/dev/null | grep -oE '^-?[0-9]+' \
         | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')" 2>/dev/null)
    log "sb${sb} eval stdev=${sd:-unavailable}"
done

log "VERDICT sb${CAND_SB} vs WARM POINT sb${REF_SB} — $GAMES games @ ${NODES}n"
TV="$ARCHON_D49/warmpoint-sb${CAND_SB}-vs-${REF_SB}.out"
ENGINE="$BIN" CONC="$CONC" bash tools/screen.sh "$CAND" "$REF" "$GAMES" "$NODES" > "$TV" 2>&1
{ echo "=== sb${CAND_SB} vs WARM POINT sb${REF_SB} ==="; grep -A5 'Results of' "$TV" | tail -5; } >> "$OPS_LOG"
log "done -> $TV"

#!/usr/bin/env bash
# tools/ops/match/scale_sweep.sh — find the best `network_scale` for a checkpoint.
#
# PURPOSE
#   `network_scale` is a free MULTIPLIER on the packed net's output. It has a
#   measured INTERIOR optimum — too small and too large both cost Elo — and that
#   optimum was fitted for whatever net was canonical at the time. A new net
#   inherits nothing. This sweeps it and ranks the arms.
#
# USAGE
#   SB=1758 bash tools/ops/match/scale_sweep.sh
#   SB=1758 SCALES="600 756 900 1068" GAMES_PER_PAIR=600 bash tools/ops/match/scale_sweep.sh
#
#   Environment:
#     SB               checkpoint superbatch (REQUIRED)
#     CKPT_PREFIX      default d49sfb        RUN_SPEC default d49_sfblend_anneal.json
#     SCALES           default "600 756 900 1068"
#     GAMES_PER_PAIR   default 600      NODES default 25000     CONC default 3
#
# WHY A ROUND-ROBIN, NOT N SCREENS AGAINST CANONICAL
#   Screening each arm separately against canonical spends every game measuring
#   the arm-vs-canonical gap, which we do not care about, and then compares two
#   noisy differences to rank two arms. A round-robin spends every game on a
#   comparison BETWEEN arms, which is the quantity we actually want, and each
#   game informs the whole ranking. Same games, much better resolution on the
#   thing being decided.
#
# WHY FIXED NODES
#   Scale changes the eval SPREAD, which changes how the hardcoded search margins
#   bite, which changes the node count per move. At a real clock that confounds
#   scale with speed. Fixed nodes removes it.
#
# READ THE RESULT WITH THE BAR IN MIND
#   Pairwise bar is 383/sqrt(games_per_pair). At 600 that is +/-15.6 Elo. Scale
#   effects measured previously were ~12-20 Elo across the useful range, so this
#   resolves the ENDS of the range confidently and adjacent arms only weakly.
#   **If the top two land inside each other's bars, they are tied — pick the one
#   closer to the incumbent scale rather than inventing a winner.**

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

SB=${SB:?SB (checkpoint superbatch) required}
CKPT_PREFIX=${CKPT_PREFIX:-d49sfb}
RUN_SPEC=${RUN_SPEC:-d49_sfblend_anneal.json}
SCALES=${SCALES:-"600 756 900 1068"}
GAMES_PER_PAIR=${GAMES_PER_PAIR:-600}
NODES=${NODES:-25000}
CONC=${CONC:-3}
EXP_DIR=${EXP_DIR:-$ARCHON_ROOT/data/nnue/experiments/d49-1280}
OUT="$ARCHON_ROOT/data/nnue/tmp/scale"; mkdir -p "$OUT"
OPS_LOG=${OPS_LOG:-$ARCHON_D49/scale_sweep.log}; export OPS_LOG
BIN=$(asset target-sota/release/chessengine || asset engines/bin/archon-main-6b59a8cc)
FASTCHESS=$(asset engines/bin/fastchess)
BOOK=$(asset engines/books/UHO_Lichess_4852_v1.epd)
CKPT="$EXP_DIR/$CKPT_PREFIX-sb${SB}-ema-raw.bin"
[ -s "$CKPT" ] || { log "FATAL: checkpoint missing: $CKPT"; exit 2; }

log "scale sweep: sb${SB}  scales='${SCALES}'  ${GAMES_PER_PAIR} games/pair @ ${NODES}n"

# One requant, many packs: the quantised intermediate is scale-independent, so
# packing N scales costs one requant plus N cheap packs.
uv run --no-project --with numpy==2.5.1 python tools/nnue/requant_threats.py \
    "$CKPT" "$OUT/q${SB}.bin" --scale 256 --run-spec "tools/nnue/runs/$RUN_SPEC" >/dev/null 2>&1 \
    || { log "FATAL: requant failed"; exit 3; }

ENGINE_ARGS=()
for s in $SCALES; do
    cenn="$OUT/${CKPT_PREFIX}-sb${SB}-${s}.cenn"
    D40_ACC=1280 D40_L2=16 D40_L3=32 D40_POLICY=1 D40_POL_DIM=16 \
    D40_THREAT_QUANT=256 D40_NETWORK_SCALE="$s" \
      uv run --no-project --with numpy==2.5.1 python tools/nnue/netpack.py \
        "$OUT/q${SB}.bin" "$cenn" d40 >/dev/null 2>&1 \
        || { log "FATAL: netpack failed at scale $s"; exit 3; }
    sd=$("$BIN" nnue-eval "$(readlink -f "$cenn")" < tools/nnue/calib/wl50.fens 2>/dev/null \
         | grep -oE '^-?[0-9]+' \
         | python3 -c "import sys,statistics as st;v=[int(x) for x in sys.stdin];print(f'{st.pstdev(v):.1f}')" 2>/dev/null)
    log "  packed scale $s  (eval stdev ${sd:-?})"
    ENGINE_ARGS+=(-engine cmd="$BIN" name="s${s}" proto=uci option.EvalFile="$(readlink -f "$cenn")")
done

N=$(echo $SCALES | wc -w)
PAIRS=$(( N*(N-1)/2 ))
ROUNDS=$(( GAMES_PER_PAIR * PAIRS / 2 ))
log "  ${N} arms, ${PAIRS} pairs, round-robin, ${ROUNDS} rounds (~$((ROUNDS*2)) games total)"
log "  pairwise bar = +/-$(python3 -c "print(f'{383/($GAMES_PER_PAIR**0.5):.1f}')") Elo"

SW="$ARCHON_D49/scale-sb${SB}.out"
"$FASTCHESS" "${ENGINE_ARGS[@]}" \
  -each nodes="$NODES" tc=inf \
  -openings file="$BOOK" format=epd order=random \
  -games 2 -rounds "$ROUNDS" -repeat -concurrency "$CONC" \
  -recover -ratinginterval 200 > "$SW" 2>&1

{ echo "=== SCALE SWEEP sb${SB} ==="; grep -A"$((N+4))" 'Rank Name' "$SW" | tail -"$((N+3))"; } | tee -a "$OPS_LOG"
log "done -> $SW"

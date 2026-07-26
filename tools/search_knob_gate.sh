#!/usr/bin/env bash
# search_knob_gate.sh — POST-D47 gate for the 2026-07-26 search-gap knobs.
#
# The two candidates (docs/research/value-foundation-state.md ~05:3x entry):
#   NMP  : NMP_EVAL_BASE 374->140, NMP_EVAL_DEPTH 14->12  (SF16-shape bar,
#          eligibility crossover d~27 -> d~12; measured tree x3.11 -> x2.15)
#   NMP+LMR : + RED_BASE 1027->2051 (+1 ply base reduction; -> x1.06)
#
# Two-stage gate, both self-play vs the UNMODIFIED build (same net, policy on):
#   stage 1  fixed 50k nodes, 200g  — sanity: harder pruning must not collapse
#            quality at equal nodes (expect >= -20 Elo; it SHOULD gain: the
#            saved nodes become depth)
#   stage 2  10+0.1 wall-clock, 200g — the real verdict (pruning pays here)
# Pre-registered: node parity != Elo; NMP guards SF16 carries that we lack
# (eval>=beta, statScore damper) are the first fix if stage 1 fails on
# tactics. Requires an IDLE BOX (no trainer) for stage 2.
#
#   bash tools/search_knob_gate.sh <net.cenn> nmp|both [games] [conc]
set -uo pipefail
cd "$(dirname "$0")/.."
NET="${1:?net .cenn}"
ARM="${2:?nmp|both}"
GAMES="${3:-200}"
CONC="${4:-4}"
ENG=engines/bin/archon-tune          # sota+tune build; rebuild after any pull
BOOK=engines/books/UHO_Lichess_4852_v1.epd

OPTS=(option.NMP_EVAL_BASE=140 option.NMP_EVAL_DEPTH=12)
[ "$ARM" = both ] && OPTS+=(option.RED_BASE=2051)

run() {
  local tag="$1"; shift
  local out="data/nnue/experiments/knobgate-$ARM-$tag"
  mkdir -p "$out"
  engines/bin/fastchess \
    -engine cmd="$ENG" name=knob "option.EvalFile=$NET" "${OPTS[@]}" \
    -engine cmd="$ENG" name=base "option.EvalFile=$NET" \
    -each proto=uci "$@" option.Threads=1 option.Hash=16 \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds $((GAMES/2)) -concurrency "$CONC" -recover \
    -draw movenumber=40 movecount=8 score=15 \
    -resign movecount=4 score=700 -maxmoves 250 \
    -srand 20260726 -pgnout file="$out/games.pgn" notation=lan \
    -log file="$out/fastchess.log" level=warn 2>&1 | tail -22 | tee "$out/stdout.log"
}

echo "=== STAGE 1: fixed 50k nodes | arm=$ARM | ${GAMES}g ==="
run nodes tc=inf nodes=50000
echo
echo "=== STAGE 2: 10+0.1 wall-clock | arm=$ARM | ${GAMES}g (IDLE BOX ONLY) ==="
run et tc=10+0.1

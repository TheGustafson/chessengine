#!/usr/bin/env bash
# BEFORE/AFTER snapshot vs SF16 with the ARCHON policy head FULLY ON
# (sidecar .cpol auto-loaded for the TT-substitute path + P-ORD quiet-move
# prior via CPL_ORD_W). Measured at EQUAL TIME and EQUAL NODES.
#
# Re-run this EXACT script after the v1.5 warm leg, after the optimization
# pass, and after SPSA to get a comparable progression. Keep games/seed/book
# fixed across runs or the comparison is meaningless.
#
#   bash tools/nnue/sf_baseline.sh <net.cenn> <tag> [games] [concurrency]
#
# The equal-time leg REQUIRES AN IDLE BOX (no trainer, no other matches).
set -uo pipefail
cd "$(dirname "$0")/../.."
NET="${1:?net .cenn (needs a sibling .cpol for policy-on)}"
TAG="${2:?tag, e.g. d47p-pre-v15}"
GAMES="${3:-200}"
CONC="${4:-4}"
PORD=engines/bin/archon-d47-pord10
SF=engines/bin/stockfish16
BOOK=engines/books/UHO_Lichess_4852_v1.epd
[ -s "${NET%.cenn}.cpol" ] || echo "WARNING: no sibling .cpol -> policy head OFF"
run() {
  local out="data/nnue/experiments/$1"; shift
  mkdir -p "$out"
  engines/bin/fastchess \
    -engine cmd="$PORD" name=archon "option.EvalFile=$NET" \
    -engine cmd="$SF" name=sf16 \
    -each proto=uci "$@" option.Threads=1 option.Hash=16 \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds $((GAMES/2)) -concurrency "$CONC" -recover \
    -draw movenumber=40 movecount=8 score=15 \
    -resign movecount=4 score=700 -maxmoves 250 \
    -srand 20260726 -pgnout file="$out/games.pgn" notation=lan \
    -log file="$out/fastchess.log" level=warn 2>&1 | tail -22 | tee "$out/stdout.log"
}
echo "=== EQUAL TIME 10+0.1 | ${GAMES}g conc ${CONC} | ARCHON(policy ON) vs SF16 ==="
run "sfbase-$TAG-et" tc=10+0.1
echo
echo "=== EQUAL NODES 50000 | ${GAMES}g conc ${CONC} ==="
run "sfbase-$TAG-nodes" tc=inf nodes=50000

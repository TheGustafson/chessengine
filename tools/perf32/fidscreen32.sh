#!/usr/bin/env bash
# perf study 32 -- measure the STRENGTH cost of an i8 quantisation scheme with
# zero engine work.
#
# The offline eval-delta battery says how far the score moves; it cannot say
# what that costs in Elo.  This can, and it needs no build: both sides run the
# SAME binary and differ only in EvalFile, so a fixed-node screen of the
# i8-simulating net against the reference net IS the fidelity cost, isolated
# from every implementation risk.
#
#   bash tools/perf32/fidscreen32.sh <scheme> [games] [conc]
set -euo pipefail
cd "$(dirname "$0")/../.."
SCHEME=${1:?scheme, e.g. col}
GAMES=${2:-1466}
CONC=${3:-6}
V=/home/nick/projects/chessengine/data/nnue/nets/perf32
CAND=$V/d48x-i8-$SCHEME.cenn
OUT=$PWD/engines/runs/s32-$SCHEME
mkdir -p "$OUT"
CONC=$CONC OUT=$OUT nohup bash tools/screen.sh "$CAND" engines/canonical/net.cenn \
  "$GAMES" 25000 > "$V/screen-$SCHEME.log" 2>&1 &
echo "started $SCHEME pid $! -> $V/screen-$SCHEME.log"

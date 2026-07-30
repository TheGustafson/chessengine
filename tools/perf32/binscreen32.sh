#!/usr/bin/env bash
# perf study 32 — the definitive strength gate: the LANDED binary against the
# pre-change binary, same net file, FIXED NODES.
#
#   bash tools/perf32/binscreen32.sh <cand-bin> <ref-bin> [games] [conc]
#
# Fixed nodes hides the throughput win on purpose. What this measures is the
# only thing perf cannot: whether halving the feature-transformer row costs
# strength. Bar = 383/sqrt(n) (tools/screen.sh derives it), so 1466 games is
# +/-10 Elo and nothing under that is a verdict.
set -euo pipefail
cd "$(dirname "$0")/../.."
CAND=${1:?candidate binary}
REF=${2:?reference binary}
GAMES=${3:-1466}
CONC=${4:-6}
NODES=${5:-25000}
NET=$(readlink -f engines/canonical/net.cenn)
BOOK=$PWD/engines/books/UHO_Lichess_4852_v1.epd
OUT=$PWD/engines/runs/s32-bin
mkdir -p "$OUT"
python3 -c "import math;print(f'bar ~+/-{383/math.sqrt($GAMES):.1f} Elo at $GAMES games')"
exec "$PWD/engines/bin/fastchess" \
    -engine cmd="$(readlink -f "$CAND")" name=I8 option.Hash=16 option.Threads=1 \
        option.EvalFile="$NET" \
    -engine cmd="$(readlink -f "$REF")" name=BASE option.Hash=16 option.Threads=1 \
        option.EvalFile="$NET" \
    -each proto=uci nodes="$NODES" \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds $((GAMES / 2)) -maxmoves 250 \
    -concurrency "$CONC" -recover -ratinginterval 500 \
    -pgnout file="$OUT/games.pgn" \
    -log file="$OUT/fastchess.log" level=warn \
    2>&1 | tee "$OUT/run.out"

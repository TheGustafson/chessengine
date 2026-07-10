#!/usr/bin/env bash
# tools/gauntlet.sh — calibration gauntlet vs reference opponents (design/14 §3.8, §4 step 11).
#
# Gauntlets tell us WHERE WE ARE on the rating ladder; they are never the
# merge gate (that is tools/sprt.sh vs our own previous version). The engine
# plays every opponent over the same openings; ratings come from ordo (with
# error bars) over the produced PGN.
#
# Usage:
#   tools/gauntlet.sh [ENGINE] [extra fastchess args...]
#
#   ENGINE defaults to target/release/chessengine.
#
# Opponents — two sources, combined:
#   1. Calibrated Stockfish (installed via `apt-get install stockfish`,
#      /usr/games/stockfish, SF16 here): one gauntlet opponent per Elo in
#      $SF_ELOS using UCI_LimitStrength + UCI_Elo (SF16 supports 1320..3190).
#      NOTE: UCI_Elo is CCRL-blitz-anchored only approximately, and
#      limit-strength SF plays with an unnatural style; treat these rungs as
#      coarse calibration until real reference engines are pinned.
#   2. Real reference engines (the doc 14 §3.8 ladder: TSCP ~1700,
#      Vice ~2000, Fruit 2.1 ~2700, Stash tags 1300..3400): build/download
#      into engines/ (gitignored) and pass via $OPPONENTS, semicolon-
#      separated "name=path" pairs:
#        OPPONENTS="tscp=engines/tscp/tscp;stash-v10=engines/stash/stash-v10" tools/gauntlet.sh
#
# Tunables (env):
#   TC=10+0.1              time control
#   HASH=16                Hash MB where the engine supports it
#   ROUNDS=125             opening rounds per pairing (x2 games: 250
#                          games/opponent; >= 1000 total at 4 opponents,
#                          the doc 14 §3.8 rung size)
#   SF_ELOS="1320 1500 1700 1900"   Stockfish rungs ("" to disable)
#   STOCKFISH=/usr/games/stockfish
#   CONCURRENCY=           default: physical cores - 1
#   BOOK=tests/data/openings/openings-8ply.epd
#   OUT=engines/runs/gauntlet-<ts>   output dir (PGN, log, ordo ratings; gitignored)
#
# Rating: if ordo is available ($ORDO, PATH, engines/bin/ordo — built from
# https://github.com/michiguel/Ordo, `make`), it is run on the PGN with the
# candidate anchored at 2000; record the table in docs/strength-log.md.
set -euo pipefail
cd "$(dirname "$0")/.."

ENGINE="${1:-target/release/chessengine}"
[ $# -ge 1 ] && shift
[ -x "$ENGINE" ] || { echo "error: engine '$ENGINE' not found (cargo build --release)" >&2; exit 1; }

FASTCHESS="${FASTCHESS:-$(command -v fastchess || true)}"
[ -n "$FASTCHESS" ] || { [ -x engines/bin/fastchess ] && FASTCHESS=engines/bin/fastchess; }
[ -n "${FASTCHESS:-}" ] || { echo "error: fastchess not found — see tools/sprt.sh header" >&2; exit 1; }

STOCKFISH="${STOCKFISH:-$(command -v stockfish || echo /usr/games/stockfish)}"
SF_ELOS="${SF_ELOS-1320 1500 1700 1900}"
TC="${TC:-10+0.1}"
HASH="${HASH:-16}"
ROUNDS="${ROUNDS:-125}"
BOOK="${BOOK:-tests/data/openings/openings-8ply.epd}"
OUT="${OUT:-engines/runs/gauntlet-$(date +%Y%m%d-%H%M%S)}"

physical_cores() {
    if command -v lscpu > /dev/null; then
        lscpu -p=CORE,SOCKET 2> /dev/null | grep -v '^#' | sort -u | wc -l
    else
        nproc
    fi
}
if [ -z "${CONCURRENCY:-}" ]; then
    CONCURRENCY=$(($(physical_cores) - 1))
    [ "$CONCURRENCY" -ge 1 ] || CONCURRENCY=1
fi

# --- assemble opponents ------------------------------------------------------
opponent_args=()
if [ -n "$SF_ELOS" ]; then
    [ -x "$STOCKFISH" ] || { echo "error: stockfish not found at '$STOCKFISH' (apt-get install stockfish, or SF_ELOS='' with \$OPPONENTS)" >&2; exit 1; }
    for elo in $SF_ELOS; do
        opponent_args+=(-engine cmd="$STOCKFISH" name="sf-elo$elo"
            option.UCI_LimitStrength=true option.UCI_Elo="$elo")
    done
fi
if [ -n "${OPPONENTS:-}" ]; then
    IFS=';' read -ra pairs <<< "$OPPONENTS"
    for pair in "${pairs[@]}"; do
        name=${pair%%=*}
        path=${pair#*=}
        [ -x "$path" ] || { echo "error: opponent '$path' not found" >&2; exit 1; }
        opponent_args+=(-engine cmd="$path" name="$name")
    done
fi
[ ${#opponent_args[@]} -gt 0 ] || { echo "error: no opponents (set SF_ELOS and/or OPPONENTS)" >&2; exit 1; }

mkdir -p "$OUT"
echo "gauntlet: $ENGINE vs [${SF_ELOS:+SF@ }$SF_ELOS${OPPONENTS:+ + $OPPONENTS}]"
echo "tc=$TC rounds=$ROUNDS (x2 games/opponent) concurrency=$CONCURRENCY book=$BOOK"
echo "output: $OUT/"

"$FASTCHESS" \
    -tournament gauntlet -seeds 1 \
    -engine cmd="$ENGINE" name=candidate option.Hash="$HASH" option.Threads=1 \
    "${opponent_args[@]}" \
    -each proto=uci tc="$TC" \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds "$ROUNDS" \
    -draw movenumber=40 movecount=8 score=15 \
    -maxmoves 250 \
    -concurrency "$CONCURRENCY" -recover -ratinginterval 25 \
    -pgnout file="$OUT/games.pgn" \
    -log file="$OUT/fastchess.log" level=warn \
    -config outname="$OUT/config.json" \
    "$@" 2>&1 | tee "$OUT/gauntlet.out"

# --- ordo ratings (error bars) ----------------------------------------------
ORDO="${ORDO:-$(command -v ordo || true)}"
[ -n "$ORDO" ] || { [ -x engines/bin/ordo ] && ORDO=engines/bin/ordo; }
if [ -n "${ORDO:-}" ] && [ -s "$OUT/games.pgn" ]; then
    # -a 2000 -A candidate: anchor our engine at 2000 so opponent ratings
    # read as offsets; -s100 simulations give the error bars; -W white
    # advantage auto; -G forces calculation when a swept pairing (all
    # wins/losses) would otherwise disconnect the gauntlet's star graph —
    # such rungs show as +/-inf and just mean "outside this rung's range".
    "$ORDO" -q -a 2000 -A candidate -W -G -s100 -p "$OUT/games.pgn" -o "$OUT/ordo.txt"
    cat "$OUT/ordo.txt"
    echo "record the rating row in docs/strength-log.md (design/14 §5)"
else
    echo "note: ordo not found — build it from https://github.com/michiguel/Ordo into engines/bin/ for rated output"
fi

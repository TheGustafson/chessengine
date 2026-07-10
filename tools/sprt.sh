#!/usr/bin/env bash
# tools/sprt.sh — SPRT strength test between two engine builds (design/14 §3.8, D25).
#
# The merge gate for functional search/eval changes: candidate vs baseline
# under a Sequential Probability Ratio Test with fastchess (pentanomial
# GSPRT, normalized Elo). Strength claims come from games only — a verdict
# (H1/H0) gets one row in docs/strength-log.md (design/14 §5).
#
# Usage:
#   tools/sprt.sh CANDIDATE BASELINE [extra fastchess args...]
#
# Both arguments are engine binary paths. Build the baseline from the merge
# base, the candidate from your branch, with IDENTICAL RUSTFLAGS (D16), e.g.:
#   git worktree add /tmp/base <merge-base> && (cd /tmp/base && cargo build --release)
#
# Tunables (env vars):
#   TC=10+0.1          time control (10+0.1 = our STC; confirm milestones at 60+0.6)
#   NODES=             if set, fixed-node proxy games instead of TC
#                      (tc=inf nodes=N: immune to machine load/SMT — never
#                      valid for time-management changes, design/14 §3.8)
#   ELO0=0 ELO1=5      nElo hypothesis bounds; simplifications use [-5, 0]:
#                      ELO0=-5 ELO1=0 tools/sprt.sh ...
#   HASH=16            Hash MB per engine (16 STC / 64 LTC)
#   ROUNDS=25000       max opening rounds (2 games each); SPRT stops earlier
#   CONCURRENCY=       default: physical cores - 1 (never SMT siblings for
#                      timed games — hyperthread contention corrupts clocks)
#   BOOK=tests/data/openings/openings-8ply.epd
#                      committed 43-line balanced book. For serious runs use
#                      an UHO book (fishtest-style unbalanced openings):
#                        mkdir -p engines/books && cd engines/books
#                        curl -sfLO https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip
#                        unzip UHO_Lichess_4852_v1.epd.zip
#                      then BOOK=engines/books/UHO_Lichess_4852_v1.epd
#   OUT=engines/runs/sprt-<ts>   output dir (PGN + log; gitignored)
#
# Match runner: fastchess (pinned v1.8.0). Resolution order: $FASTCHESS,
# PATH, engines/bin/fastchess. Install (documented in TESTING.md):
#   mkdir -p engines/bin && cd /tmp
#   curl -sfLO https://github.com/Disservin/fastchess/releases/download/v1.8.0-alpha/fastchess-linux-x86-64.tar
#   tar xf fastchess-linux-x86-64.tar
#   cp fastchess-linux-x86-64/fastchess <repo>/engines/bin/
# (cutechess-cli is the fallback runner but is not packaged for this distro;
# it only offers logistic-Elo SPRT — bounds translate to roughly [0, 3]
# logistic, design/14 §9.3.)
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; }

[ $# -ge 2 ] || { usage; exit 1; }
CANDIDATE=$1
BASELINE=$2
shift 2

for bin in "$CANDIDATE" "$BASELINE"; do
    [ -x "$bin" ] || { echo "error: engine '$bin' not found/executable" >&2; exit 1; }
done

# --- resolve fastchess -------------------------------------------------------
FASTCHESS="${FASTCHESS:-$(command -v fastchess || true)}"
[ -n "$FASTCHESS" ] || { [ -x engines/bin/fastchess ] && FASTCHESS=engines/bin/fastchess; }
[ -n "${FASTCHESS:-}" ] || {
    echo "error: fastchess not found (\$FASTCHESS, PATH, engines/bin/) — see header for install" >&2
    exit 1
}

# --- honest concurrency: physical cores - 1 (design/14 §3.8) ----------------
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

TC="${TC:-10+0.1}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
HASH="${HASH:-16}"
ROUNDS="${ROUNDS:-25000}"
BOOK="${BOOK:-tests/data/openings/openings-8ply.epd}"
OUT="${OUT:-engines/runs/sprt-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

# Fixed-node proxy (design/14 §3.8): removes clock jitter; never valid for
# time-management changes.
if [ -n "${NODES:-}" ]; then
    limit_args=(tc=inf nodes="$NODES")
else
    limit_args=(tc="$TC")
fi

echo "SPRT [$ELO0, $ELO1] nElo  $CANDIDATE vs $BASELINE"
echo "limit: ${limit_args[*]}  Hash=$HASH  concurrency=$CONCURRENCY  book=$BOOK"
echo "output: $OUT/  (record the verdict in docs/strength-log.md, D25)"

"$FASTCHESS" \
    -engine cmd="$CANDIDATE" name=candidate \
    -engine cmd="$BASELINE" name=baseline \
    -each proto=uci "${limit_args[@]}" option.Hash="$HASH" option.Threads=1 \
    -openings file="$BOOK" format=epd order=random \
    -repeat -games 2 -rounds "$ROUNDS" \
    -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 model=normalized \
    -draw movenumber=34 movecount=8 score=20 \
    -resign movecount=3 score=600 \
    -concurrency "$CONCURRENCY" -recover -ratinginterval 10 \
    -pgnout file="$OUT/games.pgn" \
    -log file="$OUT/fastchess.log" level=warn \
    -config outname="$OUT/config.json" \
    "$@" 2>&1 | tee "$OUT/sprt.out"

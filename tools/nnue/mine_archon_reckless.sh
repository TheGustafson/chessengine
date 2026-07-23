#!/usr/bin/env bash
# Mine ARCHON-vs-RECKLESS hard negatives (user 2026-07-20). Reckless 0.9.0 is a
# top-2 engine with a DIFFERENT style than SF18 -> ARCHON loses on DIFFERENT
# positions -> broader weakness coverage. Deep-label these with SF18@50k after
# (tools/nnue/deeplabel_*.sh). Same shape as mine_archon_losses.sh.
#   mine_archon_reckless.sh <rounds> <concurrency> <workdir> [limit]
set -uo pipefail
cd "$(dirname "$0")/../.."
ROUNDS="${1:?rounds}"; CONC="${2:?concurrency}"; WORK="${3:?workdir}"; LIMIT="${4:-nodes=20000}"
ARCHON=engines/bin/archon-sota-native
NET=data/nnue/nets/d40-uho-720sb.cenn
RECKLESS=engines/bin/reckless
BOOK=engines/books/UHO_Lichess_4852_v1.epd
FC=engines/bin/fastchess
mkdir -p "$WORK"; PGN="$WORK/games.pgn"
for f in "$ARCHON" "$NET" "$RECKLESS" "$BOOK" "$FC"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
say(){ echo "[mine-reckless $(date -u +%H:%M:%S)] $*"; }

say "fastchess: ARCHON vs Reckless, $ROUNDS rounds x2, conc=$CONC, limit=$LIMIT"
"$FC" \
  -engine cmd="$ARCHON" name=ARCHON option.EvalFile="$NET" \
  -engine cmd="$RECKLESS" name=Reckless \
  -each "$LIMIT" option.Hash=64 option.Threads=1 \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" \
  -resign movecount=4 score=500 twosided=true -draw movenumber=40 movecount=8 score=10 \
  -pgnout file="$PGN" notation=uci -recover >> "$WORK/fastchess.log" 2>&1
say "games -> $PGN ($(grep -c '^\[Event' "$PGN" 2>/dev/null || echo 0))"
say "mining ARCHON-loss decision points"
uv run --no-project --with chess python tools/nnue/pgn_mine_losses.py "$PGN" --archon-name ARCHON --out "$WORK/mined.tsv"
say "DONE: $WORK/mined.tsv ($(wc -l < "$WORK/mined.tsv" 2>/dev/null || echo 0) positions). Deep-label with SF18 next."

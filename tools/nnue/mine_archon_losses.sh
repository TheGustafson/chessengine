#!/usr/bin/env bash
# Mine high-value training positions from ARCHON-vs-SF18 games (user, 2026-07-19).
# SF18 crushes ARCHON — the positions ARCHON loses are where our eval is most
# wrong, i.e. the highest information-per-position data. We generate bulk games
# fast, extract the losing-side decision points, and (next step) deep-label them
# with SF18 @ 50k nodes into the corpus.
#
#   mine_archon_losses.sh <rounds> <concurrency> <workdir> [limit]
#
# <limit> is a fastchess -each token: "nodes=40000" (LOAD-IMMUNE — safe to run
# while the box is busy) or "tc=8+0.08" (needs a quiet box). Default nodes=40000.
# Emits <workdir>/mined.tsv  (FEN\twdl, White-relative game result) — same shape
# as the fleet's c1_orig.tsv, so the fleet deep-label + WDL-merge + bf-convert
# path builds the corpus unchanged.
#
# Deep-label step (run after, LOCAL, when the box has cores):
#   cut -f1 <workdir>/mined.tsv > <workdir>/mined.fens
#   target/release/root-analysis --engine engines/stockfish-18-src/src/stockfish \
#       --fens <workdir>/mined.fens --nodes 50000 --reference-only \
#       --out <workdir>/mined.ceobs
#   target/release/obs-tool export-value <workdir>/mined.vals <workdir>/mined.ceobs
#   # then WDL-merge mined.tsv by FEN + bf-convert (see build_corpus_from_shards.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
ROUNDS="${1:?rounds}"; CONC="${2:?concurrency}"; WORK="${3:?workdir}"; LIMIT="${4:-nodes=20000}"
ARCHON=engines/bin/archon-sota-native
NET=data/nnue/nets/d40-uho-720sb.cenn
SF18=engines/bin/stockfish18
BOOK=engines/books/UHO_Lichess_4852_v1.epd
FC=engines/bin/fastchess
mkdir -p "$WORK"
PGN="$WORK/games.pgn"
say(){ echo "[mine $(date -u +%H:%M:%S)] $*"; }

for f in "$ARCHON" "$SF18" "$BOOK" "$FC"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

say "fastchess: ARCHON vs SF18, $ROUNDS rounds x2, conc=$CONC, limit=$LIMIT"
"$FC" \
  -engine cmd="$ARCHON" name=ARCHON option.EvalFile="$NET" \
  -engine cmd="$SF18" name=SF18 \
  -each "$LIMIT" option.Hash=64 option.Threads=1 \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" \
  -resign movecount=4 score=500 twosided=true -draw movenumber=40 movecount=8 score=10 \
  -pgnout file="$PGN" notation=uci \
  -recover >> "$WORK/fastchess.log" 2>&1
say "games -> $PGN ($(grep -c '^\[Event' "$PGN" 2>/dev/null || echo 0) games)"

say "extracting ARCHON-loss decision points"
# keep ALL positions (no eval/loss filtering — data-hungry; SF18 labels every one)
uv run --no-project --with chess python tools/nnue/pgn_extract_all.py "$PGN" --out "$WORK/mined.tsv" --min-ply 0
say "DONE: $WORK/mined.tsv ($(wc -l < "$WORK/mined.tsv" 2>/dev/null || echo 0) positions). Deep-label next (see header)."

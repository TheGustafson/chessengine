#!/usr/bin/env bash
# Six-engine round robin (user-commissioned 2026-07-22): Reckless + SF16 +
# the ARCHON lineage (PSQT / best-d40 / d47-stage2 / d47-stage2-nopol).
# ~1020 games (34 rounds x 15 pairings x 2), 10+0.1, UHO book, conc 8.
# Run ONLY with the GPU/trainer idle and >4G RAM free (post-arc window).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=engines/bin
BOOK=engines/books/UHO_Lichess_4852_v1.epd
NET_D40=data/nnue/nets/d40-uho-v1-240sb.cenn
NET_D47=data/nnue/nets/d47c-stage2-sb240.cenn
NET_D47_NOPOL=data/nnue/nets/ablate/d47c-stage2-sb240-nopol.cenn
OUT=data/nnue/experiments/rr-lineage-$(date +%m%d)
mkdir -p "$OUT" data/nnue/nets/ablate
[ -s "$NET_D47" ] || { echo "missing $NET_D47 (finale pack first)"; exit 2; }
[ -s "$NET_D47_NOPOL" ] || cp "$NET_D47" "$NET_D47_NOPOL"   # no .cpol beside it => policy OFF
ROUNDS="${1:-34}"

"$BIN/fastchess" \
  -engine cmd="$BIN/reckless" name=reckless \
  -engine cmd="$BIN/stockfish16" name=sf16 \
  -engine cmd="$BIN/archon-d47" name=archon-d47s2 option.EvalFile="$PWD/$NET_D47" \
  -engine cmd="$BIN/archon-d47" name=archon-d47s2-nopol option.EvalFile="$PWD/$NET_D47_NOPOL" \
  -engine cmd="$BIN/archon-sota-native" name=archon-d40 option.EvalFile="$PWD/$NET_D40" \
  -engine cmd="$BIN/archon-d47" name=archon-psqt \
  -each tc=10+0.1 option.Threads=1 option.Hash=64 \
  -tournament roundrobin -rounds "$ROUNDS" -games 2 -concurrency 8 \
  -openings file="$BOOK" format=epd order=random \
  -pgnout file="$OUT/games.pgn" \
  -ratinginterval 30 2>&1 | tee "$OUT/fastchess.log" | tail -40

grep -aE "^Rank|^\s+[0-9]+\s" "$OUT/fastchess.log" | tail -10 || true
echo "PGN: $OUT/games.pgn  — run ordo for the final crosstable:"
echo "  $BIN/ordo -Q -p $OUT/games.pgn -a 3190 -A sf16 -o $OUT/ordo.txt && cat $OUT/ordo.txt"

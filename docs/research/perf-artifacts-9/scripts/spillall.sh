#!/usr/bin/env bash
# Dump + spill-analyse every hot symbol of a perf.data in one pass.
#   bash spillall.sh <perf.data> <outdir>
set -euo pipefail
DATA=${1:?perf.data}
OUT=${2:?outdir}
HERE=$(dirname "$0")
mkdir -p "$OUT"

syms=(
  '<chessengine::search::Worker>::search::<chessengine::search::NonPv>|search-nonpv'
  '<chessengine::eval::nnue::sota_threats::ThreatContext>::at_full|at_full'
  '<chessengine::movepick::MovePicker>::init_quiets|init_quiets'
  '<chessengine::board::Board>::make_move_with::<chessengine::search::TtPrefetch>|make_move_with'
  '<chessengine::eval::nnue::sota_accumulator::HalfkaContext>::current_full|current_full'
  '<chessengine::search::Worker>::do_move|do_move'
  '<chessengine::eval::nnue::sota_head::SotaHead>::forward_into|forward_into'
  '<chessengine::search::Worker>::qsearch::<chessengine::search::NonPv>|qsearch'
  'chessengine::eval::nnue::sota_threats::record_square_touch|record_square_touch'
)

for entry in "${syms[@]}"; do
  sym=${entry%|*}
  tag=${entry##*|}
  perf annotate -i "$DATA" --stdio -s "$sym" --percent-limit 0 2>/dev/null > "$OUT/annot-$tag.txt"
  echo "===== $tag"
  python3 "$HERE/spill8.py" "$OUT/annot-$tag.txt"
done

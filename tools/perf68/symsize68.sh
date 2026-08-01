#!/usr/bin/env bash
# perf study 68 — hot-body byte sizes for a fixed symbol list, one binary per
# column. With the machine function splitter on, `nm` reports the HOT body of a
# split function under its own name and the cold remainder as `<name>.cold`;
# this prints the hot body only, which is the span the frame actually fetches.
#
#   bash tools/perf68/symsize68.sh <label> [label...]
set -euo pipefail
cd "$(dirname "$0")/../.."
SYMS=(
  'search::<chessengine::search::NonPv>'
  'search::<chessengine::search::Pv>'
  'qsearch::<chessengine::search::NonPv>'
  'qsearch::<chessengine::search::Pv>'
  'iterative_deepening'
  'MovePicker>::init_quiets'
  'MovePicker>::init_captures'
  'i8_deltas_into_inner'
  'HalfkaContext>::finny_refresh_into'
  'SotaHead>::forward_into'
)
printf '%-46s' 'symbol (hot body bytes)'
for L in "$@"; do printf '%14s' "$L"; done
printf '%14s\n' 'sum-cold'
for s in "${SYMS[@]}"; do
  printf '%-46s' "${s:0:45}"
  for L in "$@"; do
    v=$(nm -SC "engines/bin/$L" | grep -F -- "$s" | grep -v '\.cold' \
        | awk '{print strtonum("0x"$2)}' | head -1)
    printf '%14s' "${v:-0}"
  done
  printf '\n'
done
printf '%-46s' '.text total'
for L in "$@"; do
  printf '%14s' "$(readelf -SW "engines/bin/$L" | awk '$2==".text"{print strtonum("0x"$6)}')"
done
printf '\n'
printf '%-46s' 'split-out cold bytes (n symbols)'
for L in "$@"; do
  printf '%14s' "$(nm -S "engines/bin/$L" | grep '\.cold' | awk '{s+=strtonum("0x"$2)} END{printf "%d/%d", s+0, NR}')"
done
printf '\n'

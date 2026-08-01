#!/usr/bin/env bash
# perf study 69 — per-symbol SIZE diff between two binaries, plus `.text`.
#   bash tools/perf69/symdiff69.sh <labelA> <labelB> <symbol-substring> ...
set -euo pipefail
cd "$(dirname "$0")/../.."
A=engines/bin/${1:?a}; B=engines/bin/${2:?b}; shift 2
printf '%-52s %10s %10s %8s\n' symbol A B delta
for s in "$@"; do
  a=$(nm -SC "$A" | grep -F -- "$s" | grep -v '\.cold' | head -1 | awk '{print strtonum("0x" $2)}')
  b=$(nm -SC "$B" | grep -F -- "$s" | grep -v '\.cold' | head -1 | awk '{print strtonum("0x" $2)}')
  printf '%-52s %10s %10s %+8d\n' "${s:0:52}" "${a:-?}" "${b:-?}" "$(( ${b:-0} - ${a:-0} ))"
done
ta=$(size -A "$A" | awk '/^\.text/{print $2}')
tb=$(size -A "$B" | awk '/^\.text/{print $2}')
printf '%-52s %10s %10s %+8d\n' '.text' "$ta" "$tb" "$((tb - ta))"

#!/usr/bin/env bash
# perf study 69 — disassemble a window around a symbol+offset branch site.
#   bash tools/perf69/dis69.sh <binary> <symbol-substring> <hex-offset> [before] [after]
# `mispagg69.py` names sites as `<symbol>+0xNNN`; this prints the machine code
# around one so the branch can be read.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${1:?binary}; S=${2:?symbol substring}; OFF=${3:?hex offset}; BEF=${4:-64}; AFT=${5:-32}
read -r START SIZE <<<"$(nm -SC "$B" | grep -F -- "$S" | grep -v '\.cold' \
  | awk '{print $1, $2}' | head -1)"
[ -n "${START:-}" ] || { echo "no symbol matching '$S' in $B" >&2; exit 1; }
A=$((0x$START + OFF))
objdump -d --start-address=$((A - BEF)) --stop-address=$((A + AFT)) "$B" \
  | sed -e "s/^ *$(printf %x $A):/>>> $(printf %x $A):/"

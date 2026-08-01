#!/usr/bin/env bash
# perf study 69 — the offsets of every instruction inside a symbol that
# references one exact `static+offset` (e.g. a `LazyLock`'s `Once` state word).
#   bash tools/perf69/loadoffs69.sh <label> <symbol-substring> <static+off>
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?label}; S=${2:?symbol substring}; ST=${3:?static+offset}
B="engines/bin/$L"
read -r START SIZE <<<"$(nm -SC "$B" | grep -F -- "$S" | grep -v '\.cold' \
  | awk '{print $1, $2}' | head -1)"
END=$((0x$START + 0x$SIZE))
objdump -d --start-address="0x$START" --stop-address="$END" "$B" \
  | awk -v base=$((0x$START)) -v st="$ST" '
    index($0, st) > 0 { split($1, a, ":"); printf "+0x%x\n", strtonum("0x" a[1]) - base }'

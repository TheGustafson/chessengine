#!/usr/bin/env bash
# perf study 18 (instrument inherited from 17) — dump one symbol's disassembly from a binary.
#
#   bash dis17.sh <binary> <symbol-regex> [outfile]
#
# `objdump -d` over the whole binary once, then slice out the symbol.  Static,
# so it says what the compiler EMITTED, not what ran -- pair it with a counter.
set -euo pipefail
BIN=${1:?binary}
SYM=${2:?symbol regex}
OUT=${3:-/dev/stdout}
objdump -d --no-show-raw-insn -C "$BIN" \
  | awk -v pat="$SYM" '
      /^[0-9a-f]+ </ { inside = ($0 ~ pat) }
      inside { print }
    ' > "$OUT"

#!/usr/bin/env bash
# Map a list of hex text addresses back to source lines (inlining chain).
#   bash where8.sh <binary-with-line-tables> <hexaddr>...
set -euo pipefail
BIN=${1:?binary}
shift
for a in "$@"; do
  printf '%-8s ' "$a"
  addr2line -e "$BIN" -f -C -i "0x$a" 2>/dev/null | tr '\n' ' ' | cut -c1-200
  echo
done

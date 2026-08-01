#!/usr/bin/env bash
# perf study 69 — the OFFSET histogram of every reference to one named static
# inside a symbol. `statics68.sh` says "60 references to TABLES"; this says
# WHICH BYTES of it, which is how the `LazyLock` state word is told apart from
# the magic data it guards.
#   bash tools/perf69/refoffs69.sh <label> <symbol-substring> <static-substring>
set -euo pipefail
cd "$(dirname "$0")/../.."
L=${1:?label}; S=${2:?symbol substring}; ST=${3:?static substring}
B="engines/bin/$L"
read -r START SIZE <<<"$(nm -SC "$B" | grep -F -- "$S" | grep -v '\.cold' \
  | awk '{print $1, $2}' | head -1)"
END=$((0x$START + 0x$SIZE))
objdump -d --start-address="0x$START" --stop-address="$END" "$B" \
  | grep -oE "# [0-9a-f]+ <[^>]*${ST}[^>]*>" \
  | sed -E "s/.*${ST}//; s/>$//" \
  | sort | uniq -c | sort -rn

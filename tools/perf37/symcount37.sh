#!/usr/bin/env bash
# perf study 37 — STATIC instruction census of one symbol in one binary.
#
#   bash tools/perf37/symcount37.sh <binary> <symbol-substring>
#
# Why static and not `perf annotate`: study 36 §2.3 established that
# `instructions:u` per-INSTRUCTION attribution is NOT a census on this box (a
# 20x skew among instructions that retire equally often). A disassembly count
# has no sampling in it at all, so it answers "did the codegen change, and how"
# exactly — which is the whole question for an ATTRIBUTE arm, where the source
# is identical and only LLVM's choices differ.
#
# Prints the symbol's address, byte size, total instruction count and the
# mnemonic histogram, so two binaries can be diffed line for line.
set -euo pipefail
BIN=${1:?binary}
SYM=${2:?symbol substring}

read -r ADDR SIZE NAME < <(
  nm -C "$BIN" | awk '$2 ~ /^[tT]$/' | sort | awk -v want="$SYM" '
    NR > 1 && index(prev_name, want) {
      printf "%s %d %s\n", prev_addr, strtonum("0x" $1) - strtonum("0x" prev_addr), prev_name
      exit
    }
    { prev_addr = $1; $1 = ""; $2 = ""; prev_name = $0 }
  '
)
[ -n "${ADDR:-}" ] || { echo "symbol matching '$SYM' not found in $BIN" >&2; exit 1; }

END=$(printf '%x' $(( 0x$ADDR + SIZE )))
echo "== $BIN"
echo "   symbol : $NAME"
echo "   address: 0x$ADDR .. 0x$END   bytes: $SIZE"
objdump -d --start-address="0x$ADDR" --stop-address="0x$END" "$BIN" > /tmp/symcount37.$$ || true
printf '   instrs : %s\n' "$(grep -cP '^\s+[0-9a-f]+:' /tmp/symcount37.$$)"
echo "   mnemonics:"
grep -oP '^\s+[0-9a-f]+:\s+(?:[0-9a-f]{2} )+\s*\K[a-z0-9]+' /tmp/symcount37.$$ \
  | sort | uniq -c | sort -rn | head -20 | sed 's/^/     /'
rm -f /tmp/symcount37.$$

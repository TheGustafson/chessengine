#!/usr/bin/env bash
# perf study 33 -- second half of the instrument null (see textid33.sh).
#
#   bash tools/perf33/symdiff33.sh <labelA> <labelB>
#
# textid33 hashes raw .text bytes, which differ whenever a RIP-relative
# displacement moves even though every opcode is identical.  This is the
# sharper test: compare (symbol size, symbol name) for every symbol, then
# compare a NORMALISED disassembly of `search::<NonPv>` in which addresses,
# raw byte columns and RIP-relative targets are stripped, leaving only the
# opcode + operand-shape stream.  If that stream is identical the two builds
# execute the same instructions in the same order and a per-line profile taken
# on one is a profile of the other.
set -euo pipefail
cd "$(dirname "$0")/../.."
A=${1:?labelA}
B=${2:?labelB}
# The v0 mangled name embeds the crate DISAMBIGUATOR (`Cs<hash>_`), which is
# derived from `-C metadata` and therefore differs between cargo profiles.  So
# the symbol is resolved per binary by pattern, not quoted literally.
PAT=${3:-6Worker6search.*5NonPv}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for l in "$A" "$B"; do
  # (size, name) only -- addresses are expected to move.  The disambiguator is
  # normalised out of the name so the two profiles' symbol tables are
  # comparable at all.
  nm -S "bin33/$l" 2>/dev/null | awk 'NF>=4{print $2, $4}' \
    | sed -E 's/Cs[0-9A-Za-z]+_/Cs_/g' | sort > "$tmp/$l.sym"
  sym=$(nm "bin33/$l" 2>/dev/null | awk '{print $NF}' | grep -E "$PAT" | head -1)
  [ -n "$sym" ] || { echo "no symbol matching /$PAT/ in bin33/$l" >&2; exit 1; }
  echo "$l: $sym" >&2
  # The ENCODED BYTES of the target symbol, one instruction per line.  Symbol
  # names and absolute addresses in objdump's annotation column differ between
  # profiles for reasons that have nothing to do with codegen (the v0
  # disambiguator, and a 32-byte shift of the whole .text), so the annotation
  # is discarded and only the machine code is compared.
  objdump -d --disassemble="$sym" "bin33/$l" 2>/dev/null \
    | grep -oP '^\s+[0-9a-f]+:\t\K[0-9a-f ]+(?=\t)' \
    | sed -e 's/[[:space:]]*$//' > "$tmp/$l.dis"
done

nsym=$(diff "$tmp/$A.sym" "$tmp/$B.sym" | grep -c '^[<>]' || true)
ndis=$(diff "$tmp/$A.dis" "$tmp/$B.dis" | grep -c '^[<>]' || true)
printf 'symbols (size,name): %d differing lines of %d\n' "$nsym" "$(wc -l < "$tmp/$A.sym")"
printf 'search::<NonPv> instruction bytes: %d differing lines of %d\n' \
  "$ndis" "$(wc -l < "$tmp/$A.dis")"
if [ "$ndis" -eq 0 ]; then
  echo "INSTRUCTION BYTES IDENTICAL -- line tables are codegen-inert at the target symbol."
else
  echo "INSTRUCTION STREAM DIFFERS -- investigate before trusting the attribution."
  diff "$tmp/$A.dis" "$tmp/$B.dis" | head -20
fi

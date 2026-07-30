#!/usr/bin/env bash
# perf study 33 -- THE NULL FOR THE SOURCE-ATTRIBUTION INSTRUMENT.
#
#   bash tools/perf33/textid33.sh <labelA> <labelB>
#
# Study 33's primary instrument is per-source-line + per-inline-frame
# attribution inside `search::<NonPv>`, read off a build carrying DWARF line
# tables.  The whole instrument rests on ONE claim: adding line tables does not
# change codegen.  This script *proves* it for the two binaries in hand by
# hashing the bytes of the executable sections, not by asserting it.
#
# If the two .text hashes match, the profiling build IS the release build plus
# metadata, and every share it reports is a share of the shipped binary.  If
# they differ, the instrument is perturbing and every number from it needs a
# separate null -- say so rather than proceeding.
set -euo pipefail
cd "$(dirname "$0")/../.."
A=${1:?labelA}
B=${2:?labelB}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for sec in .text .rodata; do
  for l in "$A" "$B"; do
    objcopy -O binary --only-section="$sec" "bin33/$l" "$tmp/$l$sec" 2>/dev/null
    printf '%-10s %-8s %10d bytes  %s\n' "$l" "$sec" \
      "$(stat -c%s "$tmp/$l$sec")" "$(sha256sum "$tmp/$l$sec" | cut -c1-24)"
  done
done

if cmp -s "$tmp/$A.text" "$tmp/$B.text"; then
  echo "TEXT IDENTICAL -- the line-table build is codegen-identical to release."
else
  echo "TEXT DIFFERS -- the instrument is NOT inert; it needs its own null."
fi

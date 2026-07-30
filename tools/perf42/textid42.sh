#!/usr/bin/env bash
# perf study 42 -- .text identity between two binaries given by PATH.
#
#   bash tools/perf42/textid42.sh <pathA> <pathB>
#
# Study 33's textid33.sh with the bin33/ prefix removed, so it can compare a
# release build against a `profiling` build living in a different directory.
# This is the NULL for every source-attributed number in this study: if the
# line-table build is byte-identical in .text, its attribution is the shipped
# binary's attribution.
set -euo pipefail
A=${1:?pathA}
B=${2:?pathB}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for sec in .text .rodata; do
  for l in "$A" "$B"; do
    n=$(basename "$l")
    objcopy -O binary --only-section="$sec" "$l" "$tmp/$n$sec" 2>/dev/null
    printf '%-14s %-8s %10d bytes  %s\n' "$n" "$sec" \
      "$(stat -c%s "$tmp/$n$sec")" "$(sha256sum "$tmp/$n$sec" | cut -c1-24)"
  done
done

if cmp -s "$tmp/$(basename "$A").text" "$tmp/$(basename "$B").text"; then
  echo "TEXT IDENTICAL -- the line-table build is codegen-identical to release."
else
  echo "TEXT DIFFERS -- the instrument is NOT inert; it needs its own null."
fi

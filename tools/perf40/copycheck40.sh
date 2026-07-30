#!/usr/bin/env bash
# perf study 40 — read the pure-copy fold's CODEGEN off each binary.
#
#   bash tools/perf40/copycheck40.sh <label>...
#
# Study 38 §3 recorded that the SAME source text compiles the 2560-byte
# accumulator copy to a `memcpy` CALL in one context and 40 inline zmm moves in
# another, and that nothing was watching which. This prints both facts per
# binary: the libc call-site census restricted to `at_full` (study 38's tool)
# and the count of `vmovups` in that symbol (80 == one inlined 2560 B copy).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p bin34/tmp40
for l in "$@"; do
  objdump -d "bin34/$l" >"bin34/tmp40/$l.asm"
  calls=$(python3 tools/perf38/memcpy38.py "bin34/tmp40/$l.asm" | grep -c at_full || true)
  vmov=$(python3 tools/perf40/symscan40.py "bin34/tmp40/$l.asm" ThreatContext7at_full |
         sed -n "s/.*('vmovups', \([0-9]*\)).*/\1/p")
  printf '%-8s at_full: libc-call refs=%-3s vmovups=%s\n' "$l" "$calls" "${vmov:-?}"
done

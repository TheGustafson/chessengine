#!/usr/bin/env bash
# perf study 66 — the wl50 eval hash for one binary in each of the three data
# modes. Bit-exactness is the gate the whole arm rests on, and the three modes
# share a `.text` byte for byte, so a difference here can only be the LAYOUT.
#   bash tools/perf66/hash66.sh <binary>
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}
NET=${2:-engines/canonical/net.cenn}
for m in ${P66_MODES:-ARCHON_NARROW=0 ARCHON_PACK=0 ARCHON_PACK=1}; do
  h=$(env "$m" "$BIN" nnue-eval "$NET" < tools/nnue/calib/wl50.fens | sha256sum | cut -c1-16)
  printf '%-16s %s\n' "$m" "$h"
done

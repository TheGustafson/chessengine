#!/usr/bin/env bash
# perf study 20 — split a total instruction count into STARTUP + per-node.
#
#   bash docs/research/perf-artifacts-20/scripts/startup20.sh <binA> <binB>
#
# An `i/n` delta at ONE node budget cannot distinguish "the hot path got more
# expensive" from "the loader got more expensive and is being amortised over the
# same tree".  Two budgets separate them: run each binary at ~0 nodes and at the
# study frame, and solve the two-point line.  (Study 20 arm 2 needed this — its
# +42 i/n turned out to be almost entirely net-loading work.)
set -euo pipefail
one() {
  local bin=$1 budget=$2 t
  t=$(mktemp)
  nice -n 19 perf stat -x, -e instructions:u -o "$t.p" -- \
    taskset -c 6 "bin20/$bin" bench 16 1 "$budget" tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn > /dev/null 2> "$t.e"
  local ins nodes
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$t.p")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$t.e" | tail -1)
  rm -f "$t" "$t.p" "$t.e"
  echo "$ins $nodes"
}
for bin in "$@"; do
  lo=$(one "$bin" 1)
  hi=$(one "$bin" 100000)
  python3 - "$bin" $lo $hi <<'PY'
import sys
bin, i0, n0, i1, n1 = sys.argv[1], *[int(x) for x in sys.argv[2:]]
slope = (i1 - i0) / (n1 - n0)
print(f'{bin:22s} startup={i0/1e9:7.3f} G  marginal_i/n={slope:8.1f}  '
      f'reported_i/n@{n1}={i1/n1:8.1f}')
PY
done

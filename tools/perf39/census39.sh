#!/usr/bin/env bash
# perf study 39 — the WORKLOAD census, which is what says whether a pinning
# probe changed only the locality of the work or the work itself.
#
#   bash tools/perf39/census39.sh [budget]
#
# Runs the `foldcount` build with and without `probe-actpin`, in both pin-fill
# modes, and prints calls-per-node for every accumulator site. If a pinning
# probe leaves those rates alone it moved only locality and a per-node (or
# per-instruction) delta is meaningful; if it moved them, the probe changed
# the workload and NO normalisation repairs that.
set -euo pipefail
cd "$(dirname "$0")/../.."
BUDGET=${1:-100000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P39_CPU:-2}
OUTDIR=docs/research/perf-artifacts-39
mkdir -p "$OUTDIR"

run() { # label binary env...
  local label=$1 bin=$2; shift 2
  local f="$OUTDIR/census-$label.txt"
  # shellcheck disable=SC2086
  env ARCHON_FOLDCOUNT="$f" "$@" taskset -c "$CPU" "./bin39/$bin" \
    bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>&1
  python3 - "$f" "$label" <<'PY'
import sys
path, label = sys.argv[1:3]
kv = {}
for line in open(path):
    for tok in line.split()[1:]:
        if '=' in tok:
            k, v = tok.split('=', 1)
            kv[k] = v
n = int(kv['nodes'])
sites = ['hka_into', 'hka_dual', 'hka_inplace', 'thr_into', 'thr_copy', 'activate', 'eval']
cells = '  '.join(f'{s.split("_")[-1] if s!="eval" else "eval"}={int(kv[s+".calls"])/n:.4f}' for s in sites)
rows = '  '.join(f'{s}R={int(kv[s+".rows"])/n:.4f}' for s in ('hka_into', 'hka_dual', 'thr_into'))
print(f'{label:14s} nodes={n:<9d} {cells}   {rows}')
PY
}

run fc-base      fcbase   env
run fc-actpin-0  fcactpin env P39_PINFILL=zero
run fc-actpin-L  fcactpin env P39_PINFILL=live

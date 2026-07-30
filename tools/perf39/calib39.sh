#!/usr/bin/env bash
# perf study 39 — what the generic `L1-dcache-*` events actually COUNT on
# Zen 4, calibrated against traffic whose geometry is known exactly.
#
#   bash tools/perf39/calib39.sh [budget]
#
# `probe-addcopy` moves exactly `DOSE × 5120 B` read + the same written per
# evaluated node, so the dose difference is a ruler with no modelling in it.
# Study 29 §2.1 calibrated `L1-dcache-loads` against issued zmm loads and found
# 2.0015 events per load (Zen 4 double-pumps a 512-bit access on a 256-bit
# datapath). It never calibrated the MISS event, and both studies quote a
# "misses per line" off it.
#
# Raw AMD events counted alongside so the mapping is visible rather than
# assumed:
#   ls_dispatch.ld_dispatch / .store_dispatch — loads and stores DISPATCHED
#   ls_mab_alloc.load_store_allocations       — miss-address buffers allocated,
#                                               i.e. distinct 64-byte lines a
#                                               demand access had to go fetch
set -euo pipefail
cd "$(dirname "$0")/../.."
BUDGET=${1:-20000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P39_CPU:-2}
EV=L1-dcache-loads:u,L1-dcache-load-misses:u,ls_dispatch.ld_dispatch:u,ls_dispatch.store_dispatch:u,ls_mab_alloc.load_store_allocations:u
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for dose in 0 4; do
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp/d$dose" -- \
    env P39_DOSE=$dose P39_RING=16 taskset -c "$CPU" ./bin39/addcopy \
    bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>"$tmp/bench"
  grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1 > "$tmp/n$dose"
done

python3 - "$tmp" <<'PY'
import sys
t = sys.argv[1]
KEYS = ['L1-dcache-loads', 'L1-dcache-load-misses', 'ls_dispatch.ld_dispatch',
        'ls_dispatch.store_dispatch', 'ls_mab_alloc.load_store_allocations']
def read(d):
    n = int(open(f'{t}/n{d}').read())
    v = {}
    for line in open(f'{t}/d{d}'):
        f = line.split(',')
        if len(f) > 2 and f[2].split(':')[0] in KEYS:
            v[f[2].split(':')[0]] = float(f[0]) / n
    return v
a, b = read(0), read(4)
# Geometry: 4 doses x 5120 B read + 5120 B written per EVALUATED node.
# `activate` fires 0.4311 times per node (tools/perf39/census39.sh, fc-base).
EVALS = 0.4311
lines_r = 4 * 80 * EVALS      # 64-byte lines READ per node
lines_w = 4 * 80 * EVALS      # 64-byte lines WRITTEN per node
print(f'\n  dose 4 − dose 0, against {lines_r:.2f} lines read + {lines_w:.2f} written per node')
print(f'  (i.e. {4*80*EVALS:.2f} zmm loads and {4*80*EVALS:.2f} zmm stores per node)\n')
for k in KEYS:
    d = b[k] - a[k]
    print(f'  {k:28s} Δ/node={d:9.2f}   per line READ={d/lines_r:6.3f}   '
          f'per line R+W={d/(lines_r+lines_w):6.3f}')
PY

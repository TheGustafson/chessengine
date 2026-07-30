#!/usr/bin/env bash
# tools/perf27/panel27.sh <base> <arm> [rounds]
#
# THREE-WAY ROUND-ROBIN with the null carried alongside the arm, after study
# 23's panel3-23.sh. base / a byte-identical copy of base / arm, cycled, with
# the ORDER ROTATED inside each round (the first slot of a round gets the boost
# clock -- study 23 measured a byte-identical B slower in 10 of 16 rounds under
# a block-structured panel).
#
# Reports the median relative delta for all four counters. The NULL row is the
# error bar: an arm delta only means something if it clears it. Instructions and
# branches should read EXACTLY +0.000% on the null row -- if they do not, the
# binaries differ or the node count moved, and nothing else on the panel is
# readable.
#
# Never min-of-N (study 22 measured min-of-N at -2.07% on identical binaries).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

BASE=${1:?base label in perf27/bin/}
ARM=${2:?arm label in perf27/bin/}
N=${3:-9}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
CPU=${P27_CPU:-7}
NULL="${BASE}__nullcopy"
cp "$ROOT/perf27/bin/$BASE" "$ROOT/perf27/bin/$NULL"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"; rm -f "$ROOT/perf27/bin/$NULL"' EXIT
raw=$tmp/raw

run() {
  local bin=$1
  nice -n 19 perf stat -x, \
    -e instructions:u,branches:u,branch-misses:u,L1-dcache-load-misses:u \
    -o "$tmp/stat" -- \
    taskset -c "$CPU" "$ROOT/perf27/bin/$bin" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  local nodes ins br bm l1
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  ins=$(awk -F, '$3~/^instructions/{print $1}' "$tmp/stat")
  br=$(awk -F, '$3~/^branches:/{print $1}' "$tmp/stat")
  bm=$(awk -F, '$3~/^branch-misses/{print $1}' "$tmp/stat")
  l1=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp/stat")
  echo "$nodes $ins $br $bm $l1"
}

for r in $(seq "$N"); do
  case $((r % 3)) in
    0) order=(A N B) ;;
    1) order=(B A N) ;;
    *) order=(N B A) ;;
  esac
  for slot in "${order[@]}"; do
    case $slot in A) bin=$BASE ;; N) bin=$NULL ;; B) bin=$ARM ;; esac
    echo "$r $slot $(run "$bin")" >> "$raw"
  done
  printf 'panel27: round %s/%s\n' "$r" "$N" >&2
done

python3 - "$raw" "$BASE" "$ARM" <<'PY' | tee -a "$ROOT/docs/research/perf-artifacts-27/panel.txt"
import collections, statistics, sys
raw, na, nb = sys.argv[1:4]
COUNTERS = ['instructions', 'branches', 'branch-misses', 'L1d-load-misses']
rounds = collections.defaultdict(dict)
for line in open(raw):
    r, slot, nodes, *vals = line.split()
    rounds[int(r)][slot] = [float(v) / float(nodes) for v in vals]
rs = [v for v in rounds.values() if len(v) == 3]
print(f'  panel {na} -> {nb}   rounds={len(rs)}')
for slot, label in (('N', 'NULL'), ('B', 'ARM ')):
    parts = []
    for i, c in enumerate(COUNTERS):
        d = sorted(100.0 * (v[slot][i] - v['A'][i]) / v['A'][i] for v in rs)
        parts.append(f'{c}={statistics.median(d):+7.3f}% [{d[0]:+7.3f},{d[-1]:+7.3f}]')
    print(f'    {label}  ' + '  '.join(parts))
print('    VERDICT: an arm delta is real only if it clears the NULL row.')
PY

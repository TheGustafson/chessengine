#!/usr/bin/env bash
# tools/perf31/stalls31.sh <base> <arm> [rounds]
#
# The LOAD-STALL panel: the counter study 28 used to show that 82.8% of our
# excess over Stockfish is the CPU idle waiting for load data.  It is a COUNT,
# not a timing, so it tolerates contention far better than `cycles:u` -- study
# 28 §1.3 admits `ex_no_retire.*` / `L1-dcache-*` for ratios above ~5% where it
# admits `cycles:u` only above ~10%.
#
# THREE-WAY ROUND-ROBIN with the order rotated inside each round, and the NULL
# is a byte-identical copy of the base carried through the same rotation.  The
# null row IS the error bar (studies 22/24/25 all read a drift as a result;
# study 24's cycles null hit -6.167% with a 44-point spread).  Slot order is
# randomised per round, not merely rotated: study 25 §0.2 measured that a
# FIXED-order round-robin cancels monotone drift but not curvature, and its
# fixed-order null came back +2.06% while its randomised one came back -1.93%.
#
# Events, per node:
#   ex_no_retire.load_not_complete  -- cycles in which nothing retired because
#                                      load data had not arrived (THE metric)
#   cycles / instructions           -- context; cycles quoted only with its null
#   L1-dcache-loads / -load-misses  -- volume and rate, the pair (study 27)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
BASE=${1:?base label}
ARM=${2:?arm label}
N=${3:-10}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
CPU=${P31_CPU:-6}
NULL="${BASE}__null"
cp "$ROOT/bin31/$BASE" "$ROOT/bin31/$NULL"
mkdir -p "$ROOT/docs/research/perf-artifacts-31"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"; rm -f "$ROOT/bin31/$NULL"' EXIT
raw=$tmp/raw

run() {
  nice -n 19 perf stat -x, \
    -e cycles:u,instructions:u,ex_no_retire.load_not_complete:u,L1-dcache-loads:u,L1-dcache-load-misses:u \
    -o "$tmp/stat" -- \
    taskset -c "$CPU" "$ROOT/bin31/$1" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  local nodes
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  awk -F, -v n="$nodes" '
    $3~/^cycles/{c=$1} $3~/^instructions/{i=$1}
    $3~/^ex_no_retire.load_not_complete/{s=$1}
    $3~/^L1-dcache-loads/{ld=$1} $3~/^L1-dcache-load-misses/{ms=$1}
    END{printf "%s %.4f %.4f %.4f %.4f %.4f\n", n, c/n, i/n, s/n, ld/n, ms/n}
  ' "$tmp/stat"
}

for r in $(seq "$N"); do
  for slot in $(printf 'A\nN\nB\n' | shuf); do
    case $slot in A) bin=$BASE ;; N) bin=$NULL ;; B) bin=$ARM ;; esac
    echo "$r $slot $(run "$bin")" >> "$raw"
  done
  printf 'stalls31: round %s/%s\n' "$r" "$N" >&2
done

python3 - "$raw" "$BASE" "$ARM" <<'PY' | tee -a "$ROOT/docs/research/perf-artifacts-31/stalls.txt"
import collections, statistics, sys
raw, na, nb = sys.argv[1:4]
COLS = ['nodes', 'cyc', 'ins', 'loadstall', 'l1load', 'l1miss']
rounds = collections.defaultdict(dict)
for line in open(raw):
    f = line.split()
    rounds[int(f[0])][f[1]] = dict(zip(COLS, map(float, f[2:])))
rs = [v for v in rounds.values() if len(v) == 3]
print(f'\n  stalls31  {na} -> {nb}   rounds={len(rs)}  (node counts: '
      f'{sorted({int(v[s]["nodes"]) for v in rs for s in v})})')
print(f'  {"counter":12s} {"base median":>13s} | '
      f'{"NULL %":>9s} {"n<0":>6s} | {"ARM %":>9s} {"n<0":>6s} | {"ARM abs":>10s}')
for c in COLS[1:]:
    base = statistics.median(v['A'][c] for v in rs)
    out = [f'  {c:12s} {base:13.2f} |']
    for slot in ('N', 'B'):
        d = [100.0 * (v[slot][c] - v['A'][c]) / v['A'][c] for v in rs]
        out.append(f'{statistics.median(d):+8.3f}% {sum(x<0 for x in d):3d}/{len(d)} |')
    arm = statistics.median(v['B'][c] - v['A'][c] for v in rs)
    out.append(f'{arm:+10.2f}')
    print(' '.join(out))
print('  An ARM row is a result only if it clears the NULL row on the same counter.')
PY

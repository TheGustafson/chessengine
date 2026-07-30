#!/usr/bin/env bash
# tools/perf27/cyclepanel27.sh <base> <arm> [rounds]
#
# A CYCLES panel, used for exactly one thing: adjudicating an arm whose
# instruction and cache counters point in OPPOSITE directions, where no other
# instrument can decide.
#
# READ THIS BEFORE QUOTING A NUMBER FROM IT. Cycles are not reliably measurable
# on this box: four independent studies measured byte-identical binaries at
# -3.40 / -0.07 / -1.51 / +1.42 % (study 22), a 44-point spread (study 24),
# +0.97 / +2.06 / -1.93 % (study 25), and -1.653% (study 26, quiet box,
# interleaved). So the NULL ROW IS THE ERROR BAR and it is measured in the same
# minutes, in the same slot rotation, as the arm. An arm delta that does not
# clear the null row is NOT a result in either direction -- study 22 read a
# -1.16% as a win and had to retract it.
#
# Three-way round-robin, order rotated within each round (the first run of a
# round gets the boost clock). Median, never min-of-N.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

BASE=${1:?base label}
ARM=${2:?arm label}
N=${3:-12}
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
CPU=${P27_CPU:-7}
NULL="${BASE}__nullcopy"
cp "$ROOT/perf27/bin/$BASE" "$ROOT/perf27/bin/$NULL"

tmp=$(mktemp -d); trap 'rm -rf "$tmp"; rm -f "$ROOT/perf27/bin/$NULL"' EXIT
raw=$tmp/raw

run() {
  nice -n 19 perf stat -x, -e cycles:u,instructions:u -o "$tmp/stat" -- \
    taskset -c "$CPU" "$ROOT/perf27/bin/$1" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  local nodes cyc
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  cyc=$(awk -F, '$3~/^cycles/{print $1}' "$tmp/stat")
  echo "$cyc $nodes"
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
  printf 'cyclepanel27: round %s/%s\n' "$r" "$N" >&2
done

python3 - "$raw" "$BASE" "$ARM" <<'PY' | tee -a "$ROOT/docs/research/perf-artifacts-27/cycles.txt"
import collections, statistics, sys
raw, na, nb = sys.argv[1:4]
rounds = collections.defaultdict(dict)
for line in open(raw):
    r, slot, cyc, nodes = line.split()
    rounds[int(r)][slot] = float(cyc) / float(nodes)
rs = [v for v in rounds.values() if len(v) == 3]
print(f'  cycles panel {na} -> {nb}   rounds={len(rs)}')
for slot, label in (('N', 'NULL (byte-identical copy)'), ('B', f'ARM  ({nb})')):
    d = sorted(100.0 * (v[slot] - v['A']) / v['A'] for v in rs)
    wins = sum(1 for x in d if x < 0)
    print(f'    {label:28s} median={statistics.median(d):+7.3f}%  '
          f'faster_in={wins}/{len(d)}  min={d[0]:+7.3f}%  max={d[-1]:+7.3f}%')
print('    VERDICT: an arm delta is real only if it clears the NULL row.')
PY

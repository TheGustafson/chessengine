#!/usr/bin/env bash
# perf study 22 — THE THREE-WAY PANEL.  A, its own COPY, and B, round-robin.
#
#   bash docs/research/perf-artifacts-22/scripts/panel3-22.sh <binA> <binAcopy> <binB> <rounds>
#
# WHY.  panel22.sh interleaves two binaries, which cancels drift *within* a
# panel but not *between* panels — and on this box the null itself moved from
# -0.07% to -1.51% over about two hours as sibling load changed.  A null
# measured at 21:30 therefore does not bound an arm measured at 23:00.  This
# runs the null and the arm in ONE round-robin so both see the same drift, and
# reports PAIRED per-round deltas (round i of B minus round i of A), which is
# the statistic that actually cancels a slow drift.  `all_same_sign` over the
# paired deltas is the significance read, not the median gap.
#
# Frame: bench 16 1 100000 wl50.fens nodes D48X = a fixed 5,001,562-node tree.
set -euo pipefail
A=${1:?binA}
NUL=${2:?binA-copy}
B=${3:?binB}
N=${4:-10}
CORE=${CORE:-2}
CMD=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn)
EV=cycles:u,instructions:u,L1-dcache-load-misses:u,l2_cache_req_stat.ls_rd_blk_c:u
run() {
  local bin=$1 tmp
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp.perf" -- \
    taskset -c "$CORE" "bin22/$bin" "${CMD[@]}" > /dev/null 2> "$tmp.eng"
  local cyc ins l1m l2m nodes
  cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  l1m=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp.perf")
  l2m=$(awk -F, '$3~/^l2_cache_req_stat.ls_rd_blk_c/{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  echo "$cyc $ins $l1m $l2m $nodes"
}
raw=$(mktemp)
run "$A" > /dev/null                       # discard the page-cache-cold round
for _ in $(seq "$N"); do
  echo "A $(run "$A")"   >> "$raw"
  echo "N $(run "$NUL")" >> "$raw"
  echo "B $(run "$B")"   >> "$raw"
done
python3 - "$raw" "$A" "$NUL" "$B" <<'PY' | tee -a docs/research/perf-artifacts-22/panel3.txt
import statistics, sys
raw, na, nn, nb = sys.argv[1:5]
rows = {'A': [], 'N': [], 'B': []}
for line in open(raw):
    t, *v = line.split()
    rows[t].append([float(x) for x in v])
NODES = 4
names = {'A': na, 'N': nn, 'B': nb}
def med(t, i):
    return statistics.median(r[i] / r[NODES] for r in rows[t])
print(f'--- 3-way round-robin, n={len(rows["A"])} per arm, core '
      f'{__import__("os").environ.get("CORE", "2")} ---')
for t in ('A', 'N', 'B'):
    print(f'{t} {names[t]:24s} cyc={med(t,0):8.1f} ins={med(t,1):8.1f} '
          f'l1m={med(t,2):7.2f} l2m={med(t,3):7.2f}')
for t, label in (('N', 'NULL  (same binary)'), ('B', 'ARM')):
    pairs = sorted(x[0] / x[NODES] - a[0] / a[NODES] for a, x in zip(rows['A'], rows[t]))
    same = all(p < 0 for p in pairs) or all(p > 0 for p in pairs)
    nneg = sum(1 for p in pairs if p < 0)
    print(f'  {label:20s} paired dcyc: median={statistics.median(pairs):+8.1f} '
          f'({100 * statistics.median(pairs) / med("A", 0):+.2f}%) '
          f'[{pairs[0]:+.0f}, {pairs[-1]:+.0f}]  neg={nneg}/{len(pairs)} '
          f'same_sign={same}')
    print(f'  {"":20s} medians: dcyc={med(t,0)-med("A",0):+8.1f} '
          f'({100*(med(t,0)-med("A",0))/med("A",0):+.2f}%) '
          f'dins={med(t,1)-med("A",1):+7.2f} ({100*(med(t,1)-med("A",1))/med("A",1):+.3f}%) '
          f'dl1m={med(t,2)-med("A",2):+6.2f} dl2m={med(t,3)-med("A",3):+6.2f}')
print(f'  nodes={sorted({int(r[NODES]) for t in rows for r in rows[t]})}')
PY
rm -f "$raw"

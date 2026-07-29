#!/usr/bin/env bash
# perf study 20 — INTERLEAVED A/B cycles panel.
#
#   bash docs/research/perf-artifacts-20/scripts/ab20.sh <binA> <binB> <rounds>
#
# The box hosts a GPU trainer, so cycle counts drift over minutes.  Two panels
# run back to back are therefore NOT comparable at the few-percent level; this
# alternates A,B,A,B,… inside one invocation so any drift hits both arms
# equally, and reports the median of each plus the paired per-round deltas.
#
# Same fixed 5,001,562-node frame as probe20.sh.  Instructions are printed but
# are NOT the gate here — this script exists for the arms whose mechanism is
# memory latency, where instructions rise by construction.
set -euo pipefail
A=${1:?binA}
B=${2:?binB}
N=${3:-9}
CMD=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn)
EV=cycles:u,instructions:u,L1-dcache-load-misses:u,ls_misal_loads.ma64:u
run() {
  local bin=$1 tmp
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp.perf" -- \
    taskset -c 6 "bin20/$bin" "${CMD[@]}" > /dev/null 2> "$tmp.eng"
  local cyc ins l1m mis nodes
  cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  l1m=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp.perf")
  mis=$(awk -F, '$3~/^ls_misal_loads.ma64/{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  echo "$cyc $ins $l1m $mis $nodes"
}
raw=$(mktemp)
for _ in $(seq "$N"); do
  echo "A $(run "$A")" >> "$raw"
  echo "B $(run "$B")" >> "$raw"
done
python3 - "$raw" "$A" "$B" <<'PY' | tee -a docs/research/perf-artifacts-20/ab.txt
import statistics, sys
raw, na, nb = sys.argv[1], sys.argv[2], sys.argv[3]
rows = {'A': [], 'B': []}
for line in open(raw):
    t, cyc, ins, l1m, mis, nodes = line.split()
    rows[t].append(tuple(float(x) for x in (cyc, ins, l1m, mis, nodes)))
def med(t, i):
    return statistics.median(r[i] / r[4] for r in rows[t])
for t, name in (('A', na), ('B', nb)):
    print(f'{t} {name:30s} n={len(rows[t])} cyc_n={med(t,0):8.1f} ins_n={med(t,1):8.1f} '
          f'ipc={med(t,1)/med(t,0):5.3f} l1m_n={med(t,2):7.2f} ma64_n={med(t,3):8.2f}')
pairs = [b[0]/b[4] - a[0]/a[4] for a, b in zip(rows['A'], rows['B'])]
pairs.sort()
print(f'  paired dcyc_n (B-A): median={statistics.median(pairs):+8.1f} '
      f'min={pairs[0]:+8.1f} max={pairs[-1]:+8.1f} '
      f'all_same_sign={all(p < 0 for p in pairs) or all(p > 0 for p in pairs)}')
print(f'  medians: dcyc_n={med("B",0)-med("A",0):+8.1f} ({100*(med("B",0)-med("A",0))/med("A",0):+.2f}%) '
      f'dins_n={med("B",1)-med("A",1):+8.2f} dl1m_n={med("B",2)-med("A",2):+7.2f} '
      f'dma64_n={med("B",3)-med("A",3):+8.2f}')
PY
rm -f "$raw"

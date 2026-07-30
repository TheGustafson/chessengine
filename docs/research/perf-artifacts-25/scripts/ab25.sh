#!/usr/bin/env bash
# perf study 25 — INTERLEAVED A/B CYCLES panel.  THIS IS THE ACCEPT GATE.
#
#   bash docs/research/perf-artifacts-25/scripts/ab25.sh <binA> <binB> <rounds> [frame]
#
# frame: "wl50" (default, 5,001,562 nodes) or "d16" (9,444,932 nodes).
#
# The box hosts a GPU trainer and three sibling perf studies, so cycle counts
# drift over minutes.  Two panels run back to back are therefore NOT comparable
# at the few-percent level; this alternates A,B,A,B,... inside one invocation so
# any drift hits both arms equally, and reports the median of each arm plus the
# PAIRED per-round deltas (the statistic to trust — study 21 §6).
#
# Gate is cycles.  Instructions are printed but are not the gate: study 20
# landed -5.99% cycles at +0.19% instructions.
set -euo pipefail
A=${1:?binA}
B=${2:?binB}
N=${3:-9}
FRAME=${4:-wl50}
CPU=${PERF_CPU:-2}
case "$FRAME" in
  wl50) CMD=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn) ;;
  d16)  CMD=(bench 16 1 16 default depth engines/canonical/net.cenn) ;;
  *) echo "unknown frame $FRAME"; exit 1 ;;
esac
EV=cycles:u,instructions:u,L1-dcache-load-misses:u,ls_misal_loads.ma64:u
run() {
  local bin=$1 tmp
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp.perf" -- \
    taskset -c "$CPU" "bin25/$bin" "${CMD[@]}" > /dev/null 2> "$tmp.eng"
  local cyc ins l1m mis nodes
  cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  l1m=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp.perf")
  mis=$(awk -F, '$3~/^ls_misal_loads.ma64/{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  echo "$cyc $ins $l1m $mis $nodes"
}
# ABBA, not ABAB.  Perf study 25's first null (n=9, AB order) came back at
# +0.97% with a right-skewed paired spread (-98 .. +1164): with three sibling
# studies and a trainer on the box, load RAMPS inside a round, and whichever
# arm runs second eats it.  That is a positional bias, and it is exactly the
# size of the effects being hunted.  Each round here is A,B,B,A and the pair
# delta is mean(B) - mean(A) within the round, so a monotone drift across the
# round cancels to first order.
raw=$(mktemp)
for _ in $(seq "$N"); do
  a1=$(run "$A"); b1=$(run "$B"); b2=$(run "$B"); a2=$(run "$A")
  echo "A $a1" >> "$raw"; echo "B $b1" >> "$raw"
  echo "B $b2" >> "$raw"; echo "A $a2" >> "$raw"
done
echo "== frame=$FRAME cpu=$CPU  A=$A  B=$B" | tee -a docs/research/perf-artifacts-25/ab.txt
python3 - "$raw" "$A" "$B" <<'PY' | tee -a docs/research/perf-artifacts-25/ab.txt
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
# One pair per ABBA round: mean of the two B runs minus mean of the two A runs.
per = lambda r: r[0] / r[4]
pairs = []
for i in range(0, len(rows['A']), 2):
    a = rows['A'][i:i + 2]
    b = rows['B'][i:i + 2]
    if len(a) < 2 or len(b) < 2:
        break
    pairs.append(sum(map(per, b)) / 2 - sum(map(per, a)) / 2)
pairs.sort()
nneg = sum(1 for p in pairs if p < 0)
print(f'  paired dcyc_n (B-A): median={statistics.median(pairs):+8.1f} '
      f'min={pairs[0]:+8.1f} max={pairs[-1]:+8.1f} '
      f'neg={nneg}/{len(pairs)}')
print(f'  medians: dcyc_n={med("B",0)-med("A",0):+8.1f} ({100*(med("B",0)-med("A",0))/med("A",0):+.2f}%) '
      f'dins_n={med("B",1)-med("A",1):+8.2f} ({100*(med("B",1)-med("A",1))/med("A",1):+.2f}%) '
      f'dl1m_n={med("B",2)-med("A",2):+7.2f} dma64_n={med("B",3)-med("A",3):+8.2f}')
PY
rm -f "$raw"

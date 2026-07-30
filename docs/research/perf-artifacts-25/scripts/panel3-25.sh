#!/usr/bin/env bash
# perf study 25 — THREE-WAY ROUND-ROBIN CYCLES PANEL.  THIS IS THE ACCEPT GATE.
#
#   bash docs/research/perf-artifacts-25/scripts/panel3-25.sh <base> <copy> <arm> <rounds> [frame]
#
# `copy` MUST be a byte-identical copy of `base` (`cp`, two names).  It is the
# NULL, and it runs INSIDE the same rounds as the arm.
#
# WHY, and this is not a refinement — it is the difference between a real result
# and a retracted one.  Study 22 ran four nulls on a binary against its own
# byte-identical copy in ONE session and got -3.40%, -0.07%, -1.51%, +1.42% — a
# 4.8-point spread on a change that does not exist.  It had already committed a
# -1.16% headline against a null taken 40 minutes earlier; the round-robin found
# the null itself at +1.42% and the headline was retracted.  A null taken BEFORE
# the arm characterises a different machine state than the arm runs in, and with
# a 400-superbatch trainer plus sibling studies on the box that state moves
# several percent over tens of minutes.  So: null interleaved with the arm, or
# no cycles claim.
#
# Each round runs six slots — two of each binary — in a RANDOM order.
#
# The first cut of this script used the fixed symmetric sequence
# `base copy arm arm copy base`, on the reasoning that positions 1&6, 2&5, 3&4
# cancel a monotone drift.  They do — but they do NOT cancel CURVATURE, and the
# fixed sequence hands `base` the two extreme slots and `arm` the two central
# ones, so any convex or concave drift inside a round becomes a systematic
# arm-vs-base bias.  Measured on that version: a byte-identical copy read
# +2.06% while the real arm read +0.70%.  Randomising the slot order leaves no
# position systematically owned by any binary, so a position effect becomes
# noise that n averages down instead of bias that it does not.
#
# MEDIAN, never min-of-N: min-of-N read -2.07% on two identical binaries.
#
# `instructions:u` is exact here (noise floor 0.0) and is reported but is not
# the gate.  An arm that cuts cycles and raises instructions is still a win
# (study 20: -5.99% cycles at +0.19% instructions).
set -euo pipefail
BASE=${1:?base}
COPY=${2:?byte-identical copy of base}
ARM=${3:?arm}
N=${4:-8}
FRAME=${5:-wl50}
CPU=${PERF_CPU:-2}
cmp -s "bin25/$BASE" "bin25/$COPY" || { echo "FATAL: $COPY is not byte-identical to $BASE"; exit 1; }
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
raw=$(mktemp)
for r in $(seq "$N"); do
  for slot in $(printf '%s\n' BASE COPY ARM BASE COPY ARM | shuf); do
    case $slot in
      BASE) b=$BASE ;; COPY) b=$COPY ;; ARM) b=$ARM ;;
    esac
    echo "$r $slot $(run "$b")" >> "$raw"
  done
done
echo "== 3-way round-robin  frame=$FRAME cpu=$CPU  base=$BASE copy=$COPY arm=$ARM rounds=$N" \
  | tee -a docs/research/perf-artifacts-25/ab.txt
python3 - "$raw" "$BASE" "$COPY" "$ARM" <<'PY' | tee -a docs/research/perf-artifacts-25/ab.txt
import statistics, sys
raw, nbase, ncopy, narm = sys.argv[1:5]
rounds = {}
for line in open(raw):
    r, slot, cyc, ins, l1m, mis, nodes = line.split()
    rounds.setdefault(int(r), {}).setdefault(slot, []).append(
        tuple(float(x) for x in (cyc, ins, l1m, mis, nodes)))
allrows = {s: [v for rd in rounds.values() for v in rd.get(s, [])] for s in ('BASE', 'COPY', 'ARM')}
def med(s, i):
    return statistics.median(r[i] / r[4] for r in allrows[s])
for s, name in (('BASE', nbase), ('COPY', ncopy), ('ARM', narm)):
    print(f'{s:5s} {name:26s} n={len(allrows[s])} cyc_n={med(s,0):8.1f} ins_n={med(s,1):8.1f} '
          f'l1m_n={med(s,2):7.2f} ma64_n={med(s,3):8.2f}')
base_med = med('BASE', 0)
for s, name in (('COPY', 'NULL  (copy-base)'), ('ARM', 'ARM   (arm -base)')):
    pairs = []
    for rd in rounds.values():
        if 'BASE' not in rd or s not in rd:
            continue
        b = sum(x[0] / x[4] for x in rd['BASE']) / len(rd['BASE'])
        v = sum(x[0] / x[4] for x in rd[s]) / len(rd[s])
        pairs.append(v - b)
    pairs.sort()
    m = statistics.median(pairs)
    neg = sum(1 for p in pairs if p < 0)
    print(f'  {name}: paired median={m:+8.1f} ({100*m/base_med:+.2f}%) '
          f'min={pairs[0]:+8.1f} max={pairs[-1]:+8.1f} neg={neg}/{len(pairs)}')
print(f'  dins_n: null={med("COPY",1)-med("BASE",1):+7.2f}  '
      f'arm={med("ARM",1)-med("BASE",1):+7.2f} '
      f'({100*(med("ARM",1)-med("BASE",1))/med("BASE",1):+.2f}%)')
PY
rm -f "$raw"

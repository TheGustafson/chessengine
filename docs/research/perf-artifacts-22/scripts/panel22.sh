#!/usr/bin/env bash
# perf study 22 — THE ACCEPT PANEL.  Interleaved A/B, six counters.
#
#   bash docs/research/perf-artifacts-22/scripts/panel22.sh <binA> <binB> <rounds>
#
# WHY THIS AND NOT ab20.sh's MEDIAN-OF-6 CYCLES.  The commissioning brief quoted
# a cycles/node noise floor of 0.6-1.1% (median of 6).  The FIRST thing this
# study ran was the null — the SAME binary, copied to two names, through ab22.sh
# at n=6 — and it read **-3.40% cycles** (paired per-round deltas from -1528 to
# +181).  A sibling perf study is running its own DRAM-heavy bench and a GPU
# trainer is live, so the memory subsystem is contended and the median-of-6
# cycles instrument cannot resolve anything under ~4%.  Two changes:
#
#   1. CYCLES ARE REPORTED AS min AND median.  Interference can only ADD cycles,
#      so min-of-N estimates "how fast can this go" and is far more robust to a
#      neighbour than the median.  N should be >= 12.
#   2. THE MECHANISM COUNTERS CARRY THE ARGUMENT.  In the same null,
#      instructions/node moved 0.000%, L1d misses 0.28% and misaligned loads
#      0.41% — those counters DO resolve.  An arm whose mechanism is "fewer
#      cache lines touched" must show it in l1m/l2m/dram/dtlb, and the cycles
#      number is then corroboration rather than the sole evidence.
#
# Frame (quote it with every number): bench 16 1 100000 wl50.fens nodes D48X
# = a fixed 5,001,562-node tree, one thread, 16 MB TT, arch sota,arch-d48x.
set -euo pipefail
A=${1:?binA}
B=${2:?binB}
N=${3:-12}
CMD=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn)
EV=cycles:u,instructions:u,L1-dcache-load-misses:u,l2_cache_req_stat.ls_rd_blk_c:u,ls_dmnd_fills_from_sys.dram_io_near:u,dTLB-load-misses:u
run() {
  local bin=$1 tmp
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp.perf" -- \
    taskset -c 2 "bin22/$bin" "${CMD[@]}" > /dev/null 2> "$tmp.eng"
  local cyc ins l1m l2m dram dtlb nodes
  cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  l1m=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp.perf")
  l2m=$(awk -F, '$3~/^l2_cache_req_stat.ls_rd_blk_c/{print $1}' "$tmp.perf")
  dram=$(awk -F, '$3~/^ls_dmnd_fills_from_sys.dram_io_near/{print $1}' "$tmp.perf")
  dtlb=$(awk -F, '$3~/^dTLB-load-misses/{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  echo "$cyc $ins $l1m $l2m $dram $dtlb $nodes"
}
raw=$(mktemp)
run "$A" > /dev/null; run "$B" > /dev/null   # discard the page-cache-cold round
for _ in $(seq "$N"); do
  echo "A $(run "$A")" >> "$raw"
  echo "B $(run "$B")" >> "$raw"
done
python3 - "$raw" "$A" "$B" <<'PY' | tee -a docs/research/perf-artifacts-22/panel.txt
import statistics, sys
raw, na, nb = sys.argv[1], sys.argv[2], sys.argv[3]
rows = {'A': [], 'B': []}
for line in open(raw):
    t, *v = line.split()
    rows[t].append([float(x) for x in v])
NODES = 6
def per(t, i, f=statistics.median):
    return f(r[i] / r[NODES] for r in rows[t])
nodeset = sorted({int(r[NODES]) for t in rows for r in rows[t]})
for t, name in (('A', na), ('B', nb)):
    print(f'{t} {name:26s} n={len(rows[t])} cycmin={per(t,0,min):8.1f} cycmed={per(t,0):8.1f} '
          f'ins={per(t,1):8.1f} l1m={per(t,2):7.2f} l2m={per(t,3):7.2f} '
          f'dram={per(t,4):7.2f} dtlb={per(t,5):6.2f}')
def d(i, f=statistics.median):
    a, b = per('A', i, f), per('B', i, f)
    return b - a, (100 * (b - a) / a if a else 0.0)
dcm, pcm = d(0, min); dcy, pcy = d(0)
di, pi = d(1); dl1, pl1 = d(2); dl2, pl2 = d(3); ddr, pdr = d(4); dtl, ptl = d(5)
print(f'  D cycmin={dcm:+8.1f} ({pcm:+.2f}%)  cycmed={dcy:+8.1f} ({pcy:+.2f}%)  '
      f'ins={di:+8.2f} ({pi:+.3f}%)')
print(f'  D l1m={dl1:+7.2f} ({pl1:+.2f}%)  l2m={dl2:+7.2f} ({pl2:+.2f}%)  '
      f'dram={ddr:+7.2f} ({pdr:+.2f}%)  dtlb={dtl:+6.2f} ({ptl:+.2f}%)  nodes={nodeset}')
PY
rm -f "$raw"

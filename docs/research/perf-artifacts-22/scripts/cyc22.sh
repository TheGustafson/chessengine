#!/usr/bin/env bash
# perf study 22 — THE CYCLES / STALL PANEL.  This is the accept gate.
#
#   bash docs/research/perf-artifacts-22/scripts/cyc22.sh <label> <rounds>
#
# Same 5,001,562-node frame as probe22.sh.  Reports per-node cycles, L1d
# misses, LLC misses and 64B-cacheline-crossing loads; one line per round plus
# a median.  Cycles carry ~0.6-1.1% noise on this box (median-of-6);
# instructions carry 0.0.  Study 20's headline arm was -5.99% cycles with
# instructions UP +0.19%, so CYCLES decide and instructions are reported
# alongside, never as the gate.
set -euo pipefail
LABEL=${1:?label}
N=${2:-6}
BIN=target-fast22/release/chessengine
OUT=docs/research/perf-artifacts-22/cycles.txt
EV=cycles:u,instructions:u,L1-dcache-load-misses:u,ls_misal_loads.ma64:u
for _ in $(seq "$N"); do
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e "$EV" -o "$tmp.perf" -- \
    taskset -c 2 "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens nodes \
    engines/canonical/net.cenn > /dev/null 2> "$tmp.eng"
  cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  l1m=$(awk -F, '$3~/^L1-dcache-load-misses/{print $1}' "$tmp.perf")
  mis=$(awk -F, '$3~/^ls_misal_loads.ma64/{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  python3 -c "print(f'cyc={\"$LABEL\":28s} nodes=$nodes cyc_n={$cyc/$nodes:8.1f} ins_n={$ins/$nodes:8.1f} ipc={$ins/$cyc:5.3f} l1m_n={$l1m/$nodes:7.2f} ma64_n={$mis/$nodes:8.2f}')" | tee -a "$OUT"
done
python3 - "$LABEL" <<'PY' | tee -a "$OUT"
import statistics, sys
lab = sys.argv[1]
rows = [l for l in open('docs/research/perf-artifacts-22/cycles.txt') if l.startswith(f'cyc={lab:28s}')]
if not rows: raise SystemExit
def col(k):
    return statistics.median(float(l.split(f'{k}=')[1].split()[0]) for l in rows)
print(f'MEDIAN({len(rows)}) {lab:28s} cyc_n={col("cyc_n"):8.1f} ins_n={col("ins_n"):8.1f} ipc={col("ipc"):5.3f} l1m_n={col("l1m_n"):7.2f} ma64_n={col("ma64_n"):8.2f}')
PY

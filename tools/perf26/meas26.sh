#!/usr/bin/env bash
# perf study 26 — measure an ALREADY-BUILT binary in bin26/, n times.
#
#   bash tools/perf26/meas26.sh <label> [reps] [frame]
#
# Reports the MEDIAN instructions:u and branches:u per node.  Both are exact on
# this box (a repeated run of one binary reproduces to the digit); the repeats
# exist to prove that on the day, not to average noise away.
set -euo pipefail
LABEL=${1:?label}
N=${2:-3}
FRAME=${3:-wl50}
CPU=${PERF_CPU:-2}
case "$FRAME" in
  wl50) CMD=(bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn) ;;
  d16)  CMD=(bench 16 1 16 default depth engines/canonical/net.cenn) ;;
  *) echo "unknown frame $FRAME"; exit 1 ;;
esac
raw=$(mktemp)
for _ in $(seq "$N"); do
  tmp=$(mktemp)
  nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp.perf" -- \
    taskset -c "$CPU" "bin26/$LABEL" "${CMD[@]}" > /dev/null 2> "$tmp.eng"
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
  brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
  echo "$ins $brs $nodes" >> "$raw"
  rm -f "$tmp" "$tmp.perf" "$tmp.eng"
done
python3 - "$raw" "$LABEL" "$FRAME" <<'PY'
import statistics, sys
raw, label, frame = sys.argv[1:4]
rows = [tuple(int(x) for x in l.split()) for l in open(raw)]
i = [r[0]/r[2] for r in rows]; b = [r[1]/r[2] for r in rows]
nodes = {r[2] for r in rows}
print(f'{label:26s} frame={frame} n={len(rows)} '
      f'i/n={statistics.median(i):9.3f} (spread {max(i)-min(i):.4f})  '
      f'br/n={statistics.median(b):8.3f} (spread {max(b)-min(b):.4f})  nodes={nodes}')
PY
rm -f "$raw"

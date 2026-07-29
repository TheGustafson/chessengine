#!/usr/bin/env bash
# perf study 16 — the FIXED (per-process) instruction cost inside `bench`.
#
#   bash startup16.sh
#
# Every study since #8 has quoted `instructions / nodes` over a `bench` run, so
# whatever the process does BEFORE it searches is divided into the per-node
# number. Two runs at different node budgets separate the constant from the
# slope:  instr(n) = fixed + per_node * nodes.
#
# Reported for BOTH engines, because the comparison is a ratio and a constant
# that differs between them biases it.
set -euo pipefail
ARCH=target-fast16/release/chessengine
SF=./engines/bin/stockfish-eca43a97
NET=engines/canonical/net.cenn

run() { # engine, nodes-per-position, [net]
  local out ins nodes
  out=$(nice -n 19 perf stat -x, -e instructions:u -- taskset -c 6 \
        "$1" bench 16 1 "$2" tools/nnue/calib/wl50.fens nodes ${3:+"$3"} 2>&1)
  ins=$(printf '%s\n' "$out" | awk -F, '$3=="instructions:u"{print $1}')
  nodes=$(printf '%s\n' "$out" | grep -oP 'Nodes searched\s*:\s*\K[0-9]+' | tail -1)
  echo "$nodes $ins"
}

echo "# startup16 $(date -Is)"
for pair in "archon $ARCH $NET" "sf $SF"; do
  set -- $pair
  label=$1; bin=$2; net=${3:-}
  a=$(run "$bin" 1 ${net:+$net}); b=$(run "$bin" 20000 ${net:+$net})
  python3 - "$label" $a $b <<'PY'
import sys
label, n1, i1, n2, i2 = sys.argv[1], *map(int, sys.argv[2:6])
slope = (i2 - i1) / (n2 - n1)
fixed = i1 - slope * n1
print(f"{label:>8}: small={n1:>9,}n {i1:>15,}i | big={n2:>9,}n {i2:>15,}i"
      f" || per_node={slope:8.1f}  FIXED={fixed:>14,.0f}"
      f"  (= {fixed/15_001_568:6.1f} i/n of the 15,001,568-node headline bench)")
PY
done

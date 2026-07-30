#!/usr/bin/env bash
# perf study 42 -- paired within-round panel with a RANDOMISED order.
#
#   bash tools/perf42/panel42.sh <rounds> <label>...
#   P42_EV=  P42_PANEL=  P42_BUDGET=  P42_CPU=  P42_MB=  P42_NET=  P42_FENS=
#
# tools/perf40/panel40.sh with two changes:
#   * binaries come from bin42/ (falling back to bin34/), and
#   * the Hash size is a panel variable (`P42_MB`), because this study's whole
#     question is what the transposition table costs and the answer is allowed
#     to depend on how big it is.
#
# Reduce with tools/perf40/med40.py -- the reference label's SAME-ROUND run is
# the pair, the per-round deltas are voted, and the null must come back a coin
# flip before anything below it is quotable.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=${P42_NET:-engines/canonical/net.cenn}
FENS=${P42_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P42_CPU:-2}
MB=${P42_MB:-16}
BUDGET=${P42_BUDGET:-100000}
EV=${P42_EV:-instructions:u,cycles:u,ex_no_retire.load_not_complete:u,ex_no_retire.empty:u,ex_no_retire.thread_not_selected:u}
OUT=${P42_PANEL:-docs/research/perf-artifacts-42/panel.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

labels=("$@")
for r in $(seq "$ROUNDS"); do
  order=("${labels[@]}")
  for ((i=${#order[@]}-1; i>0; i--)); do
    j=$((RANDOM % (i+1)))
    t=${order[i]}; order[i]=${order[j]}; order[j]=$t
  done
  for label in "${order[@]}"; do
    bin="bin42/$label"
    [ -x "$bin" ] || bin="bin34/$label"
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./$bin" bench "$MB" 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $1 ~ /^[0-9]+$/ && $3 != "" { printf "%s=%.6f ", $3, $1 / n }
      END { print "" }
    ' "$tmp/stat" | sed "s|^|r$r $label nodes=$nodes |" >> "$OUT"
  done
done
echo "[panel42] $ROUNDS rounds, hash=${MB}MB -> $OUT"

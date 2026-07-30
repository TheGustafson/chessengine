#!/usr/bin/env bash
# perf study 40 — the paired within-round panel, with a RANDOMISED order.
#
#   bash tools/perf40/panel40.sh <rounds> <label>...
#   P40_EV=  P40_PANEL=  P40_BUDGET=  P40_CPU=  P40_NET=  P40_FENS=
#
# Descends from tools/perf37/panel37.sh. Two changes, both about the same
# problem — this study's headline arm is a ~0.2% CYCLES effect, an order of
# magnitude below anything studies 33-39 tried to read off that counter:
#
#  1. the order inside a round is SHUFFLED, not rotated. A fixed rotation
#     cancels monotone drift but not curvature (study 25 §0.2), and with a
#     4 s run the box's own boost/thermal curve is exactly that shape.
#  2. the run is short enough (~4 s on the study frame) that hundreds of
#     rounds are affordable, so the reduction (`med40.py`) reports a SIGN
#     COUNT over rounds as well as a median. Common-mode noise cancels in a
#     within-round pair; what survives is a per-round vote.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=${P40_NET:-engines/canonical/net.cenn}
FENS=${P40_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P40_CPU:-2}
BUDGET=${P40_BUDGET:-100000}
EV=${P40_EV:-instructions:u,cycles:u,branches:u,ex_ret_brn_misp:u}
OUT=${P40_PANEL:-docs/research/perf-artifacts-40/panel.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

labels=("$@")
for r in $(seq "$ROUNDS"); do
  # Fisher-Yates over the label list, fresh every round.
  order=("${labels[@]}")
  for ((i=${#order[@]}-1; i>0; i--)); do
    j=$((RANDOM % (i+1)))
    t=${order[i]}; order[i]=${order[j]}; order[j]=$t
  done
  for label in "${order[@]}"; do
    # A label may carry a dose: `binary:D` runs `./bin34/binary` with
    # P40_DOSE=D. One binary then serves a whole dose series, so code layout is
    # differenced away by construction (study 39 handover §10.1).
    bin=${label%%:*}
    dose=${label#*:}; [ "$dose" = "$label" ] && dose=0
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      env P40_DOSE="$dose" taskset -c "$CPU" "./bin34/$bin" \
      bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$label" -v r="$r" -v n="$nodes" '
      $1 ~ /^[0-9]+$/ && $3 != "" { printf "%s=%.6f ", $3, $1 / n }
      END { print "" }
    ' "$tmp/stat" | sed "s|^|r$r $label nodes=$nodes |" >> "$OUT"
  done
done
echo "[panel40] $ROUNDS rounds -> $OUT"

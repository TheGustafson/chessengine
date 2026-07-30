#!/usr/bin/env bash
# perf study 44 — the paired panel.
#
#   bash tools/perf44/panel44.sh <rounds> <ref-label> <label>...
#   P44_EV=  P44_PANEL=  P44_CPU=  P44_BUDGET=  P44_BIN=
#
# Protocol, inherited from study 40 §1 and study 39 §0 because both paid for it:
#
#   * the label order is SHUFFLED INSIDE EVERY ROUND. A fixed rotation cancels
#     monotone drift but not curvature (study 25 §0.2), and study 29 §5.3
#     retracted a headline that turned out to be position-in-round.
#   * every label is differenced against the SAME ROUND's reference, so
#     common-mode load from the other agents on this box cancels in the pair
#     rather than being assumed away.
#   * a BYTE-IDENTICAL null rides in the rotation as an ordinary label. If the
#     null moves as much as the arm, the panel cannot answer the question --
#     that is rule 1, and it is a gate here, not a footnote.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
REF=${1:?ref}; shift
NET=${P44_NET:-engines/canonical/net.cenn}
FENS=${P44_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P44_CPU:-2}
BUDGET=${P44_BUDGET:-100000}
BIN=${P44_BIN:-bin44}
EV=${P44_EV:-instructions:u,cycles:u,ex_no_retire.load_not_complete:u,ex_no_retire.not_complete:u}
OUT=${P44_PANEL:-docs/research/perf-artifacts-44/panel.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

ALL=("$REF" "$@")
for r in $(seq "$ROUNDS"); do
  # Fisher-Yates over the label list, fresh every round.
  order=($(printf '%s\n' "${ALL[@]}" | shuf))
  for label in "${order[@]}"; do
    nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./$BIN/$label" \
      bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v n="$nodes" '
      $1 ~ /^[0-9]+$/ && $3 != "" {
        pct = ($6 == "" ? 100 : $6)
        printf "%s=%.6f ", $3, $1 / n
        if (pct + 0 < 99.5) printf "SCALED[%s@%s%%] ", $3, pct
      } END { print "" }
    ' "$tmp/stat" | sed "s|^|r$r $label nodes=$nodes |" >> "$OUT"
  done
done
echo "[panel44] $ROUNDS rounds x ${#ALL[@]} labels -> $OUT"

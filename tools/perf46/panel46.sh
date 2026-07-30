#!/usr/bin/env bash
# perf study 44 — the paired panel.
#
#   bash tools/perf46/panel44.sh <rounds> <ref-label> <label>...
#   P46_EV=  P46_PANEL=  P46_CPU=  P46_BUDGET=  P46_BIN=
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
NET=${P46_NET:-engines/canonical/net.cenn}
FENS=${P46_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P46_CPU:-2}
BUDGET=${P46_BUDGET:-100000}
BIN=${P46_BIN:-bin46}
EV=${P46_EV:-instructions:u,cycles:u,ex_no_retire.load_not_complete:u,ex_no_retire.not_complete:u}
OUT=${P46_PANEL:-docs/research/perf-artifacts-46/panel.txt}
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

ALL=("$REF" "$@")
NLAB=${#ALL[@]}
BASEORD=()
for r in $(seq "$ROUNDS"); do
  # ---- BALANCED LATIN SQUARE, not a free shuffle -------------------------
  # Perf study 44 §4.3 found the free per-round shuffle is NOT sufficient at
  # achievable n. This frame has a real within-round drift -- mean cycles/node
  # by position in round read 4282 / 4354 / 4330 / 4449, i.e. +3.9% from first
  # slot to last -- and over 13 rounds `shuf` happened to seat the REFERENCE at
  # mean position 0.77 while the null sat at 1.62 and one arm at 2.00. That
  # imbalance alone accounts for most of a byte-identical null reading +2.4%
  # cycles, which is a broken instrument by rule 1.
  #
  # A shuffle cancels drift only IN EXPECTATION. A Latin square cancels it BY
  # CONSTRUCTION: within each block of NLAB rounds every label occupies every
  # position exactly once, so any monotone within-round trend contributes the
  # same amount to every label's total. The base order is re-shuffled per block
  # so the design still varies WHICH orderings appear (study 25 §0.2's point
  # that a single fixed rotation cannot cancel curvature).
  k=$(( (r - 1) % NLAB ))
  if [ "$k" -eq 0 ]; then
    BASEORD=($(printf '%s\n' "${ALL[@]}" | shuf))
  fi
  order=()
  for ((i = 0; i < NLAB; i++)); do
    order+=("${BASEORD[$(((i + k) % NLAB))]}")
  done
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

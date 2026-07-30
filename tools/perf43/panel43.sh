#!/usr/bin/env bash
# perf study 43 — paired within-round panel with a PER-LABEL RUNTIME MODE.
#
#   bash tools/perf43/panel43.sh <rounds> <label>...
#
# Labels are `<binary>@<P43_PF mode>`; the mode is exported for that run only.
# The point is that every label can be the SAME BYTES: study 29 built its
# prefetch arm as a separate binary, and its own §5.3 records a byte-different
# null moving the cycles median further than the arm did. Here `pf43@0` and
# `null@0` are the same file run twice, which is the study's null.
#
# Order rotates within the round and `med43.py` differences each label against
# the REFERENCE RUN OF THE SAME ROUND before taking a median (study 29 nearly
# shipped a -1.67% cycles result at n=13 that was position-in-round).
#
# The GATE IS CYCLES (study 20 was -5.99% cycles with instructions UP 0.19%),
# so cycles leads the row; the stall counter is carried beside it because this
# study's target is 71% load-stall.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P43_CPU:-2}
BUDGET=${P43_BUDGET:-100000}
EV=cycles:u,instructions:u,ex_no_retire.load_not_complete:u
OUT=${P43_PANEL:-docs/research/perf-artifacts-43/panel.txt}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$(dirname "$OUT")"

for r in $(seq "$ROUNDS"); do
  # SHUFFLE, do not rotate. Study 43's first panel rotated (`set -- "${@:2}"
  # "$1"`, inherited from panel34.sh) and its BYTE-IDENTICAL null came back
  # -0.5700% on cycles with 13 of 16 rounds negative. Cyclic rotation preserves
  # ADJACENCY -- the null always ran immediately after the reference -- so a
  # neighbour effect (boost ramp, page cache, THP) is systematic and pairing
  # cannot cancel it. Study 39 randomised inside every round; this does too.
  mapfile -t order < <(printf '%s\n' "$@" | shuf)
  for spec in "${order[@]}"; do
    bin=${spec%@*}; mode=${spec#*@}
    P43_PF="$mode" nice -n 19 perf stat -x, -e "$EV" -o "$tmp/stat" -- \
      taskset -c "$CPU" "./bin34/$bin" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
      >/dev/null 2>"$tmp/bench"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
    awk -F, -v l="$spec" -v r="$r" -v n="$nodes" '
      $3~/^cycles/{c=$1} $3~/^instructions/{i=$1} $3~/^ex_no_retire/{s=$1}
      END{printf "r%-3d %-14s nodes=%-9d cyc/n=%10.4f i/n=%10.4f stall/n=%10.4f\n",
          r, l, n, c/n, i/n, s/n}
    ' "$tmp/stat" | tee -a "$OUT"
  done
done

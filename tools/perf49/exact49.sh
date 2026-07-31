#!/usr/bin/env bash
# perf study 49 -- the EXACT-counter panel.
#
#   bash tools/perf48/exact49.sh <rounds> <label>...
#
# `instructions:u`, `branches:u`, `L1-dcache-loads:u` and `ls_dispatch.*` have a
# ~0 null on this box and are ordering-independent (study 45 Sec 2.5.3), which
# is exactly why this study adjudicates on them: the load count IS the quantity
# being reduced, and it is exact.  Ordering is a BLOCKED LATIN SQUARE anyway --
# rotate within a block of NLAB, reshuffle the base between blocks -- so the
# same panel can carry `cycles:u` without the position bias that broke study
# 44's and study 45's free shuffles.  A plain Latin square IS cyclic rotation
# (study 43's defect); the per-block reshuffle is what stops that.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
LABELS=("$@")
NLAB=${#LABELS[@]}
CPU=${P49_CPU:-2}
NET=${P49_NET:-engines/canonical/net.cenn}
FENS=${P49_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P49_BIN:-bin49}
EV=${P49_EV:-instructions:u,cycles:u,L1-dcache-loads:u,ls_dispatch.ld_dispatch:u,ls_dispatch.store_dispatch:u}
OUT=${P49_PANEL:-docs/research/perf-artifacts-49/panel.txt}
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
base=("${LABELS[@]}")
for r in $(seq 0 $((ROUNDS - 1))); do
  k=$((r % NLAB))
  if [ "$k" -eq 0 ]; then
    mapfile -t base < <(printf '%s\n' "${LABELS[@]}" | shuf)
  fi
  order=()
  for i in $(seq 0 $((NLAB - 1))); do
    order+=("${base[$(((i + k) % NLAB))]}")
  done
  for pos in $(seq 0 $((NLAB - 1))); do
    lab=${order[$pos]}
    echo "### round=$r pos=$pos label=$lab" >> "$OUT"
    perf stat -x, -e "$EV" -- taskset -c "$CPU" "./$BIN/$lab" \
      bench 16 1 100000 "$FENS" nodes "$NET" 2>>"$OUT" >/dev/null
  done
done
echo "[exact49] -> $OUT"

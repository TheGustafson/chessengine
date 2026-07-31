#!/usr/bin/env bash
# perf study 59 — the STARTUP panel.
#
#   bash tools/perf59/startup59.sh <rounds> <label>...
#
# A 1-node bench: the engine parses argv, reads and validates the 138 MB net,
# quantises the HalfKA table, searches one node, prints, exits. Everything
# above the noise is load. The same blocked-Latin-square ordering as
# `panel59.sh`, and the same counter licence — `instructions:u` adjudicates
# (0.000 null on this box), `cycles:u` is carried and never trusted alone.
#
# Why a separate panel at all: startup is a FIXED cost, so on the 5 M-node
# frame it is diluted ~40x and a real 30% cut reads as 0.1%. Match play spawns
# one process per game and `tools/screen.sh` spawns thousands, which is the
# population this measures.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
LABELS=("$@")
NLAB=${#LABELS[@]}
CPU=${P59_CPU:-1}
NET=${P59_NET:-engines/canonical/net.cenn}
FENS=${P59_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P59_BIN:-bin59}
EV=${P59_EV:-instructions:u,cycles:u,branches:u,L1-dcache-loads:u,ls_dispatch.ld_dispatch:u}
OUT=${P59_PANEL:-docs/research/perf-artifacts-59/startup.txt}
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
    # 1 position, 1 node: the smallest run that still loads the net.
    perf stat -x, -e "$EV" -- taskset -c "$CPU" "./$BIN/$lab" \
      bench 16 1 1 "$FENS" nodes "$NET" 2>>"$OUT" >/dev/null
  done
done
echo "[startup59] -> $OUT"

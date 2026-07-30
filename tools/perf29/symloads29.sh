#!/usr/bin/env bash
# perf study 29 — per-symbol LOAD attribution (sampled).
#
#   bash tools/perf29/symloads29.sh <label> [budget]
#
# `ls_dispatch.ld_dispatch` is the Zen 4 load-op counter `L1-dcache-loads`
# aliases. This is a SAMPLED profile: the totals are exact (perf stat), the
# per-symbol split carries the usual ~1% sampling error. It exists to
# CROSS-CHECK the exact census, not to produce it.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
BUDGET=${2:-100000}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
OUT=docs/research/perf-artifacts-29/symloads-$LABEL-$BUDGET.txt
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

nice -n 19 perf record -q -e ls_dispatch.ld_dispatch:u -c 200003 -o "$tmp/p.data" -- \
  taskset -c "$CPU" "./bin29/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  >/dev/null 2>"$tmp/bench"
nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
total=$(perf report -i "$tmp/p.data" --stdio -q -F overhead,symbol 2>/dev/null | wc -l)
{
  echo "# $LABEL  budget=$BUDGET  nodes=$nodes  event=ls_dispatch.ld_dispatch:u  period=200003"
  echo "# columns: share%  loads/node(derived)  symbol"
  perf report -i "$tmp/p.data" --stdio -q -F overhead,symbol --percent-limit 0.20 2>/dev/null
} > "$OUT"
echo "wrote $OUT  ($total symbols, nodes=$nodes)"
head -40 "$OUT"

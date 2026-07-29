#!/usr/bin/env bash
# perf study 17 — the EXACT per-function BRANCH counter.
#
#   bash cgbr17.sh <label> <binary> <bench-args...>
#
# `--branch-sim=yes` makes callgrind emit `Ir Bc Bcm Bi Bim` per source line:
# conditional branches EXECUTED, conditional mispredicts, indirect branches,
# indirect mispredicts.  The `Bc + Bi` sum is the deterministic analogue of
# `perf stat -e branches:u`, and unlike perf it is attributed to a function
# exactly.  ARCHON's branches/node is 1.84x Stockfish's and three separate
# studies refuted the obvious explanations; this instrument says WHICH FUNCTION.
set -euo pipefail
LABEL=${1:?label}
BIN=${2:?binary}
shift 2
OUTDIR=docs/research/perf-artifacts-17/cg
mkdir -p "$OUTDIR"
OUT="$OUTDIR/br-$LABEL.out"
nice -n 19 taskset -c 6 valgrind --tool=callgrind --callgrind-out-file="$OUT" \
  --cache-sim=no --branch-sim=yes --separate-threads=no \
  "$BIN" "$@" > /dev/null 2>"$OUT.log"
nodes=$(grep -oP 'Nodes searched\s+:\s*\K[0-9]+' "$OUT.log" | tail -1)
echo "label=$LABEL nodes=$nodes"
grep -E "Collected|Branches|Mispred" "$OUT.log" | tail -4

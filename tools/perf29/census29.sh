#!/usr/bin/env bash
# perf study 29 — run the instrumented binary and save the raw census.
#
#   bash tools/perf29/census29.sh <label> <nodes-per-fen> [outfile]
#
# `<label>` must be a foldcount build in bin29/. Output is the FOLDCOUNT lines
# on stderr plus the bench trailer, appended to the artifact file.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
BUDGET=${2:-100000}
OUT=${3:-docs/research/perf-artifacts-29/census-$LABEL-$BUDGET.txt}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P29_CPU:-2}
mkdir -p "$(dirname "$OUT")"
# The census goes to its own FILE, never to a stream: `bench` writes its whole
# fingerprint to stderr and the byte-identity tests count those lines.
ARCHON_FOLDCOUNT="$OUT" nice -n 19 taskset -c "$CPU" \
  "./bin29/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
  2>"$OUT.bench" >/dev/null
sed -n 's/^Nodes searched  : /bench nodes: /p' "$OUT.bench"
cat "$OUT"

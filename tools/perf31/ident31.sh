#!/usr/bin/env bash
# perf study 31 — IDENTITY + exact-counter probe for one or more arms.
#
#   bash tools/perf31/ident31.sh <label>...
#
# Prints, per arm: instructions/node, branches/node (both 0.000%-null exact at
# CGU=1), the node total, and the wl50 eval hash.  Node total and eval hash
# MUST be unchanged across arms for a data-layout change; if either moves the
# arm is not node-identical and nothing else it prints is comparable.
#
# Frame: bench 16 1 100000 tools/nnue/calib/wl50.fens nodes engines/canonical/net.cenn
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
CPU=${P31_CPU:-6}
OUT=$ROOT/docs/research/perf-artifacts-31/ident.txt
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for label in "$@"; do
  bin=$ROOT/bin31/$label
  nice -n 19 perf stat -x, -e instructions:u,branches:u -o "$tmp/stat" -- \
    taskset -c "$CPU" "$bin" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp/stat")
  brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp/stat")
  nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp/bench" | tail -1)
  hash=$("$bin" nnue-eval "$NET" < "$FENS" | sha256sum | cut -c1-16)
  python3 -c "print(f'{\"$label\":22s} i/n={$ins/$nodes:8.2f} br/n={$brs/$nodes:7.2f} nodes={$nodes:>10,d} hash=$hash')" \
    | tee -a "$OUT"
done

#!/usr/bin/env bash
# perf study 35 -- IDENTITY check for one or more arms.
#
#   bash tools/perf35/ident35.sh <label>...
#
# Prints, per arm: instructions/node (0.000%-null exact at CGU=1), the node
# total, and the wl50 eval hash.  This study's arms are pure refactors of how
# the SAME DirtyPiece is derived, so node total and eval hash MUST be identical
# across every arm; if either moves the arm changed behaviour and nothing else
# it prints is comparable.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
NET=$ROOT/engines/canonical/net.cenn
FENS=$ROOT/tools/nnue/calib/wl50.fens
CPU=${P35_CPU:-2}
OUT=$ROOT/docs/research/perf-artifacts-35/ident.txt
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for label in "$@"; do
  bin=$ROOT/bin35/$label
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

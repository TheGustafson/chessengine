#!/usr/bin/env bash
# perf study 58 — the smallest possible instrument: two labels, alternating,
# `instructions:u` + `branches:u` only, one line per run.
#
#   bash tools/perf58/pair58.sh <rounds> <labelA> <labelB> [budget]
#
# Exists because the five-event panel disagreed with the disassembly, and a
# disagreement between an instrument and a static instruction count has to be
# settled on the simplest instrument available before anything is believed.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}
A=${2:?labelA}
B=${3:?labelB}
BUDGET=${4:-100000}
CPU=${P58_CPU:-1}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
run() {
  perf stat -x, -e instructions:u,branches:u -o "$tmp/s" -- \
    taskset -c "$CPU" "./bin58/$1" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/b"
  ins=$(awk -F, '$3 ~ /^instructions/ {print $1}' "$tmp/s")
  br=$(awk -F, '$3 ~ /^branches/ {print $1}' "$tmp/s")
  nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/b" | tail -1)
  printf '%-8s %-14s %-13s nodes=%s\n' "$1" "$ins" "$br" "$nodes"
}
for r in $(seq "$ROUNDS"); do
  # alternate which label goes first, so position cannot correlate with label
  if [ $((r % 2)) -eq 0 ]; then run "$A"; run "$B"; else run "$B"; run "$A"; fi
done

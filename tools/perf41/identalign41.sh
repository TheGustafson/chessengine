#!/usr/bin/env bash
# perf study 41 — the frame gate for the layout dose.
#
#   bash tools/perf41/identalign41.sh <label>...
#
# The coordinator's frame rule: node counts must match to ~0.000% AND
# `instructions:u` to ~0.02%, or the binaries are not doing the same work and
# no per-node counter delta means anything. For a layout dose the bar is
# HARDER than that — alignment padding is never executed, so both must be
# EXACT. Anything else means `-align-all-functions` reached the instruction
# stream (e.g. by aligning a fall-through block) and the dose is dirty.
#
# Also prints `.text` size, which is the dose's independent variable.
set -euo pipefail
cd "$(dirname "$0")/../.."
NET=${P41_NET:-engines/canonical/net.cenn}
FENS=${P41_FENS:-tools/nnue/calib/wl50.fens}
CPU=${P41_CPU:-2}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

printf '%-8s %-11s %-16s %-14s %s\n' label nodes eval-hash instructions .text
for lab in "$@"; do
  nice -n 19 perf stat -x, -e instructions:u -o "$tmp/stat" -- \
    taskset -c "$CPU" "./bin34/$lab" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>"$tmp/bench"
  n=$(sed -n 's/^Nodes searched  : //p' "$tmp/bench" | tail -1)
  i=$(awk -F, '$3 ~ /instructions/ {print $1}' "$tmp/stat")
  h=$("./bin34/$lab" nnue-eval "$NET" <"$FENS" | sha256sum | cut -c1-16)
  t=$(objcopy -O binary --only-section=.text "bin34/$lab" /dev/stdout | wc -c)
  printf '%-8s %-11s %-16s %-14s %s\n' "$lab" "$n" "$h" "$i" "$t"
done

#!/usr/bin/env bash
# perf study 69 — THE BRANCH-MISPREDICT PROFILE, DOWN TO THE BRANCH SITE.
#
#   bash tools/perf69/mispred69.sh <binary> <out.txt> [reps] [period]
#
# Nothing in this series could answer "WHICH branch mispredicts". `perf record
# -e branch-misses` on AMD has no PEBS-style precise mode, so its sample IP is
# skidded and useless at instruction resolution.
#
# This box (Zen 4) has **AMD LBR v2** — `perf record -b` captures a 16-deep
# last-branch stack at every sample, and each entry carries the branch's own
# SOURCE address, its target, and a **mispredict flag** (`M/` vs `P/`). So a
# uniform sample of the *instruction* stream yields a uniform sample of the
# recent branch stream, and counting the `M`-flagged entries by source address
# is an unskidded per-branch-site mispredict histogram.
#
# `-e instructions:u -j any` — the sampling event is uniform in retired
# instructions (NOT cycles: studies 43/63 bar `cycles:u` on this shared box,
# and a cycle-uniform sample would over-weight stalled regions, which is
# exactly the population a mispredict profile must not be biased by).
# `-j any,u` is rejected by this perf build's filter parser; the event is
# already `:u`, and `mispagg69.py` drops non-`chessengine::` sources.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary path}; OUT=${2:?out}; REPS=${3:-2}; PER=${4:-2000003}
CPU=${P69_CPU:-2}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
: > "$OUT"
for i in $(seq 1 "$REPS"); do
  taskset -c "$CPU" perf record -q -e instructions:u -c "$PER" -j any \
    -o "$tmp/m-$i.data" -- "$BIN" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>&1
  # `brstacksym` prints each of the 16 entries as
  #   <from-sym+off>/<to-sym+off>/<M|P>/<...>
  # one sample per line, entries separated by spaces.
  perf script -i "$tmp/m-$i.data" -F brstacksym >> "$OUT" 2>/dev/null
done
wc -l < "$OUT"

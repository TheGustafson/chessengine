#!/usr/bin/env bash
# perf study 67 — IBS op sampling with cnt_ctl=0 (interval counted in DISPATCHED
# OPS, not cycles), so the resulting per-rip histogram is proportional to OP
# COUNT and carries no skid: IBS tags an op at issue and records that op's own
# rip. This is the instrument `instructions:u` per-instruction cannot be
# (study 36 §2.3), and it is NOT study 59/64's cnt_ctl=1 latency profile.
#   bash tools/perf67/ibsops67.sh <binary> <out.txt> [reps]
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${1:?binary}; OUT=${2:?out}; REPS=${3:-2}
CPU=${P67_CPU:-2}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
: > "$OUT"
for i in $(seq 1 "$REPS"); do
  taskset -c "$CPU" perf record -q -e ibs_op/cnt_ctl=0,swfilt=1/ -c 20000 \
    -o "$tmp/i-$i.data" -- "$BIN" bench 16 1 100000 tools/nnue/calib/wl50.fens \
    nodes engines/canonical/net.cenn >/dev/null 2>&1
  perf script -i "$tmp/i-$i.data" -F ip,sym,symoff >> "$OUT" 2>/dev/null
done
wc -l < "$OUT"

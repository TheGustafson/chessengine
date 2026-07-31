#!/usr/bin/env bash
# perf study 60 — the minimal settling instrument (study 57 §2's `pair57.sh`).
# One line per run, alternating labels, only parsed numbers.
#   bash tools/perf60/pair60.sh <rounds> <label>...
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:?rounds}; shift
CPU=${P60_CPU:-1}
NET=${P60_NET:-engines/canonical/net.cenn}
FENS=${P60_FENS:-tools/nnue/calib/wl50.fens}
BIN=${P60_BIN:-bin60}
BUDGET=${P60_BUDGET:-100000}
EV=${P60_EV:-instructions:u,cycles:u}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for r in $(seq 1 "$ROUNDS"); do
  for lab in "$@"; do
    perf stat -x, -e "$EV" -o "$tmp/s" -- taskset -c "$CPU" "./$BIN/$lab" \
      bench 16 1 "$BUDGET" "$FENS" nodes "$NET" >/dev/null 2>"$tmp/b"
    nodes=$(sed -n 's/^Nodes searched  : //p' "$tmp/b" | tail -1)
    vals=$(grep -E '^[0-9]+,' "$tmp/s" | awk -F, '{printf "%s=%s ", $3, $1}')
    echo "$lab $vals nodes=$nodes"
  done
done

#!/usr/bin/env bash
# perf study 13 — NULL FIRST.  Build and measure the SAME source three times.
# Every arm's delta must be large against this band or it is not a result.
set -euo pipefail
N=${1:-3}
for i in $(seq 1 "$N"); do
  touch src/lib.rs
  bash docs/research/perf-artifacts-13/scripts/probe13fast.sh "null-$i"
done

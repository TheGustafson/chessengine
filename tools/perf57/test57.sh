#!/usr/bin/env bash
# perf study 57 — `cargo test` under the box discipline (lock, MemoryMax, -j2).
#   bash tools/perf57/test57.sh [test-name-filter]
set -euo pipefail
cd "$(dirname "$0")/../.."
FILTER=${1:-}
LOCK=/home/nick/.cache/archon-build.lock
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[test57] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo test --features sota,arch-d48x -j2 -- $FILTER

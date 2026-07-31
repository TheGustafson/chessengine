#!/usr/bin/env bash
# perf study 57 — clippy under the box discipline (lock, MemoryMax, -j2).
#   bash tools/perf57/clippy57.sh [extra,features]
set -euo pipefail
cd "$(dirname "$0")/../.."
EXTRA=${1:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[clippy57] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo clippy --all-targets --features "$FEATURES" -j2 -- -D warnings

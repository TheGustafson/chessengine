#!/usr/bin/env bash
# perf study 65 — run the test suite under a feature set, inside the box's
# build discipline (flock, MemoryMax, oomlast, -j2, capped test threads).
#   bash tools/perf65/test65.sh <features>
set -euo pipefail
cd "$(dirname "$0")/../.."
FEATURES=${1:?features}
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[test65] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
flock /home/nick/.cache/archon-build.lock \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  ./tools/oomlast.sh env CARGO_BUILD_JOBS=2 \
  cargo test --features "$FEATURES" --quiet -- --test-threads "${T:-4}"

#!/usr/bin/env bash
# perf study 56 -- the D24 depth-6 bench fingerprint on a PORTABLE default
# build (no target-cpu), which is what the `Bench:` trailer is pinned to.
set -euo pipefail
cd "$(dirname "$0")/../.."
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
[ "$avail" -lt 3500 ] && { echo "[bench56] REFUSING: ${avail} MB" >&2; exit 1; }
flock -w 60 /home/nick/.cache/archon-build.lock \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
    --working-directory="$PWD" nice -n 19 cargo build --release -j2 \
  || systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 --quiet \
    --working-directory="$PWD" nice -n 19 cargo build --release -j2
./target/release/chessengine bench 16 1 6 default depth 2>&1 | grep -E "Nodes searched  :"

#!/usr/bin/env bash
# perf study 57 — the D24 release perft tier under the box discipline.
#   bash tools/perf57/perft57.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
LOCK=/home/nick/.cache/archon-build.lock
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo test --release -j2 -- --ignored perft_full_suite

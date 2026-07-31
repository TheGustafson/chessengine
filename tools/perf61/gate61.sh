#!/usr/bin/env bash
# perf study 61 — the full merge gate, under the study's box discipline.
#   bash tools/perf61/gate61.sh
# Runs `CHECK_NNUE_GATE=1 bash tools/check.sh` (fmt, clippy, tests, release
# perft tier, bench fingerprint, eval hash + d13 nodes) inside the build lock
# and a MemoryMax scope, then the release perft_full_suite.
set -euo pipefail
cd "$(dirname "$0")/../.."
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[gate61] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[gate61] avail=${avail}MB — waiting for the build lock"
flock /home/nick/.cache/archon-build.lock \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
    --working-directory="$PWD" \
    env CHECK_NNUE_GATE=1 CARGO_BUILD_JOBS=2 bash tools/check.sh
echo "[gate61] check.sh OK — release perft_full_suite"
flock /home/nick/.cache/archon-build.lock \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
    --working-directory="$PWD" \
    cargo test --release -- --ignored perft_full_suite
echo "[gate61] ALL GATES GREEN"

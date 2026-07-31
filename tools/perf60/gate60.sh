#!/usr/bin/env bash
# perf study 60 — the merge gate, under the box's build discipline.
#   bash tools/perf60/gate60.sh
# CHECK_NNUE_GATE=1 because everything this study touches is under
# src/eval/nnue/, where the Bench: pin is blind (Cargo.toml has no default
# feature set, so the pinned `bench 16 1 6 default depth` is a PSQT run).
set -euo pipefail
cd "$(dirname "$0")/../.."
LOCK=/home/nick/.cache/archon-build.lock
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[gate60] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[gate60] avail=${avail}MB (waiting for lock)"
flock "$LOCK" systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  --setenv=CHECK_NNUE_GATE=1 \
  --setenv=CHECK_TEST_THREADS="${CHECK_TEST_THREADS:-3}" \
  bash tools/check.sh

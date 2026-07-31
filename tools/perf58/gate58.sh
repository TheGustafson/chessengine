#!/usr/bin/env bash
# perf study 58 — the full merge gate under this box's discipline.
#   bash tools/perf58/gate58.sh
# `CHECK_NNUE_GATE=1` is mandatory for anything under src/eval/nnue/: the
# `Bench:` pin is a DEFAULT-feature build and is blind to the whole SOTA path.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOCK=/home/nick/.cache/archon-build.lock
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[gate58] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
mkdir -p docs/research/perf-artifacts-58
export CARGO_BUILD_JOBS=2
export CHECK_NNUE_GATE=1
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  --setenv=CARGO_BUILD_JOBS=2 --setenv=CHECK_NNUE_GATE=1 \
  bash tools/check.sh

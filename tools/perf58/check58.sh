#!/usr/bin/env bash
# perf study 58 — a fast `cargo check` of the sota arch, under the same box
# discipline as build58.sh (memory refusal, build lock, MemoryMax, -j2).
#   bash tools/perf58/check58.sh [extra,features]
set -euo pipefail
cd "$(dirname "$0")/../.."
EXTRA=${1:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[check58] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
flock "$LOCK" env RUSTFLAGS="-C target-cpu=native" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo check --release --features "$FEATURES" --all-targets -j2

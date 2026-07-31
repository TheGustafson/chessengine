#!/usr/bin/env bash
# perf study 48 — build one labelled arm binary into bin48/.
#
#   bash tools/perf44/build48.sh <label> [extra,features]
#
# Every arm is built by THIS script so the pair differs only in the feature
# list — study 40 §6 is explicit that two binaries built by different paths are
# not comparable, and study 39 shipped three pin binaries from three source
# states and had to say so.
#
# OPERATIONAL (study 44's commission): swap on this box is effectively
# exhausted and the D49 trainer plus other agents are live, so every build
# serialises behind the SHARED lock and runs in a memory-capped scope. `flock`
# blocks until free — there is deliberately no give-up timeout, because a build
# that gives up leaves a half-built arm that looks like a measurement.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 2500 ]; then
  echo "[build48] REFUSING: only ${avail} MB available (< 2500)" >&2
  exit 1
fi
echo "[build48] $LABEL  features=$FEATURES  avail=${avail}MB  (waiting for lock)"

mkdir -p bin48
flock "$LOCK" env RUSTFLAGS="-C target-cpu=native" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  cargo build --release --features "$FEATURES" -j2
cp target/release/chessengine "bin48/$LABEL"
echo "[build48] -> bin48/$LABEL  .text md5 $(objdump -d --section=.text "bin48/$LABEL" | tail -n +3 | md5sum | cut -c1-32)"

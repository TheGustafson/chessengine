#!/usr/bin/env bash
# perf study 65 — build one labelled arm binary into bin65/.
#   bash tools/perf65/build65.sh <label> [extra,features]
#
# Same contract as tools/perf61/build61.sh, with two additions the 2026-07-31
# OOM post-mortem asked for:
#   * the job runs under tools/oomlast.sh, so the kernel picks the BUILD and
#     not the trainer if the box runs out (`systemd-run -p OOMScoreAdjust=`
#     provably does not work for --user --scope);
#   * the refusal reads SWAP as well as MemAvailable — swap exhaustion is the
#     earlier warning and does not show in `available` until it is too late.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock
CPUFLAG=${P65_CPU_FLAG:-native}

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
swapfree=$(awk '/^SwapFree:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[build65] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[build65] $LABEL  features=$FEATURES  cpu=$CPUFLAG  avail=${avail}MB swapfree=${swapfree}MB"

mkdir -p bin65
flock "$LOCK" env RUSTFLAGS="-C target-cpu=$CPUFLAG" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  ./tools/oomlast.sh cargo build --release --features "$FEATURES" -j2
cp target/release/chessengine "bin65/$LABEL"
md5=$(objdump -d --section=.text "bin65/$LABEL" | grep -v 'file format' | md5sum | cut -c1-8)
text=$(size -A "bin65/$LABEL" | awk '$1==".text"{print $2}')
echo "[build65] -> bin65/$LABEL  .text md5 $md5  size $text"

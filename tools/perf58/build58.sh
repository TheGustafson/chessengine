#!/usr/bin/env bash
# perf study 58 — build one labelled arm binary into bin58/.
#   bash tools/perf58/build58.sh <label> [extra,features]
# Same shape as study 55's build55.sh (which descends from 48/40 §6): ONE
# script builds every arm so a pair can only differ in the feature list.
#
# Box discipline (D49 training + perf study 56 both share this machine):
#   * refuse under 3500 MB available,
#   * flock against every other study's BUILDS — and only builds. The lock is
#     for compiler memory, not for measurement: a panel or a match that takes
#     it blocks every other study's compiles for its whole duration, and
#     `screen.sh` is contention-immune by design and never needed it.
#   * MemoryMax=6G, -j2.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[build58] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[build58] $LABEL  features=$FEATURES  avail=${avail}MB  (waiting for lock)"

mkdir -p bin58
flock "$LOCK" env RUSTFLAGS="-C target-cpu=native" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo build --release --features "$FEATURES" -j2
cp target/release/chessengine "bin58/$LABEL"
# .text md5 with the objdump filename banner stripped (study 55's trap: the
# banner carries the path, so an unstripped md5 differs for identical code).
md5=$(objdump -d --section=.text "bin58/$LABEL" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[build58] -> bin58/$LABEL  .text md5 $md5"

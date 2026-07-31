#!/usr/bin/env bash
# perf study 61 — build one labelled arm binary into bin61/.
#   bash tools/perf61/build61.sh <label> [extra,features]
#
# `P61_CPU_FLAG` selects the target-cpu (default `native`). The 2026-07-31
# handoff measured study 57's arms at −30.8 i/n on `native` and −187.6 on
# `x86-64-v3`: an arm's instruction number is a property of the COMPILER
# CONFIGURATION as much as of the arm, so the flag is explicit here and every
# panel says which it used.
#
# Box discipline (a GPU training run and perf study 59 share this machine):
# refuse under 3500 MB available, flock against every other study's BUILDS
# (builds only — never a panel), MemoryMax=6G, -j2.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock
CPUFLAG=${P61_CPU_FLAG:-native}

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[build61] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[build61] $LABEL  features=$FEATURES  cpu=$CPUFLAG  avail=${avail}MB  (waiting for lock)"

mkdir -p bin61
flock "$LOCK" env RUSTFLAGS="-C target-cpu=$CPUFLAG" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo build --release --features "$FEATURES" -j2
cp target/release/chessengine "bin61/$LABEL"
# .text md5 with the objdump filename banner stripped (study 55's trap: the
# banner carries the path, so an unstripped md5 differs for identical code).
md5=$(objdump -d --section=.text "bin61/$LABEL" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[build61] -> bin61/$LABEL  .text md5 $md5"

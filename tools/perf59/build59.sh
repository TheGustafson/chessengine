#!/usr/bin/env bash
# perf study 59 — build one labelled arm binary into bin59/.
#   bash tools/perf59/build59.sh <label> [extra,features]
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
# ⚠ STUDY 59'S OWN RULE: the gate build is not the shipped build, and an arm's
# value between them can differ 6x (study 57's arms: -30.8 i/n on native,
# -187.6 i/n on x86-64-v3). `native` on this box has AVX-512 VBMI2 and already
# performs by hand what some arms do explicitly. So the target-cpu is a
# PARAMETER of every number, and it is printed with the .text md5.
CPUFLAG=${P59_CPUFLAG:-native}

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[build59] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[build59] $LABEL  features=$FEATURES  cpu=$CPUFLAG  avail=${avail}MB  (waiting for lock)"

mkdir -p bin59
# A separate target dir per target-cpu: sharing one makes every alternation a
# full rebuild, and the point of this study is to build the SAME arm twice.
TDIR=target
[ "$CPUFLAG" = native ] || TDIR="target-$CPUFLAG"
flock "$LOCK" env RUSTFLAGS="-C target-cpu=$CPUFLAG" CARGO_TARGET_DIR="$TDIR" \
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  cargo build --release --features "$FEATURES" -j2
cp "$TDIR/release/chessengine" "bin59/$LABEL"
# .text md5 with the objdump filename banner stripped (study 55's trap: the
# banner carries the path, so an unstripped md5 differs for identical code).
md5=$(objdump -d --section=.text "bin59/$LABEL" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[build59] -> bin59/$LABEL  .text md5 $md5"

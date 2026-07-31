#!/usr/bin/env bash
# perf study 59 — build a labelled PGO arm into bin59/<label>-pgo.
#
#   bash tools/perf59/pgo59.sh <label> [extra,features]
#
# THE WHOLE METHODOLOGICAL POINT OF STUDY 59: every profile and every shipped
# claim is taken on the binary we actually ship, which is PGO. A non-PGO
# profile named `i8_deltas_into_inner` as the #1 target; in the shipped binary
# that symbol does not exist — PGO inlines it into `current_full`, which goes
# 3.72% -> 12.40%. Measuring the wrong binary re-prices the whole block.
#
# The eval-file argument to build_pgo.sh is REQUIRED (measured 2026-07-18:
# bench-only training leaves the NNUE kernels unprofiled and PGO inert).
#
# Box discipline as build59.sh: memory refusal, build lock, MemoryMax, and the
# PGO scratch parked on DISK — `/tmp` is tmpfs (RAM) on this box and a
# profile-generate run there competes with a live GPU training job.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock
NET=engines/canonical/net.cenn

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[pgo59] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[pgo59] $LABEL  features=$FEATURES  avail=${avail}MB  (waiting for lock)"

mkdir -p bin59 engines/bin
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  --setenv=TMPDIR="$PWD/.pgo-tmp" \
  bash tools/build_pgo.sh "$FEATURES" "p59-$LABEL" "$NET"
cp "engines/bin/p59-$LABEL" "bin59/$LABEL-pgo"
md5=$(objdump -d --section=.text "bin59/$LABEL-pgo" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[pgo59] -> bin59/$LABEL-pgo  .text md5 $md5"

#!/usr/bin/env bash
# perf study 60 — build a labelled PGO arm into bin60/<label>-pgo.
#
#   bash tools/perf60/pgo60.sh <label> [extra,features]
#
# The shipped binary is native+PGO; every VALUE claim in this study is measured
# on one of these, and every GUARD claim (instructions:u) on a build60.sh
# binary. `build_pgo.sh` honours PGO_CPU, so `P60_CPU_FLAG` holds target-cpu
# fixed here the same way it does for build60.sh.
#
# The eval-file argument to build_pgo.sh is REQUIRED (measured 2026-07-18:
# bench-only training leaves the NNUE kernels unprofiled and PGO inert).
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock
NET=engines/canonical/net.cenn
CPUFLAG=${P60_CPU_FLAG:-native}

avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
if [ "$avail" -lt 3500 ]; then
  echo "[pgo60] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[pgo60] $LABEL  features=$FEATURES  cpu=$CPUFLAG  avail=${avail}MB  (waiting for lock)"

mkdir -p bin60 engines/bin .pgo-tmp
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  --setenv=TMPDIR="$PWD/.pgo-tmp" \
  --setenv=PGO_CPU="$CPUFLAG" \
  bash tools/build_pgo.sh "$FEATURES" "p60-$LABEL" "$NET"
cp "engines/bin/p60-$LABEL" "bin60/$LABEL-pgo"
md5=$(objdump -d --section=.text "bin60/$LABEL-pgo" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[pgo60] -> bin60/$LABEL-pgo  .text md5 $md5"

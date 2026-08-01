#!/usr/bin/env bash
# perf study 67: memory-capped PGO build wrapper.
#   bash tools/s67_build.sh <features> <out-name>
set -euo pipefail
cd "$(dirname "$0")/.."
FEAT=${1:?features}
OUT=${2:?out name}
FREE=$(free -m | awk '/^Mem:/{print $7}')
if [ "$FREE" -lt 2500 ]; then echo "REFUSING: only ${FREE} MB available"; exit 1; fi
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  ./tools/oomlast.sh env CARGO_BUILD_JOBS=2 \
  bash tools/build_pgo.sh "$FEAT" "$OUT" engines/canonical/net.cenn

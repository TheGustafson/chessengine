#!/usr/bin/env bash
# perf study 67: same PGO pipeline as tools/s67_build.sh, but with
# `debug = "line-tables-only"` forced on through cargo's env override. Line
# tables are metadata: they add DWARF sections and change no codegen, which the
# study verifies by `cmp`ing the resulting `.text` against the plain build.
#   bash tools/s67_build_dbg.sh <features> <out-name>
set -euo pipefail
cd "$(dirname "$0")/.."
FEAT=${1:?features}
OUT=${2:?out name}
FREE=$(free -m | awk '/^Mem:/{print $7}')
if [ "$FREE" -lt 2500 ]; then echo "REFUSING: only ${FREE} MB available"; exit 1; fi
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  ./tools/oomlast.sh env CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
  bash tools/build_pgo.sh "$FEAT" "$OUT" engines/canonical/net.cenn

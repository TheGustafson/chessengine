#!/usr/bin/env bash
# perf study 59 — a PGO build WITH DWARF line tables, for source attribution
# inside the shipped inlining shape.
#
#   bash tools/perf59/pgoline59.sh <label> [extra,features]
#
# `pgo59.sh` builds the binary we ship and measure; this one is the DIAGNOSTIC
# twin. It is NOT the binary any number is quoted from — line tables change
# nothing about codegen, but the PGO profile is retrained per invocation and
# study 52 §7 item 3 flags that as an uncharacterised source of variation.
# Use it to find WHERE the instructions are, then quote pgo59.sh's binary.
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
  echo "[pgoline59] REFUSING: only ${avail} MB available (< 3500)" >&2
  exit 1
fi
echo "[pgoline59] $LABEL  features=$FEATURES  avail=${avail}MB  (waiting for lock)"

mkdir -p bin59 engines/bin
flock "$LOCK" systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 --quiet \
  --working-directory="$PWD" \
  --setenv=RUSTFLAGS_EXTRA=1 \
  env CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
  bash tools/build_pgo.sh "$FEATURES" "p59-$LABEL-line" "$NET"
cp "engines/bin/p59-$LABEL-line" "bin59/$LABEL-pgoline"
echo "[pgoline59] -> bin59/$LABEL-pgoline"

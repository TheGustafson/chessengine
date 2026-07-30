#!/usr/bin/env bash
# perf study 31 — build one labelled arm.
#
#   bash tools/perf31/build31.sh <label>
#
# Frame: the SHIPPED release profile (fat LTO, CGU=1), `target-cpu=native`,
# `--features sota,arch-d48x`.  CGU=1 is not optional: at 4 CGUs even
# `instructions:u` is unreliable (study 23; see tools/perf26/probe26.sh).
# Non-PGO — this is the PROBE frame, and studies 12-15 measured it overstating
# the PGO delta by 1.12-2.0x on data-layout changes.  Quote it as such.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
mkdir -p bin31

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3500 ] && break
  echo "[build31] only ${avail} MB available, waiting 120s" >&2
  sleep 120
done

systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target31 -q

cp target31/release/chessengine "bin31/$LABEL"
echo "[build31] bin31/$LABEL built"

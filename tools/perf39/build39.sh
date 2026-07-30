#!/usr/bin/env bash
# perf study 39 — build one labelled binary into bin39/.
#
#   bash tools/perf39/build39.sh <label> <features>
#
# Same frame as studies 29 and 31 (the two this study reconciles): the SHIPPED
# release profile (fat LTO, CGU = 1), `target-cpu=native`, non-PGO. CGU = 1 is
# not optional — at 4 CGUs even `instructions:u` is unreliable (study 23).
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
FEATURES=${2:?features}

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  swap=$(free -m | awk 'NR==3{print $4}')
  if [ "$avail" -ge 2500 ] && [ "$swap" -ge 900 ]; then break; fi
  echo "[build39] avail=${avail}MB swapfree=${swap}MB — waiting 120s"
  sleep 120
done

mkdir -p bin39
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$FEATURES" \
  --target-dir "target39-$LABEL" -q
cp "target39-$LABEL/release/chessengine" "bin39/$LABEL"
md5sum "bin39/$LABEL"
echo "[build39] bin39/$LABEL  features=$FEATURES"

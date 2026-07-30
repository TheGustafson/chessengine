#!/usr/bin/env bash
# perf study 35 -- build one labelled arm into bin35/.
#
#   bash tools/perf35/build35.sh <label> [extra cargo args...]
#
# Frame: the SHIPPED release profile (fat LTO, CGU=1), `target-cpu=native`,
# `--features sota,arch-d48x`.  Identical to studies 29/31/33 so the numbers
# compose with theirs.
#
# BOX DISCIPLINE: the D49 trainer is live and this box has 15 GB.  Builds wait
# for 3000 MB available, are capped at 6G with MemorySwapMax=0 behind
# tools/oomlast.sh, run -j2 at nice 19, and are pinned OFF the measurement CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
shift || true
mkdir -p bin35

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3000 ] && break
  echo "[build35] only ${avail} MB available, waiting 120s" >&2
  sleep 120
done

systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 19 taskset -c 0,1 \
  cargo build --release --features sota,arch-d48x "$@" \
  --target-dir target35 -q

cp target35/release/chessengine "bin35/$LABEL"
echo "[build35] bin35/$LABEL built  sha=$(sha256sum "bin35/$LABEL" | cut -c1-16)"

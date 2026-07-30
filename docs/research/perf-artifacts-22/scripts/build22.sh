#!/usr/bin/env bash
# perf study 22 — the BUILD step, split out of the probe so an arm can be
# built once and measured many times (study 20 rebuilt per probe).
#
#   bash docs/research/perf-artifacts-22/scripts/build22.sh
#
# BUILD DISCIPLINE: a GPU trainer is live, a sibling study also builds, and
# the box has 15 GB.  Memory-capped cgroup + tools/oomlast.sh so the kernel
# picks THIS build over the trainer.  thin LTO + 4 CGUs, -j2.
set -euo pipefail
avail() { free -m | awk 'NR==2 {print $7}'; }
if [ "$(avail)" -lt 3500 ]; then
  echo "only $(avail) MB available (need 3500) — waiting 120 s and rechecking"
  sleep 120
  if [ "$(avail)" -lt 3500 ]; then
    echo "REFUSING: still only $(avail) MB available" >&2
    exit 1
  fi
fi
echo "available: $(avail) MB"

systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  ${EXTRA_ENV:-} nice -n 10 cargo build --release --features sota,arch-d48x \
  --target-dir "${TARGET_DIR:-target-fast22}" -q
echo "built ${TARGET_DIR:-target-fast22}/release/chessengine"

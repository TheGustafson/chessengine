#!/usr/bin/env bash
# perf study 41 — the CODE-FOOTPRINT DOSE.
#
#   bash tools/perf41/buildalign41.sh <log2-align>...
#
# Builds `bin34/a41<k>` with `-C llvm-args=-align-all-functions=<k>`, i.e. every
# function forced to a 2^k byte boundary. This is the cleanest dose knob the
# front end has:
#
#   * it is PURELY a layout change — the padding sits BETWEEN functions and is
#     never executed, so the node count and `instructions:u` must not move
#     (gated by `identalign41.sh`, and the panel re-checks it every round);
#   * it moves the ONE thing we want to sweep, the engine's instruction
#     footprint, monotonically and by a large factor;
#   * it goes in BOTH directions from the x86 default of 16 B — k=0 packs the
#     code TIGHTER than shipping `main`, so the dose series contains a
#     candidate improvement, not only a degradation.
#
# Regressing `ex_no_retire.empty` on the icache miss rate over this series is
# what prices the icache share of study 40's 192.91 cycles/node. Neither term
# is assumed from a datasheet latency; both are measured on this engine.
set -euo pipefail
cd "$(dirname "$0")/../.."
FEATURES=${P41_FEATURES:-sota,arch-d48x}

for k in "$@"; do
  for _ in $(seq 60); do
    avail=$(free -m | awk 'NR==2{print $7}')
    [ "$avail" -ge 3000 ] && break
    echo "[buildalign41] only ${avail} MB available, waiting 120s"
    sleep 120
  done
  mkdir -p bin34
  flock /home/nick/.cache/archon-build.lock \
    systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    tools/oomlast.sh env \
    RUSTFLAGS="-C target-cpu=native -C llvm-args=-align-all-functions=$k" \
    CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --features "$FEATURES" \
    --target-dir "target41-a$k" -q
  cp "target41-a$k/release/chessengine" "bin34/a41$k"
  echo "[buildalign41] bin34/a41$k  align=2^$k B"
done
bash tools/perf40/textmd5.sh bin34/a41* 2>/dev/null || true

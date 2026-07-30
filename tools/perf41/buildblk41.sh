#!/usr/bin/env bash
# perf study 41 — the STRONG code-footprint dose.
#
#   bash tools/perf41/buildblk41.sh <log2-align>...
#
# `-align-all-functions` (buildalign41.sh) only pads BETWEEN functions, and
# after fat LTO this engine has few enough of them that 128 B alignment moves
# `.text` by 7.5%. Too small a dose to fit a slope through.
#
# `-align-all-nofallthru-blocks` pads every basic block that has NO
# fall-through predecessor — hundreds of sites per hot function, so the dose is
# large. LLVM names the flag for exactly the property this study needs: a block
# nobody falls into is only ever entered by a taken branch, so the padding is
# JUMPED OVER and never executed. `instructions:u` must therefore stay exact,
# same as the function-alignment series, and `identalign41.sh` gates it.
set -euo pipefail
cd "$(dirname "$0")/../.."
FEATURES=${P41_FEATURES:-sota,arch-d48x}

for k in "$@"; do
  for _ in $(seq 60); do
    avail=$(free -m | awk 'NR==2{print $7}')
    [ "$avail" -ge 3000 ] && break
    echo "[buildblk41] only ${avail} MB available, waiting 120s"
    sleep 120
  done
  mkdir -p bin34
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    tools/oomlast.sh env \
    RUSTFLAGS="-C target-cpu=native -C llvm-args=-align-all-nofallthru-blocks=$k" \
    CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --features "$FEATURES" \
    --target-dir "target41-k$k" -q
  cp "target41-k$k/release/chessengine" "bin34/k41$k"
  echo "[buildblk41] bin34/k41$k  nofallthru-block align=2^$k B"
done
bash tools/perf40/textmd5.sh bin34/k41* 2>/dev/null || true

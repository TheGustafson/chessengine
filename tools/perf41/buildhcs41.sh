#!/usr/bin/env bash
# perf study 41 — the LOW side of the footprint dose.
#
#   bash tools/perf41/buildhcs41.sh
#
# Every other dose point in this study makes the footprint BIGGER, so pricing
# the icache term at the shipping operating point means extrapolating a line
# back to x = 0 — a long way outside the data, which is exactly the move rule 8
# says to distrust. LLVM's hot-cold splitting pass gives a point on the other
# side: it extracts statically-cold regions (panic and unwind edges, `noreturn`
# tails, branches its static heuristic prices as rare) out of the enclosing
# function body and, with `-enable-cold-section`, parks them in a separate
# section away from the hot text.
#
# The executed path is unchanged, so this must stay node-identical and
# instruction-identical like the alignment series — `identalign41.sh` gates it.
# If `.text` shrinks and the icache miss rate falls with it, this is not only a
# dose point, it is a candidate landing.
set -euo pipefail
cd "$(dirname "$0")/../.."
FEATURES=${P41_FEATURES:-sota,arch-d48x}
for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3000 ] && break
  echo "[buildhcs41] only ${avail} MB available, waiting 120s"
  sleep 120
done
mkdir -p bin34
flock /home/nick/.cache/archon-build.lock \
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env \
  RUSTFLAGS="-C target-cpu=native -C llvm-args=-hot-cold-split -C llvm-args=-enable-cold-section" \
  CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --features "$FEATURES" \
  --target-dir target41-hcs -q
cp target41-hcs/release/chessengine bin34/hcs41
echo "[buildhcs41] bin34/hcs41"
bash tools/perf40/textmd5.sh bin34/hcs41

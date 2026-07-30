#!/usr/bin/env bash
# perf study 32 — the three gate numbers, measured (not asserted).
#
#   bash tools/perf32/gates32.sh
#
# This change ALTERS EVAL OUTPUT, so it is neither node-identical nor
# eval-bit-identical and all three pins move. CLAUDE.md's rule is that an
# intentional change moves the pin in the same commit with the new number
# stated, which is what this prints.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p bin32 docs/research/perf-artifacts-32
NET=$PWD/engines/canonical/net.cenn
FENS=$PWD/tools/nnue/calib/wl50.fens

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 2500 ] && break
  echo "[gates32] only ${avail} MB available, waiting 60s" >&2
  sleep 60
done

# 1. The Bench: trailer — DEFAULT features, x86-64-v3, depth 6 (tools/check.sh).
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  env RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --target-dir target32def -q
echo -n "Bench: "
./target32def/release/chessengine bench 16 1 6 default depth 2>&1 \
  | grep -oP 'Nodes searched\s*:\s*\K[0-9]+' | tail -1

# 2+3. The NNUE gate — sota,arch-d48x at x86-64-v3 (the gate's own arch).
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  env RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target32v3 -q
B=./target32v3/release/chessengine
echo -n "NNUE eval hash: "
"$B" nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1
echo -n "d13 node total: "
"$B" bench 16 1 13 "$FENS" depth "$NET" 2>&1 \
  | grep -oP 'Nodes searched\s*:\s*\K[0-9]+' | tail -1

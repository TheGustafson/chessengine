#!/usr/bin/env bash
# perf study 39 — the merge gate.
#
# Everything this study adds is behind an off-by-default cargo feature, so the
# shipped build must be behaviourally untouched: `Bench: 587683`, and — because
# `src/eval/nnue/` was touched — the NNUE behavioural gate as well (eval hash
# 2e1ede95…93cf79c4, d13 nodes 1739588). The `Bench:` pin is BLIND to NNUE work,
# which is exactly why `CHECK_NNUE_GATE=1` is not optional here.
#
# The probe features are NOT in the gate and must not be: they compute wrong
# evaluations on purpose, so a green suite under `probe-dstpin` would mean the
# probe was inert. `foldcount` IS gated, because its whole claim is that it
# changes nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  swap=$(free -m | awk 'NR==3{print $4}')
  if [ "$avail" -ge 3000 ] && [ "$swap" -ge 600 ]; then break; fi
  echo "[gates39] avail=${avail}MB swapfree=${swap}MB — holding 60s"
  sleep 60
done

echo "[gates39] === default gate + NNUE behavioural gate ==="
systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=2G -- \
  env CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 CARGO_BUILD_JOBS=2 \
  bash tools/check.sh

echo "[gates39] === clippy under every feature this study adds ==="
for f in foldcount probe-addcopy probe-actpin probe-accpin probe-dstpin probe-wpin; do
  echo "[gates39] --- $f ---"
  systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 -- \
    env CARGO_BUILD_JOBS=2 cargo clippy --all-targets \
    --features "sota,arch-d48x,$f" --target-dir target39-chk 2>&1 |
    grep -E "^error|^warning: unused|clippy::" | head -5 || true
done

echo "[gates39] === foldcount must be behaviourally inert ==="
systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 -- \
  env CARGO_BUILD_JOBS=2 cargo test --features sota,arch-d48x,foldcount \
  --target-dir target39-chk -q 2>&1 | tail -15

echo "[gates39] DONE"

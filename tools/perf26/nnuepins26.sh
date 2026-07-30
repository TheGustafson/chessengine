#!/usr/bin/env bash
# perf study 26 — reproduce the two NNUE-gate pins WITHOUT the whole check.sh.
#
#   bash tools/perf26/nnuepins26.sh
#
# The gate builds `-C target-cpu=x86-64-v3`, i.e. **AVX2, not AVX-512** — so it
# is the avx2 backend of every dispatched kernel that these two numbers cover,
# not the avx512 one the probe frame measures. Worth saying out loud: a study
# that only ever builds `target-cpu=native` has not run the code the merge gate
# checks.
set -euo pipefail
NET=engines/canonical/net.cenn
ARCH=$(cat engines/canonical/arch.txt)
FENS=tools/nnue/calib/wl50.fens
for _ in $(seq 40); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3500 ] && break
  echo "[nnuepins26] only ${avail} MB available, waiting 120s"
  sleep 120
done
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$ARCH" --target-dir target-nnuegate26 -q
BIN=target-nnuegate26/release/chessengine
hash=$("$BIN" nnue-eval "$NET" < "$FENS" | sha256sum | cut -d' ' -f1)
nodes=$("$BIN" bench 16 1 13 "$FENS" depth "$NET" 2>&1 >/dev/null | sed -n 's/^Nodes searched  : //p')
echo "eval hash : $hash"
echo "pin       : 5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38"
echo "d13 nodes : $nodes"
echo "pin       : 1922947"
[ "$hash" = "5ca978e012da1bb086799709c3ea5350cf6d49923c254f7975b6d780b0098b38" ] || { echo "HASH MISMATCH"; exit 1; }
[ "$nodes" = "1922947" ] || { echo "NODE MISMATCH"; exit 1; }
echo "both NNUE pins reproduce (avx2 backend)"

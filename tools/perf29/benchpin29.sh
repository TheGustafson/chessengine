#!/usr/bin/env bash
# perf study 29 — reproduce the D24 `Bench:` fingerprint without all of
# check.sh. Default features, portable build, the pinned invocation.
set -euo pipefail
cd "$(dirname "$0")/../.."
for _ in $(seq 40); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3500 ] && break
  echo "[benchpin29] only ${avail} MB available, waiting 120s"
  sleep 120
done
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --target-dir target-benchpin29 -q
n=$(./target-benchpin29/release/chessengine bench 16 1 6 default depth 2>&1 >/dev/null \
    | sed -n 's/^Nodes searched  : //p')
echo "Bench: $n"
echo "pin  : 587683"
[ "$n" = "587683" ] || { echo "BENCH MISMATCH"; exit 1; }
echo "the D24 bench fingerprint reproduces"

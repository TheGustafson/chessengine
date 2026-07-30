#!/usr/bin/env bash
# perf study 25 — PGO build, memory-capped, output into bin25/.
#
#   bash docs/research/perf-artifacts-25/scripts/pgo25.sh <label>
#
# Same pipeline as tools/build_pgo.sh (features sota,arch-d48x; the eval-file
# argument is REQUIRED for NNUE binaries or the NNUE kernels stay unprofiled
# and PGO is inert) but: -j2, capped cgroup, oomlast, and it drops the binary
# in bin25/<label> instead of engines/bin so the shared checkout stays clean.
set -euo pipefail
LABEL=${1:?label}
FEATURES=sota,arch-d48x
TD=target-pgo25
PROF=$PWD/target-pgo25-prof
EVALFILE=engines/canonical/net.cenn
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found"; exit 1; }

wait_mem() {
  for _ in $(seq 40); do
    avail=$(free -m | awk 'NR==2{print $7}')
    [ "$avail" -ge 3500 ] && return 0
    echo "[pgo25] only ${avail} MB available, waiting 120s"
    sleep 120
  done
}

build() {
  wait_mem
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native $1" CARGO_BUILD_JOBS=2 \
    nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
}

rm -rf "$PROF"; mkdir -p "$PROF"
echo "[pgo25] phase 1: instrumented"
build "-C profile-generate=$PROF"
echo "[pgo25] phase 2: training workload"
"./$TD/release/chessengine" bench 16 1 8 default depth > /dev/null
{ printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 14\n' "$EVALFILE"
  sleep 25
  printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\ngo depth 12\n'
  sleep 25; echo quit; } | "./$TD/release/chessengine" > /dev/null
echo "[pgo25] phase 3: merge"
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"
echo "[pgo25] phase 4: optimized rebuild"
build "-C profile-use=$PROF/merged.profdata"
mkdir -p bin25
cp "$TD/release/chessengine" "bin25/$LABEL"
echo "[pgo25] done: bin25/$LABEL"

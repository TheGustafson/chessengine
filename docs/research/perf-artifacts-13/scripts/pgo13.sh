#!/usr/bin/env bash
# perf study 13 — worktree-local PGO build (tools/build_pgo.sh writes into the
# SHARED engines/bin, which this study must not touch).  Same pipeline, same
# flags; output goes to bin13/<name>.
#   bash pgo13.sh <out-name> [features]
set -euo pipefail
OUT=${1:?out name}
FEATURES=${2:-sota,arch-d48x}
NET=engines/canonical/net.cenn
TD=target-pgo13
PROF=/home/nick/pgo13-prof
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found"; exit 1; }
free -g | awk 'NR==2 {if ($7 < 5) {print "REFUSING: only " $7 " GB available"; exit 1}}'
mkdir -p bin13

rm -rf "$PROF"; mkdir -p "$PROF"
RUSTFLAGS="-C target-cpu=native -C profile-generate=$PROF" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
"./$TD/release/chessengine" bench 16 1 8 default depth > /dev/null 2>&1
{ printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 14\n' "$NET"
  sleep 25
  printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\ngo depth 12\n'
  sleep 25; echo quit; } | "./$TD/release/chessengine" > /dev/null 2>&1
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"
RUSTFLAGS="-C target-cpu=native -C profile-use=$PROF/merged.profdata" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
cp "$TD/release/chessengine" "bin13/$OUT"
echo "[pgo13] done: bin13/$OUT"

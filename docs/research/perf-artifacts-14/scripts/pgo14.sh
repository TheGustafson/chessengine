#!/usr/bin/env bash
# perf study 14 — worktree-local PGO build (tools/build_pgo.sh writes into the
# SHARED engines/bin, which this study must not touch).  Same pipeline, same
# flags; output goes to bin14/<name>.
#   bash pgo14.sh <out-name> [features]
set -euo pipefail
OUT=${1:?out name}
FEATURES=${2:-sota,arch-d48x}
NET=engines/canonical/net.cenn
TD=target-pgo14
PROF=/home/nick/pgo14-prof
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found"; exit 1; }
free -m | awk 'NR==2 {if ($7 < 4500) {print "REFUSING: only " $7 " MB available (need 4500)"; exit 1}}'
mkdir -p bin14

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
cp "$TD/release/chessengine" "bin14/$OUT"
echo "[pgo14] done: bin14/$OUT"

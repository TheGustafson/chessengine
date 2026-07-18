#!/usr/bin/env bash
# R3 (D45a speed program): PGO build pipeline for machine-local binaries.
#   bash tools/build_pgo.sh <features> <out-binary-name>
#   e.g. bash tools/build_pgo.sh sota probe-sota-pgo
#        bash tools/build_pgo.sh nnue-l1-2048 probe-l2048-pgo
# Produces engines/bin/<out> built with target-cpu=native + profile-use.
# The training workload is the pinned bench mix (16 positions, depth 8) plus
# a startpos depth-15 search — search+eval hot paths, both movegen shapes.
# PGO binaries are MACHINE-LOCAL research tools (D16): never used for the
# Bench: trailer (that stays on the portable default build) and SPRT pairs
# must both be PGO or both not.
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=${1:?features (e.g. sota | nnue-l1-2048)}
OUT=${2:?output binary name under engines/bin/}
TD=target-pgo-$FEATURES
PROF=/tmp/pgo-prof-$FEATURES
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found (rustup component add llvm-tools)"; exit 1; }

rm -rf "$PROF"; mkdir -p "$PROF"
echo "[pgo] phase 1: instrumented build ($FEATURES)"
RUSTFLAGS="-C target-cpu=native -C profile-generate=$PROF" \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
echo "[pgo] phase 2: training workload (bench depth 8 + startpos depth 15)"
"./$TD/release/chessengine" bench 16 1 8 default depth > /dev/null
{ printf 'uci\nisready\nposition startpos\ngo depth 15\n'; sleep 20; echo quit; } \
  | "./$TD/release/chessengine" > /dev/null
echo "[pgo] phase 3: merge profiles"
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"
echo "[pgo] phase 4: optimized rebuild"
RUSTFLAGS="-C target-cpu=native -C profile-use=$PROF/merged.profdata" \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
cp "$TD/release/chessengine" "engines/bin/$OUT"
echo "[pgo] done: engines/bin/$OUT"

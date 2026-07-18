#!/usr/bin/env bash
# R3 (D45a speed program): PGO build pipeline for machine-local binaries.
#   bash tools/build_pgo.sh <features> <out-binary-name> [eval-file]
#   e.g. bash tools/build_pgo.sh sota probe-sota-pgo data/nnue/nets/d40-uho-720sb.cenn
#        bash tools/build_pgo.sh nnue-l1-2048 probe-l2048-pgo data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn
# Produces engines/bin/<out> built with target-cpu=native + profile-use.
# The training workload MUST exercise the real inference path: bench runs the
# DEFAULT embedded eval only (its 4th arg is a FEN file), so a bench-only
# profile leaves the feature build's NNUE kernels unprofiled and PGO inert
# (measured 2026-07-18: pgo == native on the sota path). Pass the eval-file
# the binary will actually serve; the workload is bench (search coverage,
# default eval) + a UCI session on that net (deep startpos + kiwipete).
# PGO binaries are MACHINE-LOCAL research tools (D16): never used for the
# Bench: trailer (that stays on the portable default build) and SPRT pairs
# must both be PGO or both not.
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=${1:?features (e.g. sota | nnue-l1-2048)}
OUT=${2:?output binary name under engines/bin/}
EVALFILE=${3:-}
TD=target-pgo-$FEATURES
PROF=/tmp/pgo-prof-$FEATURES
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found (rustup component add llvm-tools)"; exit 1; }

rm -rf "$PROF"; mkdir -p "$PROF"
echo "[pgo] phase 1: instrumented build ($FEATURES)"
RUSTFLAGS="-C target-cpu=${PGO_CPU:-native} -C profile-generate=$PROF" \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
echo "[pgo] phase 2: training workload (bench depth 8 + real-net UCI searches)"
"./$TD/release/chessengine" bench 16 1 8 default depth > /dev/null
if [ -n "$EVALFILE" ]; then
  { printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 14\n' "$EVALFILE"
    sleep 25
    printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\ngo depth 12\n'
    sleep 25; echo quit; } | "./$TD/release/chessengine" > /dev/null
else
  { printf 'uci\nisready\nposition startpos\ngo depth 15\n'; sleep 20; echo quit; } \
    | "./$TD/release/chessengine" > /dev/null
fi
echo "[pgo] phase 3: merge profiles"
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"
echo "[pgo] phase 4: optimized rebuild"
RUSTFLAGS="-C target-cpu=${PGO_CPU:-native} -C profile-use=$PROF/merged.profdata" \
  nice -n 10 cargo build --release --features "$FEATURES" --target-dir "$TD" -q
cp "$TD/release/chessengine" "engines/bin/$OUT"
echo "[pgo] done: engines/bin/$OUT"

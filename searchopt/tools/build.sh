#!/usr/bin/env bash
# Build the instrumented binary for searchopt work.
#
# BOTH features are mandatory and each has bitten us:
#   search-control  -> the tracer emits rows at all
#   tune            -> params.rs scalars become setoption-able; WITHOUT IT every arm
#                      returns byte-identical results and you will conclude "no effect"
#
# The output is copied to engines/bin/archon-trace because a later default-feature
# `cargo build --release` silently overwrites target/release/chessengine and the tracer
# goes quiet with no error.
set -euo pipefail
cd "$(dirname "$0")/../.."
free -g | awk 'NR==2 {if ($7 < 3) {print "REFUSING: only "$7"G available; a build here has OOMed the box before"; exit 1}}'
RUSTFLAGS="-C target-cpu=x86-64-v3" nice -n 5 cargo build --release \
    --features sota,arch-d47v,search-control,tune -j 4
cp target/release/chessengine engines/bin/archon-trace
echo "ok -> engines/bin/archon-trace"
# The default build must stay bit-identical: the Bench pin is 587683.
cargo build --release -j 4 >/dev/null 2>&1
./target/release/chessengine bench 16 1 6 default depth 2>&1 | grep -i "nodes searched  :"

#!/usr/bin/env bash
# tools/perf27/build27.sh <label>
#
# Perf study 27's build step. CODEGEN_UNITS=1 + thin LTO, always.
#
# WHY CGU=1: at CGU=4 a source change in one file reshuffles inlining across
# the whole crate and `instructions:u` stops being a gate -- study 23 measured
# +48 i/n on an arm whose own loop body shrank 189 -> 102 static instructions,
# with the increase attributed to a function the patch never touched. At CGU=1
# the inert-source-change null is 275 instructions in 30.0 G (1e-6 %).
#
# Memory-gated (a live 400-superbatch trainer owns ~3 GB) and cgroup-capped so
# a build cannot OOM the box.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

label=${1:?usage: build27.sh <label>}
BINDIR=$ROOT/perf27/bin
mkdir -p "$BINDIR"

while :; do
    avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    [ "$avail" -ge 3500 ] && break
    printf 'build27: only %s MB available (<3500), sleeping 120s\n' "$avail" >&2
    sleep 120
done
printf 'build27: %s MB available, building %s\n' "$avail" "$label" >&2

env \
    CARGO_PROFILE_RELEASE_LTO=thin \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_BUILD_JOBS=2 \
    RUSTFLAGS="-C target-cpu=native" \
    systemd-run --user --quiet --scope -p MemoryMax=3G -p MemorySwapMax=0 \
      "$ROOT/tools/oomlast.sh" \
      nice -n 10 cargo build --release --features sota,arch-d48x \
        --target-dir "$ROOT/target/p27" -j2 --quiet

cp "$ROOT/target/p27/release/chessengine" "$BINDIR/$label"
printf 'build27: stashed %s\n' "$BINDIR/$label" >&2

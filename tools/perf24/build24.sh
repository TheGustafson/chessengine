#!/usr/bin/env bash
# tools/perf24/build24.sh <label> [profile]
#
# Perf study 24's build step. Builds the CURRENT working tree into a stashed
# binary at perf24/bin/<label>, under a memory cgroup so it cannot OOM the
# live trainer, and refuses to start when the box is tight.
#
#   profile = probe   thin LTO / 4 CGU  (fast screening builds, ~1 min)
#   profile = ship    fat  LTO / 1 CGU  (the shipping codegen, ~3 min)
#
# Always: --features sota,arch-d48x, -C target-cpu=native, non-PGO.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD

label=${1:?usage: build24.sh <label> [probe|ship]}
profile=${2:-probe}

BINDIR=$ROOT/perf24/bin
mkdir -p "$BINDIR"

# --- the memory gate (commission: free -m before EVERY build) --------------
while :; do
    avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -ge 3500 ]; then break; fi
    printf 'build24: only %s MB available (<3500), sleeping 120s\n' "$avail" >&2
    sleep 120
done
printf 'build24: %s MB available, building %s (%s)\n' "$avail" "$label" "$profile" >&2

case $profile in
    probe) TD=$ROOT/target/p24-probe; LTO=thin; CGU=4 ;;
    ship)  TD=$ROOT/target/p24-ship;  LTO=fat;  CGU=1 ;;
    *) echo "unknown profile $profile" >&2; exit 2 ;;
esac

env \
    CARGO_PROFILE_RELEASE_LTO="$LTO" \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS="$CGU" \
    CARGO_BUILD_JOBS=2 \
    RUSTFLAGS="-C target-cpu=native" \
    systemd-run --user --quiet --scope -p MemoryMax=3G -p MemorySwapMax=0 \
      "$ROOT/tools/oomlast.sh" \
      cargo build --release --features sota,arch-d48x --target-dir "$TD" -j2 --quiet

cp "$TD/release/chessengine" "$BINDIR/$label"
printf 'build24: stashed %s\n' "$BINDIR/$label" >&2

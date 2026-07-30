#!/usr/bin/env bash
# perf study 33 -- a PGO build that ALSO carries DWARF line tables.
#
#   bash tools/perf33/buildpgo33.sh <label>
#
# WHY THIS EXISTS.  Study 28's headline for this study -- `search::<NonPv>` is
# 19.90% of the profile and stalls +303.4 cycles/node over Stockfish -- was
# measured on a PGO binary.  The study-29/31 probe frame is NON-PGO, and in it
# the same symbol is only 8.94% of load-stall samples.  A symbol's share is a
# function of what the compiler inlined INTO it, and PGO inlines on measured
# hotness, so the two builds do not have the same symbol at all.  Nothing about
# the 19.90% can be checked without reproducing the frame it came from.
#
# Same pipeline as tools/build_pgo.sh (which is the project standard) with
# three changes: `--profile profiling` so the result carries line tables
# (proven codegen-inert by tools/perf33/textnull33.py), the profile directory
# on DISK rather than tmpfs /tmp, and the box discipline this study runs under
# (2 jobs, cores 0/1, 6G cap, oomlast so the D49 trainer is never the victim).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
LABEL=${1:?label}
FEATURES=sota,arch-d48x
NET=$ROOT/engines/canonical/net.cenn
TD=$ROOT/target33-pgo
PROF=$ROOT/perf33data/pgoprof
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata | head -1)
[ -n "$PROFDATA" ] || { echo "llvm-profdata not found"; exit 1; }
mkdir -p bin33
rm -rf "$PROF"; mkdir -p "$PROF"

wait_mem() {
  for _ in $(seq 60); do
    avail=$(free -m | awk 'NR==2{print $7}')
    [ "$avail" -ge 3000 ] && return 0
    echo "[pgo33] only ${avail} MB available, waiting 120s" >&2
    sleep 120
  done
}

build() {  # $1 = extra RUSTFLAGS
  wait_mem
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
    tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native $1" CARGO_BUILD_JOBS=2 \
    nice -n 19 taskset -c 0,1 \
    cargo build --profile profiling --features "$FEATURES" --target-dir "$TD" -q
}

echo "[pgo33] phase 1: instrumented build"
build "-C profile-generate=$PROF"

echo "[pgo33] phase 2: training workload"
BIN=$TD/profiling/chessengine
nice -n 19 taskset -c 0,1 "$BIN" bench 16 1 8 default depth > /dev/null 2>&1
{ printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 14\n' "$NET"
  sleep 25
  printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -\ngo depth 12\n'
  sleep 25; echo quit; } | nice -n 19 taskset -c 0,1 "$BIN" > /dev/null 2>&1

echo "[pgo33] phase 3: merge"
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"

echo "[pgo33] phase 4: optimised rebuild"
build "-C profile-use=$PROF/merged.profdata"

cp "$BIN" "bin33/$LABEL"
echo "[pgo33] bin33/$LABEL built"

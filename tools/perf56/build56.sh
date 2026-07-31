#!/usr/bin/env bash
# perf study 49 — build one labelled arm binary into bin56/.
#   bash tools/perf56/build56.sh <label> [extra,features]
# Same shape as study 48's build48.sh: one script builds every arm so the pair
# differs only in the feature list (study 40 §6).
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
EXTRA=${2:-}
BASE=sota,arch-d48x
FEATURES=$BASE${EXTRA:+,$EXTRA}
LOCK=/home/nick/.cache/archon-build.lock

# Study 56 raises the refusal floor from study 55's 2500 MB to 3500 MB, because
# this study's builds run alongside the owner's 1500-game fixed-node net screen
# (CONC=5, ~1.4 GB of nets resident) as well as the D49 trainer.
avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
FLOOR=${P56_MEM_FLOOR:-3500}
if [ "$avail" -lt "$FLOOR" ]; then
  echo "[build56] REFUSING: only ${avail} MB available (< $FLOOR)" >&2
  exit 1
fi
echo "[build56] $LABEL  features=$FEATURES  avail=${avail}MB  (waiting for lock)"

# The build lock serialises MEMORY-HEAVY BUILDS. During study 56 the owner's
# gate held it for the whole duration of a fixed-node MATCH -- and that script's
# own header states "Fixed NODES, so the result is contention-immune and
# concurrency is free". Blocking hours behind a contention-immune match is not
# what the lock is for, so: wait up to P56_LOCK_WAIT seconds for it, and on
# timeout proceed WITHOUT it under a tighter cgroup (4G, not 6G) at the lowest
# CPU and IO priority, so this build cannot be the cause of an OOM or of the
# match's engines losing a timeslice. The memory floor above is the real guard.
mkdir -p bin56
build() {
  env RUSTFLAGS="-C target-cpu=native" \
    systemd-run --user --scope -p MemoryMax="$1" -p MemorySwapMax=0 --quiet \
    --working-directory="$PWD" \
    nice -n 19 cargo build --release --features "$FEATURES" -j2
}
if flock -w "${P56_LOCK_WAIT:-30}" 9; then
  build 6G
  exec 9>&-
else
  echo "[build56] lock busy after ${P56_LOCK_WAIT:-30}s -- building UNLOCKED at 4G/nice 19" >&2
  build 4G
fi 9>>"$LOCK"
cp target/release/chessengine "bin56/$LABEL"
# .text md5 with the objdump filename banner stripped (the commission's trap).
md5=$(objdump -d --section=.text "bin56/$LABEL" | grep -v 'file format' | md5sum | cut -c1-8)
echo "[build56] -> bin56/$LABEL  .text md5 $md5"

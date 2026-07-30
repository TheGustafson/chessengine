#!/usr/bin/env bash
# perf study 42 -- build one labelled arm into bin42/.
#
#   bash tools/perf42/build42.sh <label> [features] [profile]
#
# Frame: the SHIPPED release profile (fat LTO, CGU=1), `target-cpu=native`,
# `--features sota,arch-d48x` plus whatever the arm adds. Descends from
# tools/perf34/build34.sh; the third argument selects the `profiling` profile
# (release codegen + line tables) for the IBS source-attribution runs.
#
# BOX DISCIPLINE: a D49 trainer and three other agents share this machine, and
# SWAP IS EXHAUSTED (~60 MB free against ~9.2 GB RAM). Two concurrent builds is
# 12 GB of MemoryMax against 9.2 GB with no swap headroom to absorb the
# overshoot -- the exact configuration that OOMed this box once already, taking
# the trainer with it. So, in order:
#   1. wait for 3000 MB available (below), then
#   2. `flock` a SHARED lock so builds across all agents QUEUE instead of
#      colliding -- no timeout, because waiting is the point, then
#   3. a 6G cgroup with swap disabled behind tools/oomlast.sh, -j2, pinned to
#      cores 0,1 at nice 19 so a build never lands on the measurement core.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
FEATURES=${2:-sota,arch-d48x}
PROFILE=${3:-release}

if [ "$PROFILE" = release ]; then
  PROF_ARG=(--release)
else
  PROF_ARG=(--profile "$PROFILE")
fi

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3000 ] && break
  echo "[build42] only ${avail} MB available, waiting 120s" >&2
  sleep 120
done

mkdir -p bin42
LOCK=${ARCHON_BUILD_LOCK:-$HOME/.cache/archon-build.lock}
# `touch`, never `> "$LOCK"`: truncation is pointless on a zero-byte lock and
# opening another agent's in-use file with O_TRUNC is a gratuitous risk.
mkdir -p "$(dirname "$LOCK")"; touch "$LOCK" 2>/dev/null || true
flock "$LOCK" \
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 19 taskset -c 0,1 \
  cargo build "${PROF_ARG[@]}" --features "$FEATURES" \
  --target-dir "target42-$LABEL" -q

cp "target42-$LABEL/$PROFILE/chessengine" "bin42/$LABEL"
echo "[build42] bin42/$LABEL  features=$FEATURES  profile=$PROFILE"

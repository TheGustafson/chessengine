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
# BOX DISCIPLINE: a D49 trainer and two other studies share this machine.
# Builds wait for 3000 MB available, run under a 6G cgroup with swap disabled
# behind tools/oomlast.sh, at -j2, pinned to cores 0,1 at nice 19 so they never
# land on the measurement core.
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
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 19 taskset -c 0,1 \
  cargo build "${PROF_ARG[@]}" --features "$FEATURES" \
  --target-dir "target42-$LABEL" -q

cp "target42-$LABEL/$PROFILE/chessengine" "bin42/$LABEL"
echo "[build42] bin42/$LABEL  features=$FEATURES  profile=$PROFILE"

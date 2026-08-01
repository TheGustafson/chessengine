#!/usr/bin/env bash
# perf study 69 — memory-capped PGO build wrapper (study 67's build67.sh, with
# the profile directory forced onto real disk because /tmp is tmpfs=RAM on this
# box and this worktree builds several configurations back to back).
#   bash tools/perf69/build69.sh <features> <out-name>
set -euo pipefail
cd "$(dirname "$0")/../.."
FEAT=${1:?features}
OUT=${2:?out name}
FREE=$(free -m | awk '/^Mem:/{print $7}')
if [ "$FREE" -lt 2500 ]; then echo "REFUSING: only ${FREE} MB available"; exit 1; fi
mkdir -p "$PWD/target-prof"
export PGO_PROF_DIR="$PWD/target-prof"
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  ./tools/oomlast.sh env CARGO_BUILD_JOBS=2 "PGO_PROF_DIR=$PGO_PROF_DIR" \
  "PGO_EXTRA_RUSTFLAGS=${PGO_EXTRA_RUSTFLAGS:-}" \
  "CARGO_PROFILE_RELEASE_DEBUG=${CARGO_PROFILE_RELEASE_DEBUG:-false}" \
  bash tools/build_pgo.sh "$FEAT" "$OUT" engines/canonical/net.cenn

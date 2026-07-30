#!/usr/bin/env bash
# perf study 32 — build one labelled arm in the study-31 probe frame.
#
#   bash tools/perf32/build32.sh <label> [git-ref]
#
# Frame: the SHIPPED release profile (fat LTO, CGU=1), `target-cpu=native`,
# `--features sota,arch-d48x`, non-PGO.  CGU=1 is not optional (study 23).
# With a git-ref it builds that ref's tree in a detached worktree, which is how
# the baseline arm is produced without stashing.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
REF=${2:-}
mkdir -p bin32

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 2500 ] && break
  echo "[build32] only ${avail} MB available, waiting 60s" >&2
  sleep 60
done

SRC=$PWD
if [ -n "$REF" ]; then
  SRC=$PWD/bin32/src-$LABEL
  rm -rf "$SRC"
  git worktree remove --force "$SRC" 2>/dev/null || true
  git worktree add --detach "$SRC" "$REF" >/dev/null
fi

systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 10 cargo build --manifest-path "$SRC/Cargo.toml" --release \
  --features sota,arch-d48x --target-dir "$PWD/target32rel-$LABEL" -q

cp "target32rel-$LABEL/release/chessengine" "bin32/$LABEL"
echo "[build32] bin32/$LABEL built"

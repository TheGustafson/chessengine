#!/usr/bin/env bash
# perf study 69 — a NON-PGO release build, as the PGO-confound CONTROL.
#   bash tools/perf69/buildnp69.sh <features> <out-name>
#
# Two PGO builds of the SAME source are byte-identical on this box (proved
# again here: `s69-b0` vs `s69-b0n`, `cmp` clean). Two PGO builds of DIFFERENT
# source are not merely "the arm": the instrumented run's counters differ, the
# merged profile differs, and LLVM re-decides inlining in code the arm never
# touched -- study 69 §2.4 measured `forward_into` moving +0.71 share points and
# `i8_deltas_into_inner` -0.81 on a diff that touches neither.
#
# A non-PGO pair has no profile to differ, so it isolates the SOURCE change.
# It is a control, never a verdict: CLAUDE.md's rule that only the PGO binary
# may be profiled still stands, and the shipped number is still the PGO one.
set -euo pipefail
cd "$(dirname "$0")/../.."
FEAT=${1:?features}
OUT=${2:?out name}
FREE=$(free -m | awk '/^Mem:/{print $7}')
if [ "$FREE" -lt 2500 ]; then echo "REFUSING: only ${FREE} MB available"; exit 1; fi
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  ./tools/oomlast.sh env CARGO_BUILD_JOBS=2 RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --features "$FEAT" --target-dir target-np -q
cp target-np/release/chessengine "engines/bin/$OUT"
echo "[np] done: engines/bin/$OUT"

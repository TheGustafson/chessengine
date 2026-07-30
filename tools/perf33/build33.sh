#!/usr/bin/env bash
# perf study 33 -- build one labelled arm into bin33/.
#
#   bash tools/perf33/build33.sh <label> [extra cargo args...]
#
# Frame: the SHIPPED release profile (fat LTO, CGU=1), `target-cpu=native`,
# `--features sota,arch-d48x`.  Same frame as studies 29/31 so numbers compose.
#
# P33_PROFILE=profiling builds the line-table variant instead of --release.
# The `profiling` profile INHERITS release and adds `debug =
# "line-tables-only"`; study 33 verifies that this is codegen-identical by
# sha256-ing the .text section of both (tools/perf33/textid33.sh).  That check
# is the NULL for the whole source-attribution instrument.
#
# BOX DISCIPLINE: study 32 is measuring on CPU 6 (sibling 14).  Builds are
# pinned to cores 0,1 at nice 19 so they do not land on that core, and capped
# at 6G with MemorySwapMax=0 behind tools/oomlast.sh so the D49 trainer is
# never the OOM victim.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
shift || true
mkdir -p bin33

PROFILE=${P33_PROFILE:-release}
if [ "$PROFILE" = release ]; then
  PROF_ARG=(--release)
else
  PROF_ARG=(--profile "$PROFILE")
fi

for _ in $(seq 60); do
  avail=$(free -m | awk 'NR==2{print $7}')
  [ "$avail" -ge 3000 ] && break
  echo "[build33] only ${avail} MB available, waiting 120s" >&2
  sleep 120
done

systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=2 \
  nice -n 19 taskset -c 0,1 \
  cargo build "${PROF_ARG[@]}" --features sota,arch-d48x "$@" \
  --target-dir target33 -q

cp "target33/$PROFILE/chessengine" "bin33/$LABEL"
echo "[build33] bin33/$LABEL built ($PROFILE)"

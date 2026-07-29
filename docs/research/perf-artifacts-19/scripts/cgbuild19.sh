#!/usr/bin/env bash
# perf study 19 (instrument inherited from 17) — build the AVX2 (x86-64-v3) binary the callgrind instrument
# needs.  valgrind cannot execute AVX-512, and `-C target-cpu=native` on this
# box (Zen 4) emits it even in non-SIMD code — the smoke test died with an
# illegal opcode inside `std::rt::lang_start_internal`.
#
# Same OOM discipline as probe19.sh: own cgroup, no swap, oomlast.sh.
set -euo pipefail
free -m | awk 'NR==2 {if ($7 < 2600) {print "REFUSING: only " $7 " MB available (need 2600)"; exit 1}}'
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 -- \
  tools/oomlast.sh env RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
  nice -n 10 cargo build --release --features sota,arch-d48x --target-dir target-cg19 -q
echo "built target-cg19/release/chessengine"

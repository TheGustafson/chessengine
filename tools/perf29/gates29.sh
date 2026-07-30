#!/usr/bin/env bash
# perf study 29 — the merge gates for this study's surface.
#
#   bash tools/perf29/gates29.sh
#
# The instruments this study lands are all behind OFF-BY-DEFAULT features, so
# the gate has to be run per feature set: a `cfg`-gated counter that does not
# compile under `probe-wpin` is exactly the kind of breakage `cargo check`
# with default features cannot see.
#
# RUST_TEST_THREADS=3: the sota suite OOM-kills itself silently at 16 threads
# in a 3 GB cgroup — no `test result` line, reads exactly like a hang.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUST_TEST_THREADS=3
export CARGO_BUILD_JOBS=2
TD=target-gate29
run() {
  echo "===== $* ====="
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
    tools/oomlast.sh nice -n 10 "$@"
}
run cargo fmt --check
# The project gate (tools/check.sh) is DEFAULT features with -D warnings.
run cargo clippy --all-targets --target-dir "$TD" -- -D warnings
# The sota clippy is advisory (main carries 8 pre-existing lints under this
# feature set); diff its output rather than gating on an empty one.
for f in sota,arch-d48x foldcount,arch-d48x probe-accpin,arch-d48x probe-wpin,arch-d48x; do
  run cargo clippy --all-targets --features "$f" --target-dir "$TD"
done
run cargo test --target-dir "$TD" -q
run cargo test --features sota --target-dir "$TD" -q
run cargo test --features sota,arch-d48x --target-dir "$TD" -q
run cargo test --features foldcount,arch-d48x --target-dir "$TD" -q
echo "===== ALL PERF-29 GATES PASSED ====="

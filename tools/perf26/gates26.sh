#!/usr/bin/env bash
# perf study 26 — the merge gates, all five, in one memory-capped run.
#
#   bash tools/perf26/gates26.sh
#
# RUST_TEST_THREADS=3: the sota suite OOM-kills itself silently at 16 threads
# in a 3 GB cgroup — no `test result` line, reads exactly like a hang.
set -euo pipefail
export RUST_TEST_THREADS=3
export CARGO_BUILD_JOBS=2
run() {
  echo "===== $* ====="
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- \
    tools/oomlast.sh nice -n 10 "$@"
}
run cargo fmt --check
# The project gate (tools/check.sh line 22) is DEFAULT features with -D warnings.
run cargo clippy --all-targets --target-dir target-gate26 -- -D warnings
# The sota clippy is advisory: main already carries 8 lints under this feature
# set (dead `read_i8s`/`NNZ_IOTA`, and six in `simple_halfka`/`protected_halfka`
# which are outside every current study's scope), so it cannot be run with
# `-D warnings`. Diff its output instead of gating on an empty one.
run cargo clippy --all-targets --features sota,arch-d48x --target-dir target-gate26
run cargo test --target-dir target-gate26 -q
run cargo test --features sota --target-dir target-gate26 -q
run cargo test --features sota,arch-d48x --target-dir target-gate26 -q
echo "===== ALL GATES PASSED ====="

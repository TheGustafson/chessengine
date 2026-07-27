#!/usr/bin/env bash
# tools/check.sh — the merge gate (design/14-performance-testing.md §3.9).
#
# Fails loudly, runs everything. A change is mergeable only when this script
# exits 0 — see TESTING.md for the full per-change checklist (SPRT for
# functional search/eval changes is a human step this script cannot run).
#
# Steps that need components from future phases probe the release binary and
# SKIP loudly until the phase lands. When the corresponding subcommand exists
# the step runs automatically — no edits needed here; the TODO(Pn) markers
# only track when each SKIP is expected to disappear.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$*"; }
skip() { printf 'SKIP: %s\n' "$*"; }

step "cargo fmt --check"
cargo fmt --check

step "cargo clippy --all-targets -- -D warnings"
cargo clippy --all-targets -- -D warnings

# THE FEATURE MATRIX. The step above compiles DEFAULT features only, which
# leaves roughly a third of src/ — everything behind sota/datagen/policy/
# search-control/tune — never compiled, never linted, never type-checked by
# this gate. That is precisely the code the match and training binaries run:
# the NNUE head and SIMD kernels, the observation store, the SPSA tunables.
# A `cargo check` per combination is far cheaper than clippy and catches the
# failure mode that actually bites (a change that compiles under default
# features and breaks a feature build). Set CHECK_FEATURE_MATRIX=0 to skip
# when iterating; the merge gate should not skip it.
#
# NOTE the arch-* features are MUTUALLY EXCLUSIVE by construction (each sets
# different L1/L2 consts and a CENN is only loadable by a matching build), so
# they are checked one at a time, never unioned.
if [ "${CHECK_FEATURE_MATRIX:-1}" != "0" ]; then
    step "cargo check across the feature matrix (the ~1/3 of src/ the gate misses)"
    for feats in \
        "sota" \
        "sota,arch-d47v" \
        "sota,arch-d48x" \
        "sota,arch-hka2048" \
        "sota,tune" \
        "datagen" \
        "policy" \
        "search-control"
    do
        printf '  checking --features %s\n' "$feats"
        cargo check --quiet --all-targets --features "$feats" || {
            printf 'FAIL: feature combination "%s" does not check\n' "$feats" >&2
            exit 1
        }
    done
fi

step "cargo test  (debug: unit tests + fast perft tier)"
cargo test

# Release binary built up front so the phase probes below can interrogate it.
step "cargo build --release"
cargo build --release
BIN=target/release/chessengine

# --- Full perft tier (D24 gate; design/06 §3.5 tier 2) ----------------------
# P3 landed (M2): this step is unconditional. The SF-format `perft`
# subcommand (design/06 §3.2) is itself part of the contract — losing it
# would break tools/perft_diff.py — so its absence FAILS the gate.
step "perft divide sanity  (chessengine perft 1 — SF-format output)"
"$BIN" perft 1 | grep -q "Nodes searched" || {
    printf 'FAIL: %s perft 1 did not print SF-format divide output\n' "$BIN" >&2
    exit 1
}

step "release perft suite  (cargo test --release -- --ignored perft_full_suite)"
cargo test --release -- --ignored perft_full_suite
# Tier 3 (perft_extended, ~18G nodes, < 4 min) is run at maintainer
# discretion before releases / after movegen rewrites — not part of this
# gate (design/06 §3.5).

# --- Bench perft fingerprint (design/06 §3.6; master plan P4 exit) ----------
# P4 landed: this step is unconditional — losing the perft-mode bench would
# be a regression. Everything in its output except the 'Total time (ms)' /
# 'Nodes/second' lines is a pure function of the suite (D24), so two
# consecutive runs must be byte-identical once those two lines are stripped.
# Depth 3 (~1.8M nodes/run) keeps the gate fast; output is on stderr, stdout
# stays a clean UCI stream (design/06 §3.6).
step "bench perft fingerprint  (two runs, deterministic output identical)"
run_bench_perft() {
    "$BIN" bench 16 1 3 default perft 2>&1 >/dev/null \
        | grep -v -e '^Total time (ms) :' -e '^Nodes/second    :'
}
run1=$(run_bench_perft)
run2=$(run_bench_perft)
if [ "$run1" != "$run2" ]; then
    printf 'FAIL: two consecutive bench-perft runs differ\n' >&2
    diff <(printf '%s\n' "$run1") <(printf '%s\n' "$run2") >&2 || true
    exit 1
fi
printf '%s\n' "$run1" | tail -n 2

# --- Bench depth fingerprint (D24; design/14 §3.2, design/06 §3.6) ----------
# P5 landed: the search fingerprint is live. 'Nodes searched' from the
# pinned invocation below is THE number for the `Bench: <nodes>` commit
# trailer — functional commits carry the new count, intended-non-functional
# commits must reproduce the old one bit-for-bit (rerun this gate on both
# sides to verify; TESTING.md). The SF-shaped default (`bench` = 16 MB TT,
# 1 thread, depth 13) is the long-term fingerprint invocation; depth 13 is
# still infeasible pre-PVS/pruning, so the pin rises phase by phase. P5
# pinned depth 3 (~390 M nodes pre-TT); the P6 exit audit raised it to
# depth 6 (~335 M nodes, ~34 s — the TT flattens the curve, measured
# d3/d4/d5/d6/d7 = 207/213/276/335/508 M nodes). Determinism is gated by
# the release test
# bench_depth_fingerprint_is_deterministic_over_the_default_suite.
# TODO(P7+): keep raising toward depth 13 as PVS/aspiration (P7) and the
# pruning waves (P9-P10) shrink the tree; keep a single run under ~60 s.
step "bench fingerprint  (bench 16 1 6 default depth — the Bench: trailer)"
# Footer goes to stderr (stdout stays a clean UCI stream, design/06 §3.6).
bench_out=$("$BIN" bench 16 1 6 default depth 2>&1 >/dev/null)
printf '%s\n' "$bench_out" | tail -n 4
bench_nodes=$(printf '%s\n' "$bench_out" | sed -n 's/^Nodes searched  : //p')
if [ -z "$bench_nodes" ]; then
    printf 'FAIL: bench printed no "Nodes searched" total\n' >&2
    exit 1
fi
# THE PIN. Until now this step printed the number and trusted a human to
# notice when it moved — which made D24's central invariant enforceable only
# by memory. It is a snapshot test, so treat it like one: a mismatch fails
# loudly, and an INTENTIONAL functional change is landed by updating the pin
# in the same commit that changes the number (that is exactly what the
# `Bench:` trailer records). Set BENCH_PIN= (empty) to skip, e.g. while
# bisecting.
BENCH_PIN="${BENCH_PIN-385329}"
if [ -n "$BENCH_PIN" ] && [ "$bench_nodes" != "$BENCH_PIN" ]; then
    printf '\nFAIL: depth-6 bench is %s, pin is %s\n' "$bench_nodes" "$BENCH_PIN" >&2
    printf '  If this change is INTENTIONALLY functional (search, eval,\n' >&2
    printf '  movegen/make-unmake, ordering, TT), that is expected: update\n' >&2
    printf '  BENCH_PIN in this script and CLAUDE.md to %s in the same\n' "$bench_nodes" >&2
    printf '  commit, and put "Bench: %s" in the trailer.\n' "$bench_nodes" >&2
    printf '  If it is meant to be NON-functional (refactor, speedup, docs),\n' >&2
    printf '  the tree just changed search behaviour — investigate.\n' >&2
    exit 1
fi

printf '\ncommit trailer for functional changes (TESTING.md, design/14 §3.2):\n'
printf 'Bench: %s\n' "$bench_nodes"

printf '\nAll checks passed.\n'

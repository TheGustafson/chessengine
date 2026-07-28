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
        "sota,arch-d48w" \
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
BENCH_PIN="${BENCH_PIN-587683}"
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

# ---------------------------------------------------------------------------
# THE NNUE BEHAVIOURAL GATE.
#
# The BENCH_PIN above cannot see NNUE work at all. `Cargo.toml` has no
# `default = [...]`, so the pinned `bench 16 1 6 default depth` runs a
# DEFAULT-FEATURE build, and every SOTA path — the accumulator, the SIMD
# kernels, `sota_threats.rs` — sits behind `sota`. Verified 2026-07-27 three
# ways: no `default` line in Cargo.toml, `#[cfg(feature = "sota")]` on the
# module, and `nm` on the default release binary finding ZERO threat symbols.
# A no-net `bench` is a PSQT run (`EvalFile` empty ⇒ NNUE off, uci.rs).
#
# So a rewrite of the incremental threat maintenance could be arbitrarily
# wrong and still leave 385329 untouched. That is the hole this closes. Two
# pins, both on the V arch, both established against `archon-d47v-pgo` and
# reproduced by a from-source non-PGO build (PGO changes layout, not
# semantics: the NNUE path is integer-quantised, so it is bit-exact):
#
#   1. EVAL IDENTITY — sha256 of the forward-pass score for all 50 wl50 FENs.
#      Catches a wrong accumulator that search happens to paper over.
#   2. NODE IDENTITY — the depth-13 wl50 node total with the net loaded.
#      Catches an eval that is right per-position but perturbs the tree.
#
# Default OFF: it needs a second release build and an 84 MB gitignored net.
# It is MANDATORY before merging anything touching src/eval/nnue/.
if [ "${CHECK_NNUE_GATE:-0}" != "0" ]; then
    nnue_net="${NNUE_GATE_NET:-data/nnue/nets/screens/d47v-a4b2-cal400.cenn}"
    nnue_fens="${NNUE_GATE_FENS:-tools/nnue/calib/wl50.fens}"
    if [ ! -f "$nnue_net" ]; then
        printf 'FAIL: CHECK_NNUE_GATE=1 but the net is missing: %s\n' "$nnue_net" >&2
        printf '  Point NNUE_GATE_NET at a V-arch CENN, or unset the gate.\n' >&2
        exit 1
    fi
    step "NNUE behavioural gate (the sota build the bench pin cannot see)"
    # A separate target dir: sharing one with the default build would force a
    # full rebuild on every alternation between the two feature sets.
    nnue_target="target/nnue-gate"
    RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release \
        --features sota,arch-d47v --target-dir "$nnue_target" --quiet || {
        printf 'FAIL: the sota,arch-d47v release build failed\n' >&2
        exit 1
    }
    nnue_bin="$nnue_target/release/chessengine"

    eval_hash=$("$nnue_bin" nnue-eval "$nnue_net" < "$nnue_fens" | sha256sum | cut -d' ' -f1)
    NNUE_EVAL_PIN="${NNUE_EVAL_PIN-cd190d1b18e52f621cb0867e264359d3ce2cac2e6bdf65cdcdfc2005e9747f3d}"
    printf '  eval hash  : %s\n' "$eval_hash"
    if [ -n "$NNUE_EVAL_PIN" ] && [ "$eval_hash" != "$NNUE_EVAL_PIN" ]; then
        printf '\nFAIL: NNUE eval hash is %s,\n      pin is %s\n' \
            "$eval_hash" "$NNUE_EVAL_PIN" >&2
        printf '  The forward pass changed. If that is INTENTIONAL (a new arch,\n' >&2
        printf '  a quantisation change), update NNUE_EVAL_PIN in the same commit.\n' >&2
        printf '  If it is meant to be a pure speedup, the eval is now WRONG.\n' >&2
        exit 1
    fi

    # bench writes its whole report to STDERR (same idiom as the pin above:
    # stderr to the capture, stdout discarded). Piping stdout instead yields
    # an empty string and a spurious pass-turned-failure.
    nnue_nodes=$("$nnue_bin" bench 16 1 13 "$nnue_fens" depth "$nnue_net" 2>&1 >/dev/null \
        | sed -n 's/^Nodes searched  : //p')
    NNUE_NODE_PIN="${NNUE_NODE_PIN-2068701}"
    printf '  d13 nodes  : %s\n' "$nnue_nodes"
    if [ -n "$NNUE_NODE_PIN" ] && [ "$nnue_nodes" != "$NNUE_NODE_PIN" ]; then
        printf '\nFAIL: NNUE d13 wl50 node total is %s, pin is %s\n' \
            "$nnue_nodes" "$NNUE_NODE_PIN" >&2
        printf '  Same rule as the bench pin: intentional ⇒ update the pin here\n' >&2
        printf '  in the same commit; unintentional ⇒ the tree changed, investigate.\n' >&2
        exit 1
    fi
else
    printf '\nNOTE: NNUE behavioural gate SKIPPED (CHECK_NNUE_GATE=0).\n'
    printf '  The bench pin above is a PSQT number and cannot see NNUE changes.\n'
    printf '  Run CHECK_NNUE_GATE=1 before merging anything under src/eval/nnue/.\n'
fi

printf '\ncommit trailer for functional changes (TESTING.md, design/14 §3.2):\n'
printf 'Bench: %s\n' "$bench_nodes"

printf '\nAll checks passed.\n'

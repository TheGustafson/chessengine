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

# --- Bench fingerprint (D24; design/14 §3.2, design/06 §3.6) ----------------
# TODO(P5): SKIPs until P5 (search core v1 + bench depth mode) lands. The
# probe is a depth-1 bench run; the real run uses the SF-shaped defaults
# (16 MB TT, 1 thread, depth 13) whose 'Nodes searched' total is THE
# fingerprint for the `Bench:` commit trailer.
step "bench fingerprint  (chessengine bench)"
if "$BIN" bench 16 1 1 default depth 2>&1 | grep -q "Nodes searched"; then
    # Footer goes to stderr (stdout stays a clean UCI stream, design/06 §3.6).
    "$BIN" bench 2>&1 | tail -n 4
    printf "paste 'Nodes searched' into the commit's 'Bench:' trailer (TESTING.md)\n"
else
    skip "bench fingerprint — TODO(P5): binary has no 'bench' subcommand yet"
fi

printf '\nAll checks passed.\n'

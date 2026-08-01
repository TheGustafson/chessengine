#!/usr/bin/env bash
# tools/ops/match/sprt.sh — SPRT a built engine against a reference engine.
#
# PURPOSE
#   The merge gate for functional SEARCH changes, and the only instrument that
#   can see a change's TIME cost. Wraps tools/sprt.sh with binary validation and
#   the pairing rules that make the result mean something.
#
# USAGE
#   bash tools/ops/match/sprt.sh <candidate-binary> [reference-binary]
#
#   CAND=engines/bin/archon-main-abc1234 bash tools/ops/match/sprt.sh
#   TC=2+0.1 bash tools/ops/match/sprt.sh engines/bin/archon-main-abc1234
#
#   Environment:
#     CAND / REF   engine paths (REF defaults to engines/canonical/archon)
#     TC           time control (default 10+0.1, our STC)
#     CONCURRENCY  default 3
#
# PRODUCES
#   $ARCHON_D49/sprt-<cand>.log
#
# WHY A REAL TIME CONTROL, AND WHEN NOT TO USE ONE
#   Fixed-node screens are load-immune and are the right tool for NET changes.
#   They are BLIND TO SPEED, so they cannot price a search change that costs
#   instructions per node: study 71 was credited +11.00 Elo by a fixed-node SPRT
#   while raising instructions/node +3.99 %, a cost that test structurally
#   cannot charge for. Search changes therefore need a real clock — and a real
#   clock is load-sensitive, so the box must be otherwise quiet for the absolute
#   number to be meaningful (the comparison stays fair either way, since both
#   engines suffer equally).
#
#   Ultra-bullet (2+0.1) resolves fastest but over-weights speed relative to
#   search quality: a PASS there is strong evidence, a FAIL is ambiguous between
#   "the change hurt" and "this control punishes our node cost specifically".
#
# PAIRING RULES (D16/D25)
#   Both binaries PGO or both not; identical RUSTFLAGS. A mismatched pair
#   measures the build, not the change.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

CAND=${CAND:-${1:?usage: sprt.sh <candidate-binary> [reference-binary]}}
REF=${REF:-${2:-$ARCHON_ROOT/engines/canonical/archon}}
TC=${TC:-10+0.1}
CONCURRENCY=${CONCURRENCY:-3}
OPS_LOG=${OPS_LOG:-$ARCHON_D49/sprt-$(basename "$CAND").log}; export OPS_LOG

# Resolve relative paths against ARCHON_ROOT, never the caller's cwd — that
# mismatch is what made an SPRT fail with "candidate missing" after a PGO build
# wrote its binary into a different worktree.
case "$CAND" in /*) ;; *) CAND="$ARCHON_ROOT/$CAND";; esac
case "$REF"  in /*) ;; *) REF="$ARCHON_ROOT/$REF";;  esac

[ -x "$CAND" ] || { log "FATAL: candidate not executable: $CAND"; exit 2; }
[ -x "$REF"  ] || { log "FATAL: reference not executable: $REF";  exit 2; }

log "SPRT $(basename "$CAND") vs $(basename "$(readlink -f "$REF")") @ TC=$TC conc=$CONCURRENCY"
log "  candidate bench: $("$CAND" bench 16 1 6 default depth 2>&1 | grep -oE 'Nodes searched *: *[0-9]+' | grep -oE '[0-9]+$')"
log "  reference bench: $("$REF"  bench 16 1 6 default depth 2>&1 | grep -oE 'Nodes searched *: *[0-9]+' | grep -oE '[0-9]+$')"

TC="$TC" CONCURRENCY="$CONCURRENCY" bash tools/sprt.sh "$CAND" "$REF" >> "$OPS_LOG" 2>&1
rc=$?
grep -E 'Elo:|LOS:|Games:|Ptnml' "$OPS_LOG" | tail -4 | sed 's/^/    /'
log "sprt exit=$rc -> $OPS_LOG"
exit $rc

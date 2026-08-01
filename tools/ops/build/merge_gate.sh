#!/usr/bin/env bash
# tools/ops/build/merge_gate.sh — run the full merge gate under a memory cap.
#
# PURPOSE
#   `CHECK_NNUE_GATE=1 tools/check.sh` is the gate every functional change must
#   pass: fmt, clippy, tests, release perft tier, the bench fingerprint, plus the
#   NNUE behavioural pins. This runs it without being able to take the box down,
#   and echoes the pins so a caller can see WHICH numbers it checked.
#
# USAGE
#   bash tools/ops/build/merge_gate.sh
#
#   Environment:
#     MIN_MEM_MB  refuse below this  (default 2500)
#     MEM_CAP     cgroup cap         (default 4G)
#     JOBS        cargo parallelism  (default 2)
#     GATE_NET    net for the NNUE gate (default engines/canonical/net.cenn)
#     SKIP_NNUE=1 run only the default gate (faster; NOT valid for eval changes)
#
# PRODUCES
#   $ARCHON_D49/merge_gate.log, and exits with check.sh's own status.
#
# WHY THE NNUE GATE IS SEPARATE AND MANDATORY FOR eval/nnue/ CHANGES
#   The `Bench:` fingerprint is BLIND to NNUE work: Cargo.toml has no default
#   features, so the pinned bench invocation is a default-feature build and the
#   whole SOTA path (accumulator, SIMD kernels, threats) sits behind `sota`. A
#   no-net bench is a PSQT run. So an accumulator rewrite can be arbitrarily
#   wrong and leave the bench number untouched. CHECK_NNUE_GATE=1 pins the eval
#   hash and a d13 node total against the CANONICAL net instead.
#
# WHEN THE PINS MOVE
#   An intentional functional change moves the pin IN THE SAME COMMIT — both
#   here and in CLAUDE.md. Verify the doc's CURRENT pin line moved, not its
#   history: a promotion once updated check.sh correctly but wrote the CLAUDE.md
#   edit onto the PRECEDING-pin line, so the doc named a superseded gate for a
#   day and silently overwrote an earlier study's real hash.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

MIN_MEM_MB=${MIN_MEM_MB:-2500}
MEM_CAP=${MEM_CAP:-4G}
JOBS=${JOBS:-2}
GATE_NET=${GATE_NET:-$ARCHON_ROOT/engines/canonical/net.cenn}
SKIP_NNUE=${SKIP_NNUE:-0}
OPS_LOG=${OPS_LOG:-$ARCHON_D49/merge_gate.log}; export OPS_LOG

need_mem "$MIN_MEM_MB" || exit 1
if unit_running d49train pump_train; then
    log "NOTE: trainer payload is live; gate runs capped at $MEM_CAP with $JOBS jobs"
fi

log "pins currently in tools/check.sh:"
grep -nE 'BENCH_PIN=|NNUE_NODE_PIN=|NNUE_HASH_PIN=' "$ARCHON_ROOT/tools/check.sh" 2>/dev/null \
    | sed 's/^/    /' | tee -a "$OPS_LOG"

ENVS=(-E "CARGO_BUILD_JOBS=$JOBS")
if [ "$SKIP_NNUE" != "1" ]; then
    ENVS+=(-E "CHECK_NNUE_GATE=1" -E "NNUE_GATE_NET=$GATE_NET")
    log "running FULL gate (NNUE pins vs $(basename "$(readlink -f "$GATE_NET")"))"
else
    log "running DEFAULT gate only (SKIP_NNUE=1) — not valid for eval/nnue changes"
fi

systemd-run --user --unit=mergegate --quiet --wait --collect \
    -p MemoryMax="$MEM_CAP" -p MemorySwapMax=0 "${ENVS[@]}" \
    --working-directory="$ARCHON_ROOT" \
    /usr/bin/bash -c "bash tools/check.sh" >> "$OPS_LOG" 2>&1
rc=$?

tail -6 "$OPS_LOG" | sed 's/^/    /'
if [ "$rc" -eq 0 ]; then log "MERGE GATE PASSED"; else log "MERGE GATE FAILED (rc=$rc)"; fi
exit "$rc"

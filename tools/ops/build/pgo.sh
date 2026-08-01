#!/usr/bin/env bash
# tools/ops/build/pgo.sh — memory-gated PGO build that always lands where the
# consumers look.
#
# PURPOSE
#   Wraps `tools/build_pgo.sh` with the three things every caller has to get
#   right and one of them got wrong tonight: a memory gate, a capped cgroup, and
#   an output path that does not depend on which worktree invoked the build.
#
# USAGE
#   bash tools/ops/build/pgo.sh <out-name> [features] [eval-file]
#
#   bash tools/ops/build/pgo.sh archon-main-abc1234
#   bash tools/ops/build/pgo.sh archon-试 sota,arch-d48x engines/canonical/net.cenn
#
#   Environment:
#     MIN_MEM_MB  refuse below this   (default 2500)
#     MEM_CAP     cgroup cap          (default 6G)
#     JOBS        cargo parallelism   (default 2)
#
# PRODUCES
#   $ARCHON_ROOT/engines/bin/<out-name>   — ALWAYS the primary checkout, and the
#   path is echoed on success so a caller can consume it without guessing.
#
# WHY THE OUTPUT PATH IS FORCED
#   `tools/build_pgo.sh` takes a BARE NAME and writes to `engines/bin/$OUT`
#   relative to its cwd. On 2026-08-01 a build was launched with
#   `--working-directory=chessengine-mainwt`, so the binary landed in the
#   worktree while the SPRT that consumed it looked in the primary checkout and
#   died with "candidate missing". Passing an ABSOLUTE path does not help — it
#   produces `engines/bin//home/nick/.../archon-...` and fails on cp. So this
#   wrapper takes a bare name, builds from ARCHON_ROOT, and reports the absolute
#   result.
#
# WHY THE EVAL FILE IS MANDATORY FOR NNUE BUILDS
#   A bench-only PGO profile leaves the NNUE kernels unexercised, so PGO has no
#   data for the hottest code and the optimisation is effectively inert
#   (measured 2026-07-18). `build_pgo.sh` requires it; we default it to the
#   canonical net rather than letting a caller forget.
#
# WHY IT REFUSES INSTEAD OF QUEUING
#   Builds next to a live trainer are how this box OOMs. Refusing is a decision
#   the caller can act on; silently queueing hides the contention.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

OUT_NAME=${1:?usage: pgo.sh <out-name> [features] [eval-file]}
FEATURES=${2:-$(cat "$ARCHON_ROOT/engines/canonical/arch.txt" 2>/dev/null || echo "sota,arch-d48x")}
EVAL_FILE=${3:-$ARCHON_ROOT/engines/canonical/net.cenn}
MIN_MEM_MB=${MIN_MEM_MB:-2500}
MEM_CAP=${MEM_CAP:-6G}
JOBS=${JOBS:-2}
OPS_LOG=${OPS_LOG:-$ARCHON_D49/pgo.log}; export OPS_LOG

case "$OUT_NAME" in
    */*) log "FATAL: pass a BARE NAME, not a path (build_pgo.sh prefixes engines/bin/)"; exit 2;;
esac
[ -e "$EVAL_FILE" ] || { log "FATAL: eval file not found: $EVAL_FILE"; exit 2; }

need_mem "$MIN_MEM_MB" || exit 1
if unit_running d49train pump_train; then
    log "WARNING: a trainer payload is live; building under MemoryMax=$MEM_CAP with $JOBS jobs"
fi

log "PGO build: out=$OUT_NAME features=$FEATURES net=$(basename "$(readlink -f "$EVAL_FILE")")"
systemd-run --user --unit="pgo-$OUT_NAME" --quiet --wait --collect \
    -p MemoryMax="$MEM_CAP" -p MemorySwapMax=0 \
    -E CARGO_BUILD_JOBS="$JOBS" \
    --working-directory="$ARCHON_ROOT" \
    /usr/bin/bash -c "bash tools/build_pgo.sh '$FEATURES' '$OUT_NAME' '$EVAL_FILE'" \
    >> "$OPS_LOG" 2>&1
rc=$?

RESULT="$ARCHON_ROOT/engines/bin/$OUT_NAME"
if [ "$rc" -ne 0 ] || [ ! -x "$RESULT" ]; then
    log "PGO build FAILED (rc=$rc); see $OPS_LOG"
    exit "${rc:-1}"
fi
log "PGO build OK: $RESULT ($(stat -c %s "$RESULT") bytes)"
echo "$RESULT"

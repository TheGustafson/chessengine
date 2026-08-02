#!/usr/bin/env bash
# tools/ops/train/d49_launch.sh — launch or resume a D49 training run.
#
# PURPOSE
#   Keeps a JAX trainer alive across crashes, OOM kills and reboots. Resumes
#   from the newest checkpoint on disk, holds when the box is short of memory,
#   and retries after a GPU-settle delay. This is the ONLY supported way to
#   start a D49 run.
#
# USAGE
#   # as a durable service (correct for real runs — survives session death):
#   systemd-run --user --unit=d49train -p MemorySwapMax=0 \
#       /path/to/tools/ops/train/d49_launch.sh
#
#   # foreground, for debugging only:
#   bash tools/ops/train/d49_launch.sh
#
#   Environment:
#     RUN_SPEC   run config under tools/nnue/runs/   (default d49_sfblend_500.json)
#     CKPT_GLOB  checkpoint prefix to resume from    (default d49sfb)
#     TARGET_SB  stop once resume point >= this      (default: read from spec)
#     MIN_MEM_MB hold threshold                      (default 3000)
#
# PRODUCES
#   Appends to  data/nnue/experiments/d49-1280/cyc.log
#   Checkpoints data/nnue/experiments/d49-1280/<CKPT_GLOB>-sb<N>-ema-raw.bin
#
# NOTES ON THE GUARDS (each one is a bug that happened)
#   * TARGET_SB is read from the run spec, NOT hardcoded. A previous version
#     hardcoded `-ge 1642`; when the run was extended to 2130 the copy kept the
#     old literal, declared itself complete on the first iteration and exited.
#   * The jax/ tree is resolved from ARCHON_ROOT. A previous version cd'd into
#     an AGENT WORKTREE, so the run died the moment that worktree could be
#     pruned — and it also silently required `archon_mask.so`, a gitignored
#     build artifact that existed only in that tree.
#   * The retry sleep is 120s, not 30s: after a kill, CUDA memory takes longer
#     than 30s to release and the relaunch fails with CUDA_ERROR_OUT_OF_MEMORY.
#   * Memory is checked with hold_for_mem, i.e. the loop PAUSES rather than
#     dying, so a transient spike does not end the run.

source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

RUN_SPEC=${RUN_SPEC:-d49_sfblend_500.json}
CKPT_GLOB=${CKPT_GLOB:-d49sfb}
MIN_MEM_MB=${MIN_MEM_MB:-3000}

# EXACT dependency set. Do NOT paraphrase or "simplify" this line: dropping
# `optax` and unpinning jax made the trainer die instantly with
# ModuleNotFoundError while the loop cheerfully retried every 120 s. Pinned
# versions because a silent jax upgrade changes numerics mid-run.
UV_DEPS=${UV_DEPS:-'--with numpy==2.5.1 --with jax[cuda12]==0.11.0 --with optax==0.2.8'}
EXP_DIR="$ARCHON_ROOT/data/nnue/experiments/d49-1280"
SPEC_PATH="$ARCHON_ROOT/tools/nnue/runs/$RUN_SPEC"
OPS_LOG="$EXP_DIR/cyc.log"; export OPS_LOG

[ -f "$SPEC_PATH" ] || { log "FATAL: run spec not found: $SPEC_PATH"; exit 2; }

# Read the finish line from the spec so it can never drift from the config.
TARGET_SB=${TARGET_SB:-$(python3 -c "import json;print(json.load(open('$SPEC_PATH'))['superbatches'])")}
log "d49_launch: spec=$RUN_SPEC target=sb$TARGET_SB root=$ARCHON_ROOT"

while :; do
    RESUME=$(ls "$EXP_DIR/$CKPT_GLOB"-sb*-ema-raw.bin 2>/dev/null \
             | sed -E 's/.*-sb([0-9]+)-ema-raw\.bin/\1/' | sort -n | tail -1)
    RESUME=${RESUME:-0}

    if [ "$RESUME" -ge "$TARGET_SB" ]; then
        log "run complete at sb${RESUME} (target sb${TARGET_SB})"
        break
    fi

    hold_for_mem "$MIN_MEM_MB" 60

    log "launching from sb${RESUME} (spec $RUN_SPEC)"
    (
        cd "$ARCHON_ROOT/jax" || exit 9
        # Cap JAX's preallocation so the run coexists with a desktop session.
        # JAX grabs ~90% of VRAM by default; with Chrome's GPU process holding
        # ~720 MiB the trainer could not fit and died with CUDA_ERROR_OUT_OF_MEMORY
        # on every retry. 0.78 leaves room for a browser without starving it.
        XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION:-0.78} \
        PYTHONUNBUFFERED=1 ARCHON_JAX_ALLOW_GPU=1 ARCHON_JAX_FT_IMPL=pallas \
            uv run --no-project $UV_DEPS \
            python tools/pump_train.py "../tools/nnue/runs/$RUN_SPEC" "$RESUME"
    ) >> "$OPS_LOG" 2>&1
    rc=$?
    log "trainer exited rc=$rc"

    # GPU memory release after a kill is slower than it looks (measured: >30s).
    sleep 120
done

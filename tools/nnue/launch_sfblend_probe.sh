#!/usr/bin/env bash
# Launch the SF-net-teacher probe as a systemd transient SERVICE.
#
# WORKING DIRECTORY. This runs from the WORKTREE's `jax/`, not the shared
# checkout's. The shared checkout sits on another agent's branch and does not
# carry the teacher code; running there would silently train the plain arm.
# The worktree carries symlinks (`data`, `engines/bin`) so pump_train's
# ROOT-relative paths land on the real artifacts.
#
# `import jax` must not be shadowed by the repo's own `jax/` directory, which
# is why the cwd is that directory rather than the repo root.
#
# ARCHON_JAX_ALLOW_GPU=1 is mandatory: without it train.py refuses to import
# jax and the unit dies in ~1.5 s with no other symptom.
set -euo pipefail

WT="${WT:-/home/nick/projects/chessengine/.claude/worktrees/agent-ac79f75828e3be36d}"
SPEC="${SPEC:-../tools/nnue/runs/d49_sfblend_probe.json}"
RESUME="${RESUME:-1630}"
UNIT="${UNIT:-archon-d49sfb}"
LOG="${LOG:-/home/nick/projects/chessengine/data/nnue/experiments/d49-1280/cyc.log}"

test -d "$WT/jax" || { echo "no worktree at $WT" >&2; exit 1; }
test -f "$WT/${SPEC#../}" || { echo "no spec $WT/${SPEC#../}" >&2; exit 1; }

echo "[sfblend] launching $UNIT from sb$RESUME $(date -Is)" >> "$LOG"

exec systemd-run --user --unit="$UNIT" \
  --working-directory="$WT/jax" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=ARCHON_JAX_ALLOW_GPU=1 \
  --setenv=ARCHON_JAX_FT_IMPL=pallas \
  --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --property=StandardOutput="append:$LOG" \
  --property=StandardError="append:$LOG" \
  -- /usr/bin/uv run --no-project \
       --with numpy==2.5.1 --with 'jax[cuda12]==0.11.0' --with optax==0.2.8 \
       -- python tools/pump_train.py "$SPEC" "$RESUME"

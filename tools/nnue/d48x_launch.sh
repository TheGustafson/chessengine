#!/usr/bin/env bash
# D48X launcher / RESTARTER — one command, resumes from wherever it got to.
#
# Written 2026-07-27 after a global OOM killed the run at sb257 and recovery
# took ~15 minutes of archaeology. Three things went wrong that this fixes:
#
#   1. D48X had no launcher of its own (d48w did), so the exact invocation had
#      to be reverse-engineered from sibling scripts.
#   2. Plain `uv run` FAILS on this project — resolution dies on the
#      `notebook` extra under python 3.13. The working form is
#      `--no-project` with the three pins named explicitly. That detail is
#      the whole difference between a 10-second restart and a hunt.
#   3. The unit had no Restart=, so a silent kill just stopped everything.
#      The log ends mid-superbatch with no traceback and nothing notices.
#
# The resume point is DETECTED, never hardcoded. That is what makes
# Restart=on-failure safe: systemd re-runs this script, which re-scans for the
# newest checkpoint, so a crash at sb400 resumes at sb390 rather than replaying
# from whatever a stale literal said. A hardcoded resume plus auto-restart
# would be worse than no auto-restart at all.
#
#   bash tools/nnue/d48x_launch.sh            # launch or relaunch
#   bash tools/nnue/d48x_launch.sh --exec     # run the trainer in THIS process
#                                             # (what the systemd unit calls)
set -eu
cd /home/nick/projects/chessengine

CFG=tools/nnue/runs/d48x_1280.json
OUT=data/nnue/experiments/d48x-1280
NET_ID=d48x-1280
LOG=/tmp/d48x_train.log

# Newest checkpoint with BOTH halves present. pump_train needs the raw params
# and the EMA copy; a run killed mid-write can leave one without the other, so
# require the pair rather than trusting the highest number.
latest_resume() {
    local sb best=0
    for sb in $(ls "$OUT" 2>/dev/null | grep -oP "${NET_ID}-sb\K\d+(?=-raw\.bin)" | sort -n); do
        [ -s "$OUT/$NET_ID-sb$sb-raw.bin" ] || continue
        [ -s "$OUT/$NET_ID-sb$sb-ema-raw.bin" ] || continue
        best=$sb
    done
    echo "$best"
}

if [ "${1:-}" = "--exec" ]; then
    # Scan for the checkpoint from the REPO ROOT (where $OUT is rooted), then
    # move into jax/ — pump_train.py lives there and takes its config as a
    # path relative to it, which is why every sibling launcher sets
    # --working-directory to jax/. Doing the cd here instead means the unit
    # cannot be started from the wrong directory.
    RESUME=$(latest_resume)
    printf '[d48x] resuming from sb%s at %s\n' "$RESUME" "$(date -Is)" >> "$LOG"
    cd jax
    # `--no-project` is load-bearing: see the header.
    exec uv run --no-project \
        --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \
        python tools/pump_train.py "../$CFG" "$RESUME" >> "$LOG" 2>&1
fi

RESUME=$(latest_resume)
[ "$RESUME" -gt 0 ] || { echo "no complete checkpoint pair in $OUT — refusing to guess"; exit 1; }
echo "[d48x] newest complete checkpoint: sb$RESUME"

systemctl --user reset-failed archon-d48x.service 2>/dev/null || true
systemctl --user stop archon-d48x.service 2>/dev/null || true

# MemoryHigh/Max are the SAME values the run has used all along and were NOT
# the problem: the kill was a GLOBAL oom (constraint=CONSTRAINT_NONE), the
# kernel picking this 8.2 G process because the box as a whole ran dry while a
# PGO rebuild's two LTO rustc jobs were resident. Tightening the cgroup would
# not have helped and MemoryHigh throttling has bitten this run before
# (~30% throughput loss, 2026-07-26). The real rule is operational: do not run
# multi-GB builds beside the trainer.
#
# RestartSec is generous because a resume re-JITs the Pallas kernel (~3 min of
# CPU before the first superbatch); a tight restart loop would thrash.
systemd-run --user --unit=archon-d48x --working-directory="$PWD/jax" \
    -p MemoryHigh=9G -p MemoryMax=11G -p MemorySwapMax=0 --collect \
    -p Restart=on-failure -p RestartSec=60 -p StartLimitBurst=5 \
    -E PYTHONUNBUFFERED=1 -E ARCHON_JAX_ALLOW_GPU=1 -E ARCHON_JAX_FT_IMPL=pallas \
    /bin/bash "$PWD/tools/nnue/d48x_launch.sh" --exec

echo "[d48x] launched (Restart=on-failure, resume auto-detected). Log: $LOG"

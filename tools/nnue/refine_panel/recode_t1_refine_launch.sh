#!/usr/bin/env bash
# Refinement Tier-1 panel launcher (richer-representation study §6).
#
# DELIBERATELY NOT SELF-ARMING: the GPU queue at study time was
# identity2 -> d49x (~13 h), and the standing rule is that long GPU runs
# start only on the owner's say-so. Arm it with:
#
#   systemd-run --user --unit=recode-t1-refine \
#     -p MemoryMax=8G -p MemorySwapMax=0 \
#     bash <this file>
#
# It then WAITS for the GPU to be both free of the d49x unit and idle for
# 10 consecutive minutes before running the four arms SEQUENTIALLY
# (~55 min each at panel speeds; ~3.7 h total):
#
#   oracle_rank1      the value upper bound (trainer-only, 77,104 rows)
#   headbucket_rank1  the refresh-free head-side form (16 output buckets)
#   refine_rank1      the constant-row deployable-shaped candidate
#   refine_null       the content-free-conditioning control (same code,
#                     hash-bit region)
#
# PRE-RUN CHECKLIST (the launcher enforces none of these — read the study):
#   * rebuild refine-rank1-r60720.npy with the REAL probe artifact if the
#     probe has been rerun since the spec was written (refine_code.py refine
#     <bit>), and re-check the chosen bit against the probe ranking.
#   * every arm's sb0 heldout must print 0.006350 (bit-parity with the
#     incumbent) — a different number voids that arm.
#
# The decision rule is pre-registered in
# docs/research/richer-representation-study-2026-07-31.md §6 — read it
# BEFORE reading any arm's sb12.
set -euo pipefail

WT="$(cd "$(dirname "$0")/../../.." && pwd)"
ROOT=/home/nick/projects/chessengine
LOGDIR="$ROOT/data/nnue/experiments/recode-t1"
mkdir -p "$LOGDIR"

echo "[refine-launch] waiting for d49x + 10 idle-GPU minutes..."
idle=0
while true; do
    if systemctl --user is-active --quiet d49x.service; then
        idle=0
    else
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$util" -lt 10 ] && [ "$mem" -lt 1500 ]; then
            idle=$((idle + 1))
        else
            idle=0
        fi
    fi
    [ "$idle" -ge 10 ] && break
    sleep 60
done
echo "[refine-launch] GPU free, starting arms"

cd "$WT/jax"
for arm in oracle_rank1 headbucket_rank1 refine_rank1 refine_null; do
    echo "[refine-launch] arm $arm"
    ARCHON_JAX_FT_IMPL=pallas uv run --no-project \
        --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" --with optax==0.2.8 \
        python tools/recode_tier1.py \
        "$WT/tools/nnue/refine_panel/recode_t1_${arm}.json" \
        > "$LOGDIR/${arm}.log" 2>&1
    echo "[refine-launch] arm $arm exit $?"
done
echo "[refine-launch] panel complete"

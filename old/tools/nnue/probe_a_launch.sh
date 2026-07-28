#!/usr/bin/env bash
# Probe A launcher (docs/research/value-foundation-state.md): trains the three
# long-schedule cells serially on the GPU, then touches <campaign>/DONE.
# NEVER interrupt a running SF-binpack training (resume is unsafe: the reader
# ignores skip_count). If this script dies mid-cell, record it in the state
# log and re-plan; do not blindly rerun.
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG=tools/nnue/configs/value-scale-probe-a-v1.json
CAMPAIGN=data/nnue/campaigns/value-scale-probe-a-v1
mkdir -p "$CAMPAIGN"

echo "[probe-a $(date +%H:%M:%S)] cell 1/3: fishpack classic256"
uv run python tools/nnue/value_matrix.py --config "$CFG" --source binpack --cell classic256
echo "[probe-a $(date +%H:%M:%S)] cell 2/3: fishpack classic1024"
uv run python tools/nnue/value_matrix.py --config "$CFG" --source binpack --cell classic1024
echo "[probe-a $(date +%H:%M:%S)] cell 3/3: lichess classic1024"
uv run python tools/nnue/value_matrix.py --config "$CFG" --source lichess --cell classic1024

touch "$CAMPAIGN/DONE"
echo "[probe-a $(date +%H:%M:%S)] ALL CELLS COMPLETE"

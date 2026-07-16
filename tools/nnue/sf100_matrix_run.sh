#!/usr/bin/env bash
# 4-net architecture matrix on sf-100 (the known-good data), schedules scaled
# per architecture (docs/research/value-foundation-state.md). Decision-grade
# 300g @5k endpoint screens; 400g @20k for cells above -50.
set -euo pipefail
cd "$(dirname "$0")/../.."
for cell in classic256 classic1024 king768x256 d40-full; do
  echo "[sf100-matrix $(date +%H:%M:%S)] cell $cell"
  uv run python tools/nnue/value_matrix.py \
    --config "tools/nnue/configs/sf100-matrix-v1-$cell.json" \
    --source sf100 --cell "$cell"
done
touch data/nnue/campaigns/sf100-matrix-v1/DONE
echo "[sf100-matrix $(date +%H:%M:%S)] MATRIX COMPLETE"

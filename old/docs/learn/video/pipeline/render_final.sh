#!/usr/bin/env bash
# Final master render: 1920x1080 @ 30fps, one clip per scene. nice + limited threads.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SCENES="S1Hook S2Squares S3Bitboard S4SetOps S5Knight S6Tricks S7Mailbox S8Layout S9Recap"
for S in $SCENES; do
  echo ">>> FINAL $S (1080p30)"
  nice -n 19 venv/bin/manim render -r 1920,1080 --fps 30 scenes/pilot02.py "$S" \
    2>&1 | grep -viE "onnxruntime|GetGpuDevices|ReadFileContents" | tail -1
done
echo "FINAL RENDER COMPLETE"

#!/usr/bin/env bash
# Render every pilot scene. Usage: render_scenes.sh <ql|qh> [fps]
set -euo pipefail
cd "$(dirname "$0")/.."
Q="${1:-ql}"; FPS="${2:-15}"
SCENES="S1Hook S2Squares S3Bitboard S4SetOps S5Knight S6Tricks S7Mailbox S8Layout S9Recap"
for S in $SCENES; do
  echo ">>> rendering $S ($Q ${FPS}fps)"
  nice -n 19 venv/bin/manim render -$Q --fps "$FPS" scenes/pilot02.py "$S" \
    2>&1 | grep -viE "onnxruntime|GetGpuDevices|ReadFileContents" | tail -2
done
echo "ALL SCENES RENDERED ($Q)"

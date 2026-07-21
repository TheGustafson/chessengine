#!/usr/bin/env bash
# Deploy an engine binary WITH PROOF it can load CEN2 nets (uciok is not
# enough — a featureless build answers uciok and silently PSQT-evals; this
# trap fired twice on 2026-07-22).
#   bash tools/deploy_engine.sh <dest-path>
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:?dest path}"
NET="${VERIFY_NET:-data/nnue/nets/d40-uho-v1-240sb.cenn}"
cp target/release/chessengine "$DEST.new"
echo "$(head -1 tools/nnue/calib/scale-check-800.fens)" | "$DEST.new" nnue-eval "$NET" >/dev/null 2>&1 \
  || { echo "DEPLOY REFUSED: $DEST.new cannot load CEN2 ($NET) — built without --features sota?"; rm -f "$DEST.new"; exit 1; }
mv "$DEST.new" "$DEST"
echo "deployed $DEST (CEN2-load verified)"

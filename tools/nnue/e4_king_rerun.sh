#!/usr/bin/env bash
# E4: rerun the king768x256 cell on sf-100 with the sparse-init fix (the only
# change vs the -372 +/- 31 result). Gates on the E1 size-control (end of
# chain). A large recovery localizes the historical king/HalfKA-classic
# failures to the trainer-init bug, not the architecture or data.
set -euo pipefail
cd "$(dirname "$0")/../.."
until [ -f data/nnue/experiments/e1-size-control/DONE ]; do sleep 60; done
uv run python tools/nnue/value_matrix.py \
  --config tools/nnue/configs/sf100-matrix-v1-king768x256.json \
  --source sf100 --cell king768x256-fixedinit 2>/dev/null || true
# the driver keys jobs by cell name; simplest correct path: fresh campaign id
python3 - <<'EOF'
import json
cfg = json.load(open('tools/nnue/configs/sf100-matrix-v1-king768x256.json'))
cfg['campaign_id'] = 'sf100-king-fixedinit-v1'
json.dump(cfg, open('tools/nnue/configs/sf100-king-fixedinit-v1.json', 'w'), indent=2)
EOF
uv run python tools/nnue/value_matrix.py \
  --config tools/nnue/configs/sf100-king-fixedinit-v1.json \
  --source sf100 --cell king768x256
touch data/nnue/campaigns/sf100-king-fixedinit-v1/DONE
echo "[e4] KING RERUN COMPLETE"

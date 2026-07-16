#!/usr/bin/env bash
# E1 size control: sf-100 truncated to E1's exact corpus size (4,064,476
# records), identical recipe/schedule, 300g screen. Distinguishes "fishpack
# positions are worse" from "4M positions is just less than 25M" when
# reading E1. Gates on E3 DONE (end of the GPU chain).
set -euo pipefail
cd "$(dirname "$0")/../.."
EC=data/nnue/experiments/e1-size-control
mkdir -p "$EC"
until [ -f data/nnue/experiments/uho-transplant/DONE ]; do sleep 60; done

N=4064476
head -c $((N * 32)) data/nnue/train.bin > "$EC/train.bin"
ls -la "$EC/train.bin"
python3 - <<'EOF'
import json
cfg = json.load(open('data/nnue/campaigns/sf100-matrix-v1/resolved-configs/sf100-matrix-v1-sf100-classic256.json'))
cfg['net_id'] = 'e1ctl-sf100-4m-classic256'
cfg['notes'] = 'E1 size control: sf-100 prefix truncated to 4,064,476 records (E1 corpus size), net-0 recipe.'
cfg['corpus'] = {'id': 'sf-100-prefix-4m', 'raw_dir': None,
                 'train_bin': 'data/nnue/experiments/e1-size-control/train.bin'}
cfg['train']['checkpoint_dir'] = '/root/projects/learnrust/chessengine/data/nnue/experiments/e1-size-control/ckpt'
env = cfg['train']['env']
env['NET0_DATA'] = 'data/nnue/experiments/e1-size-control/train.bin'
env['NET0_FORMAT'] = 'bulletformat'
env['NET0_SUPERBATCHES'] = '30'
env['NET0_LR_END_SB'] = '30'
env['NET0_SAVE_RATE'] = '30'
json.dump(cfg, open('data/nnue/experiments/e1-size-control/config.json', 'w'), indent=2)
EOF
uv run python tools/nnue/train_run.py --config "$EC/config.json" \
  --only preflight,train,verify,pack,sanity
uv run python tools/nnue/eval_match.py \
  --candidate data/nnue/nets/e1ctl-sf100-4m-classic256.cenn \
  --baseline data/nnue/nets/net-0.cenn \
  --games 300 --nodes 5000 --concurrency 4 --seed 20260715 \
  --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
  --output "$EC/screen-n5k"
python3 -c "import json;s=json.load(open('$EC/screen-n5k/summary.json'));print(f'[e1ctl] RESULT: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
touch "$EC/DONE"
echo "[e1ctl] E1 SIZE CONTROL COMPLETE"

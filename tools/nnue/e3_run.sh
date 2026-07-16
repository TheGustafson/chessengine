#!/usr/bin/env bash
# E3 data transplant (docs/research/value-foundation-state.md): net-0's exact
# recipe on the official stage-1 corpus (UHO 10GiB prefix). Self-gates on E1
# DONE (GPU serial: matrix -> E1 -> E3).
set -euo pipefail
cd "$(dirname "$0")/../.."

E1=data/nnue/experiments/fishpack-sf16-relabel
E3=data/nnue/experiments/uho-transplant
mkdir -p "$E3"
echo "[e3 $(date +%H:%M:%S)] waiting for E1 DONE"
until [ -f "$E1/DONE" ]; do sleep 60; done
bash tools/nnue/apply_bullet_patches.sh

python3 - <<'EOF'
import json
cfg = json.load(open('data/nnue/campaigns/sf100-matrix-v1/resolved-configs/sf100-matrix-v1-sf100-classic256.json'))
net_id = 'e3-uho-transplant-classic256'
cfg['net_id'] = net_id
cfg['notes'] = 'E3: net-0 exact recipe on official stage-1 UHO prefix; div 1.9 (units->SF16 frame), skip75 (their rfs=3).'
cfg['corpus'] = {'id': 'nodes5000pv2-UHO-head10GiB', 'raw_dir': None,
                 'train_bin': 'data/corpora/official-stockfish-master-binpacks/main/samples/nodes5000pv2_UHO.binpack.head10737418240'}
cfg['train']['checkpoint_dir'] = '/root/projects/learnrust/chessengine/data/nnue/experiments/uho-transplant/ckpt'
env = cfg['train']['env']
env['NET0_DATA'] = 'data/corpora/official-stockfish-master-binpacks/main/samples/nodes5000pv2_UHO.binpack.head10737418240'
env['NET0_FORMAT'] = 'sfbinpack'
env['NET0_SF_BUFFER_MB'] = '256'
env['NET0_SUPERBATCHES'] = '30'
env['NET0_LR_END_SB'] = '30'
env['NET0_SAVE_RATE'] = '30'
env['SFBINPACK_SCORE_DIV'] = '1.9'
env['NET0_RANDOM_SKIP_PCT'] = '75'
json.dump(cfg, open('data/nnue/experiments/uho-transplant/e3-config.json', 'w'), indent=2)
EOF
echo "[e3 $(date +%H:%M:%S)] training"
uv run python tools/nnue/train_run.py --config "$E3/e3-config.json" \
  --only preflight,train,verify,pack,sanity
echo "[e3 $(date +%H:%M:%S)] screening 300g @5k vs net-0"
uv run python tools/nnue/eval_match.py \
  --candidate data/nnue/nets/e3-uho-transplant-classic256.cenn \
  --baseline data/nnue/nets/net-0.cenn \
  --games 300 --nodes 5000 --concurrency 4 --seed 20260715 \
  --candidate-engine engines/bin/probe-default --baseline-engine engines/bin/probe-default \
  --output "$E3/screen-n5k"
python3 -c "import json;s=json.load(open('$E3/screen-n5k/summary.json'));print(f'[e3] RESULT: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
touch "$E3/DONE"
echo "[e3 $(date +%H:%M:%S)] E3 COMPLETE"

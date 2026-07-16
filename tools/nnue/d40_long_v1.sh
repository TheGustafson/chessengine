#!/usr/bin/env bash
# Overnight D40 stage-1 scale run: 720 SB on full UHO = 18B presentations
# (3x the 240sb run; official stage-1 for this arch family is 40-60B).
# Waits for GPU idle (the s2 hop train), then runs ~7h. Runs DURING the
# equal-time gate by standing exception: symmetric load, precedent 0
# forfeits at higher loads today — caveat recorded with any result.
# Screen next morning: h2h vs d40-uho-v1-240sb (same arch).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[d40long $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for GPU idle (s2 hop train)"
while true; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" -eq 0 ] && break
  sleep 120
done

name=d40-uho-720sb
d=data/nnue/experiments/d40-long-v1
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/d40-uho-v1/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'd40-uho-720sb'
cfg['notes'] = 'D40 stage-1 scale run: 720 SB = 18B presentations on full UHO (3x the 240sb run).'
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/d40-long-v1/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
cfg['train']['quantised'] = 'd40-uho-720sb-sb720-quant.bin'
cfg['parity']['raw'] = f'{ckpt}/d40-uho-720sb-final-raw.bin'
env = cfg['train']['env']
env['D40_OUT'] = ckpt
env['D40_ID'] = 'd40-uho-720sb'
env['D40_SUPERBATCHES'] = '720'
env['D40_LR_END_SB'] = '720'
env['D40_SAVE_RATE'] = '720'
json.dump(cfg, open('data/nnue/experiments/d40-long-v1/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/$name.cenn" ]; then
  say "training $name (720 SB, ~7h)"
  CUDA_PATH=/usr uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,parity,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train/pack/parity FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK (parity included)"
fi
say "D40 LONG TRAIN DONE (screen when box is quiet)"

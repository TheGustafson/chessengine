#!/usr/bin/env bash
# UHO-10GiB schedule + capacity ladder (auto-research loop, 2026-07-16).
# Four GPU trains, serial, launched once the GPU is idle (E4 done):
#   u256-60sb, u256-120sb   — does more optimization on the same 10GiB pay?
#   u512-60sb, u1024-120sb  — does capacity unlock on billions-scale data?
# TRAIN ONLY (through pack+sanity). Screens run separately when the CPU is
# quiet (after the E3 equal-time gate).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[ladder $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for GPU idle (E4 training)"
while true; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" -eq 0 ] && break
  sleep 60
done
say "GPU idle; generating configs"

python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/uho-transplant/e3-config.json'))
cells = [
    ('u256-60sb',   256,  60),
    ('u256-120sb',  256, 120),
    ('u512-60sb',   512,  60),
    ('u1024-120sb', 1024, 120),
]
for name, hidden, sb in cells:
    cfg = copy.deepcopy(base)
    net_id = f'uho-ladder-v1-{name}'
    cfg['net_id'] = net_id
    cfg['arch'] = f'768x2x{hidden}x1-screlu'
    cfg['pack']['l1'] = hidden
    cfg['engine_features'] = None if hidden == 256 else f'nnue-l1-{hidden}'
    cfg['notes'] = (f'UHO ladder: {name} on nodes5000pv2_UHO 10GiB clean prefix; '
                    'e3 recipe otherwise (div 1.9, skip75, WDL 0.3).')
    ckpt = f'/root/projects/learnrust/chessengine/data/nnue/experiments/uho-ladder-v1/{name}/ckpt'
    os.makedirs(ckpt, exist_ok=True)
    cfg['train']['checkpoint_dir'] = ckpt
    env = cfg['train']['env']
    env['NET0_HIDDEN'] = str(hidden)
    env['NET0_SUPERBATCHES'] = str(sb)
    env['NET0_LR_END_SB'] = str(sb)
    env['NET0_SAVE_RATE'] = str(sb)
    json.dump(cfg, open(f'data/nnue/experiments/uho-ladder-v1/{name}/config.json', 'w'), indent=2)
    print(f'wrote {name}')
EOF

for name in u256-60sb u256-120sb u512-60sb u1024-120sb; do
  d=data/nnue/experiments/uho-ladder-v1/$name
  if [ -f "data/nnue/nets/uho-ladder-v1-$name.cenn" ]; then
    say "$name already packed, skipping"; continue
  fi
  say "training $name"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    && say "$name packed OK" \
    || say "$name FAILED (see $d/run.log)"
done
say "LADDER TRAINS COMPLETE"

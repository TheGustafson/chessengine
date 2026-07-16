#!/usr/bin/env bash
# Full-feature track, run 1: D40 (SF HalfKAv2_hm + FullThreats, sota engine)
# stage-1 on the full 40.3GB UHO corpus — the first rich-feature net ever
# given billions-scale data. Recipe mirrors the proven classic path:
# div 1.9, skip75, WDL 0.3, 240 SB. bench NPS sota ~= l2048, so a fixed-node
# win should transfer to equal time.
# Screen: h2h vs s4-2048-t74-240sb (current working-best), conc 1 (sota
# engine RAM), 300g @5k.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
UHO=data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack
say() { echo "[d40uho $(date +%H:%M:%S)] $*" | tee -a "$R"; }

name=d40-uho-240sb
d=data/nnue/experiments/d40-uho-v1
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/campaigns/sf100-matrix-v1/resolved-configs/sf100-matrix-v1-sf100-d40-full.json'))
cfg = copy.deepcopy(base)
UHO = 'data/corpora/official-stockfish-master-binpacks/main/raw/nodes5000pv2_UHO.binpack'
cfg['net_id'] = 'd40-uho-v1-240sb'
cfg['corpus'] = {'id': 'uho-stage1-full', 'raw_dir': None, 'train_bin': UHO}
cfg['notes'] = ('Full-feature track run 1: D40 stage-1 on full UHO (first rich-feature net '
                'at billions-scale data); div 1.9, skip75, WDL 0.3, 240 SB.')
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/d40-uho-v1/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
env = cfg['train']['env']
env['D40_DATA'] = UHO
env['D40_FORMAT'] = 'sfbinpack'
env['D40_OUT'] = ckpt
env['D40_ID'] = 'd40-uho-v1-240sb'
env['D40_SUPERBATCHES'] = '240'
env['D40_LR_END_SB'] = '240'
env['D40_SAVE_RATE'] = '240'
env['SFBINPACK_SCORE_DIV'] = '1.9'
env['D40_RANDOM_SKIP_PCT'] = '75'
# parity raw path follows the new checkpoint dir
cfg['parity']['raw'] = f'{ckpt}/d40-uho-v1-240sb-final-raw.bin'
json.dump(cfg, open('data/nnue/experiments/d40-uho-v1/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/d40-uho-v1-240sb.cenn" ]; then
  say "training $name (D40 on full UHO; expect hours)"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train/pack FAILED (see $d/run.log — check the parity gate)"; exit 1; }
  say "$name packed OK (parity gate passed)"
fi
if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h screening $name vs s4-2048-t74-240sb (300g @5k, conc 1 - sota RAM)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/d40-uho-v1-240sb.cenn \
    --baseline data/nnue/nets/stage2-v1-s4-2048-t74-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 1 --seed 20260730 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-l2048 \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs s4-working-best] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "D40-UHO RUN1 DONE"

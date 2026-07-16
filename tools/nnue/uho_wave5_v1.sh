#!/usr/bin/env bash
# Wave 5: width-4096 capacity rung on full UHO (gated on the farseer stage-2
# wave releasing the GPU). h2h vs the incumbent uf2048-240sb (cross-width),
# plus bench NPS. Equal-time gate decided separately if the h2h wins.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[wave5 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for the farseer stage-2 wave to finish"
until grep -aqE "^\[s2w3 [0-9:]+\] S2W3 DONE" "$R"; do sleep 120; done

name=uf4096-240sb
d=data/nnue/experiments/uho-wave5/$name
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/uho-wave4/uf2048-240sb/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'uho-wave5-uf4096-240sb'
cfg['arch'] = '768x2x4096x1-screlu'
cfg['pack']['l1'] = 4096
cfg['engine_features'] = 'nnue-l1-4096'
cfg['notes'] = 'Wave 5: width-4096 rung on full UHO; e3 recipe (div 1.9, skip75, WDL 0.3).'
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/uho-wave5/uf4096-240sb/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
cfg['train']['env']['NET0_HIDDEN'] = '4096'
json.dump(cfg, open('data/nnue/experiments/uho-wave5/uf4096-240sb/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/uho-wave5-$name.cenn" ]; then
  say "training $name (expect ~2h)"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK"
fi
nps=$(./engines/bin/probe-l4096 bench 16 1 5 default depth 2>&1 | grep -i "nodes/second" | grep -oE "[0-9]+" | tail -1)
say "probe-l4096 bench nps=$nps"
if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h screening $name vs incumbent uf2048-240sb (300g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/uho-wave5-$name.cenn \
    --baseline data/nnue/nets/uho-wave4-uf2048-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260726 \
    --candidate-engine engines/bin/probe-l4096 --baseline-engine engines/bin/probe-l2048 \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs uf2048-240sb] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "WAVE5 DONE"

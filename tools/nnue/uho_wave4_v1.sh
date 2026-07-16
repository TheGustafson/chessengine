#!/usr/bin/env bash
# Wave 4 (auto-research loop): the width-2048 capacity rung on full UHO.
# Gated on wave3 finishing (its trainer owns target/release until then).
#   uf2048-240sb — WDL 0.3 for comparability with uf1024-240sb
# Screen: h2h vs uf1024-240sb, 300g @5k (cross-width: probe-l2048 vs l1024),
# plus a bench-NPS read of probe-l2048.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[wave4 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

# NOTE: the wait phrase must NOT appear in our own log lines (say() appends
# to $R — an earlier version gated on a phrase it had just written itself).
say "waiting for wave3 to finish"
until grep -aqE "^\[wave3 [0-9:]+\] WAVE3 COMPLETE" "$R"; do sleep 120; done

name=uf2048-240sb
d=data/nnue/experiments/uho-wave4/$name
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/uho-full-v1/uf1024-240sb/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'uho-wave4-uf2048-240sb'
cfg['arch'] = '768x2x2048x1-screlu'
cfg['pack']['l1'] = 2048
cfg['engine_features'] = 'nnue-l1-2048'
cfg['notes'] = 'Wave 4: width-2048 rung on full UHO; e3 recipe (div 1.9, skip75, WDL 0.3).'
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/uho-wave4/uf2048-240sb/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
cfg['train']['env']['NET0_HIDDEN'] = '2048'
json.dump(cfg, open('data/nnue/experiments/uho-wave4/uf2048-240sb/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/uho-wave4-$name.cenn" ]; then
  say "training $name"
  uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK"
fi

nps=$(./engines/bin/probe-l2048 bench 16 1 5 default depth 2>&1 | grep -i "nodes/second" | grep -oE "[0-9]+" | tail -1)
say "probe-l2048 bench nps=$nps"

if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h screening $name vs uf1024-240sb (300g @5k)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/uho-wave4-$name.cenn \
    --baseline data/nnue/nets/uho-full-v1-uf1024-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260720 \
    --candidate-engine engines/bin/probe-l2048 --baseline-engine engines/bin/probe-l1024 \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name screen FAILED"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'[$name vs uf1024-240sb] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "WAVE4 COMPLETE"

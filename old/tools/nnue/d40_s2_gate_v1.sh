#!/usr/bin/env bash
# D40 stage-2 hop + promotion gate chain:
#   1. d40-s2-farseer-120sb — warm start (D40_INIT_RAW) from d40-uho-v1's
#      raw checkpoint, farseerT76, div 1.9, LR/2, WDL 0.25, 120 SB.
#   2. h2h vs d40-uho-v1-240sb (same arch, conc 1, 300g @5k).
#   3. EQUAL-TIME GATE: the better of the two D40 nets vs s4 working-best,
#      200g 10+0.1, conc 1 (sota engine 3.6GB => no parallel games), quiet.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
T76=data/corpora/official-stockfish-master-binpacks/main/raw/farseerT76.binpack
say() { echo "[d40s2 $(date +%H:%M:%S)] $*" | tee -a "$R"; }

name=d40-s2-farseer-120sb
d=data/nnue/experiments/d40-s2-v1
mkdir -p "$d"
python3 - <<'EOF'
import json, copy, os
base = json.load(open('data/nnue/experiments/d40-uho-v1/config.json'))
cfg = copy.deepcopy(base)
cfg['net_id'] = 'd40-s2-farseer-120sb'
cfg['parent'] = 'd40-uho-v1-240sb'
cfg['corpus'] = {'id': 'farseert76-full', 'raw_dir': None,
                 'train_bin': 'data/corpora/official-stockfish-master-binpacks/main/raw/farseerT76.binpack'}
cfg['notes'] = 'D40 stage-2 hop: warm from d40-uho-v1 raw ckpt, farseerT76, LR/2, WDL 0.25, div 1.9.'
ckpt = '/root/projects/learnrust/chessengine/data/nnue/experiments/d40-s2-v1/ckpt'
os.makedirs(ckpt, exist_ok=True)
cfg['train']['checkpoint_dir'] = ckpt
cfg['train']['quantised'] = 'd40-s2-farseer-120sb-sb120-quant.bin'
cfg['parity']['raw'] = f'{ckpt}/d40-s2-farseer-120sb-final-raw.bin'
env = cfg['train']['env']
env['D40_DATA'] = cfg['corpus']['train_bin']
env['D40_OUT'] = ckpt
env['D40_ID'] = 'd40-s2-farseer-120sb'
env['D40_INIT_RAW'] = 'data/nnue/experiments/d40-uho-v1/ckpt/d40-uho-v1-240sb-final-raw.bin'
env['D40_LR'] = '0.0005'
env['D40_WDL'] = '0.25'
env['D40_SUPERBATCHES'] = '120'
env['D40_LR_END_SB'] = '120'
env['D40_SAVE_RATE'] = '120'
json.dump(cfg, open('data/nnue/experiments/d40-s2-v1/config.json', 'w'), indent=2)
EOF

if [ ! -f "data/nnue/nets/$name.cenn" ]; then
  say "training $name (warm from d40-uho raw, ~1h)"
  CUDA_PATH=/usr uv run python tools/nnue/train_run.py --config "$d/config.json" \
    --only preflight,train,verify,pack,parity,sanity >> "$d/run.log" 2>&1 \
    || { say "$name train/pack/parity FAILED (see $d/run.log)"; exit 1; }
  say "$name packed OK (parity included)"
fi

if [ ! -f "$d/h2h/summary.json" ]; then
  say "h2h $name vs d40-uho-v1-240sb (300g @5k, conc 1)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/$name.cenn \
    --baseline data/nnue/nets/d40-uho-v1-240sb.cenn \
    --games 300 --nodes 5000 --concurrency 1 --seed 20260734 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-sota \
    --output "$d/h2h" > /dev/null 2>&1 || { say "$name h2h FAILED"; exit 1; }
fi
H2H=$(python3 -c "import json;s=json.load(open('$d/h2h/summary.json'));print(f'{s[\"candidate_elo\"]:.1f} {s[\"candidate_elo_se\"]:.1f} {s[\"candidate_score\"]:.3f}')")
echo "[$name vs d40-uho] RESULT: elo/se/score = $H2H" | tee -a "$R"

# pick gate candidate: the hop if h2h elo > 0, else the stage-1 net
BEST=$name
if python3 -c "import sys;sys.exit(0 if float('$H2H'.split()[0]) <= 0 else 1)"; then
  BEST=d40-uho-v1-240sb
fi
say "equal-time gate: $BEST vs s4 (200g 10+0.1, conc 1, ~3h)"
G=data/nnue/experiments/d40-s2-v1/gate-tc10
if [ ! -f "$G/summary.json" ] || ! grep -q '"status": "complete"' "$G/summary.json" 2>/dev/null; then
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/$BEST.cenn \
    --baseline data/nnue/nets/stage2-v1-s4-2048-t74-240sb.cenn \
    --games 200 --tc 10+0.1 --concurrency 1 --seed 20260735 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-l2048 \
    --output "$G" > /dev/null 2>&1 || { say "gate FAILED to run"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$G/summary.json'));print(f'[GATE $BEST vs s4 EQUAL-TIME] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "D40 S2+GATE DONE"

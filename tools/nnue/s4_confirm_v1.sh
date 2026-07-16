#!/usr/bin/env bash
# s4 promotion confirm: second 300g block (fresh seed) for s4-2048-t74-240sb
# vs the s3 incumbent, pooled to 600g. Gated on the SF16 match (timed games
# own the box until then).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
say() { echo "[s4confirm $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for the SF16 match to release the box"
until grep -aqE "^\[sf16match [0-9:]+\] SF16 MATCH DONE" "$R"; do sleep 60; done
sleep 15

d=data/nnue/experiments/stage2-v1/s4-2048-t74-240sb
if [ ! -f "$d/h2h-b/summary.json" ]; then
  say "confirm block: 300g @5k, fresh seed"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/stage2-v1-s4-2048-t74-240sb.cenn \
    --baseline data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn \
    --games 300 --nodes 5000 --concurrency 4 --seed 20260729 \
    --candidate-engine engines/bin/probe-l2048 --baseline-engine engines/bin/probe-l2048 \
    --output "$d/h2h-b" > /dev/null 2>&1 || { say "confirm block FAILED"; exit 1; }
fi
python3 - <<'EOF' | tee -a data/nnue/experiments/chain-v2-results.log
import json, math
a = json.load(open('data/nnue/experiments/stage2-v1/s4-2048-t74-240sb/h2h/summary.json'))
b = json.load(open('data/nnue/experiments/stage2-v1/s4-2048-t74-240sb/h2h-b/summary.json'))
w = a['candidate_wins']+b['candidate_wins']; l = a['candidate_losses']+b['candidate_losses']; d = a['draws']+b['draws']; g = w+l+d
s = (w+0.5*d)/g
elo = 400*math.log10(s/(1-s))
var = (w*(1-s)**2 + l*(0-s)**2 + d*(0.5-s)**2)/(g-1)
se = 400/math.log(10)*math.sqrt(var/g)/(s*(1-s))
print(f'[s4 POOLED {g}g vs s3-incumbent] RESULT: elo={elo:.1f} se={se:.1f} score={s:.3f}')
EOF
say "S4 CONFIRM DONE"

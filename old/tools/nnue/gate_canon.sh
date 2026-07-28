#!/usr/bin/env bash
# Gate a candidate NNUE: (1) eval-scale check vs the incumbent (a scale-broken
# net evals ~0.5x — see the 2026-07-20 root-cause), (2) fixed-node 5k match vs a
# baseline. Usage: gate_canon.sh <cand.cenn> [baseline.cenn] [games]
set -uo pipefail
cd "$(dirname "$0")/../.."
CAND="${1:?candidate .cenn}"
BASE="${2:-data/nnue/nets/d40-uho-v1-240sb.cenn}"
GAMES="${3:-200}"
ENG=engines/bin/archon-sota-native
INC=data/nnue/nets/d40-uho-720sb.cenn
FENS=/tmp/gate_cmp.fens
[ -s "$FENS" ] || head -1000 data/nnue/canon-v1/inputs/tasks/000000.fens > "$FENS"

echo "=== [1a] HARD scale check (mandatory — a FAIL aborts before any games) ==="
# 2026-07-21: this used to be advisory; canon-clean printed 'scale 0.505x' here
# and the 160-game match ran anyway. Never again: net_scale_check.sh is the
# single source of truth (band 0.85-1.18 + sign>=95%); FORCE=1 to override for
# a deliberate calibration experiment.
if ! bash tools/nnue/net_scale_check.sh "$CAND" "$INC" "$ENG"; then
  if [ "${FORCE:-0}" = 1 ]; then
    echo "[gate] scale check FAILED but FORCE=1 — continuing (calibration experiment mode)"
  else
    echo "[gate] ABORT: candidate fails the output-scale check. Fix SCORE_MULT /"
    echo "[gate] the recipe before burning games (FORCE=1 to override)."
    exit 1
  fi
fi

echo "=== [1b] eval agreement vs incumbent d40-uho-720sb (advisory diagnostics) ==="
# Slope ALONE is a broken instrument: the -662 fulldist net read 0.95x scale yet
# was 1000cp wrong on 5-7% of positions (Fable diagnosis 2026-07-20). Play quality
# is governed by the p99 TAIL (search minimaxes into it), not the mean. So report
# correlation + tail, with a health verdict. Healthy: r>=0.97, p99~260cp, >400cp~0.
cat "$FENS" | "$ENG" nnue-eval "$CAND" 2>/dev/null > /tmp/gate_cand.ev
cat "$FENS" | "$ENG" nnue-eval "$INC" 2>/dev/null > /tmp/gate_inc.ev
paste /tmp/gate_cand.ev /tmp/gate_inc.ev | python3 -c "
import sys, statistics, math
xs=[];ys=[]
for l in sys.stdin:
    p=l.split()
    if len(p)<2: continue
    try: c=int(p[0]); i=int(p[1])
    except: continue
    xs.append(i);ys.append(c)   # xs=incumbent(ref), ys=candidate
n=len(xs)
if n==0:
    print('  NO DATA'); sys.exit(0)
sxx=sum(x*x for x in xs); k=sum(x*y for x,y in zip(xs,ys))/sxx if sxx else 0
mx=statistics.mean(xs); my=statistics.mean(ys)
sdx=math.sqrt(sum((x-mx)**2 for x in xs)); sdy=math.sqrt(sum((y-my)**2 for y in ys))
r=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(sdx*sdy) if sdx and sdy else 0
diffs=sorted(abs(y-x) for x,y in zip(xs,ys))
p99=diffs[min(int(0.99*n), n-1)]; tail=sum(1 for d in diffs if d>400)
verdict='PASS' if (r>=0.97 and tail<=0.01*n) else ('WARN' if r>=0.94 else 'FAIL')
print(f'  scale {k:.3f}x  r={r:.4f}  p99|diff|={p99}cp  >400cp={tail}/{n}  mean|cand|={statistics.mean(abs(y) for y in ys):.0f}  [{verdict}]')
print(f'  (healthy: r>=0.97, p99~260, >400cp~0  |  poison: r~0.91, p99~1000, >400cp 50-70/1000)')
"
echo "=== [2] fixed-node 5k match: candidate vs $(basename "$BASE" .cenn), ${GAMES}g ==="
OUT="data/nnue/experiments/gate-$(basename "$CAND" .cenn)"
python3 tools/nnue/eval_match.py \
  --candidate "$CAND" --baseline "$BASE" \
  --nodes 5000 --games "$GAMES" --concurrency 8 \
  --candidate-engine "$ENG" --baseline-engine "$ENG" \
  --openings engines/books/UHO_Lichess_4852_v1.epd \
  --output "$OUT" 2>/dev/null
python3 -c "import json;d=json.load(open('$OUT/summary.json'));print(f'  RESULT: score={d[\"candidate_score\"]:.3f}  elo={d[\"candidate_elo\"]:.1f} +/- {d[\"candidate_elo_se\"]:.1f}')"

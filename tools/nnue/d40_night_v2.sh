#!/usr/bin/env bash
# Overnight D40 screen+gate chain, deferred from the failed d40_s2_gate run:
# the hop failed the runtime-parity gate (max 0.241 vs 0.20; mean 0.011 clean;
# worst fen is an OOD bare endgame -> quant-clipping tail, not a transform
# bug), and RAM forbids sota-vs-sota games while the 720sb trainer is
# resident (2 engines ~3.6GB each on a 7.6GB box). Waits for the train to
# finish, then on the quiet box:
#   1. h2h hop vs d40-uho-v1-240sb — DIAGNOSTIC ONLY (parity-failed nets are
#      not gate candidates; user decides waiver/rescale tomorrow).
#   2. if d40-uho-720sb packed: parse its parity from run.log, h2h vs 240sb.
#   3. EQUAL-TIME GATE vs s4 with the best PARITY-PASSING D40 net
#      (720sb if it passed parity and won its h2h, else d40-uho-v1-240sb),
#      200g 10+0.1 conc 1 — the promotion #4 decision.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[d40night $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for the 720sb train to finish (~02:15)"
while pgrep -f 'examples/d40_features' >/dev/null || pgrep -f 'd40_long_v1[.]sh' >/dev/null; do
  sleep 300
done
while [ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -lt 6000000 ]; do
  say "RAM still tight after train end, waiting"
  sleep 300
done

D=data/nnue/experiments/d40-night-v2
mkdir -p "$D"

h2h() { # name outdir seed  -> prints RESULT line; rc 1 on match failure
  local name=$1 out=$2 seed=$3
  if ! grep -q '"status": "complete"' "$out/summary.json" 2>/dev/null; then
    uv run python tools/nnue/eval_match.py \
      --candidate "data/nnue/nets/$name.cenn" \
      --baseline data/nnue/nets/d40-uho-v1-240sb.cenn \
      --games 300 --nodes 5000 --concurrency 1 --seed "$seed" \
      --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-sota \
      --output "$out" >/dev/null 2>&1 || { say "$name h2h FAILED"; return 1; }
  fi
  local e
  e=$(python3 -c "import json;s=json.load(open('$out/summary.json'));print(f'elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')") || return 1
  echo "[$name vs d40-uho-v1-240sb] RESULT: $e" | tee -a "$R"
}

say "h2h #1: hop d40-s2-farseer-120sb (parity-failed, diagnostic only)"
h2h d40-s2-farseer-120sb "$D/h2h-hop" 20260736 || true

GATE_CAND=d40-uho-v1-240sb
if [ -f data/nnue/nets/d40-uho-720sb.cenn ]; then
  P=$(grep -a 'runtime mean/max=' data/nnue/experiments/d40-long-v1/run.log | tail -1 | sed 's/.*runtime mean\/max=//;s/ .*//')
  say "720sb runtime parity mean/max = $P"
  OK=$(python3 -c "m,x='$P'.split('/');print(int(float(m)<=0.03 and float(x)<=0.2))" 2>/dev/null || echo 0)
  say "h2h #2: d40-uho-720sb vs d40-uho-v1-240sb"
  if h2h d40-uho-720sb "$D/h2h-720" 20260737 && [ "$OK" = "1" ]; then
    W=$(python3 -c "import json;print(1 if json.load(open('$D/h2h-720/summary.json'))['candidate_elo']>0 else 0)")
    [ "$W" = "1" ] && GATE_CAND=d40-uho-720sb
  fi
  [ "$OK" = "1" ] || say "720sb failed the parity thresholds — not a gate candidate"
else
  say "no 720sb net on disk (train failed?) — gating the stage-1 net"
fi

say "EQUAL-TIME GATE: $GATE_CAND vs s4 (200g 10+0.1, conc 1, ~3h)"
G="$D/gate-tc10"
if ! grep -q '"status": "complete"' "$G/summary.json" 2>/dev/null; then
  uv run python tools/nnue/eval_match.py \
    --candidate "data/nnue/nets/$GATE_CAND.cenn" \
    --baseline data/nnue/nets/stage2-v1-s4-2048-t74-240sb.cenn \
    --games 200 --tc 10+0.1 --concurrency 1 --seed 20260738 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-l2048 \
    --output "$G" >/dev/null 2>&1 || { say "gate FAILED to run"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$G/summary.json'));print(f'[GATE $GATE_CAND vs s4 EQUAL-TIME] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"
say "D40 NIGHT CHAIN COMPLETE"

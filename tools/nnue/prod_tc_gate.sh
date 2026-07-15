#!/usr/bin/env bash
# Equal-time promotion gate (user, 2026-07-15: candidates must beat net-0 at
# EQUAL TIME). Self-gates on the prod-fishpack-cal-v1 fixed-node stage: any
# candidate whose 5k-node screen Elo >= the width-specific floor gets a
# 200-game 10+0.1 real-TC match vs net-0. Floors: 256-width -10 (same
# inference cost as net-0, fixed-node ~= equal-time); 1024-width +20 (must
# out-earn its NPS tax).
set -euo pipefail
cd "$(dirname "$0")/../.."

C=data/nnue/campaigns/prod-fishpack-cal-v1
echo "[tc-gate $(date +%H:%M:%S)] waiting for fixed-node stage DONE"
until [ -f "$C/DONE" ]; do sleep 60; done

ran_any=0
for width in 256 1024; do
  id="prod-fishpack-cal-v1-classic$width"
  summary="$C/matches/$id-screen-n5k/summary.json"
  [ -f "$summary" ] || { echo "[tc-gate] $id: no fixed-node screen, skip"; continue; }
  elo=$(python3 -c "import json;print(json.load(open('$summary'))['candidate_elo'])")
  floor=-10; cand_engine=engines/bin/probe-default
  [ "$width" = 1024 ] && floor=20 && cand_engine=engines/bin/probe-l1024
  pass=$(python3 -c "print(1 if $elo >= $floor else 0)")
  if [ "$pass" != "1" ]; then
    echo "[tc-gate] RESULT $id: fixed-node elo=$elo < floor $floor — equal-time gate NOT run"
    continue
  fi
  ran_any=1
  echo "[tc-gate $(date +%H:%M:%S)] $id passed floor (elo=$elo); running 200g 10+0.1 vs net-0"
  uv run python tools/nnue/eval_match.py \
    --candidate "data/nnue/nets/$id.cenn" --baseline data/nnue/nets/net-0.cenn \
    --games 200 --tc "10+0.1" --concurrency 4 --seed 20260715 \
    --candidate-engine "$cand_engine" --baseline-engine engines/bin/probe-default \
    --output "$C/matches/$id-tc-10-01"
  python3 -c "import json;s=json.load(open('$C/matches/$id-tc-10-01/summary.json'));print(f'[tc-gate] RESULT $id EQUAL-TIME 10+0.1: score={s[\"candidate_score\"]:.3f} elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f}')"
done

[ "$ran_any" = "0" ] && echo "[tc-gate] no candidate reached the equal-time gate"
touch "$C/DONE-TC"
echo "[tc-gate $(date +%H:%M:%S)] EQUAL-TIME GATE COMPLETE"

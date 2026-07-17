#!/usr/bin/env bash
# Quiet-window gate sequence (D45/D45a, 2026-07-17). Invoke MANUALLY when the
# box is quiet (big-v1 labeling done, current SPSA campaign finished, no
# trainer): TC games need wall-clock fairness.
#   1. c1 corrhist SPSA ship gate: tuned-vs-default, 10+0.1, SPRT [0,10].
#   2. d40-uho-720sb equal-time gate vs s4 (parity-recalibration admittee).
#   3. Speed measurements: R4-stack NPS A/B + R1 native-build delta.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=data/nnue/experiments/chain-v2-results.log
say() { echo "[gates $(date +%H:%M:%S)] $*" | tee -a "$R"; }
S3=data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn

busy=$(pgrep -f "spsa_tune.py|root-analysis|d40_features" | wc -l || true)
if [ "$busy" -gt 0 ]; then
  say "ABORT: box not quiet ($busy tuner/labeler/trainer procs)"; exit 1
fi

# --- 1. c1 ship gate -------------------------------------------------------
THETA=data/nnue/experiments/spsa/c1-corrhist/theta_final.json
if [ -f "$THETA" ] && ! grep -q "C1 GATE RESULT" "$R"; then
  OPTS=$(python3 -c "
import json
f=json.load(open('$THETA'))['final']
print(' '.join(f'option.{k}={v}' for k,v in f.items()))")
  say "c1 ship gate: tuned corrhist vs defaults (10+0.1, SPRT [0,10], cap 600g)"
  OUT=data/nnue/experiments/spsa/c1-corrhist/tc-gate.log
  # shellcheck disable=SC2086
  engines/bin/fastchess \
    -engine cmd=engines/bin/tune-l2048 name=tuned $OPTS \
    -engine cmd=engines/bin/tune-l2048 name=default \
    -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=16 \
      "option.EvalFile=$S3" \
    -openings file=data/nnue/openings/UHO_Lichess_4852_v1.epd format=epd order=random \
    -repeat -games 2 -rounds 300 -concurrency 3 -recover \
    -draw movenumber=40 movecount=8 score=15 \
    -resign movecount=4 score=700 -maxmoves 250 \
    -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 model=logistic \
    -srand 20260721 > "$OUT" 2>&1 || true
  tail -30 "$OUT" | grep -E "Games:|LLR:" | tail -4 | sed 's/^/[gates c1] /' | tee -a "$R"
  echo "[C1 GATE RESULT] $(grep -E 'LLR:' "$OUT" | tail -1) | $(grep -E 'Games:' "$OUT" | tail -1)" | tee -a "$R"
fi

# --- 2. 720sb equal-time gate ---------------------------------------------
G=data/nnue/experiments/d40-720sb-gate
if ! grep -q '"status": "complete"' "$G/summary.json" 2>/dev/null; then
  say "720sb equal-time gate vs s4 (200g 10+0.1, conc 3)"
  uv run python tools/nnue/eval_match.py \
    --candidate data/nnue/nets/d40-uho-720sb.cenn \
    --baseline data/nnue/nets/stage2-v1-s4-2048-t74-240sb.cenn \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-l2048 \
    --games 200 --tc 10+0.1 --concurrency 3 --seed 20260722 \
    --output "$G" >/dev/null 2>&1 || { say "720sb gate FAILED to run"; exit 1; }
fi
python3 -c "import json;s=json.load(open('$G/summary.json'));print(f'[GATE d40-uho-720sb vs s4 EQUAL-TIME] RESULT: elo={s[\"candidate_elo\"]:.1f} se={s[\"candidate_elo_se\"]:.1f} score={s[\"candidate_score\"]:.3f}')" | tee -a "$R"

# --- 3. speed measurements --------------------------------------------------
say "speed: R4-stack NPS A/B (pre-r4 vs current probe-sota, D40 net, depth 13 x3)"
nps_probe() { # binary net -> median nps of 3 fixed-depth searches
  local bin=$1 net=$2 total=""
  for _ in 1 2 3; do
    n=$(printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 13\nquit\n' "$net" \
        | "$bin" 2>/dev/null | grep -o "nps [0-9]*" | tail -1 | cut -d' ' -f2)
    total="$total $n"
  done
  echo "$total" | tr ' ' '\n' | grep -v '^$' | sort -n | sed -n 2p
}
D40=data/nnue/nets/d40-uho-720sb.cenn
PRE=$(nps_probe engines/bin/probe-sota.pre-r4 "$D40")
CUR=$(nps_probe engines/bin/probe-sota "$D40")
say "R4 stack: pre-r4=$PRE nps, current=$CUR nps (D40 720sb net, depth 13 median-of-3)"
PRE2=$(nps_probe engines/bin/probe-l2048.pre-r4 "$S3")
CUR2=$(nps_probe engines/bin/probe-l2048 "$S3")
say "classic: pre-r4=$PRE2 nps, current=$CUR2 nps (s3 net)"
say "NIGHT GATES COMPLETE"

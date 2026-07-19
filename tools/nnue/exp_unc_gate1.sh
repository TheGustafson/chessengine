#!/usr/bin/env bash
# EXP-UNC-1 Gate 1 (search study bet 1): value parity of the NLL-mean twin
# (d40-uho-unc-240sb) vs the MSE control (d40-uho-v1-240sb) at FIXED NODES —
# isolates net quality. Kill the whole bet if the twin's value regressed
# (elo <= -15); otherwise proceed to the runtime plumbing + Gate 2.
#
# Self-firing: polls until the twin CENN lands (train_run.py: train -> pack ->
# parity), then runs the screen. Launch nohup'd; reads GATE1-DONE when finished.
#
# Arg 1 (optional): the train_run.py pid to wait on (default: poll for the CENN).
set -uo pipefail
cd "$(dirname "$0")/../.."

TWIN=data/nnue/nets/d40-uho-unc-240sb.cenn
CTRL=data/nnue/nets/d40-uho-v1-240sb.cenn
EXPDIR=data/nnue/experiments/d40-uho-unc-v1
OUT="$EXPDIR/gate1"
LOG="$EXPDIR/gate1.log"
WAIT_PID="${1:-}"

mkdir -p "$EXPDIR"
echo "[gate1] $(date -u +%H:%M:%S) armed; waiting for twin $TWIN" | tee -a "$LOG"

# 1. Wait for the trainer to finish (if a pid was given), then for the CENN.
if [ -n "$WAIT_PID" ]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[gate1] train pid $WAIT_PID exited" | tee -a "$LOG"
fi
for _ in $(seq 1 60); do [ -f "$TWIN" ] && break; sleep 30; done
if [ ! -f "$TWIN" ]; then
    echo "[gate1] ERROR: twin CENN never appeared — train/pack/parity likely failed; investigate $EXPDIR/train.log" | tee -a "$LOG"
    exit 1
fi

# 2. Fixed-node value screen (isolates net quality; not equal-time). SPRT
#    early-stops once the twin is provably not-worse-than -15 or clearly worse.
echo "[gate1] $(date -u +%H:%M:%S) twin present ($(stat -c%s "$TWIN") B); running 400g @5k nodes, conc 1" | tee -a "$LOG"
uv run python tools/nnue/eval_match.py \
    --candidate "$TWIN" --baseline "$CTRL" \
    --games 400 --nodes 5000 --concurrency 1 --seed 20260719 \
    --sprt-elo0 -15 --sprt-elo1 5 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-sota \
    --output "$OUT" 2>&1 | tee -a "$LOG"

# 3. Verdict.
python3 - "$OUT/summary.json" <<'PY' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1]))
elo, se = s["candidate_elo"], s["candidate_elo_se"]
lo = elo - 2 * se
if elo <= -15:
    verdict = "KILL — NLL value regression; the (mean,logvar) head hurts value quality"
elif lo > -15:
    verdict = "PASS -> build runtime consumers + Gate 2"
else:
    verdict = "MARGINAL — extend games before deciding"
print(f"[gate1] RESULT score={s['candidate_score']:.3f} elo={elo:+.1f} +/- {se:.1f} (2σ lo {lo:+.1f}) => {verdict}")
PY
touch "$EXPDIR/GATE1-DONE"
echo "[gate1] $(date -u +%H:%M:%S) done" | tee -a "$LOG"

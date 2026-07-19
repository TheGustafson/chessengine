#!/usr/bin/env bash
# EXP-UNC-1 g2 gate: the REAL Gate 1 + sidecar. The g2 retrain
# (d40-uho-unc-g2-240sb) is the first net whose l3_var actually TRAINED (the
# twin's was output-unreachable → untrained → effectively MSE, Gate 1 +5.2±14.2).
# So g2's mean is variance-weighted and needs its own value-parity screen vs the
# control. Self-firing: waits for the retrain (pid arg) + the g2 CENN, runs the
# screen, and on PASS extracts the runtime sidecar (variance head).
set -uo pipefail
cd "$(dirname "$0")/../.."
EXP=data/nnue/experiments/d40-uho-unc-g2
LOG="$EXP/gate.log"
CENN=data/nnue/nets/d40-uho-unc-g2-240sb.cenn
CTRL=data/nnue/nets/d40-uho-v1-240sb.cenn
RAW="$EXP/ckpt/d40-uho-unc-g2-240sb-final-raw.bin"
SIDE=data/nnue/nets/d40-uho-unc-g2-240sb.sidecar
RETRAIN_PID="${1:-}"
say(){ echo "[g2-gate $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$EXP"

say "waiting for the g2 retrain to finish + pack"
if [ -n "$RETRAIN_PID" ]; then while kill -0 "$RETRAIN_PID" 2>/dev/null; do sleep 60; done; fi
for _ in $(seq 1 60); do [ -f "$CENN" ] && break; sleep 30; done
if [ ! -f "$CENN" ]; then say "ERROR: g2 CENN never appeared — retrain/pack failed; see $EXP/retrain.log"; exit 1; fi

say "g2 present; extracting sidecar (variance head) first"
.venv/bin/python3 tools/nnue/extract_unc_head.py "$RAW" "$SIDE" --validate "$CENN" >> "$LOG" 2>&1 \
    && say "sidecar OK: $SIDE" || say "sidecar extraction/validation FAILED (check $LOG)"

say "REAL Gate 1: g2 mean vs control, 400g @5k nodes, conc 1, SPRT[-15,5]"
uv run python tools/nnue/eval_match.py \
    --candidate "$CENN" --baseline "$CTRL" \
    --games 400 --nodes 5000 --concurrency 1 --seed 20260719 \
    --sprt-elo0 -15 --sprt-elo1 5 \
    --candidate-engine engines/bin/probe-sota --baseline-engine engines/bin/probe-sota \
    --output "$EXP/gate1" >> "$LOG" 2>&1
python3 - "$EXP/gate1/summary.json" <<'PY' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1]))
elo, se = s["candidate_elo"], s["candidate_elo_se"]
lo = elo - 2 * se
verdict = ("KILL — real-NLL mean regressed" if elo <= -15
           else "PASS -> Gate 2 (2x2 c1/c2 probe)" if lo > -15
           else "MARGINAL — extend games")
print(f"[g2-gate] REAL GATE1 score={s['candidate_score']:.3f} elo={elo:+.1f} +/- {se:.1f} (2σ lo {lo:+.1f}) => {verdict}")
PY
touch "$EXP/G2-GATE-DONE"
say "done. Net=$CENN Sidecar=$SIDE — if PASS, run the 2x2 Gate-2 probe (design doc)."

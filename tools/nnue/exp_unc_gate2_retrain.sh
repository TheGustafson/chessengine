#!/usr/bin/env bash
# EXP-UNC-1 Gate-2 retrain, self-firing + double-gated. The running twin is
# GATE-1-ONLY (its l3_var was dropped — the raw ckpt persists only
# output-reachable weights). This re-runs the SAME twin config with the
# logvar-save fix (tracked trainer 459a368 → train_run.py copies+rebuilds it),
# reproducing the IDENTICAL mean (same seed → Gate 1 carries over) but now
# saving l3_var, then extracts the runtime sidecar. Fires ONLY when BOTH:
#   - Gate 1 PASSED (the NLL mean is not worse than the control), and
#   - the logvar-fix smoke confirmed l3_varw/l3_varb land in the raw ckpt.
# So it never wastes ~2h GPU on a dead branch. Launch nohup'd after the twin.
set -uo pipefail
cd "$(dirname "$0")/../.."
EXP=data/nnue/experiments/d40-uho-unc-v1
LOG="$EXP/gate2-retrain.log"
say(){ echo "[gate2-retrain $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for Gate 1 + logvar-verify to finish"
until [ -f "$EXP/GATE1-DONE" ] && [ -f "$EXP/LOGVAR-VERIFY-DONE" ]; do sleep 60; done

if ! grep -q "PASS" "$EXP/gate1.log" 2>/dev/null; then
    say "Gate 1 did NOT pass — EXP-UNC-1 killed; no retrain. Verdict:"
    grep "RESULT" "$EXP/gate1.log" 2>/dev/null | tail -1 | tee -a "$LOG"
    exit 0
fi
if ! grep -q "FIX WORKS" "$EXP/logvar-verify.log" 2>/dev/null; then
    say "logvar fix did NOT save l3_var — abort (a different save mechanism is needed):"
    grep "RESULT" "$EXP/logvar-verify.log" 2>/dev/null | tail -1 | tee -a "$LOG"
    exit 1
fi

say "BOTH gates green — launching the Gate-2 retrain (same config + logvar-save fix, ~2h)"
.venv/bin/python3 tools/nnue/train_run.py --config "$EXP/config.json" >> "$LOG" 2>&1
say "retrain pipeline finished; extracting the uncertainty sidecar"
RAW="$EXP/ckpt/d40-uho-unc-240sb-final-raw.bin"
CENN=data/nnue/nets/d40-uho-unc-240sb.cenn
SIDE=data/nnue/nets/d40-uho-unc-240sb.sidecar
.venv/bin/python3 tools/nnue/extract_unc_head.py "$RAW" "$SIDE" --validate "$CENN" >> "$LOG" 2>&1 \
    && say "sidecar ready: $SIDE (transpose validated against the mean head)" \
    || say "sidecar extraction/validation FAILED — inspect $LOG"
touch "$EXP/GATE2-RETRAIN-DONE"
say "done — next: the 2x2 (c1,c2) Gate-2 probe (design doc). Net=$CENN Sidecar=$SIDE"

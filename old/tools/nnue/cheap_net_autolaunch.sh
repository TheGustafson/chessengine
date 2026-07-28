#!/usr/bin/env bash
# Smallnet-gate cheap-net train, auto-chained AFTER the detach retrain frees the
# GPU, so the GPU never idles (max-GPU directive). Trains a HalfKA-only-768 D40
# net on the same UHO corpus as the big net (perf #3 #8). train_run.py fails fast
# on config/arch errors (bullet compile + feature-parity), and periodic saves
# (D40_SAVE_RATE 30) make a late failure resumable — so no separate smoke.
set -uo pipefail
cd "$(dirname "$0")/../.."
DETACH_PID="${1:-}"
EXP=data/nnue/experiments/d40-uho-halfka-cheap
LOG="$EXP/autolaunch.log"
CENN=data/nnue/nets/d40-uho-halfka-cheap-240sb.cenn
say(){ echo "[cheap-net $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$EXP/ckpt"

say "waiting for the detach retrain (pid ${DETACH_PID:-none}) to finish + free the GPU"
if [ -n "$DETACH_PID" ]; then while kill -0 "$DETACH_PID" 2>/dev/null; do sleep 60; done; fi
# Small settle so the detach's pack/parity fully releases the GPU.
sleep 30
say "GPU free — launching the HalfKA-only cheap-net train (240 SB, ~1.8h)"
.venv/bin/python3 tools/nnue/train_run.py --config "$EXP/config.json" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -f "$CENN" ]; then
    say "cheap-net train FAILED (rc=$rc, cenn present=$([ -f "$CENN" ] && echo yes || echo no)) — inspect $LOG"
    exit 1
fi
say "cheap-net done: $CENN"
say "NEXT (manual/next-fire): (1) micro-bench eval cost ratio HalfKA-only vs full;"
say "  (2) gate-sweep ARCHON_SMALLNET_THRESH x ARCHON_SMALLNET_MAT_GATE equal-time vs off (quiet box);"
say "  (3) ledger the net. Design: docs/research/smallnet-gate-design.md"
touch "$EXP/CHEAP-NET-DONE"

#!/usr/bin/env bash
# Fire the 1536 HalfKA-only train the instant the canon-clean trainer frees the
# GPU. Waits on the SPECIFIC pid (kill -0), never pgrep (self-match trap).
# Sequence: SMOKE validate (3+3 SB) -> if OK, full run (240+120 SB).
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=/tmp/train_1536.log; : > "$LOG"
say() { echo "[auto1536 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

CANON_PID="${1:-4030639}"
say "waiting for canon trainer pid $CANON_PID to exit (GPU busy)..."
while kill -0 "$CANON_PID" 2>/dev/null; do sleep 30; done
say "GPU free (canon pid $CANON_PID exited). Starting 1536 SMOKE validation."

# ---- SMOKE: 3+3 SB end-to-end (arch trains/saves/packs/loads) --------------
if SMOKE=1 bash tools/nnue/train_1536_halfka.sh >>"$LOG" 2>&1; then
  say "SMOKE PASSED. Measuring NPS on the smoke stage-2 net vs incumbent."
  SN=data/nnue/nets/h1536-smoke-s2-farseer-3sb.cenn
  if [ -s "$SN" ]; then
    for lbl in "NEW-1536 $SN archon-sota-1536" "INCUMBENT data/nnue/nets/d40-uho-720sb.cenn archon-sota-native"; do
      set -- $lbl
      ( printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo movetime 4000\n' "$2"; sleep 5 ) \
        | timeout 12 engines/bin/$3 2>&1 | tr '\r' '\n' | grep -aoE "nps [0-9]+" | tail -1 \
        | sed "s/^/[auto1536 NPS] $1 -> /" | tee -a "$LOG"
    done
  fi
else
  say "SMOKE FAILED — NOT launching the full run. Inspect $LOG. Aborting."
  exit 1
fi

# ---- FULL: 240 + 120 SB (~3.6h) --------------------------------------------
say "Launching FULL 1536 two-stage train (300+140 SB, ~5h)."
if bash tools/nnue/train_1536_halfka.sh >>"$LOG" 2>&1; then
  say "FULL TRAIN DONE. stage2 ship candidate = data/nnue/nets/h1536-s2-farseer-140sb.cenn"
else
  say "FULL TRAIN FAILED. Inspect $LOG."
  exit 1
fi

# ---- GATE: ship candidate vs incumbent (fixed-node + equal-time) + SF16 -----
SHIP=data/nnue/nets/h1536-s2-farseer-140sb.cenn
if [ -s "$SHIP" ]; then
  say "GATING $SHIP vs incumbent (fixed-node + equal-time) + SF16."
  LOG=/tmp/gate_1536.log ROUNDS=150 bash tools/nnue/gate_1536.sh "$SHIP" >>"$LOG" 2>&1 || say "gate had errors (see /tmp/gate_1536.log)"
  say "GATE RESULTS:"; grep -aE "\[gate1536 RESULT\]" /tmp/gate_1536.log | tee -a "$LOG"
fi
say "AUTOLAUNCH COMPLETE."

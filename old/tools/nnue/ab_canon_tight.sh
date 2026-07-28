#!/usr/bin/env bash
# ============================================================================
# ab_canon_tight.sh — the confirming A/B for the canon re-diagnosis
# (docs/research/sf18-canon-data-diagnosis-2026-07-21.md §"Minimal fix").
#
# One corpus (build_canon_tight.sh: full 5-criterion filter + real WDL join),
# two cells differing ONLY in the WDL blend, each at its own scale-matched
# SCORE_MULT (train_canon.sh auto-couples: WDL=0.3 -> 1.42, WDL=0 -> 1.94) so
# WDL is isolated from output scale:
#   Cell A: WDL=0.3  (the proven-recipe blend)   -> canon-tight-A-60sb
#   Cell B: WDL=0    (the every-catastrophic-run setting) -> canon-tight-B-60sb
# Each cell: train 60sb -> net_scale_check (hard) -> gate_canon fixed-node.
#
# Readout (from the diagnosis):
#   A lands repro-class (~-150±40): the canon DATA was always fine; the -582
#     was pure recipe. Remaining gap vs streamed = the volume/epoch term.
#   A ~ -400: residual is volume -> stop materializing; use canon as
#     D40_MIX_DATA fold-in on the streamed base.
#   B collapses while A healthy: WDL=0 confirmed as a first-class killer.
#   B ~ A: WDL was innocent; deltas 1+2 (+4) explain everything.
#
#   bash tools/nnue/ab_canon_tight.sh          # waits for corpus, runs A then B
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

CORPUS=data/nnue/canon-v1/build-tight/corpus.shuf.bin
LOG=/tmp/ab_canon_tight.log; : > "$LOG"
say(){ echo "[ab-canon $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
GAMES="${GAMES:-200}"
SB="${SB:-60}"

# Wait for the corpus builder (marker: the shuffled bin exists and the builder
# log printed DONE — size alone races the in-progress shuffle write).
say "waiting for $CORPUS (builder log /tmp/canon_tight_build.log)..."
until grep -q "DONE: .*corpus.shuf.bin" /tmp/canon_tight_build.log 2>/dev/null; do
  if grep -q "FATAL" /tmp/canon_tight_build.log 2>/dev/null; then
    say "builder hit FATAL — aborting the A/B"; exit 1
  fi
  sleep 30
done
[ -s "$CORPUS" ] || { say "builder said DONE but corpus missing?"; exit 1; }
say "corpus ready: $(stat -c%s "$CORPUS") bytes ($(( $(stat -c%s "$CORPUS") / 32 )) records)"

run_cell() { # name wdl
  local name="$1" wdl="$2" id="canon-tight-${1}-${SB}sb"
  local net="data/nnue/nets/$id.cenn"
  say "=== CELL $name: WDL=$wdl SB=$SB -> $id ==="
  if [ ! -s "$net" ]; then
    DATA="$CORPUS" WDL="$wdl" SB="$SB" NET_ID="$id" \
      bash tools/nnue/train_canon.sh >>"$LOG" 2>&1 \
      || { say "CELL $name: train FAILED (see $LOG)"; return 1; }
  else
    say "CELL $name: net exists, skipping train"
  fi
  say "CELL $name: scale check"
  bash tools/nnue/net_scale_check.sh "$net" 2>&1 | tee -a "$LOG" | grep -E "median|PASS|FAIL"
  # gate_canon re-runs the scale check as its own hard step [1a]; FORCE only
  # if you deliberately want a failed-scale cell's games anyway.
  say "CELL $name: fixed-node gate ($GAMES games) vs d40-uho-v1-240sb"
  bash tools/nnue/gate_canon.sh "$net" "" "$GAMES" 2>&1 | tee -a "$LOG" | grep -E "RESULT|ABORT|scale-check" | tail -4
}

run_cell A 0.3
run_cell B 0
say "A/B COMPLETE — summary:"
grep -E "CELL|RESULT|PASS|FAIL" "$LOG" | tail -20
say "next: ledger add-exp for both cells + update value-foundation-state.md"

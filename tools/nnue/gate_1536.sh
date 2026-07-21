#!/usr/bin/env bash
# Gate the 1536 HalfKA-only ship candidate. Direct ARCH comparison vs the
# incumbent d40-uho-720sb (768 + FullThreats, L2=16, SAME UHO data) and vs SF16.
#   1. Fixed-node 25k vs incumbent  -> isolates EVAL QUALITY (search identical,
#      tree size fixed; the only variable is the net).
#   2. Equal-time 10+0.1 vs incumbent -> the SHIP BAR (eval x NPS — the whole bet).
#   3. Equal-time 10+0.1 vs SF16      -> the north star.
# Prints Elo per comparison. Extend rounds for a tighter CI before any ship call.
set -uo pipefail
cd "$(dirname "$0")/../.."

CAND="${1:-data/nnue/nets/h1536-s2-farseer-120sb.cenn}"
ROUNDS="${ROUNDS:-150}"           # x2 games per round
INCUMBENT=data/nnue/nets/d40-uho-720sb.cenn
FC=engines/bin/fastchess
BOOK=engines/books/UHO_Lichess_4852_v1.epd
LOG="${LOG:-/tmp/gate_1536.log}"; : > "$LOG"
say(){ echo "[gate1536 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CAND" ] || { say "candidate net missing: $CAND"; exit 1; }

# HARD scale check (2026-07-21, mandatory before any games): a wrong-SCORE_MULT
# net fits its loss yet outputs at the wrong magnitude vs the search's co-adapted
# cp margins (the canon -582 mechanism). Candidate loads in the 1536 engine, the
# 768+threats incumbent in its own. FORCE=1 overrides for calibration experiments.
if ! bash tools/nnue/net_scale_check.sh "$CAND" "$INCUMBENT" engines/bin/archon-sota-1536 2>&1 | tee -a "$LOG"; then
  if [ "${FORCE:-0}" = 1 ]; then say "scale check FAILED but FORCE=1 — continuing"; else
    say "ABORT: candidate fails the output-scale check — fix the recipe before burning games (FORCE=1 to override)"
    exit 1
  fi
fi

# engineopt <engine> <net>  -> emits the -engine block (net optional; SF has its own)
eblock(){ local e="$1" n="$2" nm="$3"; if [ -n "$n" ]; then
    echo "-engine cmd=engines/bin/$e name=$nm option.EvalFile=$n option.Hash=64 option.Threads=1"
  else echo "-engine cmd=engines/bin/$e name=$nm option.Hash=64 option.Threads=1"; fi; }

run(){ # label eA netA eB netB tc-or-nodes
  local label="$1"
  say "$label  ($((ROUNDS*2)) games)"
  # shellcheck disable=SC2046
  "$FC" $(eblock "$2" "$3" cand) $(eblock "$4" "$5" base) \
    -openings file=$BOOK format=epd order=random \
    -each $6 -rounds "$ROUNDS" -games 2 -repeat -concurrency 6 -recover >>"$LOG" 2>&1
  local elo; elo=$(tr '\r' '\n' < "$LOG" | grep -aE "Elo:" | tail -1)
  echo "[gate1536 RESULT] $label -> ${elo:-<no elo parsed>}" | tee -a "$LOG"
}

say "candidate = $CAND (1536 HalfKA-only, L2=32)"
run "FIXED-NODE-25k vs incumbent (EVAL quality)" archon-sota-1536 "$CAND" archon-sota-native "$INCUMBENT" "nodes=25000"
run "EQUAL-TIME-10+0.1 vs incumbent (SHIP BAR)"  archon-sota-1536 "$CAND" archon-sota-native "$INCUMBENT" "tc=10+0.1"
run "EQUAL-TIME-10+0.1 vs SF16 (NORTH STAR)"     archon-sota-1536 "$CAND" stockfish16 "" "tc=10+0.1"
say "GATE DONE (candidate positive on eval-quality + ship-bar => strong arch win)"

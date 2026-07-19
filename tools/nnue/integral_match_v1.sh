#!/usr/bin/env bash
# ARCHON (current best sota: opt1+opt2 + lazy observer, net d40-uho-720sb) vs
# Integral v6 — equal-time reality check (user-requested 2026-07-19): "how far
# have we caught up" since the Integral v6 −70±22 anchor. 100g 10+0.1,
# Threads=1 Hash=16, policy-balanced-256 book (same as the SF16/Integral
# cross-match anchor triangle → comparable).
#
# The GPU twin train (~1.4 host cores) keeps running; the CPU relabel is the
# only timing pollutant, so we SIGSTOP it for the match and ALWAYS SIGCONT it on
# exit (trap), losing zero relabel progress. conc 1 = cleanest timing + RAM-safe.
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=data/nnue/experiments/integral-match-v1
LOG="$OUT/run.log"
ARCHON=target/release/chessengine
NET=data/nnue/nets/d40-uho-720sb.cenn
mkdir -p "$OUT"
say() { echo "[integral $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

resume_relabel() {
    pkill -CONT -f '[s]tockfish-18-src/src/stockfish' 2>/dev/null || true
    pkill -CONT -f '[r]elabel_depth.py' 2>/dev/null || true
    say "relabel resumed"
}
trap resume_relabel EXIT

say "pausing relabel for clean equal-time timing (GPU train keeps running)"
pkill -STOP -f '[r]elabel_depth.py' 2>/dev/null || true
pkill -STOP -f '[s]tockfish-18-src/src/stockfish' 2>/dev/null || true
sleep 3

say "ARCHON(sota d40-uho-720sb, opt1+opt2) vs integral-v6 — 100g 10+0.1 conc1"
engines/bin/fastchess \
    -engine cmd="$ARCHON" name=archon-sota option.EvalFile="$NET" \
    -engine cmd=engines/bin/integral-v6 name=integral-v6 \
    -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=16 \
    -openings file=data/nnue/openings/policy-balanced-256.epd format=epd order=random \
    -repeat -games 2 -rounds 50 \
    -concurrency 1 -recover \
    -draw movenumber=40 movecount=8 score=15 \
    -resign movecount=4 score=700 -maxmoves 250 \
    -srand 20260719 -pgnout file="$OUT/games.pgn" notation=lan \
    -log file="$OUT/fastchess.log" level=warn > "$OUT/stdout.log" 2>&1

say "match finished; result:"
grep -E "Elo difference|Games:|Score of|LOS|Ptnml" "$OUT/stdout.log" | tail -8 | tee -a "$LOG"
say "INTEGRAL MATCH DONE (archon-sota is player 1; NEGATIVE Elo = ARCHON behind Integral)"
touch "$OUT/DONE"

#!/usr/bin/env bash
# ARCHON (incumbent net) vs full-strength Stockfish 16 — the user-requested
# reality check. 100 games, 10+0.1, both Threads=1 Hash=16, quiet box
# (gated on the T74 hop finishing). SF16 = /usr/games/stockfish, the
# historical anchor engine (net-0 was "~3400 anchored" against its rungs).
set -uo pipefail
cd "$(dirname "$0")/../.."

R=data/nnue/experiments/chain-v2-results.log
OUT=data/nnue/experiments/sf16-match-v1
say() { echo "[sf16match $(date +%H:%M:%S)] $*" | tee -a "$R"; }

say "waiting for the T74 hop to release the box"
until grep -aqE "^\[s2w4 [0-9:]+\] S2W4 DONE" "$R"; do sleep 60; done
sleep 15

mkdir -p "$OUT"
say "starting: ARCHON(s3-2048-farseer) vs SF16 full strength, 100g 10+0.1"
engines/bin/fastchess \
  -engine cmd=engines/bin/probe-l2048 name=archon-incumbent \
    option.EvalFile=data/nnue/nets/stage2-v1-s3-2048-farseer-120sb.cenn \
  -engine cmd=/usr/games/stockfish name=stockfish16 \
  -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=16 \
  -openings file=data/nnue/openings/policy-balanced-256.epd format=epd order=random \
  -repeat -games 2 -rounds 50 \
  -concurrency 3 -recover \
  -draw movenumber=40 movecount=8 score=15 \
  -resign movecount=4 score=700 -maxmoves 250 \
  -srand 20260728 -pgnout file="$OUT/games.pgn" notation=lan \
  -log file="$OUT/fastchess.log" level=warn > "$OUT/stdout.log" 2>&1
grep -E "Elo difference|Games:" "$OUT/stdout.log" | tail -4 | tee -a "$R"
say "SF16 MATCH DONE"

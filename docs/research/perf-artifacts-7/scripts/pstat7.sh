#!/bin/bash
# perf-study-7 interleaved counter rounds: one run of every engine per round,
# pinned core 4, nice 19, per-round ratios computed downstream.
# usage: pstat7.sh <tag> <events> <rounds>
set -u
S=/tmp/claude-1000/-home-nick-projects-chessengine/61b8a705-638b-4a89-a73b-e4fc2a5170a2/scratchpad
cd /home/nick/projects/chessengine
A=./engines/bin/archon-d47v-pgo
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
SF18=./engines/bin/stockfish18
SF16=./engines/bin/stockfish16
TAG=$1; EV=$2; ROUNDS=$3; FENS=${4:-$S/wl50.fens}
for r in $(seq 1 "$ROUNDS"); do
  for eng in A S18 S16; do
    case $eng in
      A)   CMD="$A bench 16 1 300000 $FENS nodes $NET";;
      S18) CMD="$SF18 bench 16 1 300000 $FENS nodes";;
      S16) CMD="$SF16 bench 16 1 300000 $FENS nodes";;
    esac
    nice -n 19 perf stat -x, -e "$EV" -o "$S/runs/ps-$TAG-$eng-r$r.csv" -- \
      taskset -c 4 $CMD > "$S/runs/out-$TAG-$eng-r$r.txt" 2>&1
  done
done
echo "DONE $TAG" >> "$S/runs/pstat7.done"

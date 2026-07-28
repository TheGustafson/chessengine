#!/bin/bash
# Threat-strip verification: interleaved rounds of full vs halfka-only vs study net.
set -u
S=/tmp/claude-1000/-home-nick-projects-chessengine/61b8a705-638b-4a89-a73b-e4fc2a5170a2/scratchpad
cd /home/nick/projects/chessengine
A=./engines/bin/archon-d47v-pgo
for r in $(seq 1 5); do
  for cfg in full halfka study; do
    case $cfg in
      full)   NET=$S/v-a4-full.cenn;;
      halfka) NET=$S/v-a4-halfka.cenn;;
      study)  NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn;;
    esac
    nice -n 19 perf stat -x, -e instructions:u,cycles:u -o "$S/runs/strip-$cfg-r$r.csv" -- \
      taskset -c 4 $A bench 16 1 300000 $S/wl50.fens nodes $NET \
      > "$S/runs/strip-out-$cfg-r$r.txt" 2>&1
  done
done
echo STRIP-DONE

#!/bin/bash
# movetime-3s probe: reproduce last night's instrument under the pinned protocol.
set -u
S=/tmp/claude-1000/-home-nick-projects-chessengine/61b8a705-638b-4a89-a73b-e4fc2a5170a2/scratchpad
cd /home/nick/projects/chessengine
MIDFEN="r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14"
NET=data/nnue/nets/screens/d47v-a4b2-cal400.cenn
probe() { # $1 binary, $2 evalfile-or-empty, $3 outfile
  { echo "uci"; echo "setoption name Hash value 64";
    [ -n "$2" ] && echo "setoption name EvalFile value $2";
    echo "isready";
    echo "position startpos"; echo "go movetime 3000"; sleep 4;
    echo "position fen $MIDFEN"; echo "go movetime 3000"; sleep 4;
    echo "quit"; } | nice -n 19 taskset -c 4 "$1" > "$3" 2>&1
}
for r in 1 2 3; do
  probe ./engines/bin/archon-d47v-pgo "$NET" "$S/runs/mt-A-r$r.txt"
  probe ./engines/bin/stockfish18 "" "$S/runs/mt-S18-r$r.txt"
done
echo MT-DONE

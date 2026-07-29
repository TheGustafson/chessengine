#!/usr/bin/env bash
# perf study 8 — the matched fixed-node counter instrument.
#
#   bash pstat8.sh <rounds> <nodes-per-pos> <fenfile> <outfile> <label:cmd>...
#
# Each <cmd> must contain the literal token @BENCH@, which is replaced by
#   bench 16 1 <nodes> <fenfile> nodes
# so an ARCHON arm can append its net as bench's optional 6th argument
# ("./engines/bin/archon-x @BENCH@ engines/canonical/net.cenn") while an SF
# arm just ends there ("./engines/bin/stockfish18 @BENCH@").
#
# Runs one invocation of EVERY arm per round (interleaved, study #5 §8), each
# pinned to one core with `taskset -c 4` and `nice -n 19`, and records
# `instructions:u` (deterministic, contention-immune — the instrument that
# carries the conclusions) alongside cycles/branches/NPS (bands only).
#
# Node counts are captured per arm per round so any tree divergence between
# arms is visible rather than silently normalised away.
set -euo pipefail
ROUNDS=${1:?rounds}
NODES=${2:?nodes per position}
FENS=${3:?fen file}
OUT=${4:?output file}
shift 4

EVENTS=instructions:u,cycles:u,branches:u,branch-misses:u

: > "$OUT"
{
  echo "# pstat8  rounds=$ROUNDS nodes=$NODES fens=$FENS"
  echo "# date=$(date -Is)  uptime=$(uptime)"
  echo "# arms:"
  for a in "$@"; do echo "#   $a"; done
} >> "$OUT"

for r in $(seq 1 "$ROUNDS"); do
  for arm in "$@"; do
    label=${arm%%:*}
    cmd=${arm#*:}
    tmp=$(mktemp)
    full=${cmd//@BENCH@/bench 16 1 $NODES $FENS nodes}
    # shellcheck disable=SC2086
    nice -n 19 perf stat -x, -e "$EVENTS" -o "$tmp.perf" -- \
      taskset -c 4 $full > /dev/null 2> "$tmp.eng" || true
    nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
    nps=$(grep -oP 'Nodes/second\s*:\s*\K[0-9]+' "$tmp.eng" | tail -1)
    ins=$(awk -F, '$3=="instructions:u"{print $1}' "$tmp.perf")
    cyc=$(awk -F, '$3=="cycles:u"{print $1}' "$tmp.perf")
    brs=$(awk -F, '$3=="branches:u"{print $1}' "$tmp.perf")
    bmi=$(awk -F, '$3=="branch-misses:u"{print $1}' "$tmp.perf")
    echo "round=$r arm=$label nodes=$nodes nps=$nps instructions=$ins cycles=$cyc branches=$brs branch_misses=$bmi" >> "$OUT"
    rm -f "$tmp" "$tmp.perf" "$tmp.eng"
  done
done
echo "wrote $OUT"

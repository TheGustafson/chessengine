#!/usr/bin/env bash
# perf study 9 — the SMP scaling curve, four studies overdue.
#
#   bash smp9.sh <rounds> <depth> <outfile> <label:cmd>...
#
# Each <cmd> contains @T@ (thread count) and @BENCH@ is implicit: the command
# is expected to end in the bench invocation with @T@ where threads go.
#
# Two statistics per (arm, threads):
#   * NPS scaling     — aggregate throughput; the raw parallel-efficiency number
#   * time-to-depth   — the EFFECTIVE speedup (a shared TT makes threads search
#                       overlapping trees, so nodes grow with T and NPS scaling
#                       flatters the result; wall time to a FIXED depth does not)
# Rounds are interleaved across arms and thread counts so a drift in box state
# hits every cell equally.
set -euo pipefail
ROUNDS=${1:?rounds}
DEPTH=${2:?depth}
OUT=${3:?outfile}
shift 3
THREADS="1 2 4 8 16"

: > "$OUT"
{
  echo "# smp9 rounds=$ROUNDS depth=$DEPTH threads='$THREADS'"
  echo "# date=$(date -Is) uptime=$(uptime)"
} >> "$OUT"

for r in $(seq 1 "$ROUNDS"); do
  for t in $THREADS; do
    for arm in "$@"; do
      label=${arm%%:*}
      cmd=${arm#*:}
      full=${cmd//@T@/$t}
      full=${full//@D@/$DEPTH}
      s=$(date +%s.%N)
      # shellcheck disable=SC2086
      log=$(nice -n 0 $full 2>&1 | tail -5)
      e=$(date +%s.%N)
      nodes=$(grep -oP 'Nodes searched\s*:\s*\K[0-9]+' <<<"$log" | tail -1)
      nps=$(grep -oP 'Nodes/second\s*:\s*\K[0-9]+' <<<"$log" | tail -1)
      ms=$(grep -oP 'Total time \(ms\)\s*:\s*\K[0-9]+' <<<"$log" | tail -1)
      wall=$(python3 -c "print(f'{$e-$s:.3f}')")
      echo "round=$r arm=$label threads=$t nodes=$nodes nps=$nps ms=$ms wall=$wall" >> "$OUT"
    done
  done
done
echo "wrote $OUT"

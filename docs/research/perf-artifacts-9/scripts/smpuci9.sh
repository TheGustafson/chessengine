#!/usr/bin/env bash
# perf study 9 — the SMP scaling curve, driven through UCI.
#
#   bash smpuci9.sh <rounds> <depth> <outfile> <label:binary[:netarg]>...
#
# WHY NOT `bench`: ARCHON's `bench` PARSES its threads argument and never uses
# it (src/bench.rs — `params.threads` has no reader), so a bench-driven curve
# silently measures Threads=1 at every point. That is exactly the failure that
# made this study's first SMP panel worthless; only `setoption name Threads`
# reaches the pool (src/uci.rs `Threads` arm -> ThreadPool::resize).
#
# Statistic: the LAST `info depth <D>` line's nps/nodes for a `go depth D` from
# a fixed position, which is the aggregate across the pool.
set -euo pipefail
ROUNDS=${1:?rounds}
DEPTH=${2:?depth}
OUT=${3:?outfile}
shift 3
THREADS="1 2 4 8 16"
FEN1="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN2="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

: > "$OUT"
{
  echo "# smpuci9 rounds=$ROUNDS depth=$DEPTH threads='$THREADS'"
  echo "# date=$(date -Is) uptime=$(uptime)"
} >> "$OUT"

for r in $(seq 1 "$ROUNDS"); do
  for t in $THREADS; do
    for arm in "$@"; do
      label=${arm%%:*}
      rest=${arm#*:}
      bin=${rest%%:*}
      net=${rest#*:}
      [ "$net" = "$bin" ] && net=""
      for fen in "$FEN1" "$FEN2"; do
        {
          echo uci
          [ -n "$net" ] && echo "setoption name EvalFile value $net"
          echo "setoption name Threads value $t"
          echo "setoption name Hash value 1024"
          echo isready
          echo "position fen $fen"
          echo "go depth $DEPTH"
          sleep 90
          echo quit
        } > /tmp/smpin-$$.txt
        s=$(date +%s.%N)
        log=$("$bin" < /tmp/smpin-$$.txt 2>&1 | grep -E "^info depth $DEPTH |^bestmove" | tail -3)
        e=$(date +%s.%N)
        nodes=$(grep -oP ' nodes \K[0-9]+' <<<"$log" | tail -1)
        nps=$(grep -oP ' nps \K[0-9]+' <<<"$log" | tail -1)
        ms=$(grep -oP ' time \K[0-9]+' <<<"$log" | tail -1)
        wall=$(python3 -c "print(f'{$e-$s:.3f}')")
        pos=$([ "$fen" = "$FEN1" ] && echo start || echo kiwi)
        echo "round=$r arm=$label pos=$pos threads=$t nodes=${nodes:-0} nps=${nps:-0} ms=${ms:-0} wall=$wall" >> "$OUT"
        rm -f /tmp/smpin-$$.txt
      done
    done
  done
done
echo "wrote $OUT"

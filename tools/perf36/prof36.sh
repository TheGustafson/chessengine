#!/usr/bin/env bash
# perf study 36 — the post-i8 profile census.
#
#   bash tools/perf36/prof36.sh <label> [events...]
#
# Records one perf.data per event on the study frame and prints the symbol
# table for each.  Symbol SHARES are within-run attributions and are therefore
# immune to the between-run drift that made `cycles:u` DELTAS unusable on this
# box (studies 33/34/35).  Absolute per-node numbers from a loaded box are not
# quotable; shares are.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}; shift
EVENTS=("$@")
[ ${#EVENTS[@]} -eq 0 ] && EVENTS=(cycles:u instructions:u L1-dcache-loads:u ex_ret_brn_misp:u ex_no_retire.load_not_complete:u)
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
CPU=${P36_CPU:-2}
BUDGET=${P36_BUDGET:-100000}
DIR=docs/research/perf-artifacts-36
mkdir -p "$DIR"

for EVENT in "${EVENTS[@]}"; do
  case "$EVENT" in
    instructions:u|L1-dcache-loads:u) PERIOD=200003 ;;
    ex_ret_brn_misp:u)                PERIOD=2003   ;;
    *)                                PERIOD=50021  ;;
  esac
  DATA="$DIR/perf-$LABEL-${EVENT%%:*}.data"
  nice -n 19 perf record -q -e "$EVENT" -c "$PERIOD" -o "$DATA" -- \
    taskset -c "$CPU" "./bin34/$LABEL" bench 16 1 "$BUDGET" "$FENS" nodes "$NET" \
    >/dev/null 2>"$DIR/bench-$LABEL.txt"
  nodes=$(sed -n 's/^Nodes searched  : //p' "$DIR/bench-$LABEL.txt" | tail -1)
  tot=$(perf report -q -i "$DATA" --no-children --stdio -F overhead,period 2>/dev/null \
        | awk '{s+=$2} END{print s+0}')
  echo "### $EVENT  period=$PERIOD  nodes=$nodes  total/node=$(awk -v t="$tot" -v n="$nodes" 'BEGIN{printf "%.1f", t/n}')"
  perf report -q -i "$DATA" --no-children --percent-limit 0.30 --stdio 2>/dev/null \
    | sed 's/^ *//' | awk -v n="$nodes" -v p="$PERIOD" \
        '{pct=$1; sub("%","",pct); sym=""; for(i=4;i<=NF;i++) sym=sym" "$i;
          printf "%8s  %s\n", $1, sym}' | head -28
  echo
done

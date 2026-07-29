#!/usr/bin/env bash
# perf study 8 — INSTRUCTION-attributed flat profile (not cycles).
#
#   bash flat8.sh <label> <period> <outdir> <cmd...>
#
# `perf record -e instructions:u -c <period>` makes every sample worth exactly
# <period> retired user instructions, so the per-symbol sample share IS the
# per-symbol instruction share.  That is the contention-immune analogue of the
# cycles flat profiles studies 3/7 used, and it is what a study whose target is
# an instruction budget needs.
#
# `-e branches:u` variant via EV=branches:u for the branch decomposition.
set -euo pipefail
LABEL=${1:?label}
PERIOD=${2:?sample period}
OUTDIR=${3:?outdir}
shift 3
EV=${EV:-instructions:u}
mkdir -p "$OUTDIR"
data="$OUTDIR/perf-$LABEL-${EV%%:*}.data"
nice -n 19 perf record -e "$EV" -c "$PERIOD" -o "$data" -- \
  taskset -c 4 "$@" > /dev/null 2>&1 || true
perf report -i "$data" --stdio --no-children --percent-limit 0.05 -s sym 2>/dev/null \
  | sed -n '1,80p' > "$OUTDIR/flat-$LABEL-${EV%%:*}.txt"
echo "wrote $OUTDIR/flat-$LABEL-${EV%%:*}.txt"

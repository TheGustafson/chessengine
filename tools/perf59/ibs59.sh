#!/usr/bin/env bash
# perf study 59 — IBS op sampling on one labelled binary, into a perf-script text
# file the study-52 census tools read.
#
#   bash tools/perf59/ibs59.sh <label> <reps> <out.txt>
#
# WHY IBS AND NOT `perf record -e instructions:u`: verified again in this study
# — a single stack load inside `search::<NonPv>` collects 13.32% of that
# symbol's `instructions:u` samples, and a single source line collects 13.15% of
# the ENGINE's. Neither can be a retired-instruction count. Study 36 §2.3 killed
# per-mnemonic attribution of `instructions:u`; this confirms it at the
# per-instruction and per-LINE level too. IBS tags a micro-op at ISSUE and
# records that op's own rip, so there is no skid by construction.
#
# PER-SYMBOL `instructions:u` shares remain usable (study 58 §8) — it is only
# WITHIN a symbol that the counter cannot be believed.
set -euo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}
REPS=${2:-4}
OUT=${3:?out.txt}
CPU=${P59_CPU:-1}
NET=engines/canonical/net.cenn
FENS=tools/nnue/calib/wl50.fens
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
: > "$OUT"
for i in $(seq 1 "$REPS"); do
  taskset -c "$CPU" perf record -q -e ibs_op/cnt_ctl=1,swfilt=1/ -c 8000 -d -W \
    -o "$tmp/ibs-$i.data" -- "./bin59/$LABEL" bench 16 1 100000 "$FENS" nodes "$NET" \
    >/dev/null 2>&1
  perf script -i "$tmp/ibs-$i.data" -F ip,sym,symoff,weight,data_src >> "$OUT" 2>/dev/null
done
echo "[ibs59] $LABEL x$REPS -> $OUT  ($(wc -l < "$OUT") lines)"

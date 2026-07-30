#!/usr/bin/env bash
# perf study 39 — the DOSE-MATCHED panel, which fixes a confound in my own
# first dose series.
#
# With a FIXED ring, raising the dose shortens the probe's own reuse distance:
# at `RING=16, DOSE=1` a slot comes round again after 16 evaluated nodes' worth
# of copies, but at `DOSE=4` after only 4. Higher doses therefore look cheaper
# per line for a reason that has nothing to do with the engine — and the first
# panel duly showed misses per dose falling from 90.5 to 80.8 across doses 1
# to 4.
#
# Scaling RING with DOSE holds the reuse distance at 16 evaluations for every
# arm, so the only thing the dose changes is how many lines are added. That is
# the curve study 29 §4.1's capacity-cliff reconciliation makes a prediction
# about.
#
# No rebuild: both knobs are runtime, so this is the SAME BYTES as the first
# panel and the two are directly comparable.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:-20}
P39_PANEL=docs/research/perf-artifacts-39/panel-dosematched.txt \
P39_CPU=${P39_CPU:-2} \
bash tools/perf39/panel39.sh "$ROUNDS" \
  d0=addcopy:P39_DOSE=0,P39_RING=16 \
  d0null=addcopy-null:P39_DOSE=0,P39_RING=16 \
  d1=addcopy:P39_DOSE=1,P39_RING=16 \
  d2=addcopy:P39_DOSE=2,P39_RING=32 \
  d4=addcopy:P39_DOSE=4,P39_RING=64 \
  d8=addcopy:P39_DOSE=8,P39_RING=128

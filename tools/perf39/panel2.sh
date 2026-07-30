#!/usr/bin/env bash
# perf study 39 — panel 2: the DOSE-MATCHED series and the SOURCE arm, together.
#
# Two corrections to panel 1, both of them to MY OWN instrument:
#
# 1. DOSE-MATCHED RINGS. With a fixed ring, raising the dose shortens the
#    probe's own reuse distance: at `RING=16, DOSE=1` a slot comes round again
#    after 16 evaluations' worth of copies, at `DOSE=4` after only 4. High
#    doses therefore look cheaper per line for a reason that has nothing to do
#    with the engine, and panel 1 shows it — misses per dose fall from 90.5 to
#    80.8 across doses 1..4. Scaling RING with DOSE holds the reuse distance at
#    16 evaluations for every arm.
#
# 2. THE SOURCE ARM. My ring copy generates 1.30 L1 miss events per 64-byte
#    line where study 31's copy generated 0.477, and `tools/perf39/calib39.sh`
#    puts a FULLY COLD line at ~1.2 events. So my copy's lines are ~100% cold
#    and study 31's were ~40% — which is also the real accumulator stack's
#    figure, and is exactly the sort of difference that could produce a 5.2x
#    disagreement out of two honest measurements. `P39_SRC=live` makes the
#    copy's READ half touch the accumulator `activate` is about to consume, so
#    the read prefetches the engine instead of adding traffic. If that
#    reproduces study 31's miss count, the reconstruction is settled.
#
# 3. THE PLACEMENT ARM. The recommendation turns on whether stall-per-line is a
#    property of the TRAFFIC or of where in the node it is issued: fusion
#    removes accumulator traffic from the FOLD, the probe adds it in
#    `activate`. The fold fires 0.8538/node against `activate`'s 0.4311, so
#    `f1` (136.6 lines/node) sits against `d2` (137.95) and `f2` (273.2)
#    against `d4` (275.9) — same volume, same reuse distance in nodes, two
#    different places in the node. If they price the same, the number
#    transfers to fusion; if not, the whole framing is wrong and that is the
#    finding.
#
# Same bytes as panel 1 for the ring arms (runtime knobs), so the two panels
# are directly comparable.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:-24}
P39_PANEL=docs/research/perf-artifacts-39/panel-dosematched.txt \
P39_CPU=${P39_CPU:-2} \
bash tools/perf39/panel39.sh "$ROUNDS" \
  d0=addcopy:P39_DOSE=0,P39_RING=16 \
  d0null=addcopy-null:P39_DOSE=0,P39_RING=16 \
  d1=addcopy:P39_DOSE=1,P39_RING=16 \
  d2=addcopy:P39_DOSE=2,P39_RING=32 \
  d4=addcopy:P39_DOSE=4,P39_RING=64 \
  d1L=addcopy:P39_DOSE=1,P39_RING=16,P39_SRC=live \
  d2L=addcopy:P39_DOSE=2,P39_RING=32,P39_SRC=live \
  d4L=addcopy:P39_DOSE=4,P39_RING=64,P39_SRC=live \
  f1=addcopy:P39_DOSE=1,P39_RING=32,P39_SITE=fold \
  f2=addcopy:P39_DOSE=2,P39_RING=64,P39_SITE=fold

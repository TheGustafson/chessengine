#!/usr/bin/env bash
# perf study 39 — panel 3: split the clean subtractive arm into its READ and
# WRITE halves, and test the one explanation I have for the residual
# additive/subtractive asymmetry.
#
# §4.3 leaves a 1.93x factor unexplained: an ADDED miss costs 2.50 stall cycles
# and a REMOVED one saves 1.30. One candidate is an artefact of the subtractive
# probe itself — `dstpin` funnels EVERY fold's stores into ONE 2560 B buffer,
# so its saving may be partly eaten by store-port and store-buffer contention
# the real engine would never have. If so the subtractive number is a LOWER
# bound and fusion is worth more than §6 says.
#
# `accpin` pins the fold `src` READS and leaves the stores in the real per-ply
# stack, so it has no funnelling at all. All three pin arms compute the same
# all-zero evaluation, search the same tree and retire the same instructions
# (5473.91 / 5475.74 / 5474.96 i/n), so all three differences are clean:
#
#   accpin − actpin0  : 74.17 READ lines/node relocated, no store funnelling
#   dstpin − accpin   : the WRITE half, where the funnelling would be
#   dstpin − actpin0  : both, i.e. §4.1's headline arm
set -euo pipefail
cd "$(dirname "$0")/../.."
ROUNDS=${1:-24}
P39_PANEL=docs/research/perf-artifacts-39/panel-readwrite.txt \
P39_CPU=${P39_CPU:-2} \
bash tools/perf39/panel39.sh "$ROUNDS" \
  actpin0=actpin:P39_PINFILL=zero \
  accpin=accpin:P39_UNUSED=1 \
  dstpin=dstpin:P39_UNUSED=1 \
  d0=addcopy:P39_DOSE=0,P39_RING=16

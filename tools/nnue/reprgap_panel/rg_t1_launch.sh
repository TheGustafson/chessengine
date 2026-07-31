#!/usr/bin/env bash
# Tier-1 panel: invisible-state atoms (repr-gap study, 2026-07-30).
# DELIBERATELY UNARMED — long GPU runs start on the owner's say-so, and the
# recode-t1 self-launcher + refine/updloc/conj panels have queue priority.
#
# ============================ PRE-REGISTERED =============================
# (fixed 2026-07-30 BEFORE any arm has run; do not modify after readout)
#
# Statistic: m(arm) = mean STATE-heldout MSE over sb8..12.
# Noise: re-derive the per-readout sd from the identity arm's own sb8..12
#   dispersion at readout (the 16.3e-6 mix-heldout pin does NOT transport to
#   this corpus); the 25e-6 gate below is inherited from the panel family
#   and stands unless the re-derived 3-sigma exceeds it, in which case the
#   re-derived value replaces it and is reported as such.
# RG0 validity: every arm prints the SAME sb0 legacy heldout AND the same
#   sb0 state heldout (zero-init rows x silent/zero atoms = bit-parity;
#   CPU-verified max|d| = 0.0 by reprgap_smoke.py). A different sb0 VOIDS
#   the arm (seam bug, not thesis).
# RG1 value:    m(state12) <= m(identity) - 25e-6.
# RG2 content:  m(state12) <= m(state12_null) - 25e-6. A RG1 pass that RG2
#   fails is capacity/optimizer effect, not information — reported as such.
# RG3 legacy guard: state12's sb12 LEGACY heldout within +25e-6 of
#   identity's (the atoms must not damage state-free positions).
# Slope guard (the only allowed extension): an arm within 2x its gate whose
#   sb8-12 slope is steeper than -5e-6/sb gets ONE 24-sb rerun + identity.
# A RG1+RG2 double fail is a measured NULL on a well-posed question and is
# reported as a success of the method.
# =========================================================================
#
# Corpus prerequisites (built by this script if missing; ~50 min CPU for the
# 40M build at 6 procs — run it BEFORE the GPU frees, it needs no GPU):
#   data/nnue/reprgap/state-uho-40M.bin       (shuffled filtered TSV rows)
#   data/nnue/reprgap/state-heldout-1M.bin    (disjoint later slice)
set -euo pipefail
cd "$(dirname "$0")/../../.."
WT=.claude/worktrees/reprgap-seam
OUT=data/nnue/reprgap
mkdir -p "$OUT" data/nnue/experiments/reprgap-t1
if [ ! -s "$OUT/state-uho-40M.bin" ]; then
  echo "[rg-t1] building state corpora (CPU)"
  shuf data/nnue/binpacks/uho_frs_filtered.tsv > "$OUT/uho_frs_filtered.shuf.tsv"
  head -n 40000000 "$OUT/uho_frs_filtered.shuf.tsv" > "$OUT/train40M.tsv"
  tail -n 1000000 "$OUT/uho_frs_filtered.shuf.tsv" > "$OUT/heldout1M.tsv"
  .venv/bin/python tools/nnue/reprgap/build_state_corpus.py \
    "$OUT/train40M.tsv" "$OUT/state-uho-40M.bin" 6
  .venv/bin/python tools/nnue/reprgap/build_state_corpus.py \
    "$OUT/heldout1M.tsv" "$OUT/state-heldout-1M.bin" 6
  rm -f "$OUT/uho_frs_filtered.shuf.tsv" "$OUT/train40M.tsv" "$OUT/heldout1M.tsv"
fi
for arm in identity state12 state12_null; do
  echo "[rg-t1] arm $arm"
  (cd "$WT" && python jax/tools/reprgap_tier1.py \
     ../../../tools/nnue/reprgap_panel/rg_t1_${arm}.json) \
    2>&1 | tee "data/nnue/experiments/reprgap-t1/${arm}.log"
done
echo "[rg-t1] panel complete — apply the pre-registered rules above"

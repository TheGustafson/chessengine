#!/usr/bin/env bash
# updloc Tier-1 "TRAINED PROPERTIES" panel launcher (learned-representation
# concept study). DELIBERATELY NOT SELF-ARMING: the GPU is committed to the
# D49 trainer (~14 h at study time), and long GPU runs start only on the
# owner's say-so. Arm it with:
#
#   systemd-run --user --unit=updloc-t1 \
#     -p MemoryMax=8G -p MemorySwapMax=0 \
#     bash <this file>
#
# It then WAITS for the GPU to be free of d49x AND of recode-t1-refine (the
# richer-representation panel, which has queue priority if armed) AND idle
# for 10 consecutive minutes, then runs the four arms SEQUENTIALLY
# (~55 min each, ~3.7 h total):
#
#   refine_pocket       the LEARNED king-context bit {g1,h1,h2} own-frame
#                       (min-flip >=30%-data-mass bipartition of the measured
#                       in-tree KTRANS matrix), constant-row (tuple x bit)
#                       code2d refinement of the threat channel
#   refine_pocket_null  content-free control: SAME code, hash-bit region
#   ftsparse_a / _b     element-wise post-CReLU L1 at lambda 0.003 / 0.01
#                       with a 2-sb D47M ramp (the 2026-07-26 lambda=0.001
#                       arm never moved density: dose ladder, not verdict)
#
# PRE-RUN CHECKLIST (the launcher enforces none of these — read the study,
# docs/research/learned-representation-study-2026-07-30.md §6):
#   * regenerate region-census-mix8M.npz (region_census.py picks up the new
#     `pocket` bit from REGION_BIT_NAMES) and re-run region_probe.py
#   * build refine-pocket-r60720.npy: refine_code.py refine pocket
#   * every arm's sb0 heldout must print 0.006350 (bit-parity with the
#     incumbent — the ftsparse ramp starts at 0, so this holds for ALL four
#     arms); a different sb0 voids that arm
#
# PRE-REGISTERED DECISION RULES (2026-07-30, before any arm has run — do not
# modify after readout; statistic m(arm) = mean heldout over sb8-12; gate
# 25e-6 = 3x the identity-pair tail-mean noise, as the refine panel):
#   U1 value:        m(refine_pocket) <= m(identity) - 25e-6
#   U2 content:      m(refine_pocket) <= m(refine_pocket_null) - 25e-6
#   U3 deployability: m(identity) - m(refine_pocket) >= 32e-6
#                    (= the bit's measured engine price, 30.1 cyc/node
#                    = 1.15 Elo, at the UNFAVORABLE end of the 12-28e-6/Elo
#                    exchange; the favorable end is 14e-6, below the
#                    detection floor — i.e. any detectable U1 gain already
#                    clears the favorable-end price)
#   U4 ftsparse:     (a) density gate: sb12 ft_density must drop >= 0.01
#                    absolute vs identity, else the verdict is "dose too
#                    low, ladder up" — never "sparsity dead";
#                    (b) exchange: PASS iff
#                    120*log2(1/(1 - 0.0615*(1 - d'/d))) > dHeldout/12e-6
#                    (head share 6.15% cycles, density-linear model; a pass
#                    still needs a Tier-2 NPS A/B — density is measurable in
#                    the deployed engine)
#   Slope guard (only allowed extension): an arm within 2x its gate with an
#   sb8-12 slope steeper than -5e-6/sb gets ONE 24-sb rerun + identity.
set -euo pipefail
cd "$(dirname "$0")/../../.."
WT=$(pwd)

echo "[updloc-t1] waiting for GPU (d49x + recode-t1-refine gone, 10 min idle)"
idle=0
while [ "$idle" -lt 10 ]; do
  sleep 60
  systemctl --user is-active --quiet d49x.service && { idle=0; continue; }
  systemctl --user is-active --quiet recode-t1-refine.service && { idle=0; continue; }
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${util:-100}" -lt 10 ] && [ "${mem:-99999}" -lt 1500 ]; then
    idle=$((idle + 1))
  else
    idle=0
  fi
done

for arm in refine_pocket refine_pocket_null ftsparse_a ftsparse_b; do
  echo "[updloc-t1] arm $arm"
  ARCHON_JAX_FT_IMPL=pallas \
  uv run --no-project --with numpy==2.5.1 --with "jax[cuda12]==0.11.0" \
    --with optax==0.2.8 python "$WT/jax/tools/recode_tier1.py" \
    "$WT/tools/nnue/updloc_panel/updloc_t1_${arm}.json"
done
echo "[updloc-t1] panel complete — read the decision rules BEFORE the sb12s"

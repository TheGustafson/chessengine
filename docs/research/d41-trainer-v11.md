# D41 Phase-1 allocation-head trainer — v1.1 anti-overfit grid (2026-07-17)

Branch `d41-trainer-v11`. Deliverable: `tools/nnue/bullet/d41_alloc11.rs`
(v1.1 trainer), the `obs-tool export-train --all-tiers` flag
(`src/bin/obs_tool.rs`), and this note. Builds on v0
(`d41-trainer-v0-smoke.md`) and v1 (state doc 16:30 / 17:20). Play matches
deferred — the TC ship gates own the box; the acceptance bar here is a packed
artifact + a policy-eval sanity probe.

## The v0/v1 diagnosis this answers (state doc 17:20)

v1's margin head generalised values (held-out rank corr **0.604**, top-1
**0.130** on big-v1-sf18) but a 60M-param trunk trained FROM SCRATCH on ~10k
positions with pure margin-MSE overfit ~180× (train 9.5e-5 vs held-out 0.0176);
in-game it was null at any knob (exp-0086). Root cause: **data starvation +
wrong loss + unfrozen trunk**. v1.1 is the three-part anti-overfit stack, plus
early stopping + head weight decay.

## v1.1 stack (all env-gated; unset = v0 behaviour)

1. **Frozen trunk** (design/19 §4.2). `D41_TRUNK_RAW` loads the
   `l0_pstw`/`l0_pstb`/`l0_threatw` tensors from the `d40-uho-720sb` raw
   checkpoint and freezes them (bullet `train_frozen`: frozen tensors stay in
   the forward graph, autograd never materialises their gradients, the
   optimiser never updates them). Only the rank-8 bilinear head learns — ~63.9M
   of ~64M params are removed from optimisation. The packed trunk prefix is
   byte-identical to v0's (81,237,504 B), so the play-time POLICY_TRUNK **is**
   `data/nnue/nets/d40-uho-720sb.cenn` (no graft).
2. **Pairwise ranking loss (RankNet)**, `D41_LOSS=pairwise`. Within each
   `(key, source, tier)` search, up to K=24 arm pairs; loss = 2-logit
   softmax-cross-entropy of `[s_i, s_j]` against the one-hot better-arm label
   (identically the RankNet logistic loss on `s_i - s_j`), weighted by
   `clamp(|margin_i-margin_j|/100, 0.1, 3)`. `D41_LOSS=margin` keeps v0's
   squashed-margin MSE, now tier-weighted.
3. **Dual-teacher multi-tier data**, `D41_DATA=dir1,dir2`. `obs-tool
   export-train --all-tiers` emits every Complete tier (4k/16k/64k node budgets)
   not just the deepest — 3× the rows; rows/pairs weighted 64k=1.0, 16k=0.7,
   4k=0.4. Ranking groups are `(key, source, tier)`; split is by `key`
   (leakage-safe across tiers and teachers).
4. **Early stopping**: every superbatch, score a deterministic capped tier-64000
   held-out subset (same split seed across configs) and save best-by-rank-corr
   separately from final. `D41_EVAL_TIER` restricts the eval to one tier so an
   all-tiers run stays comparable to a deepest-only run.
5. **Head weight decay**: AdamW `decay=D41_WD` (default 0.01). With the trunk
   frozen this binds only the head tensors by construction.

Architecture, feature framing, mover-relative encoding, and the packed
save-format are unchanged from v0 (the proven gen-0 shape).

## Grid (frozen trunk = d40-uho-720sb; rank-8; 12 sb × 250 × 4096; wd 0.01; lr 1e-3→1e-4 cosine)

Best-subset = the deterministic ~89-group tier-64000 held-out subset used for
early stopping (identical set across same-tier configs; top-1 on 89 groups is
noisy). Final-full = full held-out eval on the FINAL (sb12) weights.

| run | freeze | loss | data | best-subset rankcorr / top-1 (sb) | final-full rankcorr / top-1 (groups) |
|-----|--------|------|------|-----------------------------------|--------------------------------------|
| **v1 baseline** | **no** | margin | sf18 deepest | — | **0.604 / 0.130** |
| a-sf18 | yes | margin | sf18 deepest | 0.714 / 0.193 (sb1) | 0.694 / 0.173 (1941) |
| b-sf18 | yes | pairwise | sf18 deepest | 0.677 / 0.216 (sb1) | 0.637 / 0.208 (1941) |
| **(a)** | yes | margin | both deepest | 0.690 / 0.258 (sb2) | 0.685 / 0.175 (3882) |
| **(b)** | yes | pairwise | both deepest | 0.686 / 0.225 (sb1) | 0.666 / 0.205 (3882) |
| **(c) WINNER** | yes | pairwise | both all-tiers | **0.706 / 0.258 (sb2)** | **0.698 / 0.225** (1777†) |

† (c) final eval capped at 60,000 tier-64000 arms (1,777 groups); its
best-subset 0.706/0.258 is on the same 89-group set as (a)/(b). Margin-mode
final held-out MSE: a-sf18 0.0114, (a) 0.0115 — vs the v0/v1 ~0.0176.

## The two decisive comparisons

- **Frozen vs unfrozen** (margin loss, identical big-v1-sf18 data): **a-sf18
  0.694 / 0.173 vs v1 0.604 / 0.130 → +0.090 rank corr, +0.043 top-1 from
  freezing alone.** This is the headline: the anti-overfit stack's central lever
  works. The frozen head is tiny and cannot memorise, so train and held-out stay
  close and the peak arrives at sb1–2.
- **Pairwise vs margin** (frozen, same data): pairwise consistently trades a
  little Pearson calibration for better best-move discrimination —
  both-teacher 0.666/**0.205** vs 0.685/0.175; sf18 0.637/**0.208** vs
  0.694/0.173. Margin literally regresses the margin (higher rank corr);
  pairwise optimises order (higher top-1). For allocation, top-1 / ordering at
  the top is what matters.
- **All-tiers (3× data)**: (c) 0.706/0.258 vs (b) 0.686/0.225 on the identical
  eval set → +0.020 rank corr, +0.033 top-1. The shallow tiers add signal — the
  same data-scaling shape that decided B4 for value nets, now at the head level.

Both (b) and (c) clearly beat v1's 0.604/0.130. **Winner: (c) frozen + pairwise
+ all-tiers** (highest on both metrics).

## Packed artifact + policy-eval sanity probe (acceptance bar)

Winner best checkpoint packed with `policy_pack.py --base 81237504 bilinear
--rank 8`, then the CPOL layout byte set to **0** (the gen-0 from/to transpose
recovery — empirically required, matching v0's exp-0086 finding).

- POLICY_TRUNK: `data/nnue/nets/d40-uho-720sb.cenn` (the frozen trunk itself)
- POLICY_NET: `data/nnue/experiments/d41-v11/d41-v11-c-pairwise-alltiers.cpol`
  (3,149,856 B, CPOL v2 bilinear, layout 0)

Free-queen probe (exp-0086), `probe-policy-d41 policy-eval`,
`rnb1kbnr/pppp1ppp/8/4q3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 4|f3e5 b1c3 h2h3 a2a3`:

```
layout 0 (correct): f3e5=+4.263  b1c3=-2.846  h2h3=-1.897  a2a3=-0.080
layout 1 (wrong):   f3e5=-2.148  b1c3=+5.068  h2h3=+3.740  a2a3=+4.037
```

Layout 0 ranks **Nxe5 (capturing the free queen) clearly top** by a wide
margin; layout 1 (policy_pack.py's bilinear default) is the transpose and ranks
it last. Acceptance bar PASSED.

## Read: did the anti-overfit stack move held-out generalization?

**Yes, decisively.** The clean same-data frozen-vs-unfrozen comparison
(a-sf18 vs v1) is +0.090 rank corr / +0.043 top-1 from freezing alone, and the
train/held-out MSE gap collapses from ~180× to near-parity — the overfit
diagnosis was correct and the frozen strong trunk is the fix. All-tiers data
adds a further +0.02/+0.03, and pairwise buys +0.03 top-1 (the strategically
important cell) at a small rank-corr cost. The best config (c) reaches held-out
rank corr **0.706** / top-1 **0.258**, up from v1's 0.604 / 0.130.

Caveat unchanged from 17:20: better held-out RANKING is necessary but not
sufficient for an in-game positive — the v1 margin head had 0.60 rank corr and
was still null as a quiet-move ordering prior at 0.3M rows. The play test stays
deferred until the box frees (TC gates); the packed artifact + probe prove the
whole pack/load/rank path is correct and ready.

## Reproduce

```
# corpora (side target dir; --all-tiers is the new flag):
cargo build --release --features datagen --bin obs-tool --target-dir <side>
<side>/release/obs-tool export-train data/obs/train/big-v1-sf18-alltiers \
    data/obs/big-v1-sf18.ceobs --all-tiers        # + reckless twin

# trainer (registered as a bullet [[example]]):
cd .archon-state/toolchains/bullet
CUDA_PATH=/usr cargo build --release --features cuda --example d41_alloc11
REPO=/root/projects/learnrust/chessengine
env CUDA_PATH=/usr \
  D41_DATA=$REPO/data/obs/train/big-v1-sf18-alltiers,$REPO/data/obs/train/big-v1-reckless-alltiers \
  D41_TRUNK_RAW=$REPO/data/nnue/experiments/d40-long-v1/ckpt/d40-uho-720sb-final-raw.bin \
  D41_LOSS=pairwise D41_SUPERBATCHES=12 D41_BPS=250 D41_BATCH=4096 D41_THREADS=2 \
  D41_EVAL_TIER=64000 D41_OUT=$REPO/data/nnue/experiments/d41-v11/ckpt D41_ID=d41-v11-c-pairwise-alltiers \
  ./target/release/examples/d41_alloc11
```

Checkpoints (gitignored bytes): `data/nnue/experiments/d41-v11/ckpt/`;
packed head: `data/nnue/experiments/d41-v11/d41-v11-c-pairwise-alltiers.cpol`.

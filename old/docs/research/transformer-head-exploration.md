# Transformer (KQV attention) eval head — design exploration

Status: THINKING (2026-07-18, user-requested). No code yet. This is a side-bet
relative to the SF16+30-at-equal-time goal; the analysis below explains why,
and lays out the cheap experiment that would decide it.

## 0. The question

Replace our chess-specific eval head (the D40 Reckless-shaped
pairwise-multiply → `768→16→32→1` MLP, per 8 buckets) — and possibly the
hand-crafted FullThreats features — with a learned **KQV self-attention**
block. User's flagged concern: sparseness. Honest read below.

## 1. The tokenization problem (the crux)

Attention needs a *sequence of tokens*; our accumulator is a single 768-vector.
So "add a transformer head" is really a choice of **what the tokens are**, and
that choice dominates everything (cost, capacity, whether it can beat the MLP):

- **A. Accumulator-chunk tokens** (minimal head-swap): reshape the 768-acc into
  T tokens × d, 1 self-attn layer + FFN, pool → scalar/bucket. Keeps the fast
  incremental feature transformer (FT); isolates "MLP head vs attention head".
  Tokens carry no chess semantics — attention just learns cross-feature 2nd-order
  interactions (our pairwise-multiply already does a crude version of this).
  Cheapest to try; probably only marginal over the MLP.
- **B. Piece-token transformer** (the "real" chess value transformer): each token
  = (piece-type embed + square positional encoding + color); sequence = occupied
  squares (2–32, SPARSE + variable). Pieces attend to pieces — exactly the
  relational structure chess eval needs (attacker↔victim, king↔its attackers,
  passer↔blockers), i.e. the same interactions FullThreats hand-codes, but
  LEARNED. This is what the request is really pointing at.
- **C. Per-square FT-embedding tokens** (the strong candidate): don't SUM the FT
  into one accumulator — keep 64 per-square embeddings (each = sum of that
  square's active feature columns), then attention over 64 square-tokens. This is
  essentially the Lc0/DeepMind eval architecture (64 square tokens + attention)
  but fed by OUR HalfKA(+threats) FT — so it keeps the huge FT capacity AND gets
  semantic tokens. Most likely to actually beat the MLP head.

## 2. The sparseness tradeoff — honestly

- Piece-tokens (B): sequence length = piece count (2–32), variable.
- **Upside of sparseness:** attention is O(P²·d); P~15 midgame → ~225 pairwise
  scores — CHEAP, and cheaper still in endgames. Unlike a fixed 64-grid (64²),
  sparse piece-tokens keep attention affordable. Sparseness is arguably a
  *feature*, not a bug, for cost.
- **Downside 1 — loses NNUE's incremental update.** Our NPS comes from updating
  only the changed features on make/unmake. A piece/square transformer recomputes
  attention every eval (each move changes tokens). Over ~15 tokens it's not
  catastrophic, but it's the wrong shape for our make/unmake accumulator seam.
- **Downside 2 — capacity (the real one).** NNUE's strength lives in the HUGE FT
  (768 × (22528+60720) ≈ 64M params, per-king-square specialization). A
  piece-token transformer with small embeddings has far fewer params and may
  UNDERFIT the NNUE FT — i.e. Option B risks being a *weaker* eval overall, more
  expressive per-FLOP but a smaller model. This is why **Option C** (attention on
  top of the big FT's per-square embeddings) is the only variant likely to be
  *stronger* than our MLP head; B alone probably regresses.

## 3. The NPS / equal-time reality (the headwind)

- Our equal-time edge IS nps. Any attention head is far more FLOPs than a
  `768→16→32→1` MLP (~hundreds of MACs). Even a 1-layer attention is 10–100× the
  head cost, before counting the lost incremental-update speed.
- At EQUAL TIME, lower nps = shallower search; the eval gain must beat the depth
  loss. Empirically it doesn't in CPU alpha-beta: **no top CPU engine (SF,
  Berserk, Reckless, Obsidian…) uses a transformer head** — all use fast MLP
  NNUE heads. The transformer-eval regime that WINS is Lc0 (GPU + MCTS, low nps)
  and DeepMind's 270M "chess without search" (GPU, ~2895 Elo, no search) — a
  different engine architecture, not CPU alpha-beta at equal time.
- So a transformer head is a **different bet** than "beat SF16 at equal time via
  fast eval + deep search." On CPU alpha-beta it's historically a losing bet at
  equal time.

## 4. Why it's still worth testing — as an eval-CEILING probe

The scientific value is not an equal-time contender; it's quantifying **how much
eval quality our MLP head + hand-crafted features leave on the table.** Run at
FIXED NODES (isolate net quality from speed, per the D39 decomposition). Two
outcomes, both useful:
- If an attention head is meaningfully stronger at fixed nodes → there's real
  eval headroom → capture it CHEAPLY via **distillation** (train the transformer,
  then retrain our fast MLP head on the transformer's outputs — the MLP is a
  weaker function class but better targets still transfer). Distillation is the
  pragmatic payoff: some of the gain at MLP inference cost, keeping nps.
- If it's NOT stronger even at fixed nodes → transformer heads don't help our
  features → ruled out cheaply.

An elegant sub-hypothesis worth the same probe: can attention over HalfKA-only
tokens (NO FullThreats) MATCH HalfKA+FullThreats+MLP? If yes, attention SUBSUMES
the 60,720 hand-crafted (and slow-to-compute) threat features — a strong, clean
result that would simplify the whole eval.

## 5. Ruthless staged plan (cheap gate first, like everything else)

- **Stage 0 — offline gate (hours, pure GPU, no engine work).** On the SAME
  inputs and held-out value data, does an attention head fit the value target
  BETTER than our MLP head? Arms: (A) accumulator-chunk attention, (C) per-square
  FT-embedding attention, and the (C-minus-threats) subsumption probe. Metric:
  val eval-MAE / loss vs the MLP baseline. If none beats the MLP → STOP.
  - Trainer — the honest boundary (re-corrected 2026-07-18 after a deep read of
    bullet's API): the matmul/softmax primitives exist, BUT only at bullet's
    lower **tensor-IR** level; the **model-builder** level that examples use
    (`ModelNode`) has **no transpose, no standalone softmax, no reduce-MAX** —
    so a genuine `[T×T]` `Q·Kᵀ` self-attention is **NOT expressible** at the
    usable example level. A true KQV self-attention head therefore needs one of:
    (a) drop to bullet's tensor-IR (significant work, stays in-stack), or
    (b) **PyTorch** (attention is one line; a parallel data/export stack) — so
    PyTorch is a real option after all for *true* self-attention, contrary to my
    earlier "no PyTorch needed." What IS expressible in bullet now, and what the
    Stage-0 scaffold (`tools/nnue/bullet/d40_attn.rs`) implements, is
    **diagonal self-gating attention**: Wq/Wk/Wv projections, per-token
    importance = `softmax(diag(Q·Kᵀ))` (softmax hand-composed from
    `exp`+`reduce_sum_cols`+`repeat`+reciprocal), value-weighted token-sum. This
    is attention-*lite* — learned token gating, NOT token↔token mixing — so it is
    a weaker first probe: if even it beats the MLP head, invest in true
    self-attention; if it's flat, accumulator-chunk attention likely doesn't
    help our features. Inference still needs no framework (matmul+softmax in Rust).
  - INFERENCE needs NO framework at all: the attention forward is matmul +
    softmax in plain Rust/SIMD, hand-written like `sota_head`/`sota_simd`.
- **Stage 1 — if offline-positive: Rust scalar forward + FIXED-NODE screen** vs
  the incumbent net. Reuse the firewalled-instrument pattern (feature-gated,
  scalar OK for fixed-node). Measure fixed-node Elo AND nps (to project
  equal-time). This is the real "is there eval headroom" answer.
- **Stage 2 — if fixed-node-positive: the payoff fork.** Either SIMD/quantize the
  attention (hard, uncertain) OR **distill into our MLP head** (pragmatic, keeps
  nps). Distillation is the likely real win.

## 6. Infra + effort

- PyTorch install (CUDA, ~GB) — one-time; the box has CUDA (bullet uses it) and
  an RTX 4060 Ti (8 GB). Bullet itself can't express attention (NNUE-shaped).
- Stage 0 ≈ ~1 day (data pipe + 2–3 head trainers + compare). Stage 1 ≈ 2–3 days
  (Rust forward + screen). Stage 2 (distill) ≈ ~2 days.
- Stage 0 alone answers the core question cheaply — do it first.

## 7. Recommendation

1. This is a genuine research question but a **side-bet**: the equal-time / CPU
   alpha-beta regime disfavors transformer heads, and the realistic payoff is
   **distillation of a transformer's eval into our fast head**, not shipping a
   transformer head.
2. The promising variant is **C (attention over per-square FT embeddings,
   keeping the big FT)**, not a from-scratch piece-token transformer (likely
   underfits vs our 64M-param FT). Piece-token sparseness is cost-friendly but
   capacity-poor.
3. Do **Stage 0** (offline: does attention fit the target better?) when the box
   frees and PyTorch is set up — cheap and decisive. Gate everything after it on
   that result, exactly like the helm offline bars.
4. Sequencing vs the SF16 push: this waits behind the live eval levers (D40
   stage-2 verdict, longer stage-1, king-bucket factorization) unless Stage 0
   surprises us — it is an eval-ceiling instrument, not the equal-time path.

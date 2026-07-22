# Learned sparse embeddings (de-handcrafting the NNUE input layer)

*User question, 2026-07-22 (pre-sleep): all our inputs are handcrafted
inductive bias (HalfKA, threats, even SF's new pawn pairs) — history says
learned beats handcrafted. NNUE's special constraints: inputs must be
SPARSE and INCREMENTALLY STABLE between moves. Is there a good way to learn
an embedding respecting those properties?*

## Reframing: what is actually handcrafted

The FT embedding vectors are already fully learned. The handcrafted part is
the **index function** — which discrete conjunctions of board atoms get a
dedicated row. Sparsity forces features to be *rarely active* (conjunctive);
incrementality forces them to be *local* (each feature reads a bounded board
region so a move flips few). So the learnable object is: **which local
conjunctions deserve capacity.**

## Four paths, ranked by shippability

1. **Learn the maps, deploy as tables** (zero runtime cost, this-quarter
   engineering). Every discrete bucketing/keying (king buckets, threat
   typing, routing axes) can be trained with soft assignment offline and
   HARDENED to a lookup table at export. Deployed engine sees only a
   different index table — all NNUE properties preserved exactly.
   Precedent: learned king-bucket maps via gradient-statistic clustering.
   Validate any learned map against the handmade one with the queens-canary
   methodology (ablation suite).

2. **Enumerate local windows exhaustively** (the PP_3Wide generalization).
   Instead of a human selecting pawn-pair features, pick only the WINDOW
   (e.g., all pawn configs in a 3-file slice — finite/enumerable), give
   every configuration its own learned row, let SGD sort out what matters.
   HalfKA is secretly this recipe for (king, piece-square). Residual human
   bias shrinks to "which windows" — much weaker than "which features".
   Sparse (one code/window), incremental (pawn move flips 1–2 codes),
   buildable on existing table machinery.

3. **Sparse autoencoder + slowness constraint** (the research bet — open
   combination in this domain). Three constraints, three known mechanisms:
   - sparsity → top-k activation;
   - incrementality → **temporal-coherence penalty** on code churn between
     successive game positions (Slow Feature Analysis; our game data gives
     successive pairs for free);
   - cheap delta updates → **locality masks** (each code unit reads a
     bounded square set, making incremental encoding exact).
   Hard part: the encoder's own runtime cost. If it works it's a genuine
   contribution: an SFA-regularized sparse feature layer in an NNUE engine.

4. **Dense-teacher feature discovery** (flywheel-era). Train an
   unconstrained dense net; mine its attributions over candidate
   conjunction families; enumerate the winners into the table net (path 2).
   Automates the "which windows" choice — human fully out of the loop,
   runtime contract untouched.

## Sequencing

Not before the D47 arc settles (eval foundation first). Then: path 1 on the
king-bucket map (T0-adjacent, zero deploy risk) → path 2 on a pawn-window
family (subsumes/tests T6b) → path 3 as a dedicated research spike →
path 4 once the dense-teacher infrastructure exists (self-play era).
Relates: `arch-ablation-suite.md` (T0/T6b/T7, queens canary),
`policy-training-roadmap-2026-07-22.md`,
`stockfish_reference/08-sfnnv16-pp3wide-pawn-pairs.md`.

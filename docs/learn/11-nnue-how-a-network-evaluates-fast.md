# 11 — NNUE: how a neural network evaluates a position fast enough

Every file so far has taken the evaluation function for granted. The
search visits millions of positions per second; at each one it needs a
number — "how good is this for the side to move?" — and files 05 through
10 all assumed that number appears cheaply. File 08 showed how ARCHON's
classical evaluation produces it: material counts plus hand-tuned
piece-square tables. This file replaces that with a **neural network** —
and the entire story is about the word *fast*. A network that evaluates a
position slightly better but ten times slower would make the engine
weaker, because it would search shallower. NNUE is the set of tricks that
make a neural evaluation cheap enough to run at every node.

**NNUE** stands for "Efficiently Updatable Neural Network" (the letters
are shuffled; the original coiner wrote it "ƎUNN"). Read that name
carefully, because it names the one idea that makes the whole thing work:
*efficiently updatable*. Let us see why that matters, then build the
network from the input features up.

## 1. Why a normal neural network is too slow here

Imagine the most naive plan: at each node, take the 64 squares, encode
what is on them, feed that through a few fully-connected layers, and read
off a score. The problem is arithmetic volume. A first layer that maps,
say, 768 inputs to 256 hidden units is `768 * 256 ≈ 197,000`
multiply-adds — *per position*. At tens of millions of positions per
second, that first layer alone would need trillions of operations per
second. You would spend all your time in the network and none searching.

But look at what actually happens during a search. The engine does not
evaluate 64 unrelated positions. It walks a tree by **making and unmaking
moves** (file 04). Each move changes the board *very little* — it moves
one piece, occasionally capturing another. The position at a child node
is almost identical to its parent: 62 of the 64 squares are unchanged.

That observation is the whole game. If the expensive first layer's output
could be **updated incrementally** — adjusted for just the one or two
squares that changed, rather than recomputed from scratch — then the cost
per node would collapse from ~197,000 operations to a few hundred. That is
what "efficiently updatable" means, and it is why NNUE networks have a very
specific shape: a **huge, sparse first layer** designed to be updated
incrementally, followed by a **tiny dense tail** that runs from scratch
each time but is cheap because it is small.

## 2. Input features: from piece-square to king-relative to relational

The first design question is: what are the network's inputs? The answer
has evolved, and ARCHON contains three generations of it, so it makes a
perfect ladder to climb.

### 2.1 Raw piece-square features (768 inputs)

The simplest encoding, and the one ARCHON's first network ("net-0") uses,
is a **768-dimensional binary vector**:

```
   768 = 2 colors × 6 piece types × 64 squares
```

Each entry is 1 if "a piece of this color and type is on this square,"
else 0. A normal middlegame lights up ~32 of the 768 inputs (one per piece
on the board) and leaves the rest at zero. This **sparsity** — only a
few dozen of hundreds of inputs are ever nonzero — is the second key fact,
and it pairs with incremental updates: a move flips a small handful of
these bits off and on.

### 2.2 King-relative features: HalfKA

Raw piece-square features have a weakness: they treat "knight on f3" the
same regardless of where the kings are. But in chess, a piece's value
depends enormously on king safety and king position. So the next
generation makes the features **relative to the king**.

ARCHON's stronger network uses **HalfKAv2_hm**, a feature set adapted from
Stockfish (the index math is reimplemented in `src/eval/nnue/sota_features.rs`;
the ideas are borrowed, no code is copied — a license firewall the project
takes seriously). "HalfKA" reads as *Half* (one accumulator per
perspective, §3), *K* (king), *A* (all pieces). Each active feature now
encodes the tuple `(our king's square, this piece's color-and-type, this
piece's square)`. The network learns piece values *conditioned on where
the king sits* — a knight near an exposed king means something different
from a knight near a castled one, and the feature set lets the network
express that directly.

Two engineering tricks keep the feature space manageable. **King
bucketing**: the 64 king squares are grouped into 32 "buckets" of similar
king positions, so the network shares knowledge across nearby king squares
instead of learning each of the 64 separately. **Mirroring**: the board
has a left-right symmetry (a position and its mirror image are
strategically identical), so features are horizontally flipped to put the
king always on one side, halving the space. The result is a
22,528-dimensional feature set — much larger than 768, but still sparse
(~32 active features) and still incrementally updatable.

There is a subtlety incremental updates must respect: when the *king
itself* moves, the reference point for *every* feature changes, so the
whole accumulator must be rebuilt from scratch. ARCHON detects this and
refreshes only when the king crosses into a different bucket-or-mirror
state; ordinary moves stay cheap. A small cache (a "Finny table," keyed by
king-bucket state) further softens the cost of those rebuilds.

### 2.3 Relational features: FullThreats

The third generation, ARCHON's D40 architecture, adds **FullThreats** — a
*relational* feature set. Instead of encoding pieces in isolation, a
FullThreats feature fires for a directed *relationship* between two
pieces: "the attacker on square `from` threatens the victim on square
`to`." This is a qualitatively different kind of input. It hands the
network, for free, information it would otherwise have to *infer* — which
pieces attack which — so the network can spend its capacity on judgment
rather than on rediscovering the geometry of the board.

Relational features cost more to maintain incrementally, and this is where
NNUE's neat "one or two squares changed" story gets genuinely hard. When a
single piece moves, it can change *dozens* of attack relationships at once:
every sliding piece whose line was blocked or unblocked by the vacated or
occupied square (an "x-ray ripple"), plus every short-range piece next to
either endpoint. So the FullThreats accumulator (`src/eval/nnue/sota_threats.rs`)
emits, per move, a list of up to ~80 threat events to add or remove,
rather than the tidy two-add/two-subtract of HalfKA. The engine goes to
real trouble to compute this correctly from the post-move board plus the
record of what changed, and it verifies — over thousands of random games —
that the incrementally maintained accumulator exactly equals a
from-scratch recompute. Relational inputs matter enough to be worth that
machinery: at fixed node count, ARCHON's FullThreats network is measurably
stronger per node than its best classic-feature network.

## 3. The feature transformer and perspective accumulators

Now the central mechanism. The huge first layer is called the **feature
transformer**. It maps the sparse feature vector to a dense hidden vector
of width `L1` (256 in net-0; 768 in the D40 architecture). Because the
input is a sum of a few active features, and a matrix times a sum is the
sum of the matrix times each term, the transformer's output is simply:

```
   accumulator = bias + Σ (column of the weight matrix for each active feature)
```

The output is called the **accumulator**, and this formula is the source
of the magic. To *add* a piece, add its weight column to the accumulator.
To *remove* a piece, subtract its column. A move that moves a piece from
`f3` to `e5` is: subtract the `f3` column, add the `e5` column. Two vector
operations of width `L1`, instead of a full matrix multiply. **That is
the incremental update**, and it is exactly a make/unmake mirror of file
04: as the search makes a move it pushes the deltas onto an accumulator
stack; as it unmakes, it pops back.

### 3.1 Perspective and side-to-move symmetry

There is one more idea baked into the feature transformer:
**perspective**. Chess is symmetric between the two players — a position
is "the same" whether you describe it from White's or Black's point of
view, with colors and ranks flipped. NNUE exploits this by maintaining
**two accumulators**, one from each player's perspective, both from the
same feature transformer weights but with the board flipped for the
opponent's view.

At evaluation time the network concatenates the two accumulators with the
**side-to-move's perspective first**. This means the network always sees
"me" then "them," never "White then Black" — so it learns a single notion
of "good for the player to move" that works for both colors, effectively
doubling its training data (every position teaches both perspectives). The
`evaluate` function in `src/eval/nnue/accumulator.rs` does exactly this:
it folds both perspectives forward, then orders them `us, them` before the
output layer.

### 3.2 Lazy folding: don't update accumulators you never read

Here is a refinement that ties straight back to file 10. Most nodes the
search *makes* are **pruned before they are ever evaluated** — the whole
point of the pruning cascade. So eagerly updating the accumulator on every
made move would compute vectors nobody reads.

ARCHON therefore updates **lazily**. Each made move records its deltas and
marks the child accumulator "not yet computed." Only when the search
actually calls `evaluate` does the engine walk back to the nearest
already-computed ancestor and fold the recorded deltas forward — paying for
exactly the nodes that are evaluated and no more. The correctness linchpin
is an invalidation rule (a child marked uncomputed can never reuse a
sibling's stale accumulator) backed by a debug-only oracle,
`assert_matches_refresh`, that checks the incrementally maintained columns
against a from-scratch rebuild at every computed node — the NNUE analogue
of the board-consistency oracle from file 04.

## 4. The tail: activation and the output layers

The accumulator is the expensive-but-incremental part. Everything after it
is the **tail**: small dense layers that run from scratch at each
evaluated node, cheap because they are small.

### 4.1 The activation function: (S)CReLU

Between layers the network applies a **nonlinearity** — without one, a
stack of linear layers collapses into a single linear layer and can learn
nothing interesting. NNUE uses variants of the clamped rectified linear
unit:

- **CReLU** (clipped ReLU): `clamp(x, 0, QA)` — clip the value to the
  range `[0, QA]`. Cheap, and the upper clamp keeps values in the fixed
  range that quantized arithmetic needs (§5).
- **SCReLU** (squared clipped ReLU): `clamp(x, 0, QA)²` — the same, then
  squared. The squaring gives the network more expressive power at the
  same width, and it is ARCHON's default; net-0 is a `768 → 2×256 → 1`
  SCReLU network.

The choice of CReLU vs SCReLU is a per-network flag stored in the net file
(`src/eval/nnue/network.rs`), so a loaded net carries its own activation.

### 4.2 The output, and output buckets by material

The classic net-0 tail is trivially small: activate the two 256-wide
accumulators, concatenate to 512, and take a single weighted sum to a
scalar score. The D40 architecture uses a slightly deeper tail — the
768-wide accumulators feed a `768 → 16 → 32 → 1` stack of small dense
layers (`src/eval/nnue/sota_head.rs`) — but "small" is the operative word:
`768*16 + 16*32 + 32*1 ≈ 13,000` operations, negligible next to the
197,000 the accumulator would have cost from scratch.

The D40 tail adds one more idea: **output buckets by material**. The right
way to convert an internal score to centipawns differs between a
queens-on middlegame and a king-and-pawn endgame. So the network keeps
**8 separate output heads**, and selects which one to use based on the
number of pieces on the board (a non-uniform mapping — finer at
mid-material where most decisions happen, coarser at the extremes). Each
bucket specializes for its phase of the game. Only the selected bucket
runs, so the cost is still one small tail, not eight.

## 5. Quantization: floats to integers, and what breaks

The network is **trained** in floating point (file 12). But floating-point
arithmetic is comparatively slow and, more importantly, it does not pack
into wide SIMD vectors as densely as small integers do. So for
**inference**, the weights are **quantized**: converted to small integers.
ARCHON uses `i16` for the accumulator and `i8` for the deeper layers, with
three scale constants baked into the net format
(`src/eval/nnue/mod.rs`): `QA = 255`, `QB = 64`, `SCALE = 400`.

The scheme is straightforward once you see it. A trained floating-point
weight `w` becomes the integer `round(w * QA)`. Because the accumulator is
a sum of such integers, it lives at "scale `QA`" — that is, an integer
value of `QA` represents the real number `1.0`. The output layer weights
are quantized at scale `QB`, so after the accumulator (scale `QA`) is
multiplied by output weights (scale `QB`), the result is at scale
`QA * QB`, and a final division by `QA * QB * (something)` and multiply by
`SCALE` recovers a centipawn number. The activation clamp `clamp(x, 0, QA)`
is what keeps accumulator values inside the range `i16` can hold — the
`QA = 255` ceiling and the `i16` roof are chosen together.

**What breaks.** Quantization is lossy, and the failure modes are real,
not hypothetical:

- **Overflow.** If accumulator values exceed the `i16` range, they wrap
  around to garbage. The activation clamp and the choice of `QA` are
  precisely what prevent this; a network trained without weight
  constraints can produce weights that, once quantized and summed, blow
  the range. Trainers therefore *clip* weights during training so the
  quantized net cannot overflow.
- **Rounding error.** Each `round(w * QA)` loses a little precision;
  across a network these errors accumulate. The scales are chosen large
  enough that the rounding noise is small next to the centipawn
  granularity the search actually cares about.
- **Distribution tails.** ARCHON's own research log records a concrete
  instance: a candidate net passed everywhere except on a handful of
  bizarre out-of-distribution endgame positions, where a value clipped
  against the quantization ceiling and produced a wrong eval — a
  "quant-clipping tail," not a logic bug. Catching these is why the engine
  runs a **parity gate**: it checks that the quantized integer inference
  matches the float reference within a tight tolerance across thousands of
  positions before a net is trusted.

The trade is worth it: integer inference is several times faster than
float, and — as the next section shows — it unlocks the vector hardware.

## 6. SIMD in one intuitive pass

The last speed multiplier is **SIMD** — Single Instruction, Multiple Data.
A modern CPU can, in one instruction, apply the same operation to a whole
*vector* of values at once. An AVX2 register holds 16 `i16` values; an
AVX-512 register holds 32. So "add this weight column to the accumulator"
— a loop over 256 numbers — becomes 16 vector adds instead of 256 scalar
ones. The incremental update and the activation are perfectly shaped for
this: they apply the same operation across every lane of the accumulator.

The intuition to carry away: NNUE's data layout is *designed* so the hot
operations are "do the same thing to 16 (or 32) numbers at once." The
accumulator width `L1` is always a multiple of the SIMD width for exactly
this reason (ARCHON asserts `L1 % 16 == 0`). This is why the whole design
insists on small integers and contiguous, aligned arrays: it is what lets
the compiler and the hardware run the evaluation in wide parallel strides.

ARCHON ships multiple SIMD backends and picks one at runtime based on what
the CPU supports (`src/eval/nnue/simd.rs` and `sota_simd.rs`). The best is
**AVX-512-VNNI** — "Vector Neural Network Instructions," a set of
instructions added specifically for exactly this workload, which fuses the
multiply-and-accumulate at the heart of the integer layers into single
operations. On the D40 architecture, VNNI is the main lever for closing
the inference-speed gap that its richer features open up. There is a
plain-scalar fallback too, kept as the bit-exact reference the vector
paths must match — the same "one correct reference, several fast
implementations validated against it" discipline you have now seen
throughout the engine.

## 7. Putting the pieces together

Trace one evaluation end to end, and the design's logic snaps into focus:

1. The search **makes a move** and pushes the tiny feature delta onto the
   accumulator stack — but does no arithmetic yet (lazy).
2. If the node is **pruned** (file 10), the delta is popped on unmake and
   nothing was wasted.
3. If the node is **evaluated**, the engine folds the recorded deltas
   forward from the nearest computed ancestor — a handful of SIMD vector
   add/subtracts per perspective — producing the two accumulators.
4. It orders them **side-to-move first**, applies the **(S)CReLU
   activation**, runs the **small integer tail** through the
   **material-selected output bucket**, and rescales the integer result to
   centipawns.
5. That centipawn number flows back into alpha-beta exactly where file
   08's hand-crafted number used to.

The network is doing more sophisticated judgment than a piece-square
table ever could — king-relative, relational, phase-specialized — and yet,
because of incremental updates, sparsity, quantization, and SIMD, it costs
the search only marginally more per node. That combination is why NNUE
took engines from strong to superhuman, and it is why net-0 alone was
worth roughly +700 Elo over ARCHON's hand-tuned evaluation.

## What you now know

- **NNUE = Efficiently Updatable NN**: a huge sparse first layer that is
  updated **incrementally** as moves are made/unmade (mirroring file 04),
  plus a tiny dense tail run from scratch — the shape that makes a neural
  eval cheap enough for every node.
- Input features climb a ladder: **raw 768 piece-square** → **king-relative
  HalfKA** (king-bucketed, mirrored) → **relational FullThreats**, each
  handing the network more structure at higher maintenance cost.
- The **feature transformer** produces per-**perspective accumulators**;
  adding/removing a piece is adding/subtracting a weight column, and the
  side-to-move-first ordering encodes chess's color symmetry.
- The tail applies a **(S)CReLU** activation and selects an **output bucket
  by material**; ARCHON updates accumulators **lazily** so pruned nodes
  cost nothing.
- **Quantization** turns float weights into `i16`/`i8` (`QA/QB/SCALE`) for
  speed and SIMD-packing; its failure modes (overflow, tails) are guarded
  by weight clipping and a **parity gate**.
- **SIMD** applies one operation to 16–32 lanes at once; ARCHON's
  accumulator layout is built for it, with AVX2 and **AVX-512-VNNI**
  backends validated against a scalar reference.

We now have a network that *evaluates* fast. But where did its weights
come from? The next file is about **training**: turning millions of
labeled positions into those weights, why data volume turned out to be the
single dominant factor in ARCHON's own experiments, and the surprisingly
deep question of what a "correct" evaluation label even is. Continue to
`12-nnue-training.md`.

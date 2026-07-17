# 12 — NNUE training: learning an evaluation from data

File 11 built a network that can evaluate a position in a fraction of a
microsecond. It left one enormous question unanswered: *where do the
weights come from?* A neural network is just a big pile of numbers; those
numbers have to be *learned* from data. This file is about that learning —
framed, honestly, as a regression problem — and about the hard-won
practical lessons ARCHON's own research program produced, chief among them
a finding that surprised the project: the single dominant factor in
whether a network turned out strong was **how much data it trained on**,
by a wide margin over cleverer considerations.

This is a story with a real plot, recorded in the project's live research
log (`docs/research/value-foundation-state.md`). We will tell it, because
the plot *is* the lesson.

## 1. Training as regression

Strip away the mystique and NNUE training is a textbook supervised
regression. You have:

- a large set of chess **positions**, and
- for each position, a **label**: a number saying how good that position
  is.

You want the network to output that label. So you define a **loss
function** measuring how far the network's output is from the label,
and you nudge the weights, over millions of examples, to make that loss
small. The nudging is **stochastic gradient descent**: compute the loss on
a small batch of positions, compute which direction to move each weight to
reduce it (the gradient), take a small step, repeat. Do this for billions
of position-presentations and the weights converge to something that
predicts the labels well — and, if all goes right, *generalizes* to
positions it never saw.

Everything interesting is in two questions: **where do the labels come
from**, and **what exactly are we regressing onto**. Both are subtler than
they first appear.

## 2. Where labels come from

There are three sources of labels, and modern training blends them.

### 2.1 Teacher-engine distillation

The most direct source: take an already-strong engine (the "teacher"),
have it evaluate each position — usually with a shallow fixed-depth or
fixed-node search — and use its score as the label. The student network
learns to imitate the teacher's judgment, but *without doing the
teacher's search*: it distills the teacher's search-informed evaluation
into a single fast forward pass.

ARCHON's first strong network, net-0, was distilled from a compiled
Stockfish 16 acting as teacher. This is a deliberate bootstrapping
strategy (call it "AlphaGo-style"): use a strong external engine to
generate the first net, prove the whole training-and-inference pipeline
works, and only later retrain the *shipped* net on the engine's own
self-play data for clean provenance. The teacher's data and binary are
treated as discardable scaffolding.

### 2.2 Self-play and game outcomes (WDL)

The second source needs no teacher at all: let the engine **play games
against itself**, and label each position with the eventual **game
outcome** — Win, Draw, or Loss (WDL) for the side to move, encoded as
`1.0`, `0.5`, `0.0`. This is the AlphaZero idea. The label is not any
engine's opinion; it is *ground truth* about how the game actually turned
out from that position.

WDL labels have a complementary character to teacher scores. A teacher
score is precise but only as good as the teacher. A game outcome is
authoritative but *noisy*: a single game from a slightly-better position
can still be lost, so any one label is a coin flip biased by the position's
true value. Over millions of games the noise averages out and the network
learns the true win probability — but only in aggregate.

### 2.3 Blending the two: lambda

Because teacher scores and game outcomes have complementary strengths, the
standard practice is to **blend** them. The training target for each
position becomes a weighted mix:

```
   target = λ · (teacher eval, as a win probability)
          + (1 − λ) · (actual game result)
```

The parameter **λ** (lambda) slides between "pure distillation" (`λ = 1`,
trust the teacher's eval) and "pure outcome learning" (`λ = 0`, trust the
game result). A typical recipe leans toward the eval early (the teacher is
a cleaner signal than a noisy single game) and mixes in outcomes for
grounding. The `bullet` trainer ARCHON uses exposes this directly as its
`--wdl` result-weight knob. Interestingly, in ARCHON's large-scale
experiments lambda turned out to be a **flat** lever — varying it barely
moved results — which is itself a finding: it told the team the bottleneck
was elsewhere.

## 3. The sigmoid–centipawn bridge

There is a units problem hiding in that blend, and resolving it is one of
the most important — and most error-prone — parts of NNUE training.

Game outcomes live in `[0, 1]`: a probability of winning. Engine
evaluations live in **centipawns**: `+150` means "up about a pawn and a
half," and the scale is unbounded. You cannot average a probability with a
centipawn number directly; they are different kinds of quantity. You need
a bridge, and the bridge is the **sigmoid**.

A centipawn advantage maps to a win probability through a logistic
(sigmoid) curve:

```
   win_probability = sigmoid(cp / scale) = 1 / (1 + e^(−cp / scale))
```

Intuitively: a `0` cp position is a coin flip (probability `0.5`); a large
positive advantage saturates toward `1.0`; a large negative one toward
`0.0`; and the `scale` constant sets how many centipawns correspond to a
given probability shift. Now both the teacher's centipawn eval and the
game's WDL result live in the same `[0, 1]` space, and blending and loss
computation make sense.

The inverse direction matters just as much. When you want to express a
target win-probability *back* as the centipawn number the network should
output, you invert the sigmoid: `cp = scale · atanh(...)`, or more
precisely `cp = scale · logit(p)`. And here is a trap ARCHON hit hard
enough to write it into its permanent memory. **The label must be computed
with the trainer's own scale formula, not some engine's centipawn-*display*
formula.** Early networks failed mysteriously because a data converter used
an engine's human-facing "score display" mapping instead of the trainer's
`cp = scale · atanh(q)` relationship — the labels were quietly on the wrong
scale, and no amount of training could fix mislabeled data. The lesson,
generalized: *the label scale is part of the network's contract, and it
must match end to end.*

## 4. Loss functions

With targets in probability space, the loss is a standard choice. Two show
up:

- **Mean squared error** on the win probabilities: `(predicted_p −
  target_p)²`, averaged over the batch. Simple and stable.
- **Cross-entropy** (log loss): the natural loss for probabilities, which
  penalizes confident wrong predictions more harshly.

Either works; the field mostly uses variants of these, sometimes weighted
to emphasize decisive positions over dead-drawn ones. The loss is the least
interesting design choice in the pipeline — which is exactly why beginners
over-focus on it. The data matters far more than the loss, as ARCHON was
about to discover the hard way.

## 5. The B4 finding: data volume dominates

Here is the plot. For a long stretch of the project, *every* network
ARCHON trained failed to beat net-0. The team tried richer feature sets,
different label sources, different lambda values, different architectures.
Nothing worked. The natural hypothesis was that something was subtly
*wrong* — a target mismatch, a search-contract problem, a bug.

The actual answer, once found, was almost embarrassingly simple. Look at
the numbers. Every failed challenger had trained on somewhere between **25
million and 265 million** positions — and, more tellingly, had seen only
`0.3` to `3` passes over its data before training stopped. Net-0 itself
had won not because it was special but because its small 25-million-position
corpus had been presented **thirty times over** — it was fully *converged*,
while every bulk challenger was *starved*, stopping after a fraction of the
training it needed.

When the team finally trained networks on the scale the wider community
uses — the official Stockfish training data ("binpacks"), presented for
**billions** of position-presentations (3 to 9 billion) — the challengers
beat net-0 **decisively, on the first try.** The governing question the
whole research phase had been built around ("why can't we beat net-0?")
had a one-word answer: **volume**. Compute had never been the constraint;
nobody had turned the dial far enough.

The quantitative shape of the win is worth internalizing:

- **Width scaled cleanly with data.** Once the data was abundant, making
  the network *bigger* (accumulator width `256 → 512 → 1024 → 2048`)
  monotonically improved strength (+32, +123, +225, +282 Elo at a
  fixed-node screen versus net-0) — until it **saturated at 2048**, where
  going to 4096 added a statistically-flat +15. There is a right size for
  a given data budget, and you find it by climbing until the gains stop.
- **The feature-set failures had been data-starvation in disguise.** The
  richer HalfKA and FullThreats architectures from file 11 had "failed"
  earlier *only* because they were even hungrier for data than the plain
  768 features and had been starved worse. Given full-scale data, the D40
  FullThreats network beats the best classic-feature network at fixed
  nodes. The architecture was never the problem; the diet was.

The generalizable lesson — and the reason this file spends a whole section
on a negative result — is that in machine learning, **the most common
cause of a model that "won't learn" is not enough data or not enough
training, not a subtle bug.** The instinct to hunt for cleverness first
and check volume last is exactly backwards. ARCHON's log now states it
flatly: "data volume was the constraint end-to-end."

## 6. Two-stage training: broad base, then quality finetune

Once volume was solved, a second structural lesson emerged: **the order and
quality of data matter, in stages.**

The winning recipe is two-stage:

1. **Stage 1 — broad base.** Train from scratch on a large, diverse corpus
   (in ARCHON's case, a big "unbalanced human openings" binpack of
   moderate-depth labels). This gives the network broad *coverage* — it
   has seen a huge variety of positions and learned the general shape of
   chess evaluation.
2. **Stage 2 — high-quality finetune.** *Warm-retrain* (continue training)
   the stage-1 network on a smaller, **higher-quality** corpus — positions
   labeled by deeper, more careful searches. This sharpens the network's
   judgment on top of the broad base.

The finding that makes this principled: the stage-2 gain is **proportional
to the label quality of the stage-2 data.** ARCHON measured this directly.
A stage-2 corpus whose labels correlated `0.93` (Pearson) with a trusted
deep reference gave a large gain (+55, then +38 Elo on successive hops); a
noisier corpus correlating only `0.78` gave a much smaller one (+19, then
flat). And the hops **plateau** after about two — you cannot finetune your
way to infinity; a couple of quality passes extract most of the available
gain. Broad coverage first, then a short burst of high-quality sharpening:
that is the shape.

## 7. Hygiene: overfitting, leakage, and deduplication

Volume and staging assume one thing that is not automatic: that you are
measuring *generalization*, not memorization. Two failure modes threaten
this, and both have standard defenses.

**Overfitting** is when the network memorizes its training positions'
labels instead of learning the underlying function — it aces the training
set and flunks new positions. The universal defense is a **held-out set**:
positions the network *never trains on*, used only to measure loss. If
training loss keeps dropping while held-out loss stalls or rises, the
network is overfitting, and you stop. For a chess engine there is also a
far more honest test that sidesteps the metric entirely — *play games*
(file 14) — since the only thing that ultimately matters is whether the
new net wins, not whether its held-out loss is prettier.

**Leakage** is subtler and more dangerous: it is when information from the
test set sneaks into training, making the held-out score a lie. In chess
data the classic leak is **duplicate or near-duplicate positions**. The
same position can arise in thousands of games (every game passes through
the starting position, common openings recur constantly). If the identical
position appears in both training and held-out data, the network can
"memorize" it and post a falsely good held-out score. The defense is
**deduplication by position key** — hash each position (a Zobrist-style
key, file 04) and ensure no position, or no position too similar, straddles
the train/test split. Related discipline: skip the first several "book
plies" of each game (the opening moves are shared across huge numbers of
games and add coverage of nothing), and balance the corpus by game phase
and piece count so the network is not swamped by, say, opening positions.

## 8. The label-quality lesson: bounds are not values

We close with the deepest and least obvious lesson in the whole file,
because it is the kind of thing that silently corrupts a training pipeline
and it is entirely non-obvious until it bites you.

Recall from files 05 and 10 that alpha-beta search does not always return
a position's *exact* value. When a search runs inside a narrow window (an
aspiration window, or the zero-width windows of the pruning cascade), and
the true value lies *outside* that window, the search returns only a
**bound** — "the value is *at least* this" (a fail-high, lower bound) or
"*at most* this" (a fail-low, upper bound) — not the value itself. The
search deliberately stopped looking the moment it proved the bound,
because that was all it needed to prune.

Now suppose you build training data by having your *own* engine search
each position under a hard node limit, and you record whatever number the
search returned. A large fraction of those numbers will be **bounds, not
values.** A label that says "at least +200" is being fed to the network as
if it meant "exactly +200" — but the true value might be +600. The network
is being trained on systematically distorted targets, and the distortion
correlates with position type (sharp tactical positions fail high or low
far more often than quiet ones), so it is not even zero-mean noise that
washes out. It is *bias*.

ARCHON's response is a firm rule, recorded as a standing invariant in its
research log: labels produced by its own hard-node-limit (5,000-node)
datagen are treated as **quarantined** — they are excluded from scalar
regression losses precisely because they are frequently bounds. The
official Stockfish binpacks, by contrast, are known to be clean (they
trained world-class networks), and so are trusted. The general principle:
**a training label is only as meaningful as the process that produced it,
and a search score is not automatically a position value — check whether
it is a bound before you regress onto it.**

That, more than any loss-function choice, is the difference between a
pipeline that learns and one that quietly teaches the network the wrong
lesson. The team's deep-value labeling plan is built around exactly this
concern: use deeper, more exhaustive searches — a tiered protocol — to
produce labels that are *values*, not bounds, on the positions where it
matters.

## What you now know

- NNUE training is **supervised regression**: nudge weights via stochastic
  gradient descent so the network's output matches a per-position label.
- Labels come from **teacher distillation**, **self-play WDL outcomes**, or
  a **λ-blend** of the two; the **sigmoid** with a `scale` constant bridges
  centipawns and win probabilities, and the label scale must match the
  trainer's own formula end to end.
- ARCHON's **B4 finding**: data *volume* dominated everything. Failed nets
  trained on 25–265M positions and 0.3–3 epochs; winners trained on
  **billions** of presentations and beat net-0 immediately. Width scaled
  with data and saturated at accumulator width 2048.
- **Two-stage training** — broad base, then a short high-quality finetune —
  works, with stage-2 gains **proportional to label quality** and
  plateauing after ~2 hops.
- Hygiene means **held-out sets**, guarding against **leakage** via
  **dedup by position key**, and — for a chess engine — trusting *games*
  over held-out loss.
- The **label-quality lesson**: a search score under a narrow window is
  often a **bound, not a value**; ARCHON quarantines its own
  node-limited labels from scalar losses because bounds systematically
  distort the target.

Training turns data into a net. But every claim in this file — "+282 Elo,"
"beats net-0 decisively," "+73 from correction history" — rests on a
question we have deferred long enough: how do you actually *know* one
version is stronger than another, when chess games are so noisy? Before we
get there we need one more piece of the engine: how it talks to the outside
world and manages its clock, which is also where testing's determinism
requirements come from. Continue to `13-time-management-and-uci.md`.

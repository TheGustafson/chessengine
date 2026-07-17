# 14 — Testing and statistics: how you actually know it's stronger

Every "+X Elo" claim in this series has been an IOU. This file pays them.
The uncomfortable truth of engine development is that **you cannot tell,
by watching, whether a change made your engine stronger.** Chess results
are so noisy that your intuition is not just unreliable — it is actively
misleading, confidently wrong in a way that will waste months if you trust
it. The entire discipline of engine testing exists to replace "it looks
stronger" with a number and an error bar. This file builds that discipline
from first principles: why the noise is so bad, how Elo and its error bars
work, the sequential test (SPRT) that reaches a verdict efficiently, the
tricks that cut variance, and the parameter-tuning method (SPSA) that
squeezes gains out of the same noisy signal.

The owning design document is `design/14-performance-testing.md`.

## 1. Why "it looks stronger" is a lie

Start with the source of the noise: **draws.** Between two engines of
similar strength, most games are drawn. A tiny genuine edge — say your
change is worth a real +5 Elo — shifts the *outcome distribution* only
slightly: maybe you win 21% and lose 20% instead of a symmetric 20.5%
each, with 59% draws. That 1-percentage-point difference in win rate is
completely swamped, over any short run, by luck.

Put concrete numbers on it. Suppose you play 20 games and win 11, lose 9.
You feel great — a winning score! But 11–9 out of 20 is *entirely
consistent* with the two engines being exactly equal; you would see a
result at least that lopsided by pure chance roughly a third of the time
even if your change did nothing. Play another 20 and you might go 8–12. The
swing between "obviously better" and "obviously worse" across two
20-game blocks, with a completely unchanged engine, is larger than almost
any real single improvement. This is why watching a handful of games and
concluding anything is the cardinal sin of engine development.

The rule of thumb the field converged on: differences smaller than roughly
**15 Elo are invisible to the naked eye**, no matter how many games you
casually watch, because the variance dwarfs the signal. And most real
improvements *are* smaller than 15 Elo. Progress in a mature engine is a
long series of +2, +3, +5 Elo changes, each individually below the noise
floor of informal observation. You need statistics not as a formality but
because there is *no other way to see the signal at all.*

## 2. Elo as a logistic model

Before we can measure a difference we need to define one. **Elo** is a
model that turns "strength" into "expected score." If two players differ by
`Δ` Elo, the model predicts the stronger one's expected score (counting a
win as 1, a draw as ½) as:

```
   expected_score = 1 / (1 + 10^(−Δ / 400))
```

That is a **logistic** (sigmoid) curve — the same shape that bridged
centipawns and win probability in file 12, which is not a coincidence:
both are modeling "advantage → probability of winning." The `400` is a
convention that sets the scale: a 400-Elo gap predicts a ~91% score, a
0-Elo gap predicts exactly 50%.

Run it backward and you can *estimate* an Elo difference from a match
result. If your engine scored a fraction `s` of the available points, the
implied Elo difference is:

```
   Elo = −400 · log10(1/s − 1)
```

A score of `s = 0.53` (53%) implies about +21 Elo. A score of `0.50` is
exactly 0. This is how a match result becomes an Elo number.

### 2.1 The error bar is the whole story

But that Elo estimate is meaningless without its **error bar**, because
`s` itself is a noisy sample. The uncertainty shrinks as the square root of
the number of games — the universal `1/√N` law of averaging noisy
measurements. ARCHON's design doc records the honest figures (approximate,
for draw-heavy short time controls, 95% confidence):

```
   100 games   →  ±44 Elo
   1,000 games →  ±14 Elo
   10,000 games →  ±4.4 Elo
```

Read those and let them sink in. At **100 games**, your ±44 error bar is
*wider than almost any real improvement you will ever make.* At **200
games** — a run that takes real time to play — the bar is still about ±30
Elo, which is why the design doc states flatly that "a 200-game eyeball run
proves nothing inside ±30 Elo." To resolve a +2 Elo change you need on the
order of **tens of thousands of games**. This `1/√N` wall is the central
economic fact of engine testing: precision is *expensive*, quadratically
so, and the whole art is getting a reliable verdict while playing as few
games as possible.

## 3. SPRT: stop as soon as you know

Here is the key insight that makes testing affordable. If you fix the
number of games in advance — "I'll play exactly 10,000 and then decide" —
you are wasting games in two directions. When a change is *great*, you knew
after 2,000 games; the other 8,000 just confirmed it. When a change is
*terrible*, likewise. Only for changes right at the boundary do you need the
full sample. A fixed-N test spends its worst-case cost on *every* test.

The **Sequential Probability Ratio Test (SPRT)** fixes this. It is a test
that looks at the results *as they arrive* and stops the moment it has
enough evidence to decide — which, for clearly-good or clearly-bad changes,
is far sooner than a fixed-N test. On average it reaches a verdict in
roughly **a third of the games** a fixed-N test would need for the same
confidence.

### 3.1 The two hypotheses

SPRT does not ask "how strong is this change?" (an estimation question with
a `1/√N` cost). It asks a cheaper *decision* question: "is this change at
least a little bit good, or is it not?" You frame it as two hypotheses:

- **H0** (the null): the change is worth `elo0` — typically `0` Elo, i.e.
  "no improvement."
- **H1** (the alternative): the change is worth `elo1` — typically a small
  positive bound like `+5` Elo, i.e. "a real, if modest, gain."

ARCHON's standard gate uses `elo0 = 0`, `elo1 = 5` (in *normalized* Elo, a
draw-rate-adjusted variant), with error rates `α = β = 0.05` — a 5% chance
of accepting a bad change, a 5% chance of rejecting a good one.

### 3.2 The log-likelihood ratio

As each game-pair comes in, SPRT updates a running number, the
**log-likelihood ratio (LLR)**. Intuitively, the LLR is a running tally of
"how much more consistent is the evidence so far with H1 than with H0."
Each game nudges it: results that look like a real gain push it up, results
that look like no gain push it down. The test stops when the LLR crosses
one of two thresholds derived from `α` and `β`:

```
   LLR ≥ ln((1−β)/α) ≈ +2.94   →  accept H1 (the change is good — ship it)
   LLR ≤ ln(β/(1−α)) ≈ −2.94   →  accept H0 (the change is not good — reject)
   in between                  →  keep playing
```

The thresholds `±2.94` come straight out of the chosen error rates
(`ln(19) ≈ 2.94` for `α = β = 0.05`). You do not need to derive them; you
need the intuition: **the LLR is an evidence accumulator, and you stop when
the evidence is decisive in either direction.** No fixed sample size —
the sample size is whatever the evidence demands.

### 3.3 A real ARCHON example

This is not abstract. When ARCHON wired in correction history (file 09),
the equal-time SPRT **passed H1 in 208 games**, measuring +72.9 ± 16.2 Elo
with an LLR of 2.95 crossing the +2.94 bar. Two things to notice. First,
208 games is *tiny* — a fixed-N test targeting the same confidence for a
near-boundary change would have played thousands; the change was so
strongly positive that SPRT nailed it early. Second, the ±16.2 error bar is
still wide (that is 208 games' worth of precision), but it does not matter,
because the *decision* — "is this ≥ 0?" — was made with total confidence.
SPRT gave a fast, correct verdict without pretending to know the exact Elo.

## 4. Variance reduction: getting more signal per game

Since precision costs quadratically, anything that reduces variance *per
game* is enormously valuable — it is a `1/√N`-buster. Two techniques are
standard, and ARCHON uses both.

### 4.1 Paired openings and color swap

If every game started from the standard position, the two engines would
draw almost all of them and you would learn very little per game. Instead,
matches start from a **book** of opening positions, and — the key trick —
each opening is played **twice, with colors reversed** (`-repeat` in the
harness). Engine A gets White from opening #37, then Engine B gets White
from the same opening #37.

Why does this cut variance? Because it *controls for the opening.* If
opening #37 happens to be a great position for White, both engines get to
exploit it once, so the opening's inherent bias cancels out of the *pair's*
result. What is left is the difference attributable to the *engines*, not
the luck of the draw. Statisticians call this **pentanomial** scoring: you
score the game-*pair* (which can go 0, ½, 1, 1½, or 2 points for one side —
five outcomes) rather than individual games, and the pairing removes a
large chunk of the noise. This is why SPRT here counts game-pairs, not
games.

### 4.2 Unbalanced opening books (the UHO idea)

The second trick is counterintuitive at first. You might think a *balanced*
opening book — dead-equal starting positions — is the fairest test. It is
the *worst* test, because balanced positions produce the most **draws**, and
draws carry almost no information about which engine is stronger.

So the field deliberately uses **unbalanced** opening books: collections of
positions where one side already has a real advantage. ARCHON tests with a
UHO book — "Unbalanced Human Openings" — of exactly this kind. Starting from
a position where White is, say, +0.7, forces *decisive* games: the stronger
engine converts the advantage more often and squanders it less, so wins and
losses (the informative outcomes) replace draws (the uninformative ones).
Combined with color-swapped pairs (so *both* engines must play the winning
and losing sides), unbalanced books dramatically **raise the sensitivity**
of the test — more Elo resolution per game played. It feels like cheating to
start from lopsided positions, but it is precisely correct: you are not
measuring who is better *from equality*, you are measuring who plays chess
better, and lopsided positions surface that faster.

## 5. Fixed-node screens versus time-control gates: the NPS confound

File 13 introduced fixed-node testing; here is where it becomes a
methodology, and where it can bite you.

Recall the strength decomposition: an engine's playing strength factors
into roughly three near-independent axes —

```
   Elo ≈ f(net quality, search quality, inference speed)
```

— and the deadliest testing mistake is *measuring one axis through a test
that silently moves another.* A **fixed-node** match (both engines search
the same number of nodes per move) deliberately removes the *speed* axis:
neither engine benefits from being faster, so the result reflects **net +
search quality per node**. This makes fixed-node the right microscope for
research questions like "is network A a better evaluator than network B?"
It is also immune to a busy machine — since no clock is involved, every CPU
core can play games at full tilt without corrupting the result.

But a **real time-control** match blends *all three* axes, including speed —
and a network that evaluates better per node but takes longer to run will
search *fewer* nodes in the same wall-clock time. The two tests can
disagree, and when they do, the disagreement is the truth, not a paradox.

ARCHON lived exactly this. Its D40 FullThreats network (file 11) is
genuinely stronger per node — a fixed-node screen measured **+32.5 ± 17.3
Elo** over the best classic network. But its richer features make its
evaluation *slower*, so at **equal time** it *lost* — a promotion attempt
came back **−57.9 ± 12.6 Elo**. Same network, opposite verdicts, both
correct: it wins per node and loses on the clock, because the inference tax
outweighs the per-node gain (it would need roughly 1.7–2× the speed to
convert its node-quality edge into a clock win). The lesson is a rule:
**fixed-node is a research instrument; a real time-control match is the
ship gate.** You screen candidates fast and cheap with fixed-node, but you
never let a fixed-node win *ship* — the clock has the final say, because the
clock is what a real game runs on.

## 6. SPSA: tuning parameters against a noisy signal

The last piece. Files 09, 10, and 13 are full of magic constants —
reduction divisors, futility margins, time-management scalars, history
weights. Where do their *values* come from? They are **tuned** — and tuning
against a signal as noisy as chess results needs a special method, because
you cannot compute a gradient of "Elo" the way you would for a smooth
function.

**SPSA** — Simultaneous Perturbation Stochastic Approximation — is the
method. The core idea is remarkably clever, and it works *because* it makes
peace with the noise rather than fighting it.

You have a vector of parameters `θ` you want to optimize (say 20 search
constants). You want the gradient — which direction to move `θ` to gain
Elo — but you cannot measure Elo precisely, and evaluating 20 partial
derivatives separately would be impossibly expensive. SPSA's trick:

1. **Perturb *all* parameters at once, randomly.** Pick a random sign
   (`+1` or `−1`, a "Rademacher" perturbation) for each parameter and step
   the whole vector by `+c·δ` in that random direction, and separately by
   `−c·δ` in the opposite direction. This gives two parameter sets, `θ⁺`
   and `θ⁻`, differing in *every* coordinate simultaneously.
2. **Play a small batch of games** with `θ⁺` on one side and `θ⁻` on the
   other. The result `z` — the score difference between the two — is a
   *noisy* measurement of which perturbed vector is better.
3. **Nudge every parameter** by `z` times its own random sign. If `θ⁺` won,
   push each parameter a little in the direction its sign pointed; if it
   lost, push the opposite way.

Why does perturbing everything at once and using one noisy scalar work?
Because over many iterations, each parameter's random signs are
**uncorrelated** with the others', so the useful signal for parameter *i*
accumulates while the "noise" contributed by the *other* parameters'
perturbations averages to zero. You get an unbiased gradient estimate for
the *whole vector* from just **two** game batches per step, instead of one
per parameter. And because each step only needs to know "which side scored
better," it tolerates enormous per-batch noise — the noise washes out across
hundreds of iterations. ARCHON's tuner (`tools/spsa/spsa_tune.py`) runs
exactly this: paired Rademacher perturbations, fishtest-style step-size
schedules (the `a_k`/`c_k` sequences that shrink the step and perturbation
over time so the process converges), and — crucially — **fixed-node**
mini-matches on the UHO book, so tuning can run at high concurrency on a
busy machine without clock corruption.

Two disciplines from the earlier sections carry straight over. First, tune
on the **shipped network**, not a throwaway: margins expressed in centipawns
re-mistune when the evaluation's scale or distribution changes, so tuning
against one net and then swapping nets wastes the work. Second, a
fixed-node-tuned parameter set still must clear an **equal-time SPRT ship
gate** before it becomes the default — the fixed-node/real-time split of §5
applies to tuning exactly as it applies to networks.

## 7. The hierarchy, one more time

Zoom out and the whole methodology is a hierarchy of questions, each with
its proper tool — and the most common mistake is using the wrong tool for
the question:

- **"Is it stronger?"** → SPRT games. The only tool that answers this. Never
  a node count.
- **"Did search behavior change / did speed change?"** → the bench
  fingerprint and fixed-node timing (file 13). Never a strength claim.
- **"Did this one function get faster?"** → a micro-benchmark. Never a
  whole-engine claim.

A movegen speedup that shows up in a micro-benchmark must be re-confirmed
in the engine's node rate before it is claimed; a search change that alters
the bench node count is *functional* and must go to SPRT no matter how
harmless it looks. Each tier answers exactly one kind of question, and the
discipline is refusing to let a cheap-but-wrong tier answer an expensive
question. That refusal — the willingness to play 200 games and conclude
*nothing*, and then play the SPRT — is what separates real progress from a
random walk that feels like progress.

## What you now know

- Chess results are **draw-dominated and high-variance**: differences under
  ~15 Elo are invisible to informal observation, and most real improvements
  are that small — so statistics is the only way to see the signal.
- **Elo** is a logistic model of expected score; its **error bar** shrinks as
  `1/√N` (±44 Elo at 100 games, ±14 at 1,000, ±4.4 at 10,000), making
  precision quadratically expensive.
- **SPRT** decides between H0 (`elo0`) and H1 (`elo1`) by accumulating a
  **log-likelihood ratio** and stopping the instant it crosses ±2.94,
  reaching a verdict in ~⅓ the games of a fixed-N test — ARCHON's
  correction-history gate passed H1 in **208 games**.
- **Paired color-swapped openings** cancel opening bias (pentanomial
  scoring), and **unbalanced (UHO) books** cut draws to *raise* the test's
  sensitivity.
- **Fixed-node** screens measure net + search quality per node (the research
  microscope); **real-time** matches are the ship gate — and they can
  disagree, as ARCHON's D40 net winning per node but losing on the clock
  shows.
- **SPSA** tunes whole parameter vectors from two noisy game batches per
  step by perturbing every coordinate at once with random signs — and the
  tuned values still face an equal-time SPRT before shipping.

You now understand not just how the engine works, but how anyone *knows* it
works. The final file steps back to survey where ARCHON's research actually
stands — the frontier where all of this converges — and what you, having
read this far, are now equipped to do. Continue to
`15-the-research-frontier.md`.

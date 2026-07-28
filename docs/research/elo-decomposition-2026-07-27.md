# Where the 191 Elo to SF18 actually lives — and a pre-registered prediction

2026-07-27. Written **before** the equal-time / equal-nodes rematch runs, so
the prediction below is falsifiable rather than fitted after the fact.

## The inputs (both already measured, neither by me)

External anchors, 100 games, 10+0.1 (`CLAUDE.md`):

| opponent | ARCHON |
|---|---|
| SF16 | −78 ± 18 |
| SF18 | −191 ± 23 |
| Integral v6 | −70 ± 22 |

NPS, same-round pinned matched bench (`perf-study-7`, §1.1):

| engine | NPS | vs ARCHON |
|---|---:|---:|
| ARCHON | 1.125 M | — |
| SF18 | 1.558 M | 1.33–1.38× |
| SF16 | 2.264 M | 2.01× |

The SF18 deficit grows with depth to ~1.44–1.48× on the movetime instrument,
so 1.33× is the floor and ~1.45× the working figure.

## The decomposition

A node-count ratio converts to Elo through the speed-doubling constant, which
this project measured for itself: **~120 Elo per doubling** (exp-0154, the
1.51× ≈ +73 Elo calibration).

```
SF18:  log2(1.40) = 0.485 doublings  ->  ~58 Elo of pure speed handicap
SF16:  log2(2.01) = 1.007 doublings  ->  ~121 Elo of pure speed handicap
```

Subtract the speed handicap from the observed gap to get the per-node
decision-quality difference:

| opponent | observed | speed | **per-node quality** |
|---|---:|---:|---:|
| SF18 | −191 | −58 | **−133** |
| SF16 | −78 | −121 | **+43** |

Two things fall out, and the second is the one that matters.

**We are already better than SF16 per node, by ~43 Elo.** We overcome a full
doubling of speed deficit and still finish only 78 Elo behind. That is the net
doing its job, and it is a real achievement rather than a rounding artifact.

**Against SF18, speed is roughly 30% of the gap and decision quality is 70%.**
The implied claim — SF18 is ~176 Elo better per node than SF16 — is consistent
with the outside world, where SF18 is well ahead of SF16 at equal time *and*
slower per node, so its equal-node margin must be larger still.

## What this means for tunables

**RETRACTED, 2026-07-27.** This section originally opened "search-parameter
tuning has a bounded ceiling here, and the bound is not large." That ceiling
was never derived. I computed a ~58 Elo bound for the SPEED pool, then wrote
off the per-node pool as untunable *because I had already misattributed it to
the net* (see the correction below). Once the attribution is corrected the
ceiling argument collapses: the per-node term is exactly what configuration
tuning addresses.

The evidence for LARGE tunable headroom was already in this document:

- restoring Stockfish parity measured neutral-or-worse **four times out of
  four** — which I described as "what a joint optimum looks like when it has
  never been jointly optimised for this engine" and then failed to act on;
- **42 eval-denominated constants** transcribed from an engine whose eval
  frame is 1.56× ours, so every one of them sits in a different regime here;
- **~46% of tunables have never been in any campaign**;
- **time management was demonstrably mis-set on the first family examined
  closely** (9.98s of a 10s bank spent before move 33; 47.3% of moves under one
  second against SF18's 25.2%). If the first knob family inspected was wrong,
  the prior that the other ~250 are collectively right is low.

What remains true from the original section is only the narrow part: the SPEED
pool is ~58 Elo and no parameter set captures all of it, so prefer changes that
improve the CHOICE at a node over changes that only cut node count.

- Closing the *entire* NPS gap to SF18 buys ~58 Elo, and no realistic set of
  parameter changes closes all of it. Node-saving tunables are worth tens of
  Elo, not hundreds.
- The other ~133 Elo (revised to ~98 below) is per-node decision quality.

**CORRECTION, 2026-07-27, prompted by the user.** An earlier version of this
line read "so that is mostly **eval**, i.e. the net". That was an inference by
elimination — search structure checks out, therefore the residual must be the
net — and it was stated as a finding when it is not one. **There is no
measurement here showing our net is worse than SF18's**, and the user reports
that every net-to-net metric they have puts d47v near parity.

The evidence in this very document points the same way as theirs: over 7,686
move pairs from a 120-game match, our eval's median disagreement with SF18's
in a common frame is **−0.02 pawns**. That is a parity result, and it is
unselected, so it is the trustworthy kind.

The honest reading of a per-node deficit that is **flat across a 32× budget
range** is narrower than I wrote: it says the deficit is
*depth-independent*, and eval is only one depth-independent cause. Mistuned
search constants are another, and they fit better here, because:

- the constants were transcribed from Stockfish, where they are co-tuned with
  Stockfish's **eval frame** (ours measures 0.64× theirs) and Stockfish's
  mechanisms;
- restoring Stockfish parity mechanism-by-mechanism measured neutral-or-worse
  four times out of four, which is what you expect when a configuration is a
  joint optimum that has never been jointly optimised for *this* engine;
- ~46% of our tunables have never been in any SPSA campaign.

So the residual is best read as **configuration tuning of our own engine**,
not net quality. That is a more actionable conclusion and it raises the value
of the tuning work rather than lowering it.

Which is exactly what the project's own history says: net-0 was +696 over
PSQT, the width ladder and the data-volume finding moved hundreds, and D48X is
training right now. The tunables are worth grinding — tens of Elo are still
Elo, and they are what can move today — but they should not be mistaken for
the main lever, and no tuning campaign should be justified by the 191.

It also re-ranks my own queue. Anything whose entire benefit is fewer nodes is
competing for a ~58 Elo pool that we can only partly capture. Anything that
improves the *choice* at a given node competes for a ~133 Elo pool. Prefer the
latter when both are available at similar cost.

## The Stockfish-parity well is dry, and that is now evidence rather than a mood

Two independent lines of work converged on this within a few hours, which is
why it is worth stating as a finding rather than an impression.

**The transcription is faithful almost everywhere.** Read side by side against
`eca43a97`: the movepick stage machine, `select()`, every quiet and capture
scoring term, every history table dimension and bound, the whole LMR reduction
assembly and its adjustment order, the `lmrDivisor` table, the move-loop
futility/LMP/SEE block, hindsight depth adjustment, `correction_value`, and 42
eval-denominated constants. All match. There is no pile of transcription
errors waiting to be found.

**And where we DO deviate, restoring parity has measured neutral-or-worse, four
times out of four** (Fable's toggles, all wired at identity and screened
net-live):

| restored SF mechanism | result |
|---|---|
| `follow_pv` wired (was dead code, two live readers) | −0.09±0.10 cp at **+12.2% nodes** ≈ −15 Elo net |
| qsearch upcoming-repetition | +0.23 cp deep tier |
| ttPv rebate on TT miss | +0.34 cp, +5.3% nodes |
| ProbCut in singular verification | de facto no-op in both engines |

The `follow_pv` case is the instructive one. It looked like the find of the
session — genuinely dead code with two live consumers — and its exploratory
read was −0.40±0.26. At full corpus it shrank fourfold to −0.09±0.10 and its
node cost sank it. That is regression to the mean behaving exactly as the
playbook says it will, and it was correctly not promoted.

The mechanism behind all four: **Stockfish's constants are co-tuned with
Stockfish's mechanisms.** Importing a mechanism without importing the tuning
that surrounds it imports the cost and not the benefit. Any of these could
still pay inside a joint retune; none of them pays alone.

The strategic consequence is direct. Further Stockfish-parity work has a poor
expected value, and the case for ARCHON originals is now empirical rather than
aesthetic: parity is measurably exhausted, so the remaining Elo has to come
from things Stockfish does not do, or from tuning our own configuration as a
whole rather than importing pieces of theirs.

---

## The pre-registered prediction

The rematch plays the same match twice — equal time and equal nodes. Equal
nodes removes the speed term by fiat, so:

> **Predicted equal-nodes result: ARCHON −133 Elo vs SF18, with a working
> range of −115 to −150** (the range reflects 1.33× vs 1.48× on the NPS
> deficit and the ±23 on the anchor).

Falsification, stated in advance:

- **Better than −100** — speed was a bigger share than study 7's NPS implies,
  or the equal-node protocol flatters us (e.g. our node accounting differs from
  Stockfish's, which would be an instrument bug worth finding). Either way the
  node-saving branch is worth more than this note claims.
- **Worse than −170** — speed contributes almost nothing, the ~120 Elo per
  doubling constant does not transfer to this comparison, and NPS work is close
  to worthless for Elo. Retire it from the queue.
- **Anywhere in the predicted band** — the SPEED/quality split holds. Note
  what that does and does not license: it confirms the size of the per-node
  term, not its *cause*. Attributing that term to eval rather than to
  configuration tuning needs a separate measurement, which this leg does not
  provide. (See the correction above.)

## Second instrument, arrived before the rematch: it agrees on the shape and
## disagrees on the size

A budget sweep with an SF18 reference row, 1500 positions, cp at the 64k tier:

| nodes | 3125 | 6250 | 12500 | 25000 | 50000 | 100000 |
|---|---:|---:|---:|---:|---:|---:|
| ARCHON | 10.21 | 8.27 | 6.23 | 5.52 | 4.13 | 3.82 |
| SF18 | 9.34 | 7.56 | 5.24 | 4.17 | 3.60 | 2.62 |
| gap | 0.87 | 0.71 | 0.99 | 1.35 | 0.53 | 1.20 |

**The gap does not widen with budget** — 0.86 mean over the first three
budgets, 1.03 over the last three, across a 32× range. A gap that stays flat
as search deepens is an eval difference; one that widens is a search
difference. That is an independent confirmation of the same conclusion the Elo
arithmetic reached, from a completely different instrument.

On the size, the two disagree by about a factor of two. At the rough ~60 Elo
per cp derived above, a 0.94 cp gap is ~56 Elo, against the ~133 Elo the
anchors and the NPS table imply. Three candidate reconciliations, none yet
settled:

- The Elo-per-cp constant is an order of magnitude, not a number. Estimated
  across the sweep it ranges 60–390 depending on which doubling you take.
- The oracle labels are Stockfish's, so this comparison **flatters SF18** and
  the measured 0.94 is if anything an over-estimate of the true gap. That
  widens the discrepancy rather than closing it.
- Mean cp loss over a generic corpus may simply not be what decides games.
  Time management, the position mix of real play, and the distribution of
  errors rather than their mean are all outside this instrument.

Noise, stated honestly: each cell is an unpaired mean over a heavy-tailed
distribution at n=1500, so roughly ±0.5 cp, and the gap roughly ±0.7. "Flat"
is consistent with the data but a modest trend cannot be excluded. The
reference row is now **paired** against the control per budget, which cancels
position difficulty and should cut that noise substantially on the re-run.

---

## UPDATE: the timed leg landed, and one of my inputs was stale

**Measured: −143.8 ± 27.0 Elo, 30.42% over 120 games (+7 =59 −54)**, 10+0.1,
UHO book, PGO binary, d47v net, concurrency 3 with everything else on the box
niced.

The anchor this whole decomposition was built on — SF18 −191 ± 23 — is from
**2026-07-16** and a different net (`stage2-v1-s3-2048-farseer-120sb`). Since
then: the d47v net, the S0 search program, PGO as standard, the perf-impl-1
wins, and upcoming-repetition. −143.8 vs −191 is 47 Elo of apparent progress
at 1.3σ, so it is not a *significant* difference in isolation, but it is the
current number and the 2026-07-16 figure should not be quoted as live again.
**The anchors in `CLAUDE.md` need refreshing.**

Redoing the arithmetic with the measured value:

| | old (stale anchor) | measured |
|---|---:|---:|
| observed gap | −191 | **−143.8** |
| speed term (1.40× NPS) | −58 | −58 |
| **per-node quality** | −133 | **−86** |

**This reconciles the two instruments.** The budget sweep put the per-node gap
at 0.94 cp, which at the rough ~60 Elo per cp is ~56 Elo. Against the old
−133 that was a factor-of-2.4 disagreement I recorded but could not explain;
against −86 it is a factor of 1.5, comfortably inside the uncertainty on a
constant I only know to an order of magnitude. The discrepancy was a stale
input, not a broken instrument.

**Revised prediction for the equal-nodes leg: ≈ −86, range −70 to −105.** The
original −133 (range −115 to −150) is left standing above rather than edited
out: it was pre-registered in good faith from the best number available, and
the failure mode it illustrates — a stale anchor quietly propagating through
an otherwise sound derivation — is worth more as a visible record than as a
tidy document. The *method* is unchanged and still falsifiable: equal-nodes
should equal the timed result plus the speed term.

### From the timed leg's forensics

- **Draw-heavy: 59 of 120.** We are hard to beat and struggle to win — 7 wins.
- Loss taxonomy is an exact 50/50 tactical/grind split, 52% endgame.
- **Eval bias over all 7,686 move pairs: median −0.02 pawns.** Across a full
  match, in a common frame, our eval agrees with SF18's on average. The
  opening is the exception at −1.09 (we are more pessimistic out of book).
  This is the unselected number and therefore the trustworthy one.
- Depth at the losing move: ours 17, theirs 16. We are not searching shallower
  in plies.
- The "−5.04 pawns disagreement at the losing move" headline is **partly an
  artefact of the selection rule** and should not be quoted alone: the point
  of no return is *detected* by our own eval trail crossing a threshold, so
  the position is chosen partly for having a very negative eval of ours. The
  unselected all-pairs median above is the honest version.

---

One caveat I want on the record before the number arrives: equal-nodes matches
are not a clean instrument for a speed decomposition, because the two engines
count nodes differently and neither counts the *work per node* that the other
does. Treat a result inside the band as consistent with the decomposition, not
as proof of it, and treat a result outside the band as a reason to check the
instrument before rewriting the theory.

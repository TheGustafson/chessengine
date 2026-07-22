# Hybrid alpha-beta / PUCT search (one rich head, two search regimes)

*User idea, 2026-07-22 (overnight): output a richer policy head that serves
alpha-beta pruning, then at some depth or trigger switch to MCTS/PUCT and
explore the tree stochastically in the other style.*

## The direction question: which end of the tree gets PUCT?

The proposal as stated puts AB on top and PUCT below a trigger. The evidence
from every strong hybrid and the known failure modes says **invert it — PUCT
on top, AB at the bottom**:

- **Mean backup dilutes refutations.** MCTS backs up averages; one refuting
  line must outvote many optimistic visits before the parent flips. Minimax
  backs it up in ONE visit. The bottom of the tree is the most *tactical*
  region — exactly where PUCT is weakest and where our qsearch/SEE/futility
  stack is hyper-optimized. Handing the leaves to PUCT trades our best
  machinery for MCTS's worst regime.
- **Production precedent**: Komodo/Dragon MCTS — the only strong shipped
  hybrid — is root-PUCT with *AB probes as the visit evaluation*. Research
  agrees (Baier & Winands, MCTS-minimax hybrids): embedding shallow minimax
  INTO MCTS fixes its tactics; the reverse direction has no wins on record.
- AB is *weakest at the root* (aspiration re-search thrash, best-move
  instability between iterations) and that is where value-averaging and
  principled exploration are safest: strategic choice among a few plausible
  moves, each backed by a tactically-sound AB value.

**Steelman of the trigger version**: if the trigger is a REGIME, not a depth
— quiet position ∧ flat policy distribution ∧ high value-uncertainty (σ) —
then stochastic *widening* at such nodes is defensible: those are the nodes
where mean-vs-max matters least and AB's ordering edge is smallest. Kept as
H2 below, reframed as "exploration inside AB" rather than a search-engine
swap.

## What the idea gets right

1. **The policy head IS the missing PUCT ingredient.** PUCT without priors
   is UCT and dies on chess branching factor; d47c proves our priors are
   learnable and riding along free at inference.
2. **One richer head serves both regimes from one forward pass**: move
   priors (AB ordering + PUCT prior), WDL/σ (uncertainty → exploration
   targeting AND safer pruning margins), optionally calibrated P(best) for
   direct pruning thresholds. `wdl-head-design-2026-07-20.md` +
   `rich-multihead-net-2026-07-21.md` already spec most of this head.
3. **AB+LMR is already a soft best-first search.** ~80% of "PUCT-ness" —
   node allocation proportional to prior × promise — imports into AB with
   zero structural risk as policy-aware LMR and prior-seeded history
   (policy roadmap §6; this IS the design/19 learned-allocation program).
   The hybrid is the *radical* version of a program we already have pinned.

## The killer application: datagen, not match play

Root-PUCT **visit distributions are AlphaZero-style soft policy targets** —
they break the one-hot supervision ceiling (roadmap §2 soft targets, no
multiPV teacher needed) and give principled exploration diversity
(Dirichlet-at-root) for the self-play flywheel. The searcher that GENERATES
data does not have to be the searcher that PLAYS matches: even if the hybrid
never wins a head-to-head, it can be the data engine that trains the nets
that do. This is the strongest single reason to build H1/H3.

## Constraints that bite (pinned decisions)

- **Determinism (D24 bench trailer; RNG jitter already declined vs
  Reckless)**: all "stochastic" exploration must be Zobrist-seeded
  deterministic noise — keyed-hash pseudo-randomness gives AZ-style
  diversity with bit-reproducible benches. Datagen mode may relax to
  logged-seed-per-game.
- **Backup rule**: pure mean fails tactically even at the root — use a
  max-mixing backup (visit-annealed mean→max, or power-mean p>1).
- **CPU visit economics**: a "visit" that costs a reduced-depth AB probe is
  ~10³–10⁴ nodes; root visit budgets are hundreds, not Leela's 10⁵. The
  allocator must earn its keep at tiny visit counts (priors do the heavy
  lifting).

## Experiment ladder (sequenced behind the value foundation + design/19 S-items)

- **H0 — policy-aware AB allocation** (pure AB, no structure change):
  policy-scaled LMR, prior-seeded MovePicker history. Roadmap §6's first
  consumer; the cheap 80%. Runs as soon as a credible policy head ships.
- **H1 — root-PUCT over AB probes** (Komodo shape): root allocator with
  head priors; visit = zero-window/reduced-depth AB probe from the existing
  RootMoves machinery; max-mixed backup. Gate on best-move stability,
  hard-position suite, and Elo; MultiPV quality upgrade falls out free.
- **H2 — regime-triggered widening** (the user's trigger, refined): at
  quiet ∧ flat-policy ∧ high-σ nodes, deterministically-seeded widening
  (keep extra candidates past LMP, damp reductions). Needs the σ head.
- **H3 — datagen mode**: H1 + seeded Dirichlet-at-root + visit-distribution
  soft labels → the policy flywheel (feeds roadmap §2 soft targets). Build
  when the self-play era starts.

Prereqs: WDL/σ head (T9/D41 arc), policy calibration (a temperature on the
logits), H0 evidence that priors transfer into search at all.

**Prior source (user, 2026-07-22): the H1/H3 PUCT prior is the Lc0-SOFT
head (policy roadmap §2d head bank), not the one-hot head — and the
hybrid is where that head's full distribution earns its keep.** In AB a
prior only modulates mechanisms that work without it (value compressed by
overlap with history/LMR); in PUCT the prior IS the allocation mechanism
(`U ∝ P·√N/(1+n)`) — quality converts linearly to search efficiency, and
small visit budgets amplify it (too few visits for value to correct a bad
prior). Soft-CE calibration is load-bearing (exploration constant ↔ prior
temperature), and PUCT-visit targets → PUCT prior is on-policy use of the
data. Sequencing: build the soft head EARLY (cheap, during stage-2); AB
consumers (§6 menu) are the proving ground that de-risks H1.
Relates: `policy-training-roadmap-2026-07-22.md` §2/§6, `design/19`,
`rich-multihead-net-2026-07-21.md`, `wdl-head-design-2026-07-20.md`,
`arch-ablation-suite.md` (T9).

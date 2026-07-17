# 15 — The research frontier: where ARCHON is going, and what you can now do

You have built, in your head, a complete modern chess engine. Board
representation, move generation, state and repetition, alpha-beta with
iterative deepening and quiescence, a transposition table, move ordering
and history, the pruning cascade, a fast neural evaluation, its training,
the protocol, and the statistics that keep everyone honest. That is the
*settled* part of computer chess — the part where the answers are known and
the engineering is "merely" excellent.

This final file is about the *unsettled* part: ARCHON's actual research
program, which sits right at the edge of what anyone knows how to do. It is
a case study in engine R&D — not because ARCHON has won anything (it has
not; it is, by its own accounting, still hundreds of Elo below the very
top), but because the *shape* of its research is where the whole field is
heading, and because the discipline it uses to get there is worth as much
as any single result. We will look at four things: the provenance ledger,
the observation store and dual-teacher labeling, the learned-search-
allocation bet, and the honest open questions — then close with what you,
having read this far, are now equipped to do.

## 1. The provenance ledger: reproducibility as infrastructure

Start with something that sounds like bookkeeping and is actually the
foundation everything else rests on.

A research program that trains networks generates a combinatorial mess:
dozens of networks, each trained from some corpus, which was labeled by
some teacher, using some version of the code, and evaluated in some match
against some opponent producing some Elo number. Six months in, someone
asks a simple question — "which data was net-14 trained on, and was it
really stronger than net-9, and can we rebuild it?" — and if you have not
been disciplined, the honest answer is "we don't know," and the work is
scientifically worthless because it cannot be reproduced.

ARCHON's answer is the **experimental ledger** (`ledger/`, `design/18`).
It is deliberately humble technology: append-only JSONL files listing every
network, every corpus, and every experiment, driven by a small standard-
library-only Python tool. But it enforces two invariants that make it a
*provenance DAG* rather than a notebook:

- **Content-addressing.** A network's identity *is* the SHA-256 hash of its
  bytes. You cannot confuse two nets, cannot silently overwrite one, and can
  always prove which exact artifact a result refers to.
- **A regeneration recipe on every row.** Each entry records the git commit,
  the input hashes, and the exact command that produced the artifact. Any
  net's entire lineage — data → teacher → code → net → match result — is
  provable and, in principle, rebuildable from scratch.

The reason this belongs in a document about the research *frontier*, rather
than in a footnote, is that **the frontier is only reachable if you can
trust your own past results.** File 14 showed how expensive a single Elo
verdict is; the ledger is what stops you from paying that cost twice, or
from building a month of work on a comparison you can no longer verify. The
discipline it embodies — content-address your artifacts, record how to
regenerate them, never trust an un-provenanced number — generalizes far
beyond chess. It is what turns a pile of experiments into a science.

## 2. The observation store: capture the search, not just the label

Now the first genuinely forward-looking idea, and the one that reorganizes
everything after it.

Recall file 12's training pipeline. To train a value network you generate
positions, search each one, and write a line: `(position, score, result)`.
The moment you write that line, you **throw away** almost everything the
search just computed — the node counts, which moves it pruned and why, the
reductions it applied, how the best move changed with depth, which
alternatives it considered and rejected. You keep one number. And a deep
search is *expensive* to run and *cheap* to re-read — so throwing away its
byproducts means re-running it from scratch every time a future project
wants a signal you discarded.

ARCHON's **observation store** (`design/16`, `src/obs/`) inverts this. Its
guiding principle is a single sentence: **store expensive observations, not
just current labels.** Instead of `(position, score, result)`, it records
four layers per searched position — the full position state, the search
*context* (window, depth, node type), the search's *observations* (how the
score and best move evolved), and *per-move* observations (what happened to
each candidate: was it pruned, reduced, extended, what did it score). The
value label becomes just one cheap *derivation* from this rich record; so
do policy labels, uncertainty labels, and — the point — labels for signals
nobody has thought to train on yet.

The economics are the argument. Regenerating deep search costs core-hours;
deriving a new label from a stored observation costs milliseconds. If you
believe you will want *more than one kind of label* from your data over the
life of a research program — and you always do — then capturing the
observation once and deriving labels forever is dramatically cheaper than
re-searching for each new idea. This is the same lesson as the ledger,
one level up: **the expensive thing is generating information; arrange your
infrastructure so you generate it once.** ARCHON builds this behind a
feature flag so it never perturbs the shipping engine's search — the store
observes, it does not interfere, and the bench fingerprint of file 13 stays
put.

### 2.1 Dual-teacher labeling and information density

Two refinements sharpen the idea. First, **dual-teacher labeling**: label
each position with *two* strong engines (ARCHON uses a Reckless-based
primary and a Stockfish-18-based control) rather than one. Why two? Because
a single teacher's label is a single opinion, and you cannot tell a
confident label from a shaky one. With two, you can *measure agreement*.
ARCHON's audits found the teachers agree on the best move about 75% of the
time (a rank correlation of ~0.84) — and, crucially, that where they *do*
agree, the label is trustworthy, while where they disagree, the position is
genuinely ambiguous. That disagreement signal is itself training data: it
tells you which labels are soft.

The audits surfaced a concrete instance of the file-12 label-softness
problem: the teachers' *top move margin* was often small (a median of
13–16 centipawns separating the best move from the second), meaning the
single "best move" label is frequently a near-tie. The response is to train
on **per-move values** — the whole distribution of what each move is worth —
rather than a hard argmax "this one move is best." A soft target that
admits "these three moves are nearly equal" carries more honest information
than a hard one that pretends the top move is uniquely correct.

This leads to a concept worth naming: **information density per game.** Two
data-generation schemes can cost the same core-hours and yet produce wildly
different amounts of *usable* signal. A scheme that plays a game and records
one label per position is low-density. A scheme that records the full search
observation, from two teachers, with per-move values and agreement signals,
extracts far more learnable structure from the *same* expensive searches.
When compute is the binding constraint — and file 12's B4 finding showed it
always is — maximizing information density per game is how you make a fixed
compute budget go further. It is the observation-store principle expressed
as a metric.

## 3. Learned search allocation: the actual bet

Everything so far — ledger, observation store, dual-teacher labels — is
infrastructure for one audacious research bet, ARCHON's reason for
existing. It is set out in `design/19`, and it is worth stating carefully
because it is genuinely different from how engines are usually built.

### 3.1 The thesis

Here is the argument. Stockfish has a multi-year, thousands-of-core-years
lead in *evaluation tuning*. You will not out-evaluate it. But recall the
strength decomposition from file 14: `Elo ≈ f(net quality, search quality,
inference speed)`. At **fixed time**, playing strength is *decision quality
per node-microsecond* — not centipawn accuracy per position. Two engines
that evaluate equally well but *allocate their search differently* can
differ by hundreds of Elo.

And how does every engine, including Stockfish, currently allocate its
search? With the **hand-tuned heuristics of files 09 and 10** — the LMR
table, the futility margins, the singular rules, the history weights. Those
constants are transcribed and SPSA-tuned (file 14), but they are still, at
bottom, *hand-designed formulas with fitted numbers.* ARCHON's bet is that
these allocation decisions — *which move to try next, how much to reduce,
how long to think, whether to extend* — could instead be **learned** from
the recorded consequences of the engine's own past searches.

This is the natural payoff of the observation store. The store captures,
for every node, the search *action* taken (reduce by this much, prune this,
extend that) and its *consequence* (did the reduced search need a costly
re-search? did the prune turn out safe? did the extension change the best
move?). That is exactly a dataset of `(situation, action, outcome)` — the
raw material for learning a *policy over search actions* rather than
hand-coding one.

### 3.2 The reframing: predict consequences, then plan

The D41 formulation (`design/19`) is precise about what to learn. For a
position `x`, search context `c`, candidate move `m`, and a proposed search
action `a`, learn a function

```
   Q(x, c, m, a) → predicted search consequences
```

whose outputs are probabilities and expected costs: the chance this reduced
search will turn out wrong and need re-searching, the chance the move causes
a cutoff or raises alpha, the expected number of nodes and cycles it will
cost. Then a **planner** — deliberately chess-agnostic; all the chess
judgment lives in `Q` — chooses the action that minimizes expected cost
*subject to an explicit error-risk budget*: search cheaply, but keep the
probability of a search error below `ε`.

Look at how directly this operationalizes the "error budget" mental model
from file 10. There, every prune was described as a bet with an upside
(nodes saved) and a cost (probability of being wrong times the damage).
File 10's engine makes those bets with *hand-tuned* margins. The D41
program proposes to make them with *learned* probabilities and an *explicit*
risk constraint — turning the informal budget into a formal optimization.
The controller "chooses computation, not merely a move."

### 3.3 Why the richer inputs, and why gradually

Two hard-won refinements shape the current plan. First, **richer relational
inputs** (the FullThreats features of file 11) are not incidental — they
are there because a scalar evaluation *discards the information needed to
allocate search.* To predict whether a position is tactically volatile (and
thus whether reducing it is dangerous), the network needs to see the
*relationships* — which pieces threaten which — not just a single number.
The richer inputs exist to *absorb the richer targets* the observation
store now provides. Input richness and target richness advance together.

Second, **gradual authority.** You do not hand a freshly-trained control
model the keys to the search and hope. The learned signal is introduced to
*augment* the hand-tuned heuristics first, in low-risk seams, and is given
more authority only as it proves itself under the same equal-time SPRT
gates as everything else. The history of the project is littered with
cautionary results — an earlier attempt at naively co-training policy and
value *damaged the evaluation* — so the program advances behind games, not
behind training loss, and behind a flywheel: use the current engine to
generate observations, label them with a slower more careful search, train
a control model, deploy it cautiously, then *mine its failures* to generate
the next round of observations. State → oracle label → train → deploy →
failure-mine, repeat.

It is entirely possible this bet does not pay off — the design documents say
so plainly, and rank the ways it can fail. But it is, as ARCHON's own thesis
puts it, "the only lever where a clean, well-instrumented engine has an
advantage over a giant hand-tuned one," because the training signal — the
engine's own search decisions and their outcomes — is something ARCHON can
generate for itself and Stockfish cannot hand it. That is what makes it
research and not just engineering.

## 4. The honest open questions

A good case study does not hide its unresolved problems. Three of ARCHON's
are worth stating, because they are the *real* frontier and because they
recur in any serious ML-plus-search system:

**Label softness on near-tie positions.** As §2.1 noted, a large fraction of
positions have several nearly-equal best moves, so any single "correct
answer" label is partly fictional. How much does training on soft,
per-move-value targets actually help versus hard argmax targets? How do you
even *measure* the quality of a policy on positions where the ground truth
is a near-tie? This is unsolved, and it is the same question that limits
value-net label quality (file 12) wearing a policy costume.

**Equal-time versus per-node quality.** File 14's central confound is not a
one-time embarrassment; it is a *permanent tension* in the program. Every
learned signal that improves decision quality *per node* also costs
inference time, and thus costs nodes. A control model that allocates search
better but takes 2× longer to run must clear that 2× hurdle on the clock.
ARCHON's D40 network living exactly this — winning per node, losing on the
clock — is the whole research program in miniature: **smarts and speed are
in constant tension, and only the clock adjudicates.** Every idea in
`design/19` has to survive that adjudication.

**The speed-versus-smarts trade, structurally.** Underlying both of the
above is the deepest question: is there *room* for a smaller, smarter engine
to beat a bigger, faster, hand-tuned one at equal time? The bet says yes —
that decision quality per node-microsecond has enough headroom that learned
allocation can overcome an inference-cost penalty. But that is a hypothesis,
not a theorem, and the honest state of ARCHON's research is that it has
proven the *pieces* work (learned move information *can* improve fixed-node
search; the data platform *does* capture the needed signals) without yet
proving the *whole* clears the equal-time bar. That gap — between "the parts
work" and "the sum wins on the clock" — is precisely where the frontier is.

## 5. What you can now do

Step back. When you started this series you likely thought of a chess engine
as a black box that somehow plays good moves. You now know it is a
carefully layered system, and you understand every layer well enough to
reason about it. Concretely, here is what that gets you:

- **You can read any engine's source.** Open Stockfish, or Leela, or a
  from-scratch hobby engine, and the vocabulary is now yours: you will
  recognize the bitboards and magic tables, the make/unmake and Zobrist
  keys, the PVS and aspiration windows, the transposition table's
  replacement scheme, the MovePicker stages, the LMR reduction table, the
  NNUE accumulator and its incremental updates. The code will read as
  *choices you understand the reasons for*, not incantations.

- **You can follow engine-development discourse.** When a developer says "my
  SPRT passed at LTC with an unbalanced book," or "the net wins fixed-node
  but the inference tax kills it at equal time," or "this is a functional
  patch so it needs games, not just a bench," you now know exactly what each
  claim means and why it matters — and you can tell a rigorous claim from a
  hand-wavy one.

- **You can run your own experiments — correctly.** You know the cardinal
  sin (trusting a 20-game eyeball), the tool for each question (SPRT for
  strength, bench for behavior, micro-benchmarks for function-speed), the
  variance-reduction tricks (paired color-swapped openings, unbalanced
  books), and the confound to fear (net versus search versus speed). You
  could set up a fastchess SPRT tomorrow and interpret the verdict honestly.

- **You understand the ML-plus-search frontier** well enough to see why it is
  hard: that data volume dominates, that a search score can be a bound rather
  than a value, that labels are soft on near-ties, that smarts and speed
  trade off, and that reproducibility discipline is not optional overhead but
  the thing that makes any of it count.

That last cluster — the hard-won, non-obvious lessons — is the real prize.
Anyone can memorize that alpha-beta prunes; understanding *why ordering
matters more than evaluation at fixed depth*, *why a null-move prune needs a
zugzwang guard*, *why data volume beat every clever idea*, and *why a
fixed-node win can lose on the clock* is the difference between knowing the
facts and having the judgment. You now have the judgment.

## What you now know

- The **provenance ledger** (content-addressed artifacts, regeneration
  recipes) makes a research program reproducible — the precondition for
  reaching any frontier, because file-14-expensive verdicts must be trusted,
  not re-earned.
- The **observation store** captures the *whole search*, not just a label,
  turning every future training target into a cheap derivation; **dual-teacher
  labeling** and **per-move values** measure and respect label softness, and
  raise **information density per game** under a fixed compute budget.
- **Learned search allocation** (D41) is ARCHON's core bet: learn to predict
  search *consequences* and let a risk-budgeted planner choose the
  reductions, prunes, and extensions that files 09–10 hand-tune — powered by
  richer relational inputs and introduced with *gradual authority* behind
  equal-time game gates.
- The **honest open questions** — label softness on near-ties, equal-time vs
  per-node quality, and the structural speed-vs-smarts trade — are the real
  frontier, where "the parts work" has not yet become "the sum wins on the
  clock."
- Having read the series, you can **read any engine's source, follow the
  field's discourse, and run your own rigorous experiments** — and, more
  importantly, you have the *judgment* behind the facts.

That is the end of the series. You came in able to play chess; you leave
able to understand — and, if you like, to build and to improve — the
machines that play it better than any human ever will. The frontier
described here is genuinely open. Nothing stops you from working on it.

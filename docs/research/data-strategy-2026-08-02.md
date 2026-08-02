# Data strategy — plan of record, 2026-08-02

Owner ruling: **generate our own data.** The Lc0 download was vetoed mid-flight
and cancelled (3 tars / 475 MB removed). This is the crystallised plan; nothing
generates until the owner says go.

## Why we are regenerating rather than tuning

Measured over ~3,000 games on 2026-08-01/02: blending 40 % of a Stockfish net
into the training target produced **+30.42 ± 11.02 Elo vs canonical**
(LOS 100 %) while the label-heldout loss **rose monotonically** (0.010545 →
0.010643, EMA 473 → 560, ~1.3 % above the plain-run band and diverging).

**Moving away from our labels GAINS Elo.** If labels were merely noisy, moving
off them would cost — noise is unbiased and the mean is the best estimate.
Gaining means they are **biased**, i.e. systematically pointing somewhere worse.
That makes label quality the binding constraint on this engine, and it is why
regenerating beats any amount of trainer tuning.

Corroborating: Stockfish's net in **our** search at equal nodes is **+90.7 Elo**
on a provably identical 83,248-input feature space, while our accumulator is
1.25× *larger* than theirs. Bigger, same features, 90 Elo worse ⇒ the deficit is
in the weights, and weights are data plus recipe.

## What is being built (commissioned, in progress)

A study to design and BUILD the generation tooling. Deliverable is working tools
plus a measured operating point, not a document.

| # | tool | purpose |
|---|---|---|
| 1 | game generation, 3 modes | ARCHON self-play · ARCHON vs SF18 · ARCHON vs SF16 |
| 2 | labeller | score positions with our strongest oracle at a chosen node count |
| 3 | exporter | into trainer corpus format, honouring D46 / `data-recipe-contract.md` |
| 4 | throughput/quality table | so an operating point can be *chosen*, not guessed |

## The owner's requirements, and what each one implies

| requirement | design implication |
|---|---|
| "decent games done quickly" | throughput is a first-class axis, not an afterthought |
| "diverse positions where **WDL is meaningful**" | book + game policy must avoid dead-drawn and already-decided positions; measure the distribution we actually get |
| "labelled by our strongest oracle at a **reasonable node count**" | sweep the node curve, find the knee, justify with numbers |
| "**don't want deep labels** — a shallow NNUE can't learn them" | deeper is NOT automatically better; label depth must match model capacity |
| "accurate and **not disagreeing with WDL**" | **the central constraint** — see below |

### ⚠ The eval-outcome consistency constraint is the heart of it

A label can be individually accurate and still be **noise to the trainer** if it
disagrees with the game's eventual result. The two signals are trained together;
where they conflict, the net learns the conflict.

This must be **measured** — disagreement rate as a function of node count and
game phase — and treated as a primary output, not asserted. It is the metric the
owner named explicitly and it is the one most likely to be quietly skipped.

## Open decisions (owner's, not the agent's)

1. **Mode mix.** Self-play vs SF18 vs SF16 in what proportion. Self-play matches
   our own distribution; vs-SF gives positions we mishandle. Both matter, ratio
   unknown.
2. **Corpus size target**, which converts to wall-clock via the measured
   throughput once it exists.
3. **Whether to keep any existing corpus.** The live mix is 326 GB across 12
   corpora and is **not re-derivable on this box** — see
   `data-holdings-inventory-2026-08-02.md`. Do not delete it until generated data
   has beaten it in games.

## Disk position

282 GB free after reclaiming 155 GB (`canon-v1`, `fulldist`, `hardneg`, scratch
`tmp`). Generated data goes to `data/nnue/selfplay/`. The remaining 326 GB of
`jaxcorpora` is the safety net and stays until superseded by measurement.

## Lc0 material — kept, not deleted

`lc0-data-survey-2026-08-01.md` and `lc0-acquisition-plan-1B.md` remain valid
reference: the archive map, the run-strength timeline, the V6 format spec, and
the measured 208 B/position constant. If self-generation underperforms, that plan
is ready to execute. The fetcher is archived at `old/lc0-vetoed/`.

**Unresolved there, if we ever return to it:** test91-vs-test80 was never
settled. The survey's own SF-agreement probe is a null across the good eras
(73–79 %). test91 has deeper per-position search (~800 vs ~470 visits); test80
trained both BT3 and BT4 and has 7× the volume but is described as a debug run.
The recommendation was to hedge across both rather than pick.

# Generating ARCHON's own NNUE training data — design and operating point

**Written 2026-08-02.** Tools live in `tools/ops/data/` (indexed in
`tools/ops/README.md`). Everything below is measured on this box unless it is
explicitly flagged as unestablished.

**Nothing here has been run at production scale.** The owner's constraint was
tooling plus a measured operating point, ready to launch on their word. The
pilots below are ~450 games total.

---

## 0. Why this exists

Over ~3,000 games we measured that **our current labels point the wrong way**:
blending 40 % of a Stockfish net into the training target gave
**+30.42 ± 11.02 Elo vs canonical** while the label-heldout loss rose
monotonically (`d48y-wdl-fix-plan.md`, 2026-08-01 18:00). Moving *away* from our
labels gains Elo. Label quality is therefore the binding constraint, and that is
why data is being regenerated rather than the trainer retuned.

The one-line design consequence: **this pipeline's job is not to produce many
positions, it is to produce labels that do not fight the outcome.**

---

## 1. THE RECOMMENDED OPERATING POINT

```bash
# per mode; run one at a time, or share the box by lowering CONC
systemd-run --user --unit=dgself -p MemoryMax=6G -p MemorySwapMax=0 \
  --working-directory=/home/nick/projects/chessengine \
  -E MODE=self -E GAMES=20000 -E GAMENODES=25000 \
  -E ORACLE=engines/bin/stockfish-eca43a97 -E LABELNODES=10000 \
  -E PERGAME=80 -E CONC=3 -E WORKERS=5 \
  /home/nick/projects/chessengine/tools/ops/data/run_pipeline.sh
```

| parameter | recommendation | why (section) |
|---|---|---|
| **mode mix** | 50 % `self` / 30 % `sf18` / 20 % `sf16` | §6.2, §6.3 — **judgement, not measurement** |
| **play node budget** | 25,000 both sides | §6.3 — both SF anchors contested here |
| **oracle** | `stockfish-eca43a97` | §5, §5.3 |
| **oracle node budget** | **10,000** | §4 — the whole of it |
| **book** | `UHO_Lichess_4852_v1.epd`, sliced per run | §6.7 |
| **positions per game** | **80** (yields 73.6) | §6.1 — largest *unbiased* cap |
| **adjudication** | on (`ADJ=1`), two-sided resign | §6.4, §6.5 |
| **expected yield** | ~176k kept positions/h; ~2.8 h per 1M on 12 cores | §7 |

### 1.1 The node count, honestly

**Nothing I measured distinguishes 5,000 from 200,000 nodes on the criterion the
owner named.** Eval-outcome consistency is flat across a 100× range on three
independent instruments, each below its own noise floor (§4.1). The
representability argument — the owner's "a shallow NNUE can't learn them" —
came out a **wash** once a mate-score contamination was removed (§4.6). So the
evidence bounds a *range*, 5k–25k, rather than picking a point in it.

Within that range I recommend **10,000** because:

* **2,000 is strictly dominated** — same throughput as 5,000 (284.3 vs 292.4
  pos/s, per-position overhead rules below ~5k) and worse labels (§4.4).
* **Cost is dominated by game generation, not labelling.** 5k → 10k costs only
  **+15 %** total core-hours (28.8 → 33.2 per million). Paying that buys label
  σ 65.6 → 59.2 against the deep reference. Going on to 100k costs 2.1× (§7.2).
* It hedges the one argument that favours depth (§4.5) and the one structural
  worry the null cannot address: at 10k the oracle is not *much* weaker than the
  25k search that generated the games.

**5,000 and 25,000 are both defensible.** Anyone preferring the cheap end or the
matched-to-play end should take them; the data does not separate them.

### 1.2 What makes this pipeline better than the one it replaces

Not the node count — the **provenance and the WDL column**:

* **100 % real WDL by construction.** The result rides with the position from
  the game that produced it. `build_canon_tight.sh` ships `JOIN_MIN=33` because
  its measured join overlap was 35.6 %; two thirds of its labelled positions had
  no result at all (§2).
* **Every row traces to `(run, game, ply)`**, with sha256 of the engine, net,
  opponent and book in a manifest written *before* the games start. This is the
  granularity the D38 ledger cannot give today — it proves a corpus came from a
  recipe, not which position came from which game.
* **The filter runs where it can be correct**, and the export stage *refuses*
  audit-mode input rather than trusting the caller (§2, verified in §8).

### 1.3 Do this before training on it

1. `tools/ledger/ledger.py add-data` — identity is the corpus sha256, already in
   `corpus.meta.json`.
2. `tools/nnue/net_scale_check.sh` on the first trained net. Labels are raw SF
   UCI cp, so contract §2 says `SCORE_MULT` 1.42 at WDL=0.3 / 1.94 at WDL=0 —
   **but the gate script is the authority, not that sentence.** It is a hard
   pre-game gate for a reason.

### 1.4 Sizing

Our label problem is a *quality* problem, so this corpus does not need to rival
the 24 GB of binpacks. **5 M positions (~14 h on 12 cores)** is enough to answer
whether self-generated, fully-provenanced data beats the current mix, and 1 M
(~2.8 h) is enough to smoke-test the whole chain into the trainer.

### 1.5 The experiment that actually settles the node count

Everything above is an argument from label properties. **Games arbitrate.** Two
corpora of *equal size* from the same games — one labelled at 5k, one at 25k —
trained with identical recipes and screened against each other. That is one
extra labelling pass over positions already generated (§7.2: ~8 core-hours per
million extra at 25k), and it converts §4's null from "no measurable difference"
into a verdict.

---

## 2. The pipeline

Four stages, each resumable, each leaving a stats file. `run_pipeline.sh` chains
them.

| stage | tool | what it decides |
|---|---|---|
| 1 play | `gen_games.sh` | mode, node budget, book, adjudication |
| 2 extract | `pgn_positions.py` | which positions, how many per game |
| 3 label | `label_positions.py` | oracle, node count, **the D46 filter** |
| 4 export | `export_corpus.sh` | trainer bytes + the contract's self-checks |

### Where the D46 filter runs, and why it can only run there

The contract (`data-recipe-contract.md` §1) requires the five-criterion quiet
filter at "the layer that still has raw scores + best moves". In this pipeline
that layer is **stage 3 and nowhere else** — the exported text has lost the move,
so a downstream filter is structurally partial (the `canon_filter` lesson).

Stage 2 applies the two criteria that need neither a score nor a move (ply ≥ 16,
not in check) purely so we do not buy an oracle search for a position that is
already disqualified. It records those counts so the **full** five-criterion rate
stays reconstructable; otherwise stage 3's rate looks deceptively low and its
"did the filter run?" self-check loses its meaning.

Two details that are easy to get wrong and are implemented deliberately:

* **Criteria 4 and 5 are not redundant.** An en-passant capture lands on an
  *empty* square, so "destination empty" alone lets it through; the move-kind
  test is what catches it. `label_positions.py` follows the normative Rust
  (`d40_features.rs::sfbinpack_filter`) rather than `relabel_depth.py`, which
  reaches the same place by a different route.
* **Filter first, clamp last.** `|raw| ≤ 10000` is tested on the raw score;
  only survivors are clamped to ±2499. Stage 4 re-asserts the clamp and *dies*
  rather than clamping again — a second clamp is exactly the bug that put
  86,320 mate labels into an aux corpus.

### The WDL column is real by construction

This is the pipeline's main structural advantage over the shard path.
`build_canon_tight.sh` has to inner-join a result column onto positions labelled
from a different sample, and ships `JOIN_MIN=33` because its measured overlap was
**35.6 %** — two thirds of its labelled positions had no result and were
discarded. Here the result rides *with* the position from the game that produced
it, so the join hit-rate is **100 % by construction**; `export_corpus.sh` asserts
it and treats a miss as an upstream bug, not an acceptable loss.

---

## 3. NULL FIRST — what the instruments can and cannot resolve

Run before any arm was read.

### 3.1 Instrument null: are labels reproducible at all?

`label_positions.py --null` labels the input twice and diffs.

```
.venv/bin/python tools/ops/data/label_positions.py <200 pilot positions> \
    --out lab200.tsv --oracle engines/bin/stockfish18 --nodes 25000 \
    --workers 4 --null
→ NULL CONTROL: 200 positions relabelled identically=200 differing=0 (0.000%)
```

**200/200 bit-identical.** Labels are a pure function of
(engine, net, nodes, position); `ucinewgame` per position is what buys it. So no
node-count difference below can be run-to-run noise. This was worth checking
first: without it every tier comparison would have an unknown noise floor.

### 3.2 Sampling null: how big a difference is even resolvable?

Every statistic in the node study is also computed on two disjoint halves of the
position set **at the same node count**. The half-vs-half spread is the sampling
noise floor and is printed beside every tier as `nullspread`. **A tier-to-tier
movement smaller than the local nullspread is reported as not resolvable**, and
in §4 several of them are.

### 3.3 Oracle-comparison null

`study_oracles.py` automatically adds a `__NULLCOPY` arm: the first oracle
labelled a second time and compared with itself. It must read 100.00 %
best-move agreement and 0 cp. Any cross-oracle difference smaller than that is
inside the instrument.

### 3.4 A null that is NOT clean, stated plainly

**Throughput numbers are contaminated by box load.** `promogate` (a 1500-game
match) ran throughout, and the pilots and studies overlapped each other; load
average reached 17.98 on 16 cores. Specifically, `pilot-sf16` ran while the node
study was saturating the box and reported 1475 games/h against 2857 games/h for
`pilot-sf18`, which ran on a quieter box. **That is not evidence that SF16 is
slower than SF18** — it is the load difference, and it would be a wrong
conclusion drawn from a dirty instrument. Every wall-clock number in §7 is a
**lower bound**, and the mode-to-mode comparison of games/h is void.
Re-measuring the throughput table needs a quiet box.

---

## 4. How many nodes should the oracle get?

```
.venv/bin/python tools/ops/data/study_label_quality.py \
    --positions data/nnue/selfplay/study/combined.tsv \
    --oracle engines/bin/stockfish18 \
    --nodes 2000,5000,10000,25000,50000,100000,200000 \
    --workers 5 --limit 4200 --threshold 150 \
    --out data/nnue/selfplay/study/nodes-sf18
```

2,921 positions quiet and scored at **all seven** tiers (paired set — see §2 on
why the D46 verdict itself moves with the tier).

| nodes | disagree % | ±95 % | nullspread | **AUC** | AUC null | bmAgree vs 200k | med \|Δ\| vs 200k | drop % | pos/s |
|---|---|---|---|---|---|---|---|---|---|
| 2,000 | 0.53 | 0.51 | 0.16 | 0.9834 | 0.0086 | 47.6 % | 24 | 21.4 | 284.3 |
| 5,000 | 0.32 | 0.42 | 0.25 | 0.9851 | 0.0082 | 54.7 % | 20 | 22.1 | 292.4 |
| 10,000 | 0.42 | 0.46 | 0.04 | 0.9848 | 0.0088 | 60.4 % | 18 | 22.4 | 170.6 |
| 25,000 | 0.42 | 0.45 | 0.04 | 0.9866 | 0.0079 | 67.4 % | 13 | 22.6 | 91.5 |
| 50,000 | 0.21 | 0.35 | 0.43 | 0.9873 | 0.0073 | 73.6 % | 10 | 22.6 | 53.2 |
| 100,000 | 0.30 | 0.39 | 0.63 | 0.9872 | 0.0075 | 83.0 % | 6 | 22.9 | 37.8 |
| 200,000 | 0.20 | 0.33 | 0.41 | 0.9885 | 0.0063 | 100 % (ref) | 0 | 23.4 | 20.6 |

### 4.1 The headline is a NULL, and it is the most useful thing here

**Eval-outcome consistency does not measurably improve with oracle depth over a
100× node range.** Three instruments, all agreeing:

* **Contradiction rate at |cp| ≥ 150**: 0.53 % → 0.20 %, a 0.33 pp move against
  a 95 % CI of ±0.33–0.51 pp and a sampling nullspread reaching 0.63 pp.
* **Contradiction rate at |cp| ≥ 50** (where outcomes are genuinely in doubt):
  2.20 % → 1.99 %, CI ±0.62–0.65 pp. At |cp| ≥ 400 it is **0.00 % at every
  tier**, and on played-out games it is **0.00 % at every tier**.
* **AUC** (P(label ranks a White win above a White loss), scale- and
  frame-free): 0.9834 → 0.9885, **a 0.0051 move against an AUC null floor of
  0.0063–0.0088**. The change across the whole range is *smaller than the
  half-vs-half noise*.

The null-first rule fired exactly as intended. Reading the raw disagreement
column as a downward trend would have been reading noise.

**Why the metric saturates, mechanically:** the outcome was produced by engines
playing at 25k nodes. An oracle searching deeper than the players knows things
that *by construction did not affect this game's result*. The outcome cannot
validate oracle knowledge beyond the players' own horizon, so it is a weak
ground truth — and a weak ground truth is what a near-1.0 AUC looks like.

**A null bounds the effect; it does not prove equality.** The honest statement
is: any depth effect on outcome consistency is smaller than ~0.0086 AUC and
~0.65 pp of contradiction rate at n≈2,900.

### 4.2 What DOES move: precision, monotonically and hugely

Best-move agreement with the 200k reference runs **47.6 → 100 %** and median
\|Δ\| runs **24 → 0 cp**. Labels change a great deal with depth. They just do
not agree with the outcome any better.

### 4.3 Shallow-label error is NOISE, not BIAS — which is what makes it cheap

Deviation from the 200k reference, over ~3,000 paired positions:

| nodes | signed mean | median | stdev | \|mean\|/stdev |
|---|---|---|---|---|
| 2,000 | −5.41 | 0.0 | 72.3 | 0.075 |
| 5,000 | −3.86 | 0.0 | 65.6 | 0.059 |
| 10,000 | −3.47 | 0.0 | 59.2 | 0.059 |
| 25,000 | −2.81 | 0.0 | 47.9 | 0.059 |
| 100,000 | −1.04 | 0.0 | 29.7 | 0.035 |

**The median is exactly 0 at every tier and the signed mean is under 0.08 σ.**
A shallow label is an unbiased, noisy estimate. A regression trained on
zero-mean noise converges to the same conditional mean — it needs more data, not
different data. Bias would have been the disqualifying finding, and it is absent.

### 4.4 Below 5,000 nodes there is no throughput left to win

2,000 nodes runs at **284.3 pos/s** and 5,000 at **292.4** — the shallower tier
is *not faster*. Per-position overhead (`ucinewgame`, the UCI round trip)
dominates below ~5k: effective search rate is 0.57 Mnodes/s at 2k against 1.46
at 5k and 3.78 at 100k. **2,000 nodes is strictly dominated** — same speed,
worse labels.

### 4.5 The counter-argument, stated rather than buried

An inverse-variance reading — precision per core-hour, `n/σ²` — *favours depth*,
because σ (§4.3) falls faster than cost rises. I am not treating it as
decisive, and the reason is that **it is circular**: σ is measured against the
200k label, so it is guaranteed to fall to zero at the deepest arm. The metric
cannot rank the reference against itself. Breaking that circularity needs an
independent deep reference (e.g. eca43a97 at 200k judging SF18's tiers), which
I did not run.

The non-circular ground truth available is the game outcome, and it is flat.

### 4.6 Irreducible label noise — the owner's "a shallow NNUE can't learn them"

This is the instrument built specifically for the other half of the constraint.
Pairs of positions **identical in the network's input space** that differ only in
something it cannot see; whatever the oracle says they differ by is noise the
net can never fit. Blindness is a receipt, not an assumption: `archon
nnue-eval` returned identical output on **700/700 pairs** for both kinds.

```
.venv/bin/python tools/ops/data/study_twins.py \
    --positions data/nnue/selfplay/study/combined.tsv --kind castle|rule50 \
    --nodes 5000,25000,100000 --n 700 --oracle engines/bin/stockfish18
```

| kind | nodes | bestmove chg | \|Δ\| p50 | p90 | p99 | mean |
|---|---|---|---|---|---|---|
| **castle** | 5,000 | 41.3 % | 15 | 60 | 147 | **25.4** |
| | 25,000 | 39.9 % | 18 | 71 | 155 | **29.6** |
| | 100,000 | 41.4 % | 21 | 78 | 169 | **33.7** |
| **rule50** | 5,000 | 40.4 % | 15 | 69 | 161 | **27.6** |
| | 25,000 | 33.5 % | 12 | 49 | 181 | **22.1** |
| | 100,000 | 33.0 % | 11 | 43 | 201 | **19.9** |

**The two dimensions go OPPOSITE WAYS, and the combined effect is flat.**
Castling noise grows monotonically in every quantile (mean 25.4 → 33.7) — a
deeper search sees further into king safety and so values the rights more.
Clock noise *shrinks* (mean 27.6 → 19.9, p50 15 → 11, p90 69 → 43) — a deeper
search resolves the position confidently regardless of the clock — while its
extreme tail grows (p99 161 → 201), the rare positions where the 50-move rule
genuinely decides. Averaged: 26.5 / 25.9 / 26.8 cp at 5k / 25k / 100k.

**This corrected my own emerging story, and it is worth saying how.** An earlier
run of this arm reported the rule50 mean rising 27.6 → 64.1 → 62.5, which fit
the "deep labels are less representable" narrative neatly. It was wrong: the
max column read **29,830**, i.e. `uciprobe`'s mate mapping. A handful of pairs
where the deep search found a mate on one side of the clock twin carried the
whole mean. Those positions **cannot become training labels** — D46 criterion 3
drops |raw| > 10000 — so counting them as corpus noise overstated the case in
the direction I was already leaning. The tool now excludes them and reports the
excluded count and the max next to every quantile.

**Consequence for the decision: the representability argument is a WASH, not a
point for shallow labels.** The only clean argument left for shallow is
throughput, and the only argument for deep is circular (§4.5).

---

## 5. Which oracle?

```
.venv/bin/python tools/ops/data/study_oracles.py \
    --positions data/nnue/selfplay/study/combined.tsv \
    --oracles engines/bin/stockfish18,engines/bin/stockfish16,engines/bin/stockfish-eca43a97 \
    --nodes 25000 --limit 4200 --workers 5 --out data/nnue/selfplay/study/oracles
```

### 5.1 The choice DOES change the labels — this one is not a null

| pair | best-move agree | \|Δ\| p50 | \|Δ\| p90 | \|Δ\| mean |
|---|---|---|---|---|
| SF18 vs SF16 | 56.47 % | 16 | 68 | 62.1 |
| SF18 vs eca43a97 | 59.86 % | 15 | 86 | 37.4 |
| SF16 vs eca43a97 | 55.91 % | 16 | 69 | 59.5 |
| **SF18 vs SF18 (NULLCOPY)** | **100.00 %** | **0** | **0** | **0.0** |

Against a perfectly clean null, the three oracles agree on the best move only
**56–60 %** of the time and sit **15–16 cp apart at the median**. SF18 and
eca43a97 are closer to each other (mean 37.4) than either is to SF16 (59.5),
which is what a generation gap looks like.

### 5.2 But outcome consistency cannot rank them

Their own contradiction rates at |cp| ≥ 150 are SF18 0.60 %, SF16 0.41 %,
eca43a97 0.30 % — all inside a ±0.5 pp CI at n≈1,000. **Same null as §4.**
So the labels differ materially, and the metric that matters cannot say which
set is better. The tie has to be broken on strength.

### 5.3 So I measured strength directly — and it is decisive

```
engines/bin/fastchess \
  -engine cmd=engines/bin/stockfish-eca43a97 name=eca43a97 \
  -engine cmd=engines/bin/stockfish18 name=SF18 \
  -each nodes=25000 tc=inf option.Hash=64 option.Threads=1 \
  -openings file=UHO_Lichess_4852_v1.epd format=epd order=random start=1200000 \
  -games 2 -rounds 150 -repeat -srand 777 -concurrency 3 -maxmoves 250
```

```
Elo: +47.78 +/- 25.58   nElo: +74.52 +/- 39.32   LOS 99.99 %
Games: 300, W113 L72 D115, Points 170.5 (56.83 %)   Ptnml [2,27,62,46,13]
```

**`stockfish-eca43a97` is +47.78 ± 25.58 Elo stronger than `stockfish18` at
exactly the node budget we label at.** The CI excludes zero comfortably. This
also confirms the version reasoning from `CLAUDE.md` — eca43a97 (2026-07-06) is
223 commits *newer* than the `sf_18` tag (2026-01-31), and the release binary
named "Stockfish 18" is the older engine.

**Recommendation: `engines/bin/stockfish-eca43a97`.** Three reasons, in order:

1. It is measurably the strongest oracle at 25k nodes, so it is the best
   available proxy for a position's true value.
2. It costs nothing — 112.7 pos/s against SF18's 107.3 in the same run.
3. It is **already our pinned constants reference**. Labelling with the same
   revision the engine's tunables were transcribed from removes a whole class
   of cross-version drift between what we train toward and what we compare
   against.

Note this is the one place a "strongest" claim was worth the games: §5.1 had
already shown the choice *changes the labels* (56–60 % move agreement), so the
ranking was decision-relevant rather than cosmetic.

---

## 6. Diversity and WDL meaningfulness

### 6.1 Sampling policy: fixed count per game, not every ply

Positions are sampled evenly across each game's eligible plies. The receipt for
caring is `tools/forensics/build_oracle.py`, whose predecessor sampled without
this discipline, came out 100 % ≥21-piece positions, and was blind to 82 % of
our damage.

Games yield **98.6 quiet plies each** (150 self-play games, 14,795 quiet plies),
of which **99.8 % are distinct piece placements** — so the cost of taking more
per game is *correlation*, not duplication.

Taking every ply is not free, and the bias is measurable:

| sampling | n | W | D | L |
|---|---|---|---|---|
| every quiet ply | 14,795 | 32.3 % | **43.0 %** | 24.7 % |
| even 14/game | 2,099 | 33.3 % | **40.6 %** | 26.0 % |
| **per-GAME (unbiased reference)** | 150 | 33.3 % | **40.7 %** | 26.0 % |

**A fixed count per game reproduces the true per-game result distribution
exactly; taking every ply inflates draws by +2.4 pp.** The mechanism is direct:
drawn games run **104.3** quiet plies against **93.7** for losses and 95.5 for
wins, so per-ply sampling over-weights them. Fixed-count sampling is
self-normalising and costs nothing to keep.

```
.venv/bin/python tools/ops/data/pgn_positions.py \
    data/nnue/selfplay/pilot-self/games.pgn --out all.tsv --per-game 0
```

### 6.2 Per-mode distributions

150 games per mode, 25k nodes both sides, UHO book, 14 positions/game:

| mode | n | W | D | L | mean pieces | ≤12 | 13–20 | ≥21 |
|---|---|---|---|---|---|---|---|---|
| self | 2,099 | 33.3 % | 40.6 % | 26.0 % | 19.0 | 25.1 % | 29.0 % | 45.9 % |
| sf18 | 2,056 | 31.9 % | 48.7 % | 19.4 % | 17.5 | 33.6 % | 26.4 % | 40.0 % |
| sf16 | 2,061 | 29.6 % | 37.6 % | 32.8 % | 18.3 | 28.5 % | 29.2 % | 42.3 % |

All three are non-degenerate (contract §4 requires no value above 70 %). Phase
coverage is broad in every mode; vs-SF18 skews most toward endgames (33.6 %),
which is the mode most likely to supply conversion technique.

### 6.3 Both Stockfish anchors are genuinely contested

WDL is only meaningful if the games were in doubt. At 25k nodes both sides:

| mode | ARCHON score | Elo | draw rate |
|---|---|---|---|
| vs Stockfish 18 | 41.67 % | **−58.45 ± 34.65** | 32.0 % |
| vs Stockfish 16 | 61.67 % | **+82.59 ± 37.57** | 41.3 % |

We sit *between* SF16 and SF18 at equal nodes, so neither matchup is a blowout
and neither produces a corpus of foregone conclusions. (150 games each: these
place us on a ladder, they are not verdicts — the bar at n=150 is ±31 Elo.)

### 6.4 Adjudication: a throughput lever that costs endgames

| mode | played out | resign-adjudicated | draw-adjudicated |
|---|---|---|---|
| self | 25 % | **57 %** | 18 % |
| sf18 | 42 % | **42 %** | 16 % |
| sf16 | 29 % | **57 %** | 13 % |

Two consequences, and they pull in opposite directions:

* **Draw adjudication is cheap and honest.** A position dead level for 8
  consecutive moves after move 40 is a draw; adjudicating it costs nothing in
  label quality and saves real time.
* **Resign adjudication truncates the conversion phase.** 57 % of self-play
  games ended when both engines agreed on a ≥1600-ARCHON-cp (~773 SF-cp)
  advantage for 4 moves. Those games never played out the technique that
  converts a won position, so the corpus is systematically short of exactly the
  endgames a value net needs to price. It also makes eval-outcome agreement
  **partly circular at high |cp|**: the result was defined by an eval. The
  circularity does not reach the decision threshold used in §4 (150 SF-cp is far
  below the 773 SF-cp resign bar), and the `term_normal` slice is reported
  separately so the honest number is visible.

`ADJ=0` disables both. It is exposed precisely because this is a real trade the
owner may want to make differently.

### 6.5 The adjudication frame trap (fixed, and worth knowing)

fastchess compares its resign/draw thresholds against **each engine's printed
score**, and ARCHON prints ~2.07× Stockfish's printed cp. A one-sided
`score=1600` would therefore fire for ARCHON at ~773 SF-cp and for Stockfish at
1600 SF-cp — a **colour-correlated truncation bias landing directly in the WDL
column**. `gen_games.sh` uses `twosided=true`, which requires both engines to
agree and removes the asymmetry.

### 6.6 Self-play must not use `-repeat`

Two identical deterministic engines replay the same game with colours reversed.
Measured: 8 games under `-games 2 -rounds 4 -repeat` produced **4 distinct
(opening, move-list) signatures — every pair bit-identical** (127/140/87/143
plies). In `self` mode `-repeat` is an exact 2× CPU waste contributing zero new
positions; `gen_games.sh` uses it only in the vs-Stockfish modes, where the two
engines differ and colour balance is real.

### 6.7 Book

`engines/books/UHO_Lichess_4852_v1.epd`, **2,632,036 positions** at ~move 4.
Deliberately unbalanced openings, which is what makes games decisive and the WDL
column informative. Diversity is not a constraint at our scale: a 20,000-game run
consumes 0.8 % of the book. `gen_games.sh` derives a reproducible non-overlapping
book slice from the run tag (`BOOKSTART`, `SRAND`), so successive runs never mine
the same openings twice.

### 6.8 The mode mix — reasoning, and what would falsify it

**50 % `self` / 30 % `sf18` / 20 % `sf16`. This is a judgement, not a
measurement**, and it is the weakest recommendation in this document.

What the measurements support: all three modes produce non-degenerate result
distributions and broad phase coverage (§6.2), and both Stockfish anchors are
contested rather than one-sided (§6.3). Nothing distinguishes them on label
quality, because the label comes from the oracle regardless of who played.

So what a mode actually buys is **position distribution**, and the reasoning is:

* `self` is the cheapest per position and is the only mode that samples the
  positions *our own search actually reaches* — which is where our net's errors
  live and therefore where correcting them pays.
* `sf18` supplies the hardest opposition and the most endgames (33.6 % ≤12
  pieces, the highest of the three), i.e. the conversion technique that resign
  adjudication otherwise removes (§6.4).
* `sf16` supplies positions where we are the stronger side (+82.59 Elo), so it
  is the mode richest in *winning* positions we must learn to convert.

**What would falsify the weighting:** train equal-size corpora from single-mode
data and screen them. Since the label pipeline is mode-agnostic, that experiment
costs only game generation and reuses everything else.

---

## 7. Throughput

**Read §3.4 first: every wall-clock number here is a lower bound.** `promogate`
ran throughout and my own studies overlapped each other; load average reached
23.4 on 16 cores. I over-subscribed the box at one point by running two of my
own jobs at once — my error, recorded here rather than smoothed over. Fixed-node
work is result-safe under load, so the *measurements* stand; the *rates* are
pessimistic and the mode-to-mode games/h comparison is void.

### 7.1 Measured component rates

| component | rate | conditions |
|---|---|---|
| game generation, `self` | **3,068 games/h** | conc 2, 25k nodes both sides |
| game generation, `sf18` | **2,857 games/h** | conc 2, 25k nodes both sides |
| game generation, `sf16` | (1,475 games/h — **void**, ran under a saturated box) | |
| positions per game | **98.6 quiet plies**, 73.6 kept at cap 80 | |
| labelling, 5k nodes | **292.4 pos/s** | 5 workers |
| labelling, 25k nodes | **91.5 pos/s** | 5 workers |
| labelling, 100k nodes | **37.8 pos/s** | 5 workers |
| D46 label-stage drop | **21.4–23.4 %** | rises slightly with depth |

### 7.2 Cost of a corpus, by oracle depth

Derived from the component rates above (`--per-game 80`, conc 2 generation,
5-worker labelling):

| oracle nodes | gen cores | label cores | total cores | kept pos/h | **core-h per 1M positions** |
|---|---|---|---|---|---|
| 2,000 | 4.0 | 1.10 | 5.10 | 175,913 | 29.0 |
| **5,000** | 4.0 | 1.07 | 5.07 | 175,913 | **28.8** |
| **10,000** | 4.0 | 1.84 | 5.84 | 175,913 | **33.2** |
| 25,000 | 4.0 | 3.43 | 7.43 | 175,913 | 42.2 |
| 50,000 | 4.0 | 5.90 | 9.90 | 175,913 | 56.3 |
| 100,000 | 4.0 | 8.30 | 12.30 | 175,913 | 69.9 |
| 200,000 | 4.0 | 15.23 | 19.23 | 175,913 | 109.3 |

**Game generation dominates, and that is the important structural fact.** At
10k oracle nodes, playing costs 4.0 cores and labelling 1.8 — so the oracle
depth is a *minority* of the bill, and the throughput argument for a very
shallow oracle is much weaker than it looks. Going 5k → 10k costs **15 %**;
going 10k → 100k costs **2.1×**.

### 7.3 Converting to wall clock

At the recommended operating point (10k oracle, `--per-game 80`), on ~12 cores
left free beside a match:

| target | wall clock (12 cores) |
|---|---|
| 1 M positions | ~2.8 h |
| 5 M positions | ~14 h |
| 10 M positions | ~28 h |

These scale linearly in cores and are lower bounds.

---

## 8. Validation — the tools are exercised, not just written

| check | result |
|---|---|
| stage 1 produces games with `[FEN]` header + per-move eval/nodes | 150 games × 3 modes |
| `-repeat` duplication claim in self-play | 8 games → **4 distinct signatures**, every pair bit-identical |
| stage 2 ply accounting | 16,999 plies → 15,050 candidates, D46 stage-2 drop 11.47 % |
| stage 3 reproducibility (`--null`) | **200/200 labels bit-identical** |
| stage 3 D46 drop rate | **28.0 %** at 25k — inside the contract's 20–35 % band |
| stage 4 refuses audit-mode input | **exit 1, no `.bin` written** (933 flagged rows) |
| stage 4 happy path | 3,222 rows → 3,222 records, 32B-aligned, text/bin match, WDL 100 % |
| **`nnue_convert inspect` cross-decode** | **3,222 records, 0 piece-count mismatches, ALL CHECKS PASSED** |
| twins blindness receipt | `archon nnue-eval` identical on **700/700** pairs, both kinds |
| oracle-comparison null arm | **100.00 % agreement, 0 cp** |

`nnue_convert inspect` is the strongest available proof: it re-decodes every
record with bulletformat's own loader and asserts the piece multiset, one king
per side, `occ` popcount vs nibble count, `|score| < 2500`, `result ∈ {0,1,2}`,
`ksq`/`opp_ksq^56` consistency **and** the sign/flip canonicalisation against
the source text line.

---

## 9. What I could NOT establish, and the failed arms

Reported because they are required, and because a null is not a licence to stop.

1. **No training run, therefore no Elo.** Nothing here is games-arbitrated.
   Every recommendation is an argument from label properties, not from measured
   strength. **The decisive experiment is cheap and named in §1.5.**
2. **The primary pre-registered metric failed.** Eval-outcome contradiction rate
   saturated (0.00 % on played-out games at every tier) and could not rank node
   counts. AUC and a threshold sweep were added *after* seeing that, which is
   the honest ordering to record. All three then agreed on the same null.
3. **`n/σ²` points the other way** (§4.5) and I could not de-circularise it
   without an independent deep reference. This is the strongest argument
   *against* my own recommendation and it is unresolved.
4. **Mode mix is a judgement, not a measurement.** I measured what each mode
   *yields* (§6.2) but nothing that ranks mixes by resulting net strength.
5. **`--per-game 80` trades correlation for volume, unmeasured.** The result
   distribution stays unbiased up to a cap of ~100 (§6.1) and 99.8 % of
   placements are distinct, but whether 74 correlated positions from one game
   carry as much information as 74 from 74 games cannot be settled without
   training.
6. **Throughput is load-contaminated** (§3.4). The `sf16` games/h figure is void.
   **A quiet box is needed to re-measure the §7.1 table**; I did not take one.
7. **An SF16-vs-SF18 style question I did not answer:** whether an *ARCHON*
   oracle (our own net, which is what the trainer must eventually match) labels
   differently in a way that matters. `label_positions.py --net` supports it and
   `study_oracles.py` deliberately refuses to mix frames, so the arm is one
   command away — I ran out of quiet box, not ideas.
8. **No Syzygy tablebases on this box.** `fastchess -tb` would give *ground
   truth* endgame adjudication instead of engine opinion, which is strictly
   better than the resign rule in §6.4 and would shorten games. 3-4-5-man is
   ~1 GB against 281 GB free. Cheapest available upgrade to WDL honesty; needs
   a download, so it was out of scope.

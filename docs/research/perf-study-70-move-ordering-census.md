# Perf study 70 — the move-ordering census, and the two ordering defects it found

**Target: nodes per move, via move-ordering QUALITY.** Not per-node speed.
The per-node microarchitectural axis was closed by studies 65–69; time-to-depth
has two factors and nobody had ever measured the other one.

Everything below is reproducible from this tree:

```
cargo build --release --features sota,arch-d48x,census70   # the census
cargo build --release --features sota,arch-d48x,arm70      # the arm bench
tools/perf70/cscreen70.py 'defcapqs=56' ...                # screen on the census
tools/perf70/screen70.py  'defcapqs=56' ...                # confirm on paired nodes
```

---

## 0. The one-paragraph version

ARCHON's first-move beta-cutoff rate had never been measured. It is **84.5 %**
at NonPV nodes, and the census's first real finding is that this single number
hides a **15-point split**: at nodes that arrive with a usable TT move the rate
is **91.4 %**, and at nodes without one it is **76.5 %** — and the TT-less half
is 46 % of all NonPV cutoffs, so it owns **70 % of the wasted cutoff work in the
engine.** The two landed fixes both attack ordering where no TT move and no SEE
gate is available to save it: qsearch capture ordering, which was blind to
recaptures, and evasion capture ordering, which ignored capture history
entirely. Two arms that looked like large wins on node counts — the history
malus *shape* and a cross-kind malus — are **retracted**: their magnitude
controls reproduce the whole effect, so both were pruning changes wearing an
ordering costume.

---

## 1. The instrument had to be built twice

### 1.1 The obvious metric fails its own null, badly

The natural metric is "nodes to reach a fixed depth, summed over a position
set". It is discrete and load-immune, which is why the study was aimed at it.
It is also **useless at the sizes this project uses**.

`arm70` carries a `jitter=1` knob that adds a deterministic pseudo-random
`[-1, +1]` to every quiet's ordering score. That is an *inert* perturbation: a
one-unit change can only re-break exact ties among moves whose entire seven-plane
history stack agrees to the unit. Whatever it moves is chaos, not ordering.

| null seed | wl50 @ d13 | uho300 @ d10 | uho300 @ d11 |
|---|---|---|---|
| 1 | **+18.28 %** | +1.76 % | −4.73 % |
| 2 | +8.58 % | −0.55 % | −1.04 % |
| 3 | +10.85 % | −3.04 % | +2.24 % |
| 4 | +7.43 % | −0.39 % | −1.06 % |
| 5 | +1.88 % | −2.10 % | −2.04 % |
| 6 | +8.97 % | +2.09 % | −2.37 % |

The 50-position d13 set — the set this project's NNUE gate pins — has a **±18 %
noise floor** on an inert perturbation. The entire first arm screen (`qsortfull`
+24.9 %, `lva=16` +23.1 %, `evcap` +17.9 % on that set) was noise, and would
have been read as three findings.

A second fact falls out of the same panel: on wl50 the null is **positive in all
six seeds** (mean +9.3 %). Generation order is not a neutral tiebreak —
`partial_insertion_sort` is stable, so equal-scoring quiets keep movegen's
piece-then-square order, and destroying that order costs nodes. The ±1 jitter is
therefore a *conservative* null: it is the smallest order-changing perturbation
available and it already carries information.

### 1.2 What survives: paired per position, and the sign test

Sum-over-positions is dominated by whichever two positions exploded. Pairing
each position against itself and aggregating log-ratios fixes the tail:

| null seed (uho1000 @ d10) | geomean | median | better/worse |
|---|---|---|---|
| 1 | −0.07 % | −1.78 % | 511 / 489 |
| 2 | −0.50 % | −0.20 % | 503 / 497 |
| 3 | +1.59 % | −1.05 % | 507 / 493 |
| 4 | −0.15 % | +0.30 % | 499 / 501 |
| 5 | −1.49 % | −1.92 % | 510 / 488 |
| 6 | −0.81 % | −1.01 % | 503 / 497 |

The **sign test is the discrete statistic that works**: better/worse sits at
505 ± 6 of 1000 across all six seeds. The geometric mean has a ±1.5 % null
spread — enough to see a 3 % effect, not a 1 % one.

### 1.3 What actually resolves ordering: the census itself

Node count is ordering's *downstream consequence*, three chaotic steps removed.
The census statistics are averages over 10⁵–10⁶ events **inside one
deterministic run**, so their noise is two orders of magnitude smaller:

| statistic | events | null spread over 6 seeds |
|---|---|---|
| `fhf` NonPV | 220 k cutoffs | −0.14 … +0.06 pp |
| `fhf` NonPV-in-check | 32 k | −0.20 … +0.01 pp |
| `fhf` qsearch | 146 k | −0.22 … +0.15 pp |
| LMR re-search rate | 760 k scouts | −0.03 … +0.03 pp |

**That is the study's screening instrument.** Nodes confirm; games arbitrate.

---

## 2. The census (uho300 @ d11, 4.64 M nodes) — the numbers nobody had

| node kind | cutoffs | **fhf** | mean cut index | all-nodes | moves/all-node | cut share |
|---|---|---|---|---|---|---|
| root | 3,919 | 0.855 | 1.83 | 6,289 | 34.2 | 0.384 |
| PV | 42,580 | **0.838** | 1.43 | 63,998 | 11.2 | 0.400 |
| PV in check | 1,293 | 0.909 | 1.17 | 1,574 | 4.2 | 0.451 |
| **NonPV** | **648,988** | **0.845** | 1.41 | 574,830 | 12.1 | 0.530 |
| NonPV in check | 71,945 | **0.937** | 1.10 | 20,071 | 3.8 | 0.782 |
| qsearch | 145,577 | **0.842** | 1.20 | 433,437 | 2.3 | 0.251 |
| qsearch in check | 31,085 | 0.932 | 1.07 | 20,901 | 3.5 | 0.598 |

### 2.1 The headline: the TT-move split

| node kind | fhf **with** a TT move | fhf **without** | gap | share of cutoffs without |
|---|---|---|---|---|
| NonPV | 0.9135 | **0.7645** | **14.9 pp** | 46 % |
| PV | 0.8812 | **0.6585** | **22.3 pp** | 20 % |
| PV in check | 0.9393 | 0.6210 | 31.8 pp | 10 % |
| NonPV in check | 0.9747 | 0.9082 | 6.7 pp | 56 % |
| qsearch | 0.9764 | **0.8262** | 15.0 pp | **90 %** |

Weighted out: of NonPV's 266,182 excess move-loop iterations at cut nodes,
**~70 % are at nodes with no TT move**. The static ordering — capture MVV +
capture history, quiet seven-plane history stack — is doing all the work there,
and it is fifteen points worse than a TT hit.

Two corollaries that set the study's direction:

- **Qsearch is the extreme case**: 90 % of its cutoffs have no TT move, and it
  is the only pipeline with *no SEE gate on ordering at all* (`Qcapture` emits
  every capture in raw `7·victim + capture_history` order; the good/bad split is
  a main-pipeline stage). It is also where ordering is most decisive:
  `QS_LATE_MOVE_LIMIT = 2` means that after two non-checking, non-recapture
  captures, **every remaining one is skipped** — mis-order the list and the
  winning capture is never searched at all.
- **Evasions are NOT the weak spot.** In-check fhf is 0.937 (NonPV) and 0.932
  (qsearch), *better* than the not-in-check cases, because evasion lists are
  short. The hypothesis that the impoverished evasion scoring (main + cont₁
  only, versus the main pipeline's seven planes) was costing us was **refuted**
  — and refuted twice, because giving evasion quiets the full stack
  (`evfull=1`) reads **−0.84 pp** on in-check fhf, eight times the null. More
  history is worse there; short lists want the cheap signal.

### 2.2 The cost of mis-ordering, priced

| | NonPV | NonPV in check |
|---|---|---|
| LMR scouts | 1,992,531 | 35,658 |
| scouts that beat alpha and forced a full-depth re-search | 72,466 | 1,287 |
| **re-search rate** | **3.64 %** | 3.61 % |
| nodes spent inside those re-searches | 994,940 | 11,352 |
| as a share of the 4.64 M-node run (nested, so an upper bound) | **21.4 %** | 0.2 % |

Re-searches concentrate at the *second* move: of 25,709 NonPV re-searches in
the wl50 d13 run, **41.5 % happen at move 2** and 71 % within moves 2–4. That
is what "our runner-up ordering is weak" looks like in nodes.

Singular-extension verification costs a further 648,743 nested nodes (14.0 %).

---

## 3. What was screened, and what it cost

All figures: uho1000 @ d10, deltas in percentage points against the same
baseline run, null floor ±0.15 pp.

### 3.1 The two landed wins

Measured with the landed code A/B'd against its own predecessor **inside one
binary** (`BASE="qsdef=0,evmvv=1"`, which reproduces the pre-change node counts
to the last digit — 1,739,588 on wl50 @ d13 and 9,177,610 on uho1000 @ d10):

| arm | fhf NonPV | fhf in-check | fhf qsearch | nodes |
|---|---|---|---|---|
| null floor | ±0.06 | ±0.05 | ±0.15 | ±0.6 % |
| capture history on evasion captures | +0.03 | **+0.76** | −0.03 | −1.23 % |
| quiescence recapture discount (56/16) | +0.02 | −0.11 | **+3.63** | +0.60 % |
| **both — shipped** | −0.09 | **+0.64** | **+3.51** | **−1.31 %** |

Each change moves exactly the statistic it was built to move, five to twenty
times the null, and leaves the others alone.

**Where the constant came from, and the study's own Goodhart warning.** The
quiescence discount is `QS_CAPTURE_DEFENDED_NUM/16 · attacker_value`, and the
*principled* value is 112/16 = 7: that prices the full recapture in
`CAPTURE_MVV_FACTOR`'s own frame, so a defended capture is ordered by
`7·(victim − attacker)`. The fhf curve agrees all the way there and never turns
over —

| `defcapqs` | 4 | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|---|
| Δ fhf qsearch | +0.76 | +0.97 | +1.66 | +2.38 | +3.63 | **+4.63** |
| Δ nodes (paired geomean) | — | — | — | — | **−0.50 %** | **+1.19 %** |
| paired better/worse | — | — | — | — | **519/481** | **493/507** |

— but the search gets **worse** past ~56. `QS_LATE_MOVE_LIMIT = 2` skips every
non-checking non-recapture capture after the second, so a full-recapture
discount does not demote a defended capture, it **deletes** it — including the
ones whose defender is pinned or overloaded. The metric the term was built to
maximise keeps improving while the thing it is a proxy for degrades. 56 is
where nodes are best; the reasoning is in the constant's doc comment so the
next person to "fix" it to the principled value has to read the counter-example
first.

**Evasions deliberately do not get the discount**, and that too is measured:
applying it there cuts the in-check gain from **+0.76 pp to +0.41 pp**. An
evasion list is short and forced — "will I be recaptured" competes with "or I
could just move the king", and the capture is usually right anyway.

### 3.2 The retractions — two arms that were pruning, not ordering

`arm70`'s history knobs move history *magnitude*, and
`r -= stat_score · RED_STATSCORE_NUM()` makes magnitude feed LMR directly. So a
history arm's node win is a pruning win unless a magnitude control says
otherwise. Both of this study's biggest apparent node wins failed that control:

| arm | fhf NonPV | nodes | verdict |
|---|---|---|---|
| `malusdecay=1100` (invert the malus shape) | +0.44 | **−9.90 %** | looked like the study's headline |
| `malusstart=1900` (**magnitude only**, shipped shape) | +0.48 | **−10.42 %** | reproduces all of it |
| `malusdecay=1100,malusstart=700` (shape, magnitude pulled back) | −0.27 | **+1.52 %** | **the shape is inert** |
| `crossmalus=1` (malus the tried quiets when the best move was a capture) | +0.13 | −3.90 % | looked structural |
| `crossmalus=1,malusstart=850` (magnitude-compensated) | **−0.29** | **+5.88 %** | **refuted** |

Both are **retracted as ordering findings.** The malus magnitude is a real
lever on nodes and it is an SPSA parameter, not an ordering improvement; it is
left alone here.

### 3.3 Honest nulls with mechanisms

| arm | reading | mechanism |
|---|---|---|
| `lva=4/8/16` — least-valuable-attacker tiebreak on captures | fhf NonPV −0.02 (inside null) | the good/bad SEE split already prices attacker risk in the main pipeline; a flat LVA term adds nothing it did not already know |
| `qsortfull=1` — full quiet sort instead of the depth-scaled partial one | fhf NonPV −0.02 | **sorting completeness is not the problem**; the depth-scaled `limit` already sorts everything the search consumes |
| `evfull=1` — full history stack on quiet evasions | fhf in-check **−0.84** | evasion lists are short; the extra planes are noise on 4 moves |
| `threatw=0` — delete the quiet threat escape/entry term | fhf NonPV **−0.59** | falsification control **passes**: the term is earning its keep |
| `mainw=8` — halve butterfly-history weight | fhf NonPV **−0.52** | control passes: weight 2 is right |
| `offensew=0` — delete the ARCHON-original (D30) offense term | fhf NonPV **+0.07** (inside null) | **it is not earning its keep**; flagged, not acted on |
| capture piece values vs SF `eca43a97` | 208/781/825/1276/2538, exact | the "capture history can outweigh MVV in our compressed frame" hypothesis is **wrong** — the values are in SF's frame, not ours |
| TT-move preservation on a moveless store | already correct (`tt.rs` rule 1) | the "we clobber TT moves and that is why 46 % of nodes have none" hypothesis is **wrong** |

---

### 3.4 What the landed change COSTS, and the cheaper variant it leaves on the table

A move-ordering change owes the speed side exactly one number: **instructions
per node**, since a functional change moves nodes and total instructions for
two reasons at once. `tools/perf70/ipn70.sh`, both PGO, uho300 @ d11, five
alternating rounds with **zero spread** (`instructions:u` is deterministic):

| | i/node | nodes | total instructions |
|---|---|---|---|
| baseline | 7009.30 | 4,644,479 | 32.564 G |
| landed | **7092.21** (+1.18 %) | 4,565,705 (−1.70 %) | **32.383 G (−0.56 %)** |

The +1.18 % has a named cause and it is not the arithmetic. `Board::threats` is
**lazy on purpose** — `compute_threats` walks every enemy slider, knight and
king, and its own doc comment says "every qsearch node ... threw the result
away". Arming `all_threats()` in the quiescence capture scorer makes every
qsearch node pay that fill.

A **pawn-only** recapture map (`attacks::pawn_attacks_bb`, two shifts, no fill)
was measured as the alternative:

| recapture map | Δ fhf qsearch | Δ fhf in-check | Δ nodes |
|---|---|---|---|
| full threat map, 56/16 (**shipped**) | **+3.51 pp** | +0.64 | −1.31 % |
| pawn-only, 56/16 | +1.58 pp | +0.42 | **−1.97 %** |
| pawn-only, 80/16 | +1.56 pp | +0.62 | −0.03 % |
| pawn-only, 112/16 | +1.69 pp | +0.66 | −0.11 % |

Pawn defenders alone carry **~45 % of the ordering signal** and the pawn-only
variant *saturates* — the constant stops mattering past 56, which is what a
subset-of-the-truth map should do. It is left as a measured option rather than
shipped: the full map wins on the statistic this change exists to move, its
extra per-node cost is more than repaid in the same run (−0.56 % total
instructions), and the games are already running on it. If a future study finds
the qsearch threat fill expensive in wall clock, the swap is one line and its
cost/benefit is already priced here.

### 3.5 Games

`tools/sprt.sh engines/bin/s70-cand engines/bin/s70-base` with
`NODES=25000 BOOK=UHO_Lichess_4852_v1.epd HASH=16`, both sides PGO, both built
from the same source except the two changes. (The prompt's suggested baseline
`engines/bin/archon-s25` is **stale** — it reads 1,799,710 on the wl50 d13
fingerprint against this tree's 1,739,588, so an SPRT against it would have
confounded these changes with everything else that has landed since. A fresh
same-source PGO baseline was built instead.)

**No SPRT verdict — the match was OOM-killed twice, and that is stated as a
failure rather than dressed up as a result.** Pooled over both runs with
`tools/perf70/pool_sprt.py`:

| run | pairs | games | score | Elo | LOS |
|---|---|---|---|---|---|
| `s70-sprt` | 991 | 1,982 | 51.21 % | **+8.42 ± 9.33** | 96.16 % |
| `s70-sprt2` | 40 | 80 | 45.00 % | −34.86 ± 39.00 | 3.83 % |
| **pooled** | **1,031** | **2,062** | 50.97 % | **+6.74 ± 9.10** | **92.68 %** |

LLR reached 0.93 of the +2.94 needed for H1 before the first kill. **The bar
still crosses zero**: this is a positive trend at 2,062 games, not an accepted
change. A third run (`s70-sprt3`) is accumulating.

Pooling is legitimate here and the tool proves its own arithmetic: the runs
differ only in the book's random order, so pairs are exchangeable, and pooling
the **pentanomial** counts keeps the within-round correlation that a trinomial
Elo bar throws away. `--verify` re-derives each run's Elo, nElo and LOS from
its own `Ptnml(0-2)` and prints them beside what fastchess printed — they match
to the last digit shown (the nElo constant is calibrated against fastchess and
says so in the source; Elo and LOS need no constant).

**The operational cause, because it will happen again.** Both kills were the
OOM killer, and both were self-inflicted:

- Run 1 was wrapped in `systemd-run --user --scope -p MemoryMax=4G`. **Page
  cache counts against a cgroup limit**, and a match writing a PGN for two
  thousand games accumulates it — so the cap that was meant to be a safety
  margin is what fired.
- `tools/oomlast.sh` sets `oom_score_adj=1000` **by design**, to make the
  trainer the last thing the kernel picks. That is the standing rule and it
  worked: the trainer lived, the match died. On a memory-tight box the lever is
  therefore **concurrency, not a cgroup cap**.

`tools/perf70/chain70.sh` is the fix, running as the detached `--user` unit
`s70-chain`: short segments (125 rounds each) that each write their own
`sprt.out`, with a `free -m` wait before each so it backs off while the trainer
is hungry instead of competing with it. A kill now costs at most one segment —
and in practice less than that, because **fastchess prints `Ptnml(0-2)` every
`ratinginterval`**, so `pool_sprt.py` reads the last one and a killed segment
still contributes everything up to its final interval. That property is what
made run 1's 991 pairs recoverable at all, and it is verified live: pooling the
two dead runs with the *in-progress* segment gives 1,071 pairs / 2,142 games /
**+7.14 ± 8.89 Elo, LOS 94.24 %**.

Read the running total with:

```
tools/perf70/pool_sprt.py engines/runs/s70-*/sprt.out
```

`tools/sprt.sh` had to be fixed first: it passed **no `EvalFile`**, and an
empty `EvalFile` silently evaluates PSQT — every SPRT run through that script
before 2026-08-01 measured a PSQT engine unless the caller happened to append
the option by hand. It now takes `NET=` (defaulting to
`engines/canonical/net.cenn`) and prints it.

**And the number that looks bad, stated plainly.** The wl50 d13 pin went
1,739,588 → **1,979,892, i.e. +13.8 %**, and the depth-6 PSQT bench went
587,683 → 401,733, i.e. −31.6 %. Neither is evidence of anything: §1.1's null
panel moves the first by up to +18.3 % and the second is a 49-position
depth-6 PSQT run. They are fingerprints. The measurement is §3.1's paired
1000-position −1.31 %, and even that is inside its own ±1.4 % null — which is
why the case rests on the census statistics and the games, and is stated that
way rather than dressed up.

**Gate:** `CHECK_NNUE_GATE=1 CHECK_TEST_THREADS=2 bash tools/check.sh` exits 0.
Eval hash **`2e1ede95…79c4`, unchanged** — this is search, not eval. Both node
pins move, as a functional change must: `Bench:` 587683 → **401733**, NNUE d13
1739588 → **1979892**.

---

## 4. What the study did NOT close, with numbers

The census points at one block far larger than anything landed here, and it is
left open on purpose rather than half-attacked:

**The TT-less deficit itself.** 46 % of NonPV cutoffs arrive with no TT move and
their first-move rate is 14.9 pp worse. Two hypotheses were tested and both are
**dead**, which is why the block is still open:

- *"We clobber TT moves when storing a moveless entry."* False —
  `TtEntry::save` rule 1 already preserves the old move on a same-key moveless
  write.
- *"Our capture history can outweigh MVV because our score frame is
  2.07× compressed."* False — `PIECE_VALUES` is 208/781/825/1276/2538, exactly
  SF's `eca43a97` frame, so `7·victim` spans 1456…17766 against a ±10692
  capture-history bound, the same ratio SF has.

What remains untried, in the order the data ranks it: raising TT move
*availability* at depths 1–3 (76 % / 68 % / 51 % TT-less), and improving the
static quiet ordering specifically at the depth 3–6 band where `fhf` bottoms out
at 0.827–0.839 while `tt_yes ≈ tt_no`.

**The ARCHON-original offense term (D30) does not pay.** `offensew=0` reads
+0.07 pp of NonPV `fhf` — inside the null — for a per-quiet `attacks()` call on
every non-pawn quiet. Deleting it is a speed change, not an ordering one, so it
belongs to a per-node study; it is flagged here, not acted on.

---

## 5. Reproducing

- `tools/perf70/nodes70.sh BIN` — the three node totals (the weak metric; kept
  ONLY because it is what a `Bench:` trailer is — never read a verdict off it).
  Its multi-arm wrapper `screen70.sh` was deleted rather than kept: it is the
  script that produced §1.1's three phantom findings, and a superseded
  instrument left lying around is a trap.
- `tools/perf70/cscreen70.py ARM...` — census screen with the null panel first
- `tools/perf70/screen70.py ARM...` — paired node screen with the null panel first
- `tools/perf70/paired70.py` — the paired statistics themselves
- position sets: `tools/perf70/uho300.fens` (seed 70), `tools/perf70/uho1000.fens`
  (seed 700), both sampled from `engines/books/UHO_Lichess_4852_v1.epd`

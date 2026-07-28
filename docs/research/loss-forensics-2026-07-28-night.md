# Loss forensics, night of 2026-07-28 — the population defect

Companion to `LOSS-FORENSICS-PLAYBOOK.md` (method) and
`TRANSITION-2026-07-28.md` (prior state). Read the two REFUTED banners in the
playbook first: `bbc28fd` and `b22c9c9` retract the gate-attribution mode.

Everything below was measured tonight. Where a number contradicts an earlier
one, the earlier one is named and withdrawn.

---

## 0. THE MODEL — every failure mode, what it needs, and how sure I am

Gap to SF18 at 10+0.1: **−171 to −191 Elo**.

### ESTABLISHED — measured, controlled, and survived a null or a bias check

| finding | size | why it is solid |
|---|---|---|
| **The tuning instrument sampled the wrong POPULATION** — 100% openings, while 64% of our moves are below 21 pieces and our wp loss per move there is 5× higher | — | Needs no oracle. Confirmed against games: it false-positived the whole pfut family at 4σ while the phase corpus agreed with 1,483 games (§1, §5c) |
| **…and the wrong BUDGET** — 12,500 nodes reaching depth 9.9, where our games reach 17–20 | — | Arithmetic, plus a measured verdict flip between budgets (§3m/§3q) |
| **Speed**: SF runs 1.24–1.51× our NPS, we execute 1.45–1.66× the instructions per node, our IPC is 15–17% BETTER | **~51–70 Elo** | Quiet box, startup differenced out, cycles/node ratio matches the NPS ratio exactly. Our 1108k at 21+ matches the known V-768 idle figure (§3b-QUIET) |
| **Eval is at parity or better — static AND qsearch, in every phase** | ~0 | Scored against SF's own deep search, which favours SF, and we still drew level. Conservative (§3e, §3r) |
| **Time management misallocates ~1.29 s** — 82.1% of budget burned by move 33 against SF's 72.4%, while 87% of our wp loss is after move 33 | **~38–44 Elo** (estimate) | Arithmetic on a per-game clock reconstruction. Elo conversion is order-of-magnitude, and one game datum pushes back (§3p) |
| **Move choice is CHAOTIC in parameters** on the positions we get wrong — a ±1 change to an unrelated constant flips 13–20% of them | — | Null panels; and the dither that amplifies it is exact SF parity (§3j, §6b) |
| **The error census**: 278 verified errors → 218 already-lost, **44 decidable** (66% depth-fixable, 23% search-rejects, 11% eval) | 0.51% of moves | Cold 1M oracle with a passing positive control (§3i) |

### UNCERTAIN — measured, then undermined by an instrument fault

| claim | status |
|---|---|
| "Equal-node move quality is worth ~78 Elo" | Priced at 12,500 nodes on the openings corpus — **both** now known wrong. The unbiased replacement is wp loss: **0.005/move vs SF's 0.003**, no error bars yet |
| "Our search is weakest in the middlegame" (§3g) | Same wrong budget/population. Withdrawn-adjacent |
| "At 200k we match SF in the middlegame" (§3m) | **Withdrawn.** Top-1 against an SF oracle rewards being SF: +15.1 points of swing for SF vs +5.2 for us when the oracle deepened (§3q) |
| "The 14 central knobs are at a local optimum" (§5d) | Measured at the wrong budget. Unresolved |

### RETRACTED — six, all mine

2× NPS deficit · corpus blind to game 18 · the RFP attribution · the pfut
relaxation (4σ screen, 0 Elo over 1,483 games) · the carried-state effect (its
replay reproduced the game **less** than a cold search) · §3m.

### WHAT NEEDS TO CHANGE, ranked by evidence × size

1. **Time management** — spend less before move 33. ~38 Elo, favourable trade
   (the time comes from where we already match SF), tunable today.
   **A timed 1,400-game match is running now** on `OPT_SCALE_BASE ×0.5`, verified
   to cut 22% early / 30% mid / 19% late.
2. **Per-node cost** — 1.45–1.66× SF's instructions, of which the threat channel
   is 1.482× in the middlegame (~68 Elo of speed). The eval side of that trade
   has never been isolated; **a threats-free net at the same width** is the one
   experiment that settles D49.
3. **The instruments themselves** — nothing downstream is trustworthy until the
   screen uses a phase-representative corpus at a realistic budget, judged by
   LOSS rather than agreement. Two SF-parity deviations currently rest on the
   old instrument (§3n).

### WHAT DEMONSTRABLY DOES NOT NEED TO CHANGE

The net (parity or better in every phase, and D48X's equal-time −12.6 ± 17 is
what that predicts) · the pruning gates (refuted twice, independently) · NMP,
the dither, movepick structure, rule50 handling (all exact SF parity) · qsearch
(parity).

---

## 1. THE HEADLINE — our tuning instrument samples the wrong population

`tools/nnue/calib/oracle-obs.json` is the corpus behind **every tunable screen
this project has ever run**. It is 9,992 positions, **100% of them 21+ pieces**,
and the FENs sit around move 6–8. Against the errors we actually make in games
we lost:

| pieces | share of errors | share of WIN PROBABILITY lost | share of oracle |
|---|---|---|---|
| ≤ 7 | 66.7% | 36.8% | **0%** |
| 8–12 | 18.3% | 8.4% | **0%** |
| 13–20 | 11.1% | 36.9% | **0%** |
| 21+ | 4.0% | 17.9% | **100%** |

And by move number: **zero errors before move 21**; 87% of win-probability loss
after move 33; 64% after move 46.

**Two unconfounded cross-checks, both stronger than the table above.** The
table can be attacked two ways — lost games run long, so they over-sample the
endgame by exposure, and conditioning on losses conditions on something. Both
attacks fail:

*(a) Where we simply play our moves*, over ALL games in both matches, no
outcome conditioning and no error attribution at all (n = 22,828):

| pieces | ≤7 | 8–12 | 13–20 | 21+ |
|---|---|---|---|---|
| share of our moves | 14.5% | 21.4% | 28.4% | 35.6% |
| share of the oracle corpus | 0% | 0% | 0% | 100% |

**64% of the moves we play are below 21 pieces and the screen cannot see any of
them.** That statement needs no theory of where errors happen.

*(b) Win probability lost PER MOVE* — a rate, which divides exposure out:

| pieces | moves | wp lost | **per 100 moves** |
|---|---|---|---|
| ≤7 | 492 | 1.51 | **0.31** |
| 8–12 | 679 | 0.35 | 0.05 |
| 13–20 | 808 | 1.52 | **0.19** |
| 21+ | 1223 | 0.74 | **0.06** |

Our error rate per move at ≤7 pieces is **5× the rate at 21+**, and 13–20 is 3×
it. The endgame concentration is not an exposure artifact.

**A proxy I tried and threw away, because it is a good trap.** Taking "the ply
where our own eval first drops below −3" as the decisive moment gives the
opposite answer on corpus-v2: 74% at 21+ pieces, 50% by move 20. It is
worthless. In games we lose, our eval at our FIRST move out of book (median
move 8) is already **median −2.20**, 92% of them between −3 and −1 — the UHO
book hands us the losing side, and the proxy was timing the book, not our play.
Anything that thresholds our own eval in a UHO match measures the opening book
first.

So the screen measures the phase that holds ~18% of the damage and is blind to
the other 82% — and the phase where we demonstrably make *no* mistakes at all.

The count column overstates the endgame share (a pawn-threshold detector
inflates sparse positions — a trap already on the refuted list), which is why
the decision rests on the win-probability column. It survives the correction.

**This is not merely low power — the screen can be BLIND to a large real
effect**, and the case is on file: `HINDSIGHT_EVAL_SUM=111` came back
**+0.03 ± 0.15 cp** at the full 9,992 positions — a flat nothing — and then
lost **17.4 Elo** in games. (Precision matters here: that is blindness, not
inversion. The screen's honest verdict was "no effect" on a change worth −17
Elo. The playbook records the separate reason it was ever believed — a 4000-
position run read −0.32 ± 0.23 and 1.4σ was mistaken for a find.) A corpus that
cannot see a 17-Elo change is not a weak instrument, it is the wrong
instrument. It also reframes every dry campaign —
139 of 303 tunables were screened, all of them on openings, and the other 164
were never screened at all.

**Fixed:** `tools/forensics/build_oracle.py` samples positions from our own
games uniformly across each game, so phases appear in the proportion we play
them, and scores every legal move at the same 4k/16k/64k tiers, so
`screen.py --oracle` takes it directly. Positions come from **all** games, not
only losses: conditioning the population on the outcome would bias every future
screen toward whatever we happened to be doing while losing.

---

## 2. THE CLOCK — real, quantified, and NOT the binding constraint

Per-game clock reconstruction over the 120-game rematch (10+0.1):

| move | ARCHON left | SF18 left | diff |
|---|---|---|---|
| 10 | 8.65 | 8.63 | +0.02 |
| 25 | 4.30 | 5.01 | −0.71 |
| 33 | 2.38 | 3.67 | −1.29 |
| 40 | 1.28 | 2.76 | −1.48 |
| 45 | 0.80 | 2.30 | −1.50 |

47.3% of our moves are played with under a second left, against their 25.2%.
Spend as a fraction of the clock in hand runs 1.6× (moves 21–30), 2.35×
(31–40), 2.67× (41–50) of theirs, and by move 41 our *absolute* seconds fall
below theirs — 0.163 s to their 0.184 s. We arrive at move 45 with 35% of
Stockfish's clock, and search 1.5–2.5 plies shallower from move 41 on.

**But this does not explain the blunders, and I initially said it did.** Two
things refute the causal story:

- At moves 41–50 Stockfish gains **1.5 plies from 13% more time**. No branching
  factor converts 13% into 1.5 plies. Most of that gap is not the clock.
- On game 18 Stockfish plays the right move at **~25k nodes**; we spent
  0.049 s ≈ 50–70k nodes in the game. We had a budget at which the reference is
  already correct. More time would not have saved it.

The clock deficit is real and worth fixing on its own terms. It is not the
mechanism behind the errors.

---

## 3. TREE SHAPE — measured at equal nodes, the gap is small

Depth reached at a fixed 200k nodes, Hash 16 (as in the match):

| pieces | ARCHON | SF18 | gap |
|---|---|---|---|
| 21+ | 16.37 | 16.23 | **+0.13** |
| 13–20 | 17.30 | 17.37 | −0.07 |
| 8–12 | 16.90 | 17.50 | **−0.60** |
| ≤7 | 19.53 | 19.97 | −0.43 |

We are *deeper* than Stockfish in the opening at equal nodes and 0.4–0.6 plies
shallower in endgames. That accounts for roughly a third of the in-game
endgame depth gap. **The remainder is UNEXPLAINED** — §3b withdrew the speed
story it was originally attributed to, and the clock buys only ~0.26 plies.

---

## 3b. RETRACTED — "a 2× NPS deficit, worst where we lose"

**The NPS table below is withdrawn. Do not cite it.** I measured relative speed
four times on a saturated box and got four incompatible answers:

| method | 21+ | 13–20 | 8–12 | ≤7 |
|---|---|---|---|---|
| uciprobe, nodes-at-last-info ÷ wall | 1.64× | 1.49× | 1.67× | **2.01×** |
| bench NPS, depth 11 | 1.33× | 1.17× | 1.56× | **0.72×** |
| bench NPS, depth 14–18, first-match parse | 22× | 12× | 22× | 18× |
| bench NPS, depth 14–18, correct parse | **2.18×** | 1.17× | 1.23× | 1.38× |

Three separate faults, and it is worth naming each because they are all
reusable mistakes:

1. **Nodes-at-last-info ÷ wall time is not NPS.** Both engines emit roughly one
   info line per completed depth, so the aborted final iteration — often near
   half the work at EBF ~2 — is invisible. Which engine loses more is
   *random*, depending on where each happened to be in its iteration when the
   clock ran out. My first table was noise, not a 2× effect.
2. **`re.search` takes the FIRST match.** Our bench prints per-position lines
   before the summary, so the third row measured one position's nodes against
   the whole run's time — 22×, an impossible number I should have rejected on
   sight rather than tabulated.
3. **Load.** Even with correct parsing, sequential runs on a box at load ~15
   are not comparable to each other.

### 3b-FIXED. The speed gap, measured properly: ~1.4x, and NOT endgame-specific

The way out of four contradictory wall-clock readings was to stop comparing
runs and instead take instructions, CYCLES, nodes and time from the SAME
process invocation, at the same depth -- plus a fairness check that the box
cannot fake:

| phase | engine | i/node | cyc/node | knps | Gcyc/s |
|---|---|---|---|---|---|
| 21+ (d13) | ARCHON | 9,559 | 14,042 | 395 | 5.54 |
| 21+ (d13) | SF18 | 7,242 | **10,091** | 616 | 6.22 |
| <=7 (d16) | ARCHON | 4,981 | 6,227 | 842 | 5.24 |
| <=7 (d16) | SF18 | 3,615 | **4,961** | 1134 | 5.63 |

**The fairness check.** `cycles/node x nodes/second = cycles/second` is the
effective clock the process actually received. Ours reads 5.54 / 5.24 GHz
against Stockfish's 6.22 / 5.63 -- within 7-12%, so both got a fair share and
wall clock is usable after all. That is a MEASURED verdict on the measurement
rather than another guess about it, and it is the check I should have run
before publishing any NPS number tonight.

**And the arithmetic now closes exactly**, which is what makes this the
trustworthy version: at 21+, a 1.39x cycles/node advantage times a 1.12x clock
advantage predicts 1.56x, and the measured NPS ratio is 616/395 = 1.56x. At
<=7, 1.26 x 1.07 predicts 1.35x; measured 1134/842 = 1.35x.

**Conclusion: Stockfish spends 26-39% fewer CYCLES per node, giving it
1.35-1.56x our NPS -- roughly 60-70 Elo at ~120 per doubling.** The deficit is
real, it is worth having, and it is *slightly worse in the OPENING* than the
endgame -- the exact opposite of the withdrawn claim. It therefore does **not**
explain the endgame-specific depth gap; the clock (we reach move 45 with 35% of
their time) does.

**The one load-immune measurement, and the only speed number I will stand
behind:** instructions per searched node, same positions, same depth —
1.15× (21+), 0.97× (13–20), 1.01× (8–12), **0.90× (≤7)**, with IPC 0.74–0.89
for us against 0.56–0.76 for Stockfish. Per-node cost is comparable, and if
anything we are cheaper in the endgame. **There is no evidence of a large speed
deficit.**

What this costs the model: the endgame depth gap in games (17.7 vs 19.2 at
moves 41–50; 14.7 vs 16.7 at the decisive ply) is real and observed by two
independent instruments, but its CAUSE is now unexplained. Equal-node tree
shape accounts for ~0.5 plies and the clock for ~0.26 of ~1.5. The residual is
open. Re-measure NPS on a quiet box before building anything on it.

<details>
<summary>The withdrawn section, kept for the record</summary>

## 3b-OLD. THE BIGGEST LEVER — a 2× NPS deficit, worst exactly where we lose

corpus-v2 (129 games in, vs SF18 at 10+0.1) corroborates the phase story from
an independent instrument: **51 of our 64 losses are decided in the endgame**
(80%), and at the decisive ply we are at depth **14.7 against the winner's
16.7** — two plies short.

Equal-node tree shape (§3) accounts for ~0.5 plies and the 13% time difference
for ~0.26. The remainder is raw speed, and it is large. NPS measured
interleaved on identical positions, alternating engines:

| pieces | ARCHON | SF18 | SF / ours | ours vs own 21+ | SF vs own 21+ |
|---|---|---|---|---|---|
| 21+ | 271k | 445k | 1.64× | 1.00 | 1.00 |
| 13–20 | 337k | 504k | 1.49× | 1.24 | 1.13 |
| 8–12 | 448k | 746k | 1.67× | 1.65 | 1.68 |
| **≤7** | 545k | **1094k** | **2.01×** | 2.01 | **2.46** |

Stockfish is 1.5–2× faster at every phase, and **its NPS scales up better as
the board empties** (2.46× from its own opening baseline against our 2.01×), so
the deficit is worst in the phase that decides 80% of our losses. At the
project's ~120 Elo per doubling, a 2× endgame speed gap is on the order of
100 Elo in exactly the wrong place.

It is not that Stockfish has a cheap path: **`eca43a97` has no small net** —
zero matches for `smallNet` in `evaluate.cpp`, one network for everything. We
have a smallnet gate (`ARCHON_SMALLNET`, perf study 3 item 8) and it is off by
default for want of a trained cheap net. The standing hypothesis for the gap is
already on file in memory: SF maintains its threat features incrementally where
we replay them.

**Provisional — measured on a box at load ~11, and the playbook is explicit
that NPS needs a quiet one** (there is a logged case of reading 0.755× where
the truth was 0.845×). The ratio is interleaved and self-relative, which is the
robust form, and it sanity-checks against a known idle figure: V-768 measures
1051k idle, my loaded 271k is 3.9× depressed, implying SF ~1.7M idle, which is
ordinary for SF18 single-threaded. Re-measure quiet before quoting it.

*(That was the hedge I wrote at the time, and it was not enough: I then
built a "biggest lever" heading on a number I had just called provisional.
A hedge in the footnote does not license a claim in the title.)*

</details>

## 3b-QUIET. The speed measurement, redone on an idle box

The §3b-FIXED numbers were taken at load ~15. Repeated on a quiet box, with
startup differenced out (perf counts the whole process; bench times only the
search, and loading an 84 MB net is real work SF's embedded net does not do — so
each figure is a depth-D run minus a depth-1 run of the same command):

| phase | ARCHON knps | SF18 knps | **NPS ratio** | i/node ratio | our IPC | SF IPC |
|---|---|---|---|---|---|---|
| 21+ | 1108 | 1420 | **1.28** | 1.50 | **1.73** | 1.50 |
| 13–20 | 1301 | 1611 | 1.24 | 1.45 | 1.67 | 1.44 |
| 8–12 | 1517 | 2021 | 1.33 | 1.57 | 1.69 | 1.45 |
| ≤7 | 1950 | 2944 | **1.51** | 1.66 | 1.83 | 1.68 |

**Our 1108k NPS at 21+ matches the documented V-768 idle figure of 1051k**, which
is the sanity check the loaded measurements never had.

The diagnosis is precise and it is not what I assumed: **we execute ~1.5× more
INSTRUCTIONS per node, while our IPC is 15–17% BETTER than Stockfish's.** So the
deficit is work done, not memory behaviour — data layout and cache use are fine,
and the fix is doing less per node, not laying it out better. That is what makes
the threat channel (§3h) the right target rather than, say, prefetching.

*(Before startup was differenced out, the same run reported an "effective clock"
of 6.4–8.3 GHz — above the 7700X's maximum boost, which is proof the measurement
was wrong rather than merely noisy. An impossible intermediate is worth more than
a plausible one.)*

## 3c. THE DECOMPOSITION — where the ~180 Elo actually goes

`screen.py --reference` scores Stockfish on the SAME positions at the SAME node
budget against the SAME oracle, which removes speed from the comparison and
leaves pure move quality (eval + search shape):

| at 12,500 nodes | cp loss @64k tier | top-1 |
|---|---|---|
| ARCHON | 6.72 | 62.1% |
| SF18 | **5.48** | **64.3%** |
| **headroom** | **+1.24 ± 0.30 cp** | +2.2 pts |

Putting that beside the cycles/node result (§3b-FIXED):

| term | Elo | how it was measured |
|---|---|---|
| **Move quality at equal nodes** | **~74** | +1.24 ± 0.30 cp at the screen's ~60 Elo/cp |
| **Speed**, 1.35–1.56x cycles/node | **~64** | same-run cycles/node, fairness-checked |
| residual — time management, conversion, slop | ~35–50 | by subtraction |
| **total** | **~175** | against an observed 171–191 |

The two measured terms are **roughly equal**, and together they account for most
of the gap. That is a direct answer to the framing this investigation started
from -- "an engine that is almost as fast and has almost as good a net is
getting blown out, so search shape is all wrong." Measured: we are *not* almost
as fast (1.4x, ~64 Elo) and at equal nodes we are *not* almost as good
(1.24 cp, ~74 Elo). It is not one broken thing; it is two comparable deficits.

**Caveat that matters, and a prediction.** This is measured on the openings
corpus (§1) -- the population where our error rate per move is *lowest*
(0.06 wp per 100 moves against 0.31 at <=7 pieces). So **+1.24 cp is most
likely a FLOOR on the quality gap.** The prediction to test the moment the
phase oracle lands: re-run this per phase bucket and the headroom should be
LARGER in the endgame. If it is, the population finding and the quality gap are
the same fact seen twice, and the endgame is where the Elo is.

## 3d. THE QUALITY GAP IS SEARCH, NOT EVAL

Static eval quality against the deep oracle, frame-free (Spearman is immune to
any monotone rescale; the slope is fitted per engine, so no frame conversion can
be got wrong), n = 600 at 21+ pieces:

| | Spearman vs deep truth | RMS residual | fitted slope |
|---|---|---|---|
| ARCHON | 0.742 | 65 cp | **0.56** |
| SF18 | 0.756 | 60 cp | 0.95 |

**Our net is at parity with Stockfish's** -- 1.9% worse in rank correlation, 8%
in residual. That independently corroborates the net-only suite's "near parity"
by a completely different method. As a bonus the fitted slope of 0.56
reproduces the documented frame constant (0.55 / 0.579 / 0.605 from three prior
estimates) without being told it, which is a good sign the method is sound.

Put beside §3c, that closes an inference:

- static eval quality: **parity**
- move quality at equal nodes: **we are 1.24 cp worse**
- therefore the equal-node deficit is in **what the search does with a
  comparable eval**, not in the eval

And §3b-FIXED's speed gap is also a search/implementation property. So **both
measured terms of the ~138 Elo are search.** On this population, better nets
address approximately none of it.

That is a live tension with the current phase focus (the eval-foundation
program), and it is exactly the claim the phase oracle can overturn: this is
measured at 21+ pieces, where our error rate per move is lowest. If the endgame
buckets show our eval falling behind, the conclusion flips. **Do not act on
this until the per-phase version exists.**

## 3e. CONFIRMED ON THE RIGHT POPULATION — eval is at parity in EVERY phase

The §3d inference was flagged do-not-act-on because it was measured at 21+
pieces. The phase oracle landed, so here it is per bucket (mate band excluded --
see below):

| pieces | ARCHON rho | SF18 rho | ARCHON resid | SF18 resid | slope ratio |
|---|---|---|---|---|---|
| 21+ | 0.764 | **0.802** | 72 | 66 | 0.45 |
| 13–20 | 0.848 | 0.850 | 106 | 112 | 0.45 |
| 8–12 | 0.951 | 0.957 | 99 | 100 | 0.47 |
| **≤7** | **0.960** | 0.939 | 105 | 106 | 0.46 |

**Our static eval is at parity in every phase and slightly BETTER at <=7
pieces.** It does not degrade as the board empties. The endgame deficit is
therefore not eval, on the population where the endgame is actually represented.

Two side results worth keeping:

- **The frame does not drift by phase.** Our slope against deep truth runs
  0.39 / 0.43 / 0.51 / 0.55 and Stockfish's 0.87 / 0.95 / 1.09 / 1.19 -- both
  rise as pieces come off (static evals regress toward the mean of the deep
  value), but the RATIO is a near-constant 0.45–0.47. A single frame constant is
  legitimate; it just is not the same estimator as the 0.55–0.64 figures, which
  measure something else.
- **The mate-band filter is load-bearing.** Unfiltered, the same table reported
  RMS residuals of 1855 cp (8–12) and 4232 cp (<=7) and fitted slopes of 2.18
  and 4.33 — it was measuring mate DISTANCES, not eval quality, exactly the
  "sparse corpora are ±20000-dominated" trap already on the list. `--max-abs-truth`
  now defaults to 1000.

**So the conclusion in §3d survives the population check: both measured terms of
the ~138 Elo are search.** Better nets address approximately none of the
measured gap, in any phase. That is the sharpest strategic result of the night
and it is now safe to act on.

## 3f. THE MECHANISM — our marginal return on search effort is lower

`--reference` across a 16x node range, openings corpus (no mate contamination
there, so the cp means are clean):

| budget | ARCHON cp | SF18 cp | headroom | ARCHON top-1 | SF18 top-1 | top-1 gap |
|---|---|---|---|---|---|---|
| 3k | 10.33 | 9.51 | −0.82 ± 0.5 | 56.6% | 58.0% | 1.4 pts |
| 12.5k | 6.72 | 5.48 | −1.24 ± 0.3 | 62.1% | 64.3% | 2.2 pts |
| 50k | 4.49 | 3.52 | −0.97 ± 0.2 | 66.9% | **70.6%** | **3.7 pts** |

Two readings, and they agree:

- **The cp headroom does not shrink with budget.** If our deficit were "we need
  more nodes", 16x the nodes would close it. It does not move (0.82 / 1.24 /
  0.97). Expressed in doublings-equivalent — dividing by the local cp-per-
  doubling, 1.75 then 1.12 — the handicap is 0.47 / 0.89 / 0.87 doublings, i.e.
  flat to slightly GROWING.
- **The top-1 gap widens monotonically**, 1.4 → 2.2 → 3.7 points. Our top-1
  climbs 10.3 points over the range where Stockfish's climbs 12.6.

`screen.py`'s own guidance for this table is "if ours flattens while the
reference keeps climbing, more search stops helping us and that is a search-
shape problem." That is what it shows. **Stockfish converts marginal nodes into
correct moves better than we do**, so the deficit is not depth-limited and not
eval-limited (§3e): it is in what the search spends its nodes ON — ordering,
pruning calibration, extensions.

**And that is the same story as §5b.** Over-pruning is precisely a mechanism
that caps marginal return: you spend more nodes and keep discarding the same
lines, so extra budget buys less than it should. The pfut result (five arms
agreeing that we prune too much, the best halving the blunder rate) and this
measurement are two views of one defect. That convergence is the strongest
reason to believe the pfut direction is real rather than a screen artifact.

## 3g. WHERE THE QUALITY GAP LIVES — the MIDDLEGAME, and my prediction was backwards

Equal-node headroom against SF18, per phase, on the validated corpus:

| phase | share of our moves | ARCHON cp | SF18 cp | **headroom** | ARCHON top-1 | SF18 top-1 |
|---|---|---|---|---|---|---|
| 21+ | 35.6% | 14.11 | 11.30 | **+2.81 ± 0.87** | 54.9% | 58.7% |
| 13–20 | 28.4% | 13.43 | 11.91 | +1.52 ± 1.05 | 57.4% | 59.5% |
| 8–12 | 21.4% | 13.42 | 12.66 | +0.76 ± 1.27 | 48.7% | 50.8% |
| **≤7** | 14.5% | 18.69 | 20.70 | **−2.01 ± 2.51** | 37.4% | 38.2% |

**Monotone in piece count, and at <=7 pieces we are BETTER than Stockfish at
equal nodes.** §3c predicted the opposite ("the headroom should be LARGER in the
endgame"); that prediction is refuted. The sign at <=7 is only 0.8σ on its own,
but it agrees with the independent eval measurement (§3e: our rho 0.960 against
SF18's 0.939 at <=7), so two unrelated instruments put us ahead there.

Weighted by how often we play each phase, the quality gap is
0.356·2.81 + 0.284·1.52 + 0.214·0.76 + 0.145·(−2.01) = **1.30 cp ≈ 78 Elo**,
which with §3b-FIXED's ~64 Elo of speed accounts for ~142 of the observed
171–191, the rest plausibly time management and conversion.

### The apparent contradiction, and why both facts hold

§1 says our win-probability loss per move is highest at ≤7 pieces (0.31 per 100
moves against 0.06 at 21+). §3g says our search is *strongest* there. Both are
true and they are not in tension:

- **Where our search is weakest: the middlegame.** At equal nodes SF18 converts
  21+ positions 3.8 top-1 points better than we do.
- **Where our mistakes are most expensive: the endgame.** A single error in a
  7-piece position turns a draw into a loss, so it costs far more win
  probability per centipawn than the same error at move 25.

So the two findings point at *different work*: the middlegame is where search
quality is available, and the endgame is where error-avoidance pays. Neither is
the "endgame eval weakness" I was circling for most of the night — that
hypothesis is dead twice over (§3e, §3g).

**Caveat on the ≤7 row:** cp loss is high for BOTH engines there (18.69 / 20.70)
and top-1 is low for both (37.4% / 38.2%), because sparse positions have many
equivalent moves and the oracle's single "best" is often one of several. The ≤7
column is the noisiest in the table; trust its sign, not its size.

## 3h. WHAT NEEDS TO CHANGE — and it is a decision the project already has queued

The profile answers "which mechanism", on a quiet box, depth-14 middlegame bench:

| % cycles | function |
|---|---|
| **20.9%** | `sota_threats::ThreatContext::at_full` |
| 11.9% | `sota_head::SotaHead::forward_into` |
| 11.7% | `search::Worker::search::<NonPv>` |
| 10.4% | `sota_accumulator::HalfkaContext::current_full` |
| 7.4% | `sota_loader::load_v2_bytes` *(startup, not search)* |
| 5.9% | `movepick::init_quiets` |
| 3.9% | `HalfkaContext::finny_refresh_into` |
| 3.9% | `sota_threats::record_square_touch` |

Eval is ~53% of cycles. **Correction to my first reading of this:** `at_full` and
`current_full` are NOT from-scratch recomputes — they walk back to the nearest
computed ancestor and fold deltas; "full" means the full accumulator VALUE, not
full recompute. The incremental machinery works. The cost is inside the fold:
`build_deltas` regenerates the threat EVENTS per ply through magic-table geometry
queries, because our threat features are **relational** (piece-attacks-square) so
one move changes many of them and finding which requires geometry — where
Stockfish's positional features have O(pieces-moved) deltas. That is a
consequence of the D40 feature design, not a bug.

Priced by switching the threat channel off (the `ARCHON_SMALLNET` gate forced
everywhere gives HalfKA-only evaluation; the head is miscalibrated that way but
NPS does not care):

| phase | threats on | HalfKA-only | speedup | Elo of speed |
|---|---|---|---|---|
| 21+ | 1,100k nps | 1,631k nps | **1.482×** | **+68** |
| ≤7 | 2,304k nps | 2,468k nps | 1.071× | +12 |

**This confirms rather than discovers.** `perf-study-7` already measured
HalfKA-only at +25% to +39% instructions and "up to ~1.5–1.6× cycles on
middlegames, ~45–70 Elo of speed", against threats' eval value "bounded above by
exp-0065's +32.5 ± 17.3 (still never isolated)", and concluded the removal (D49)
is positive-EV. My 1.482× on a different instrument (bench NPS via the gate,
versus their instruction counting on a stripped net) lands inside their band.
Credit where it belongs.

**What tonight adds is the PHASE PROFILE, and it sharpens the decision.** The
prior study measured wl50 and "middlegame". The cost nearly vanishes in the
endgame — 1.071× at ≤7 against 1.482× at 21+. Put beside §3g:

| phase | our quality gap vs SF | threat speed cost |
|---|---|---|
| 21+ | **+2.81 cp (worst)** | **1.482× (worst)** |
| 13–20 | +1.52 cp | — |
| 8–12 | +0.76 cp | — |
| ≤7 | −2.01 cp (we are ahead) | 1.071× (negligible) |

**The threat channel's cost and our quality deficit sit in the same phase.**
Removing it buys ~68 Elo of speed exactly where we are weakest and costs almost
nothing where we are already ahead. That alignment was not visible in either
measurement alone, and it is the strongest argument for D49 on the table.

**What is still missing to close the gate honestly:** the eval side has never
been isolated. exp-0065's +32.5 ± 17.3 is the whole D40 arch against a classic
net, not threats alone, and my speed number cannot be paired with an eval number
from a different comparison. **A net trained without the threat channel at the
same width is the one experiment that would settle it**, and nothing tonight
substitutes for it.

## 3i. THE MECHANISM HISTOGRAM — every failure mode, counted

`harvest2` over corpus-v2 (320 games, 8,548 of our moves in lost games, cold
1M-node oracle, positive control passed at 258 cp) found **278 errors >= 100 cp
(3.3%)**. `classify.py` then split them:

| verdict | n | share of all |
|---|---|---|
| **ALREADY_LOST** (best move still leaves wp < 0.10) | **218** | **78.4%** |
| MINOR_WP (< 0.08 wp lost) | 16 | 5.8% |
| DEPTH_vs_EVAL | 15 | 5.4% |
| DEPTH | 14 | 5.0% |
| **SEARCH_REJECTS** | 10 | 3.6% |
| EVAL | 5 | 1.8% |

**Of the 44 DECIDABLE errors:**

| verdict | n | share | what fixes it |
|---|---|---|---|
| DEPTH_vs_EVAL | 15 | 34.1% | plies (a better eval would do it cheaper) |
| DEPTH | 14 | 31.8% | plies — speed or clock |
| **SEARCH_REJECTS** | **10** | **22.7%** | **a search MECHANISM change, nothing else** |
| EVAL | 5 | 11.4% | a better net |

| phase | DEPTH | DEPTH_vs_EVAL | SEARCH_REJECTS | EVAL |
|---|---|---|---|---|
| 21+ | 4 | 6 | 3 | 1 |
| 13–20 | 7 | 6 | **6** | 1 |
| 8–12 | 3 | 2 | 1 | 1 |
| ≤7 | 0 | 1 | 0 | **2** |

### What it says, and it corroborates every aggregate finding

- **Two thirds of real errors are depth-fixable** (29 of 44). That is the same
  answer as §3b-FIXED (we are 1.24–1.51× slower) and §2 (we reach move 45 with
  35% of their clock) arriving from a completely different direction.
- **Only 11% are eval** — and 2 of those 5 sit at ≤7 pieces. Consistent with
  §3e/§3g: eval is at parity, and what little eval weakness exists is in the
  endgame.
- **23% are SEARCH_REJECTS**, concentrated at 13–20 pieces (6 of 10). This is
  the bucket §3f predicted from the marginal-return curve: the search discards
  what the eval already knows. It is the only bucket a mechanism change is
  needed for, and it now has **ten instances — enough for the
  three-independent-losses rule** for the first time.

### A correction to the standing blunder rate

The bar in `CLAUDE.md` cites "1.5% of our moves lose >= 100 cp — about ONE per
lost game", from a 260-move sample. At 8,548 moves the raw rate is 3.3%, but
**78.4% of those are in positions where the best move still leaves us under 10%
win probability.** Those are not defects; they are the consequence of a UHO book
that hands us the losing side (§1: our eval at the first move out of book in a
lost game is already median −2.20) plus having already lost.

The decidable-error rate is **44/8,548 = 0.51% of moves, ~0.14 per lost game**.

This does not soften the bar — 44 real errors in 320 games is 44 too many, and
the bar is about not tolerating them rather than about their rate. But the rate
should be quoted correctly, and "one blunder per lost game" is roughly 7× too
high because it counted moves in dead positions.

## 3j. THE CAPSTONE — on the positions where we err, our move choice is CHAOTIC

The SEARCH_REJECTS bucket (§3i) is the only one needing a mechanism change, and
it had a clean-looking hypothesis. Its 10 instances are **7 quiet moves**, and in
**8 of 10 both statics are strongly negative** — we are already worse and the
right move is a quiet that limits the damage. Razoring and parent/capture
futility all fire on `eval << alpha`, which IS the losing regime. So: the gates
prune the quiet that would hold the position.

Tested over all 10 at the depth reached in game, with a null panel:

| arm | found /10 | | null arm (semantically inert) | found /10 |
|---|---|---|---|---|
| control | 2 | | `RAZOR_BASE` 465→464 | 3 |
| −razor | 4 | | `RAZOR_BASE` 465→466 | 2 |
| −pfut | 3 | | **`PFUT_BASE` 40→41** | **5** |
| −cfut | 4 | | `PFUT_BASE` 40→39 | 2 |
| −qsfut | **5** | | `CFUT_BASE` 231→232 | 4 |
| −razor −pfut | 2 | | `QS_FUTILITY_MARGIN` 335→334 | 2 |
| ALL eval gates off | 4 | | | |

**Real arms span 2–5. Nulls span 2–5.** `PFUT_BASE 40→41` ties the best real arm
and beats turning every eval gate off. The hypothesis is refuted on its own
evidence set, and so is any gate explanation of this bucket.

### What that actually means — and it is the night's most useful conclusion

**On the positions where we make decidable errors, our move choice is chaotic
with respect to parameters.** A ±1 change to an unrelated constant moves the
outcome as much as removing a whole pruning gate. These are positions where
several moves are close, the oracle's single "best" is one of a near-equal set,
and which one our search lands on is effectively a coin flip.

That single fact explains every dead end of the night:

- **why single-position debugging kept failing** — game 18, the gate census, the
  pfut attribution: all were reading a coin flip as a mechanism;
- **why gate attribution was refuted twice, independently** — there was never a
  gate signal to find;
- **why a 4σ screen result converted to 0 Elo** — tuning moved the aggregate
  statistic slightly and the per-position outcomes shuffled underneath it.

**And it tells you what tuning can and cannot do.** It cannot target specific
failures, because the parameter → move-played map is chaotic on exactly the
positions that matter. It can only shift the DISTRIBUTION the coin flip is drawn
from. Aggregate statistics say the central knobs are already at a local optimum
(§5d), so the distribution is not movable by tuning them.

**Therefore the remaining levers are non-parametric**, and both are already
identified and quantified:

1. **Speed** — the threat channel at 1.482× in the middlegame, ~68 Elo, the
   phase where our quality gap is worst (§3h). This is the D49 decision.
2. **Structural search mechanisms** — whatever Stockfish does that converts
   marginal nodes into correct moves better than ours (§3f). Not a constant;
   a mechanism.

Two thirds of decidable errors being depth-fixable (§3i) is the same statement
arrived at from the corpus: make the search see more per second, and the
distribution improves. Nothing else on the table does.

## 3k. A NEW MODE — carried search state costs us moves, and it is the HISTORY tables

The one thing tonight that survived its own null panel.

**The observation.** 11 of 29 depth-fixable errors are found by a COLD search at
or below the depth the game itself reached. That is not evidence on its own — the
corpus selects positions where the game erred, and with a chaotic argmax a cold
search would land right some of the time by luck. So the control holds position
and depth fixed and varies ONLY the state: `cold` (ucinewgame, search directly)
against `replay` (feed the real game's moves, searching at every one of ours, so
TT and histories arrive as the game left them).

| outcome | n | share |
|---|---|---|
| both_bad | 23 | 52.3% |
| **cold_ok_replay_bad** | **11** | **25.0%** |
| both_ok | 7 | 15.9% |
| replay_ok_cold_bad | 3 | 6.8% |

**11 vs 3 discordant, McNemar two-sided p ≈ 0.057.** Reproduced as counts:
**cold 18/44, replay 10/44.** Carried state costs ~8 of 44 decidable errors.

*(A round-indexing bug had to be fixed first: fastchess `-repeat -games 2` puts
TWO games under one round number, and keeping only the first silently dropped 26
of 44 rows. At half power the split was 5–3, p ≈ 0.23 — indistinguishable from
noise. The finding exists only because the bug was found.)*

**Which state?** Two candidates were separable by option, and both were tested:

| arm | found /44 | what it isolates |
|---|---|---|
| cold | **18** | nothing carried |
| replay + `Clear Hash` | 12 | histories carried, **TT fresh** |
| replay | 10 | everything carried |
| replay, correction history ZEROED | **6** | corr history removed |
| cold, correction history ZEROED | 16 | — |

- **TT staleness accounts for ~2 of the 8** (10 → 12 with a fresh TT).
- **The move-ordering history tables account for ~6 of the 8 (75%).**
- **Correction history is EXONERATED and is actively earning its place**: zeroing
  it makes replay *worse* (10 → 6) and cold worse (18 → 16). That was my first
  hypothesis and it is refuted in the opposite direction.

**The mechanism, stated as a hypothesis with a test.** Butterfly / continuation /
pawn / low-ply history generalise across positions by design, which is why they
are not cleared between moves. The proposal is that ours decays too SLOWLY, so
preferences learned earlier in the game mis-order moves later. That is testable
with the history-update knobs (bonus, malus, decay numerators, the continuation
divisor) against the 18-versus-10 target, with a null arm to keep it honest.

### RETRACTED — the replay harness does not reproduce the game

I ran two more controls and the second one killed it.

**The selection control passed, impressively.** On positions where the game
played the ORACLE's move (n=44), cold agrees 37/44 and replay only 20/44.
Selection predicts the opposite (replay mimics the condition that was right), so
the 18-vs-10 gap was not selection. The effect looked real and large.

**The fidelity control killed it.** On the error set the game played a specific
WRONG move in each position. If the replay carried the game's state, it should
reproduce that move more often than a cold search does:

| | reproduces the game's actual move |
|---|---|
| cold | **17/44 = 39%** |
| replay | **8/44 = 18%** |

**The replay reproduces the game's move LESS often than a cold search.** It is
not a state-carrying version of the game; it is a third condition. So
"84% → 45%" measures my harness, not the engine, and every number in this
section is withdrawn.

**Why the harness was wrong:** it searched every preceding move at the TARGET
position's depth. In a real game the early moves get far less depth, so the
replay did much more search, accumulated much more history and churned a 16 MB TT
far harder than the game ever did. `tools/forensics/state_effect.py` replays each
move at the depth the PGN records FOR THAT MOVE and re-runs both controls; until
its fidelity number is good, there is no state finding.

**The lesson worth keeping.** The selection control passing made this look solid,
and I nearly wrote it up as the night's one survivor. What saved it was asking a
question about the INSTRUMENT rather than the hypothesis: *does my replay
reproduce the thing it claims to reproduce?* A control that validates the
comparison is not the same as a control that validates the apparatus, and this
mode needed both.

## 3m. THE SCREEN'S BUDGET IS AS WRONG AS ITS POPULATION WAS — §3g OVERTURNED

Two more parities first, both measured tonight:

- **Static eval at the root: parity** in every phase (§3e).
- **Qsearch at the leaves: parity.** rho against deep truth at depth 1 (the
  qsearch-dominated probe) is 0.821 / 0.705 / 0.863 / 0.900 against Stockfish's
  0.835 / 0.733 / 0.871 / 0.907 — gaps of −0.03 to −0.007. At depth 4 we are
  AHEAD in three of four buckets. So nothing the search *computes* is behind.

That leaves tree shape. Measuring depth AND top-1 together at two budgets, same
corpus, same method:

| pieces | Δdepth @12.5k | Δtop-1 @12.5k | Δdepth @200k | **Δtop-1 @200k** |
|---|---|---|---|---|
| 21+ | −0.38 | −1.0 | **+0.14** | **+1.5** |
| 13–20 | −0.43 | −5.0 | −0.32 | **+2.0** |
| 8–12 | −1.05 | −5.5 | −0.75 | **−7.0** |
| ≤7 | −0.81 | −5.5 | −0.36 | −2.5 |

**At 200k nodes we match or beat Stockfish in the middlegame and lose at 8–12
pieces.** At 12,500 we lose everywhere. The verdict flips with budget.

**So §3g is overturned.** "Our search is weakest in the middlegame (+2.81 cp at
21+)" was measured at 12,500 nodes — where we reach depth **9.9**, against the
17–20 our real games reach at 10+0.1. The screen's budget describes a different
engine than the one that plays.

**On the numbers that survive** (n=200/bucket, so a 7-point top-1 gap is ~2σ and
a 2-point one is not): the only clear deficit at a realistic budget is **8–12
pieces, −7.0 top-1 and −0.75 plies**. The middlegame reversal is inside noise;
the honest statement there is parity.

**And that finally agrees with the corpus.** §1 measured our win-probability loss
per move as highest below 21 pieces (0.31 per 100 at ≤7, 0.19 at 13–20, against
0.06 at 21+), and §3i put 13 of 44 decidable errors in the 8–12 and ≤7 buckets
despite those being only 36% of our moves. Two instruments that disagreed under
the 12,500-node lens agree at 200k.

**The instrument lesson, which is the same one twice.** The screen was wrong
about WHERE we play (100% openings, §1) and it is wrong about HOW HARD we think
(12,500 nodes against a real 100k–1M). Both distort the same way: they describe a
regime we do not play in. `--budgets` exists and its own help text warns that "a
real bullet move at 8+0.08 gets ~1.3M nodes on this box, 26× the 50k default" —
the warning was in the tool and I read past it for most of the night.

## 3n. TWO SF DEVIATIONS REST ON THE DISCREDITED INSTRUMENT

`params.rs` contains exactly two constants whose default was set by a cp screen,
and both are **deliberate deviations from Stockfish parity**, left off because
the screen said they cost ~0.4 cp:

| constant | what parity would do | recorded reason |
|---|---|---|
| `QS_UPREP = 0` | run the upcoming-repetition alpha raise in qsearch too (SF `search.cpp:1606`) | "+0.47/+0.20/+0.23 cp WORSE by tier at +0.6% nodes — parity does not pay" |
| `RED_TTPV_TTMISS_REBATE = 0` | grant the ttPv reduction rebate on a TT miss, as SF's `VALUE_NONE` sentinel does | "+0.44/+0.30/+0.34 cp WORSE at +5.3% nodes — parity does not pay" |

Both were measured 2026-07-27 at **n=1600 paired**. That evidence is compromised
three separate ways, every one of them established tonight:

1. **Wrong population** — the openings corpus, which produced a coherent
   four-arm false positive on the pfut family (§5c).
2. **Wrong budget** — 12,500 nodes, where we reach depth 9.9 and the phase-by-
   phase verdict differs measurably from the 200k regime (§3m).
3. **Wrong power** — n=1600 gives ~0.8 cp of paired sem on this kind of corpus
   against effects of 0.2–0.5 cp. The playbook's own rule is `--limit 0` (9,992)
   for a verdict; 1600 is a fifth of that.

So we are deviating from the reference engine on evidence that no longer stands.
Both are cheap to re-test in the right regime, and both are in the 200k arm
sweep (`sfparity_upre`, `sfparity_ttpv`, `sfparity_both`).

This is a small audit with a general point: the invalidated instrument did not
only waste screening time, it **set defaults**. Two is a reassuringly small
number — most constants come from the SF transcription rather than from our
screens — but it is not zero, and those two were chosen precisely where we chose
to differ from Stockfish.

## 3p. TIME MANAGEMENT PRICES AT ~38-44 Elo — the best-grounded lever

§2 established the clock deficit and then argued it is **not** the blunder
mechanism, which is true and which caused me to leave it un-priced for the rest
of the night. That was a mistake: "not the blunder mechanism" does not mean "not
worth Elo". Converting the measured curve into a budget share:

| by move | budget available | ARCHON spent | SF18 spent |
|---|---|---|---|
| 10 | 11.00 s | 2.35 (21.4%) | 2.37 (21.5%) |
| 25 | 12.50 s | 8.20 (65.6%) | 7.49 (59.9%) |
| 33 | 13.30 s | 10.92 (**82.1%**) | 9.63 (**72.4%**) |
| 40 | 14.00 s | 12.72 (90.9%) | 11.24 (80.3%) |
| 45 | 14.50 s | 13.70 (**94.5%**) | 12.20 (84.1%) |

Identical through move 10, then we pull away. **By move 33 we have burned 1.29 s
more of the same budget.** Redistributed over moves 33–60 that is +0.046 s/move
against a measured baseline of 0.16 s/move — **+29% time, 0.37 doublings,
≈44 Elo on those moves.** 87% of our win-probability loss from move errors happens
after move 33, so ~38 Elo of it is realisable. Order of magnitude, not a constant.

**And the trade looks favourable rather than merely neutral.** The time comes out
of moves 1–32, which hold 13% of the win-probability loss — and §3m showed we
already match Stockfish's top-1 in the middlegame at realistic budgets, so the
marginal value of a millisecond is genuinely lower there. We would be moving time
from where we are already good to where we are not.

**This matters more if §3c's ~78 Elo quality term shrinks** (which §3m suggests it
must, since it was priced at 12,500 nodes). If quality falls toward zero, the
decomposition needs ~78 Elo from somewhere, and TM at ~38 plus the residual is
where it goes. `headroom200k.py` is measuring that now.

**A tension with this thesis, stated rather than buried.** `tm10c` finished at
400 games: arm **51.4%, +9.6 ± 12.3 Elo**, and its clock curve shows the arm held
**less** clock at every single move (−0.07 to −0.34 s) and searched **shallower**
late (ddep −0.25 at move 40, −0.26 at 45, −0.43 at 60). So an arm that moved the
clock the WRONG way did not lose. If holding clock back were worth 38 Elo,
spending it earlier should have cost some.

Two things keep that from refuting the thesis, and neither is an excuse:
- **The perturbation is a tenth the size.** tm10c shifted 0.2–0.3 s; the gap to
  Stockfish is 1.29–1.50 s. It tested a nudge, not the thesis.
- **±12.3 Elo.** +9.6 is consistent with anything from −3 to +22, so it does not
  measure the sign of a 5–10 Elo effect either way.

It is still the only game-truth datum pointing away from the thesis, and it lowers
my confidence in the ~38 Elo estimate from "well-grounded" to "worth one match".

**What is untested: an arm in the RIGHT direction.** `tm10c` tested
frame-scaled `FALLING_EVAL_*` and scored +10.8 ± 13.5 over 400 games — but the
arm-verify showed it holds **less** clock at every move (−0.10 to −0.36 s) and
searches shallower late. It moved the wrong way and still did not lose. The
correctly-directed arm — spend less before move 33 — has never been run.

The knob shape matters and the obvious ones are traps. `OPT_SCALE_CLOCK_CAP`
looks right but binds in BOTH regimes (when it binds, `optimum = 0.19404 · time`
regardless of ply, a flat 19.4%-per-move decay), and our measured spend is 4.9%
early against 31.8% late, so the cap is not what governs either end.
`OPT_CONSTANT_*` multiplies the ply term and so cuts LATE harder — backwards.
`OPT_SCALE_BASE` is the only one that is differential in the right direction:
per its own doc comment it is 42% of `opt_scale` at ply 10 and 30% at ply 40.
Halving it cuts early ~21% and late ~15%. Mild, but the right sign.

**Testing it needs TIMED games** — `sprt.sh` is explicit that fixed-node proxies
are "never valid for time-management changes" — so this one cannot ride the
node-based harness, and it wants the quiet box.

## 3q. TOP-1 AGREEMENT AGAINST AN SF ORACLE IS BIASED TOWARD SF — §3m WITHDRAWN

Measured at 200k nodes on 3,000 positions, against a **4M-node Stockfish** oracle:

| pieces | ARCHON top-1 | SF18 top-1 | ARCHON wp loss | SF18 wp loss |
|---|---|---|---|---|
| 21+ | 66.7% | **75.1%** | 0.008 | 0.004 |
| 13–20 | 71.8% | **79.9%** | 0.004 | 0.003 |
| 8–12 | 58.0% | **74.3%** | 0.003 | 0.001 |
| ≤7 | 50.0% | **62.7%** | 0.003 | 0.001 |
| **ALL** | **64.9%** | **75.1%** | 0.005 | 0.003 |

That flatly contradicts §3m, which used a **64k** oracle and had SF at 60.0% and
us at 61.5% in the middlegame. Same engines, same budget, same positions — only
the oracle's depth changed. The reason is structural:

| engine, at 200k | vs 64k oracle | vs 4M oracle | swing |
|---|---|---|---|
| SF18 | 60.0% | **75.1%** | **+15.1** |
| ARCHON | 61.5% | 66.7% | +5.2 |

**The oracle IS Stockfish, so "agreement" partly measures being Stockfish**, and
the reward grows as the oracle deepens past the arm. SF@200k agrees with SF@4M
because they are the same search; against a 64k oracle SF@200k OUTRUNS its judge
and gets marked down for knowing better. Our agreement barely moves, because we
were never the same search.

**So §3m is withdrawn.** "At 200k we match or beat Stockfish in the middlegame"
was SF being penalised for outrunning a shallow oracle, not us catching up. The
§3g reading it overturned may stand after all — and both must be redone on a
metric that is not identity-biased.

**The metric that is not biased is LOSS, not agreement.** Cp/wp loss asks how
much value the chosen move gives up, which does not care which engine proposed
the reference. That is why `screen.py` uses cp loss, and I should not have
introduced a top-1-based instrument without checking it against that.

On loss, we give up **0.005 wp per move against Stockfish's 0.003** — a real but
much smaller gap than the top-1 columns suggest, worst at 21+ (0.008 vs 0.004)
and 8–12 (0.003 vs 0.001).

**Two caveats before anyone uses those wp figures:**
- **No error bars yet.** The magnitudes are 0.001–0.008 and I have not computed
  their sem. Until I do they rank, they do not decide.
- **The cp column is mate-contaminated and unusable** — 148.7 cp at 8–12 and
  422.9 at ≤7, giving an aggregate headroom of **−11.76 ± 35.37 cp**. The
  deep-top1 oracle stores raw Stockfish scores including mate values; the
  ±1000 clip that fixed the phase corpus was never applied here. Fix the builder,
  not the reader.

## 3r. THE BIAS CUTS THE OTHER WAY FOR THE EVAL FINDINGS — they are CONSERVATIVE

§3q established that comparing two engines against a Stockfish-generated
reference rewards being Stockfish. That bias does not only invalidate things — it
also means some findings are stronger than they were stated.

**Both eval measurements are scored against SF's own deep search.** §3e correlates
each engine's STATIC eval with SF@64k's value; §3m correlates each engine's
depth-1 (qsearch) value with the same. Stockfish's static eval and its deep search
share an evaluation function at the leaves, so any systematic quirk of that net
appears in both terms and inflates SF's correlation. **Stockfish is marked by its
own examiner.**

And under that handicap we still came out level:

| | ARCHON | SF18 | with the bias removed |
|---|---|---|---|
| static rho, 21+ → ≤7 | 0.764 / 0.848 / 0.951 / **0.960** | 0.802 / 0.850 / 0.957 / 0.939 | ours is **at least** this good |
| qsearch rho at depth 1 | 0.821 / 0.705 / 0.863 / 0.900 | 0.835 / 0.733 / 0.871 / 0.907 | gaps of −0.03 shrink or reverse |

So "our eval is at parity" is the **conservative** reading; the unbiased truth is
parity-or-better. That makes the strategic conclusion firmer rather than shakier:
**a better net addresses approximately none of the measured gap**, and the D48X
result (equal-time −12.6 ± 17) is what that predicts.

**The general rule this session earned:** when a measurement uses one of the
compared parties as the reference, work out which way the bias runs BEFORE
reading the result. It invalidated the top-1 comparisons (§3q) and it fortifies
the eval ones — same bias, opposite consequences, because in one case the
reference favoured the party I was measuring against and in the other it favoured
the party I was measuring.

## 3s. THE TM THESIS IS REFUTED — by a correctly-built arm, and I know why

The arm did exactly what it was designed to do. At 119 games of a timed 10+0.1
match, `OPT_SCALE_BASE ×0.5`:

| move | arm clock | base clock | diff | arm depth | base depth | Δdepth |
|---|---|---|---|---|---|---|
| 10 | 9.15 | 8.79 | +0.35 | 18.0 | 18.5 | **−0.51** |
| 30 | 5.07 | 4.02 | +1.05 | 18.1 | 18.6 | **−0.47** |
| 45 | 3.04 | 1.91 | **+1.12** | 21.1 | 20.6 | **+0.47** |
| 60 | 1.64 | 0.96 | +0.68 | 23.1 | 22.3 | +0.83 |
| 70 | 1.09 | 0.70 | +0.38 | 24.9 | 23.4 | **+1.50** |

Time-trouble moves fall from ~35% to **16.4%**. Shallower early, banked clock,
deeper late — the mechanism is exactly right. **Elo: −20.5 ± 20.8.**

Not significant yet, but negative, and the reason is a flaw in my premise that the
data I already had exposes immediately:

| move | share of ALL errors | share of **DECIDABLE** errors | share of decidable wp lost |
|---|---|---|---|
| 1–20 | 1.8% | 9.1% | 8.8% |
| **21–32** | 5.8% | **29.5%** | **30.4%** |
| 33–45 | 11.9% | 25.0% | 24.5% |
| 46+ | **80.6%** | 36.4% | 36.2% |

**"87% of win-probability loss happens after move 33" came from the ALL-ERRORS
column.** That column is 80.6% concentrated at move 46+ — and §3i already showed
**78.4% of all errors sit in positions where the best move still leaves us under
10% win probability.** Extra depth in an already-lost position converts nothing.

Condition on decidability — the only errors that can change a result — and the
mass moves forward: **39% of decidable errors happen before move 33**, and the
densest early bucket is **moves 21–32 at 29.5%**. That is precisely the phase my
arm took time out of (−0.47 plies at move 30). The Elo is not a mystery; it is
the premise.

**The lesson, and it is one I had already written down.** §3i established the
78.4% already-lost share and even corrected the standing blunder rate for it. Then
I built a 38-Elo thesis on the uncorrected column of the same table. Knowing about
a contamination is not the same as remembering to apply it — **when a distribution
is used to justify spending a resource, re-derive it over the subset where the
resource can actually pay.**

**Status:** the match continues for a tighter bound, but the thesis is
mechanically understood and I expect it to settle negative. The clock deficit
against Stockfish (§2) remains a real observation; what is dead is the inference
that closing it is worth Elo.

## 3t. THE CLEAREST PICTURE — we do not fail to convert; we never get ahead

A failure mode I had not examined all night: conversion. The first attempt
triggered on each engine's OWN eval and reported Stockfish converting **100.0%**
at every threshold — impossible, and the tell that two different frames were
being compared. Re-run with SF's annotations as a single referee for both sides
it looked real: at +1.0 sustained, ARCHON converts 29.9% and SF 92.7%, with 64.4%
of ours becoming draws.

**That collapsed too.** Those +1.0 advantages were first reached at **median ply
10** — the UHO book — and their peak was **median 1.20, max 1.65**. A book-assigned
edge, not a won game. (The same book trap as §1, in a new costume: in a UHO match,
anything triggered on an early eval is measuring the book.)

Asking the prior question instead — do we ever get winning at all? Peak advantage
in SF's frame over all 320 games:

| | median | p75 | p90 | **≥+2** | ≥+3 | ≥+5 |
|---|---|---|---|---|---|---|
| ARCHON | 0.84 | 1.12 | 1.51 | **8%** | 8% | 8% |
| SF18 | 1.25 | 12.39 | 30.00 | **48%** | 48% | 48% |

From move 20 on (past the book) our median peak is **0.24** against SF's **1.23**.

**Two things fall out, and the second is the important one.**

1. **`≥+2` = `≥+3` = `≥+5` for BOTH engines.** Once either side reaches +2 it goes
   all the way. So an advantage of +2 simply *is* a win at this level, and
   **neither engine has a conversion problem.** Conversion is not a failure mode;
   it is not on the list.
2. **Stockfish reaches a winning position in 48% of games. We reach one in 8%.**

**So the dominant failure mode is not discrete at all.** It is that we never build
an advantage — a small, uniform, per-move deficit, exactly the 0.005-vs-0.003 wp
per move of §3q, amplified by the ~1.5× per-node cost of §3b-QUIET.

**Sizing it against the blunders**, which is the comparison the standing bar
cares about: 44 decidable errors across 320 games is **0.14 per game**, carrying
~6.35 wp in total, or **0.02 wp per game**. The per-move quality gap is 0.002 wp
× ~60 of our moves ≈ **0.12 wp per game** of pressure — an order of magnitude
more, even allowing that wp does not accumulate linearly.

**This is in tension with the framing of the standing bar, though not with its
instruction.** "Rank by tractability × Elo" still holds and blunders are still the
tractable ones — but in THIS matchup the drift is roughly 6× the blunder
contribution, and a night spent entirely on blunders would have been a night
spent on the smaller term. The bar's real value tonight was procedural: it forced
the detector-independent census that produced these numbers.

## 3u. THE EVAL QUESTION, CLOSED IN BOTH FORMS

Every eval comparison tonight — and the net-only suite before it — correlates an
engine's eval with deep truth **ACROSS positions**. That is what a training loss
measures. **A search never does that.** It compares the CHILDREN of one position,
and needs the eval to order those children the way a deep search would. An eval
can be excellent across positions and nearly useless within one, if sibling
differences are small relative to its noise. I measured the first property all
night and never the second.

Measured now — the tiered oracle already scores every legal move, so the per-move
truth was free. Spearman between each engine's static eval of the children and
the oracle's value of each move, within each position:

| pieces | n | moves/pos | ARCHON rho | SF18 rho | gap | ARCHON top-1 | SF18 top-1 |
|---|---|---|---|---|---|---|---|
| 21+ | 77 | 35.4 | 0.510 | 0.546 | −0.036 | **18.2%** | 15.6% |
| 13–20 | 40 | 27.6 | 0.499 | 0.497 | +0.001 | 20.0% | **22.5%** |
| 8–12 | 25 | 17.9 | 0.413 | 0.402 | +0.011 | **24.0%** | 16.0% |

**Parity again**, with us ahead on top-1 in two buckets of three. Frame-free by
construction: a rank correlation within a position cannot be moved by any
monotone rescaling, so the 0.46× frame difference cannot touch it.

Two things worth keeping beyond the verdict:

- **Within-position ranking is a far harder problem than across-position.** Both
  engines score rho ~0.41–0.55 here against ~0.76–0.96 across positions, and a
  static eval picks the oracle's best child only **15–24%** of the time. That is
  the quantitative reason search exists, and it is a useful calibration for any
  future "the net should just know" argument.
- **The eval hypothesis is now dead in every form tested**: across positions
  (§3e), at the leaves after qsearch (§3m), and within a position's move list
  (here) — each of them against a reference that favours Stockfish (§3r).

**So the equal-node deficit is not the eval, by three independent measurements.**
What remains is the search's use of an eval that is demonstrably as good as
Stockfish's, and every structural component compared so far — gates, NMP, the
dither, movepick staging, rule50, qsearch — is at exact SF parity.

## 4. WHAT SURVIVES — the search-efficiency gap

Like-for-like MultiPV-1 ladders on game 18: **Stockfish commits to `a5c5` at
depth 5 / 618 nodes and never leaves it. We hold `h5e2` from depth 6 to 20 and
switch at depth 21 / 410,613 nodes.** The forensics agent's node-budget
measurement puts it at 25k vs 600–800k. Either way the deficit is one to two
orders of magnitude in nodes, on a position where:

- our own **static eval already prefers the right move**, by 108 cp in our
  frame against Stockfish's 31 — the net is not the problem here;
- no pruning gate is responsible (§5);
- the clock is not responsible (§2).

That leaves move ordering, extensions, and near-equal-move resolution. The
agent measured first-move cutoff at **89.4% against Stockfish's 93–95%**. That
is the surviving hypothesis and the next target.

---

## 4b. GAME 18, closed out — it is null-window root promotion, not knowledge

Two measurements finished this position off, and both went against the story I
was building.

**The eval is not merely adequate here, it is better than Stockfish's.**
Ranking all 20 of Black's replies to Be2 against deep truth (SF @400k):

| | Spearman vs deep truth | its value for the refutation `d7b5` |
|---|---|---|
| **ARCHON static** | **0.618** | **+45** (correctly positive for Black) |
| SF18 static | 0.179 | −71 (wrong sign) |

Both statics pick `c3c2` as Black's best; the truth is `d7b5`. Neither net
understands the deflection — but ours ranks the field three times better and
gets the key move's sign right. **No eval fix is available here, because our
eval is already the better of the two.**

**And Stockfish does not "know" the answer either.** Reading both candidates
out of ONE tree at MultiPV 3:

| depth | ARCHON a5c5 / h5e2 | SF18 a5c5 / h5e2 |
|---|---|---|
| 6 | −1 / **+63** | −12 / −25 |
| 10 | −32 / **+1** | −6 / **+3** |
| 12 | −25 / **−1** | −5 / **+3** |

**Stockfish's own full-window values prefer `h5e2` from depth 10 on** — the
losing move — while its single-PV search picks `a5c5` at depth 5 and never
leaves it, at every node budget from 25k up. So SF's correctness here is not
superior evaluation and not deeper resolution. In single-PV the second root
move is scouted with a null window around alpha and must EXCEED it to be
promoted; SF's scout proves `h5e2 <= alpha` and ours promotes it.

The localization is therefore precise and narrow: **root-move promotion under
the null window.** Not eval, not the gates (§5), not the clock (§2), not
movepick structure — our stage list is an exact match to SF's at `eca43a97`,
and neither engine has killers, so D17 stands.

This is a localization, not a fix, and one position cannot support a change.
It goes on the list for corpus-v2 to confirm or kill.

## 5. GATES — refuted, including my own attribution

`gate_depth.py` on game 18 with defaults read out of the binary said parent
futility cost 12 plies of discovery and 27× the nodes, where RFP-off made the
position *worse*. That overturned the standing "RFP is the only gate that flips
game 18" claim. Then a null panel overturned mine:

| arm | 25k | 50k | 100k | 200k | 400k | 800k |
|---|---|---|---|---|---|---|
| control | h5e2 | h5e2 | h5e2 | h5e2 | a5c5 | a5c5 |
| pfut-off | a5c5 | a5c5 | a5c5 | a5c5 | a5c5 | a5c5 |
| **null `PFUT_BASE` 40→41** | h5e2 | **a5c5** | **a5c5** | h5e2 | h5e2 | a5c5 |
| null `RAZOR_BASE` 465→464 | h5e2 | h5e2 | h5e2 | a5c5 | a5c5 | a5c5 |

A **±1 change to an unrelated constant** also "fixes" the position. The argmax
here is chaotic (the agent measured a 7.9–14.9% null flip rate against 30.9%
for gates-off), and its aggregate node-matched control **inverts** the whole
gates-off effect: +24.1 ± 6.3 depth-matched, **−25.7 ± 5.1 node-matched**,
monotone in the node ratio. Pruning is doing its job.

`pfut-off` is still the only arm correct at every budget, and no null matches
that. It is a curiosity, not a finding: **one position cannot support a
mechanism claim on a chaotic argmax.**

---

## 5b. THE ONE ELO LEAD — parent futility prunes too much, and my sign was wrong

Full corpus (n=9,992, openings — see §1 for why that matters), arms at 12,500
nodes:

| arm | what it does to the margin | cp@16k | nodes | blunder rate |
|---|---|---|---|---|
| control | — | 6.32 | — | 0.2% |
| `pf64c` constants ×0.64 | prune MORE | **+0.19 ± 0.15** | −1.8% | 0.3% |
| `pf64all` all four ×0.64 | mixed | −0.24 ± 0.15 | −0.8% | 0.2% |
| `pfutbase120` base ×3 (agent) | prune less | −0.38 ± 0.14 | +3.0% | — |
| `pfmult64` ply coeff ×0.64 | prune less when `lmr_depth<0` | −0.44 ± 0.15 | +4.4% | 0.2% |
| **`pfinv` constants ×1.56** | **prune less** | **−0.58 ± 0.14** | +4.4% | **0.1%** |

Five arms, four of them independently derived, and they order themselves
perfectly by how much they RELAX the bound. The one arm that tightens it
(`pf64c`) is the only one that hurts, and it is also the only one that fattens
the tail (0.3%). The best arm halves the blunder rate, 0.2% → 0.1%, which is
the property the screen's own trailer says to prefer: the mean is mostly tail,
and a blunder converts a game where a 2 cp inaccuracy does not.

**`pfinv` was built as a falsification control for my own hypothesis, and it
won.** I derived from the frame that our margins were too LARGE and should be
scaled by 0.64; the data says they are too SMALL. The dimensional argument
picked the right constants to move and the wrong direction — worth recording
precisely, because the same rule shipped the aspiration fix and will be reached
for again. It locates candidates; it does not predict signs.

At the screen's own conversion (~60 Elo per cp, and `log2(1+node_cost)·120`
against) `pfinv` is nominally ~35 − 7.5 ≈ **+27 Elo**. Do not believe that
number yet:

1. It is measured on the openings corpus (§1), which is the wrong population
   and is the very instrument this document exists to indict.
2. Relaxation cannot be monotone — gates-off is worse once node-matched — so
   the optimum is bracketed but not located. `c1.56 / c2.5 / c4 / c6 / off` is
   running.
3. `hind111` is the standing reminder that a screen number is not Elo.

**The node cost is ~zero, possibly negative.** A single middlegame position
showed the arm building a 78% larger tree at depth 13, which would have cost
~100 Elo of speed and killed it. Aggregated over 60 positions across phases at
depth 12 it reverses:

| pieces | base nodes | arm nodes | ratio |
|---|---|---|---|
| 21+ | 572,469 | 533,012 | 0.93 |
| 13–20 | 418,672 | 410,042 | 0.98 |
| 8–12 | 319,768 | 332,985 | 1.04 |
| ≤7 | 399,404 | 346,870 | **0.87** |
| **all** | 1,710,313 | 1,622,909 | **0.95** |

Against the screen's +4.4% on its own bench suite, so the honest reading is
**±5%, i.e. no meaningful node cost** — and the cp gain is not swamped. There
is a mechanism for less pruning costing fewer nodes rather than more: when
parent futility prunes it LIFTS `best_value` to the futility bound, and that
inflated value is stored as a TT lower bound. Fewer inflated lifts means better
TT data downstream. That also predicts the effect should be largest where trees
are deepest and TT reuse heaviest, which matches the ≤7 column.

Sequence before it can be promoted: bracket on the phase corpus → SPRT.
A fixed-node SPRT (`tc=inf nodes=200000`, load-immune, legitimate here because
a pruning margin has no NPS effect) is **running now** at `engines/runs/pfut-sprt`,
arm verified live by tree change (20,622 → 36,654 nodes, restore exact).

**Why a 4σ effect was still lying around — and a correction.** I first wrote
that PFUT_\* were among the 164 of 303 tunables in no campaign. That is wrong:
they are named in four campaign configs, and `c5-futility-see` bounds
(`PFUT_BASE` [20,80], `NO_BESTMOVE` [65,280], `LMRDEPTH_MULT` [55,235],
`EVAL_ABOVE_ALPHA` [45,180]) contain every value `pfinv` uses. The effect was
reachable.

It was never reached because **none of those campaigns was ever run.**
`tools/spsa/out/` holds runs for exactly two: `c2-reduction` and `s2-stageA`,
and stage-A's only PFUT entry is `PFUT_BASE`. `c5-futility-see`,
`c6-gates-counters` and `s4-searchgap` have no output directory at all.

So the coverage metric is itself misleading: `param_provenance.py` counts a
tunable as "covered" when it is **named in a config file**, not when it has
been tuned. It reports 139/303 covered; the number actually exercised by a run
is far smaller. Worth fixing — a coverage number that counts intentions rather
than runs will keep pointing campaigns away from unexplored ground.

## 5c. THE pfut ARM IS DEAD — and killing it VALIDATED the new corpus

Three measurements, and the candidate does not survive any of them:

| arm | openings corpus (n=9,992) | **phase corpus (n=3,352)** |
|---|---|---|
| `m75` PFUT_LMRDEPTH_MULT 75 | −0.26 ± 0.16 | **+1.10 ± 0.68** |
| `c1.56` constants ×1.56 | **−0.35 ± 0.15** | **+0.80 ± 0.54** |
| `base120` PFUT_BASE 120 | −0.26 ± 0.15 | **+0.54 ± 0.62** |
| `m60` PFUT_LMRDEPTH_MULT 60 | — | −0.10 ± 0.62 |

Every relaxation arm flips sign. And the games agree with the phase corpus:

| games | Elo |
|---|---|
| **12,500-node SPRT** (the screen's own budget), n=676 | **−1.5 ± 10.7** |
| 200,000-node SPRT, n=264 | −5.3 ± 16.1 |

**The predicted +27 Elo is excluded at 2.7σ.** This is not one arm going wrong:
the openings corpus produced a false positive on the ENTIRE FAMILY, four arms,
all in the same direction, at up to 4σ.

**One explanation was tested and refuted.** The obvious defence was budget
mismatch — screened at 12.5k nodes, raced at 200k, and `screen.py` warns that
"an arm whose gain shrinks or flips as the budget grows was tuned for a search
shallower than the one it will play in." So I ran a second SPRT at 12,500 nodes,
the screen's exact budget. Still zero. The arm is not shallow-tuned; it simply
does not work.

**A second self-inflicted lesson.** An exploratory 900-position run on the phase
corpus read `m75` at **−1.10** and I reported it as confirmation. At full power
the same arm reads **+1.10**. The sign flipped with sample size, at 900
positions — the "4000 is a coin flip" trap, one level worse. `screen.py` printed
the earlier value next to the new one, which is the only reason it was caught.

**But the important result is what this says about the INSTRUMENT, not the arm.**
The two corpora disagreed, and the games adjudicated:

- openings corpus: −0.58 cp (better) — **contradicted by the games**
- phase corpus: +1.10 cp (worse) — **consistent with the games** (−1.0 ± 15.1)

So on the one case where the old and new instruments disagree, **the
phase-representative corpus agrees with reality and the openings corpus produced
a false positive.** That is exactly the failure mode §1 predicted, caught in the
act, and it is empirical support for the population fix rather than merely
theoretical support. It cost me the night's only Elo candidate to get it, which
is a fair price: a screen that reports +27 Elo on nothing will keep doing so.

**Standing consequence: no tunable should be promoted on `oracle-obs.json`
again.** Re-screen the 139 tunables that were "covered" on openings, and treat
the pfut family as unresolved rather than dead — the direction still has the
§3f marginal-return mechanism behind it, but nothing has been demonstrated.

## 5d. THE 78 Elo OF QUALITY IS NOT PARAMETRIC — 15 arms, every one worse

With a corpus that has been shown to agree with games (§5c), a +-25% sweep of the
central pruning / reduction / ordering knobs. cp@64k, paired deltas:

| arm | −25% | +25% | | arm | −25% | +25% |
|---|---|---|---|---|---|---|
| `RED_TABLE_NUM` | +0.41 | +1.02 | | `RFP_MULT_BASE` | +1.26 | **+0.23** |
| `RED_BASE` | +1.28 | +1.55 | | `RFP_MULT_SLOPE` | +0.84 | +0.67 |
| `RED_MOVECOUNT` | +1.24 | +0.88 | | `LMP_BASE` | +0.37 | +1.32 |
| `RAZOR_BASE` | +0.45 | +0.81 | | `IIR_REDUCTION` | +0.96 | — |

**Every single arm is worse.** The best is `RFP_MULT_BASE` +25% at +0.23 ± 0.68,
i.e. neutral. Nothing is negative at the deepest tier.

**The instrument was verified before this was believed.** A null arm — setting a
parameter to its OWN default — reads `+0.00 ± 0.00` at all three tiers and
reproduces the control's top-1, blunder rate, p90 and node count exactly. So the
uniform positive sign is the engine, not a bias in the screen. (This is the same
self-check that caught a real TT-carryover bug in `spsa_cp.py`; it earns its
keep every time.)

**What it means.** These values were transcribed from Stockfish's tuner, which
optimised them against a similar objective, so finding them at a local optimum
is the expected result rather than a surprise. But combined with §3g it says
something sharper: **the ~78 Elo of equal-node quality gap is not reachable by
moving these knobs.** It is a MECHANISM difference — something Stockfish's search
does that ours does not, or does differently in structure — not a mis-tuning.

That is a direct answer to "there is 90-ish Elo hidden in the tunables": not in
these fourteen, at this resolution, on the corpus that predicts games.

**CAVEAT ADDED LATER, and it is a big one (§3m).** Every arm here was scored at
**12,500 nodes** — a budget at which we reach depth 9.9, where our real games
reach 17–20, and at which the phase-by-phase verdict is measurably *different*
from the 200k regime. So "the central knobs are at a local optimum" is
established **for a search we do not play**. It has to be re-run at a realistic
budget before it can be believed, which needs an oracle deeper than the arms
(`deep_top1.py` / `arm_top1.py`). Until then this section is suggestive, not
settled.

**Honest limits of the claim.**
- ±25% is coarse. A ±5–10% optimum could exist; it would be worth far less, and
  finding it needs SPSA rather than a grid.
- 14 of 303 tunables. The other 289 are untested here — and the 139 previously
  "covered" were screened on the corpus §5c just discredited, so they are
  effectively untested too.
- Single-parameter moves cannot find interactions. A joint optimum can sit where
  every one-at-a-time step is uphill; that is what SPSA is for.

## 5e. THE PHASE CORPUS IS THE RIGHT POPULATION BUT UNDERPOWERED

A 522-arm coarse ranking pass (`--limit 1500`) over every plausibly
strength-relevant tunable produced **78 of the first 106 arms "better", best
−2.40 cp**. That is not credible against §5d, where every ±25% arm on the central
knobs was worse at full power. So I checked it with a control, and the control
killed it:

| arm | cp@16k | cp@64k | top-1 |
|---|---|---|---|
| `RED_BASE` **1027 → 1028** (a +1 change) | **−1.13 ± 0.83** | +0.36 ± 1.01 | 51.7% |
| `RED_BASE` 1027 → 1284 (+25%) | −0.39 ± 0.92 | **−0.22 ± 1.00** | 52.3% |

Two things follow, and both are fatal to the sweep:

- **A ±1 change reads −1.13 cp** — the same magnitude as every "best" arm in the
  ranking. Nothing in that top-14 list is separable from a semantically trivial
  perturbation.
- **`RED_BASE` +25% reads −0.22 at `--limit 1500` and +1.55 at `--limit 0`.** The
  coarse limit does not merely lose power, it **flips signs**.

**Root cause, and it is not a bug — it is corpus power:**

| corpus | control cp | n | paired sem |
|---|---|---|---|
| openings (`oracle-obs`) | 6.72 | 9,992 | **0.15** |
| phase (`oracle-phase-clip`) | 15.0 | 3,352 | **~0.55** |

3× fewer positions (√3 = 1.7×) and 2.2× the cp variance, because the positions
are genuinely harder and full of near-equal moves — the chaotic regime of §3j.
1.7 × 2.2 ≈ 3.7, and 0.15 × 3.7 = 0.55. The arithmetic closes.

**So the phase corpus is the right population (§5c proved that against games) and
simultaneously too small to rank tunables on.** That is the honest state of the
instrument, and it is why the sweep was retired rather than mined.

**Fix in flight:** `oracle-phase2`, 12,000 positions from both matches
(11.4 / 19.1 / 29.9 / 39.5% across the four phase buckets), which should give
~0.28 cp sem — the openings corpus's power on the correct distribution. The build
was also 4× too slow for a stupid reason: every move-search is preceded by
`ucinewgame` for reproducibility, and clearing a **128 MB** hash dominated
everything. At 16 MB the searches are the cost again.

## 6. INSTRUMENT BUGS FOUND TONIGHT (five, all mine)

1. **Bound lines read as scores.** `uciprobe` matched `score cp X upperbound`
   as an exact value. Harmless for a depth ladder; fatal for a node-limited
   search, whose last iteration is cut off mid-window. A position worth −228
   read as −40.
2. **Cold vs warm TT.** `harvest.py` analyses unrestricted-then-forced back to
   back, so the forced search inherits a warm TT: 212 cp warm, 0 cold, same
   budget. Its cp values are order-dependent. This produced a false retraction
   ("the corpus is blind to game 18") which is itself now retracted.
3. **A hand-typed control.** Three of eight gate "defaults" were invented
   (`NMP_R_BASE` 4658 for 7, `RED_TABLE_NUM` 1155 for 2834, `LMP_BASE` 3387 for
   3). Every delta in that table was meaningless while looking quantitative.
   Defaults now come from `<engine> tunables`. **A control you typed is not a
   control.**
4. **MultiPV confound.** Stockfish's ladder was run at MultiPV 6 and compared
   against our MultiPV 1. MultiPV gives every root move a full window and
   surfaces refutations plies earlier. That manufactured most of an "11-ply
   gap".
5. **Cumulating per-move means.** Averaging time-spent at move N across games
   of different lengths and summing gives a *negative* time-left column. Clocks
   must be simulated per game.

The pattern in 1, 2 and 3: each instrument's sensitivity or baseline was set by
convenience rather than checked against a known value. `harvest2.py` now
refuses to run until it re-detects game 18.

---

## 6b. QUEUED, NEEDS A QUIET BOX

1. **Re-measure the §3b NPS table idle.** Everything else tonight is fixed-node
   and load-immune; this one is not, and the playbook has a logged case of an
   NPS read being wrong by 12% under load.
2. **The smallnet gate as an endgame speed lever.** `ARCHON_SMALLNET` +
   `ARCHON_SMALLNET_THRESH` + `ARCHON_SMALLNET_MAT_GATE` are implemented and
   off by default for want of a trained cheap net — but passing the EXISTING
   d47v net as the cheap net already yields a HalfKA-only evaluation with the
   FullThreats channel skipped, which the code says is "the gate's entire speed
   win". So the speed side is testable tonight without training anything; the
   accuracy side is not, because the head was trained with threats present and
   its output is miscalibrated without them. Two obstacles to note before
   anyone tries: the gate is ENV-driven, so `screen.py` cannot A/B it in one
   run (env reaches control and arms alike, and the runs would be unpaired),
   and a pure speed change has no deterministic proxy — node count does not
   move with eval cost. Wall clock on a quiet box, or nothing.
3. **Fix `param_provenance.py` coverage** to count runs, not config membership.

## 6b. THREE MORE DEAD ENDS, KILLED PROPERLY

Chasing the retracted state effect to the bottom produced three sub-hypotheses.
All three are dead; the reasons are worth more than the hypotheses were.

**1. "The played move comes from an incomplete deeper iteration."** A timed
search is aborted mid-iteration, so it can promote a move the last completed
depth never chose — which would explain why a depth-limited replay cannot
reproduce a time-limited game. Measured over 220 timed searches at the games'
own per-move times: the final `bestmove` differs from the last completed depth's
PV move in **8/220 = 4%** of searches. Of those 8, judged by SF18 @800k: 4 worse,
2 better, 2 tie. **4% cannot explain a 39%-vs-16% fidelity gap**, and n=8 decides
nothing about whether the switches help. Closed.

**2. "The equal-value dither is the chaos source."** `search/mod.rs:3895` gives a
move that TIES `best_value` an `inc=1` bonus when `(nodes & DITHER_NODE_MASK)==0`
— so tie-breaks are keyed on the node counter, and any perturbation that shifts
the node trajectory reshuffles them. That is a genuine chaos amplifier and it
would explain §3j. Measured as flip rate under an inert ±1 change to
`RAZOR_BASE`, 30 positions:

| | flips (+1) | flips (−1) | mean |
|---|---|---|---|
| dither ON (mask 14, shipped) | 7/30 | 5/30 | **20%** |
| dither OFF (mask 2⁶²−1) | 5/30 | 3/30 | **13%** |

It contributes perhaps a third — but 12 vs 8 flips out of 60 comparisons
establishes nothing, and **most of the chaos survives with the dither off.** And
it is exact Stockfish parity: `eca43a97` line 1457 is
`(value == bestValue && ss->ply + 2 >= rootDepth && (int(nodes) & 14) == 0 && ...)`,
matching our `DITHER_PLY_MARGIN=2` / `DITHER_NODE_MASK=14` exactly. Not ours to
remove, and Stockfish carries the same instability by construction. Closed.

**3. "History-driven pruning cuts the right move."** Carried state shrinks the
tree hard (median replay/cold nodes **0.66**, geomean 0.59) and the shrinkage
coincided with the errors, which points at history buying reduction depth and
pruning outright rather than at ordering. Relaxing exactly those knobs against
the replay target: `QHIST_PRUNE_MULT` ×0.5 → 10 (no change), ×0.25 → 8 (worse).
And weakening what history *remembers* — bonus ×0.5 → 9, malus ×1.5 → 9, quiet
bonus ×0.5 → 8 — all worse. Then the harness itself was invalidated, so these are
moot as well as negative. Closed.

**What the three have in common:** each was a plausible mechanism with a clean
story, and each died to a measurement rather than to an argument. The 4% number,
the 13%-with-dither-off number and the SF-parity grep each took minutes. That is
the whole method: the cheap check that could kill the idea, run before the
expensive one that would confirm it.

## 6c. TRAINING RUN + AN OPERATIONAL RISK

**D48X is complete and no trainer is running.** `TRAIN EXIT 0`, sb520/520,
heldout value 0.006350 = ema 0.006350; the final artifact is
`data/nnue/experiments/d48x-1280/d48x-1280-sb520-ema-quant.bin`. The only live
units are the TB shim and the TB server, both tailing a finished log — harmless
but idle. Its already-recorded verdict stands: better than d47v *and* SF18 on the
net-only suite, equal-node +8.9 ± 17, **equal-time −12.6 ± 17 — not a gain.**

**Tonight's evidence bears directly on what that program should do next.** D48X
is a 1280-wide FullThreats net, and the two measurements that matter for it now
point the same way:

- our eval is **at parity with Stockfish's in every phase** (§3e), so more eval
  capacity is buying something we do not appear to be short of;
- the threat channel costs **1.482× speed in the middlegame** (§3h), the phase
  where our equal-node quality gap is worst (§3g).

So the marginal net program is spending speed on eval quality that measures as
already-sufficient. **The indicated next net is threats-FREE at the same width**
— which is also the one experiment that would close the D49 gate honestly, since
exp-0065's +32.5 ± 17.3 was never isolated to threats.

**Should the threats-free net be launched tonight?  No, and for two concrete
reasons rather than caution.** The GPU is idle (0% util) and `d48x_launch.sh` +
a JSON config is the whole launch path, so it looked tempting. But:

1. **Disk.** 22 GB free; D48X's own footprint is 40 GB across 144 checkpoints. A
   new run fills the disk and takes other things down with it.
2. **It is not a config flag.** `d48x_1280.json` has no threats/feature key — the
   feature set comes from the frozen warm-start trunk (`_arch_ruling`: "the A3
   trunk transfers VERBATIM ... full trunk (frozen)"). A threats-free net at the
   same width needs a feature-set design decision, not a parameter change.

So it stays the recommended next experiment and an explicit ask, not something to
start unattended.

**Operational risk, flagged not acted on:** the disk is at **98%, 22 GB free of
937 GB**. `data/nnue/experiments/` holds ~105 GB in its top five directories
alone, and d48x-1280 is 40 GB across **144 checkpoints** of which four (the sb520
quartet) are the ones anything references. Pruning intermediate checkpoints is
the obvious reclaim, but deleting training artifacts is not reversible and is not
mine to decide. Nothing currently running is at risk (the 12k oracle writes
~17 MB), but the next big build or PGO link could fail on ENOSPC.

## 7. STATE

| thing | state |
|---|---|
| **`pfut-sprt`** | **RUNNING** — `PFUT_BASE=62, NO_BESTMOVE=215, EVAL_ABOVE_ALPHA=140` vs base, fixed-node (`nodes=200000`), SPRT [0,6] normalized, cap 3000 games. `engines/runs/pfut-sprt`. Arm verified live behaviourally. This is the night's one real Elo test |
| tm10c (frame-scaled `FALLING_EVAL_*`, 10+0.1) | finished 400 games; at 321 the arm was 51.6% = **+10.8 ± 13.5 Elo**. Not significant. The arm holds *less* clock and searches shallower, so it does **not** validate the §2 story |
| corpus-v2 | ~290/700 vs SF18 at 10+0.1. Already 108 losses. NOT Elo-comparable to rematch-v1 (ran on a loaded box); fine as a corpus, since the oracle judges positions, not clocks |
| phase oracle | building, 3,350 positions, 12.5/25/25/37.5% across the four buckets |
| `harvest2` | chained behind corpus-v2 — cold 1M-node oracle over both matches, refuses to start unless it re-detects game 18 |
| phase bracket | chained behind the oracle: `c1.56 / c2.5 / c4 / m75 / base120 / c2.5+m75 / off` |

### The order to read results in, in the morning

1. **`pfut-sprt`** — the only game-truth number. If it clears H1, that is the
   night's Elo. If it is still running without a verdict past ~800 games, the
   effect is under ~10 Elo whatever the screen said.
2. **phase bracket** — does the pfut direction survive on the population that
   matters, and where is the optimum?
3. **`harvest2` output** — the corpus that finally makes the
   three-independent-losses rule possible. The agent's blocking finding was
   that the 120-game match contained only two clean cases.

### What I would NOT spend more time on

- Game 18 as a single position. It is localized (§4b) and further single-
  position probes on a chaotic argmax produce confident nonsense; three of my
  retractions tonight came from exactly that.
- Gate attribution. Refuted twice, independently (§5).
- The clock as a blunder mechanism (§2). Real deficit, wrong causal story.

**Blocking constraint (the agent's finding, and it is right):** of 19
adjudicated candidates in the 120-game match, only **two** had an oracle stable
across 400k/1.6M/8M nodes. One match does not contain three clean cases of any
mode. More games is the prerequisite for the three-independent-losses rule, not
more analysis of these.

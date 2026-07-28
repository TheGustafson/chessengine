# Loss forensics, night of 2026-07-28 — the population defect

Companion to `LOSS-FORENSICS-PLAYBOOK.md` (method) and
`TRANSITION-2026-07-28.md` (prior state). Read the two REFUTED banners in the
playbook first: `bbc28fd` and `b22c9c9` retract the gate-attribution mode.

Everything below was measured tonight. Where a number contradicts an earlier
one, the earlier one is named and withdrawn.

---

## 0. THE MODEL — every failure mode, and what each one needs

Gap to SF18 at 10+0.1: **−171 to −191 Elo**. Decomposed, all terms measured:

| term | Elo | how measured | what it needs |
|---|---|---|---|
| **Equal-node move quality** | **~78** | +1.30 cp weighted over phases, §3c/§3g | **Mechanism, not tuning** — 15 ±25% arms all worse (§5d) |
| **Speed** — 1.26–1.39× cycles/node | **~64** | same-run cycles/node with a fairness check, §3b-FIXED | Per-node cost engineering |
| Residual — TM, conversion | ~30–50 | by subtraction | TM: we reach move 45 with 35% of their clock (§2) |

**Where each thing lives:**

| question | answer | evidence |
|---|---|---|
| Where is our search weakest? | **Middlegame.** Headroom +2.81 cp at 21+, falling to **−2.01 (we are better)** at ≤7 | §3g |
| Where do errors cost most? | **Endgame.** 0.31 wp lost per 100 moves at ≤7 vs 0.06 at 21+ | §1 |
| Is the eval behind? | **No — parity in every phase**, slightly ahead at ≤7 | §3e |
| Is it depth-limited? | **No.** Headroom is flat over a 16× node range; the top-1 gap WIDENS 1.4→3.7 pts | §3f |
| Is any pruning gate the culprit? | **No.** Refuted twice, independently | §5, playbook §14 |
| Is it the clock? | Real deficit, but **not** the blunder mechanism | §2 |
| Is it tunables? | **Not the 14 central ones.** Every ±25% move is worse | §5d |

**The one-sentence model:** our eval is at parity, our search costs ~1.3× more
per node, and it converts the nodes it does get into correct moves less well than
Stockfish's — worst in the middlegame — while our clock policy concentrates what
remains into the endgame, where each error is most expensive.

**What that rules out:** better nets (eval is at parity in every phase), more
depth (the gap does not shrink with 16× nodes), and tuning the central pruning
knobs (all worse). **What it points at:** per-node cost, and structural search
mechanisms Stockfish has that we do not — ordering, extensions, what gets
searched rather than how much.

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

**Honest limits of the claim.**
- ±25% is coarse. A ±5–10% optimum could exist; it would be worth far less, and
  finding it needs SPSA rather than a grid.
- 14 of 303 tunables. The other 289 are untested here — and the 139 previously
  "covered" were screened on the corpus §5c just discredited, so they are
  effectively untested too.
- Single-parameter moves cannot find interactions. A joint optimum can sit where
  every one-at-a-time step is uphill; that is what SPSA is for.

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

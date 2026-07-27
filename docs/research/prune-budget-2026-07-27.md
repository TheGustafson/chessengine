# Prune budget — which mechanisms actually shape our tree (2026-07-27)

Follow-on to the eval-frame work, which ended by concluding that our search
constants sit at a **joint** optimum rather than an independently correct
one: our tree is ~35% smaller than SF16's at equal depth, so we must prune
harder somewhere, and the interior margins are compensating
(`eval-frame-scale-model-2026-07-27.md` §11.4).

That story named no mechanism. This measures one.

Re-runnable: `python3 tools/prune_budget.py` (needs a `--features tune`
build). Turn one gate off, measure how much the tree grows, and the growth is
that gate's contribution.

---

## 1. The budget

Depth 10, `wl50.fens`, NNUE (`d47v-a4b2-cal400`), baseline **1,002,862** nodes.

| mechanism | nodes without it | growth |
|---|---|---|
| **LMR log term** | 3,476,572 | **+248.2%** |
| **RFP** | 1,623,990 | **+62.6%** |
| **LMP** | 1,465,444 | **+46.8%** |
| quiet SEE | 1,202,567 | +20.4% |
| parent futility | 1,059,742 | +6.1% |
| capture futility | 1,003,146 | +0.5% |
| IIR | 968,391 | −3.0% |
| **NMP** | 957,267 | **−4.1%** |
| ProbCut | 947,403 | −5.1% |
| conthist pruning | 944,730 | −5.4% |
| **razoring** | 871,706 | **−12.7%** |

Two readings, and the second is the surprising one.

**LMR is the tree.** Removing just its logarithmic term multiplies the tree
by 3.5×. RFP and LMP together are worth another ~110%. Everything else is
noise by comparison. That is the mechanism behind the joint-optimum story:
**we prune overwhelmingly by reduction, not by margin**, so interior margins
have little tree left to remove and mostly cost accuracy when tightened —
exactly the uniform interior-margin tax the earlier fixed-node screens found.

**Five mechanisms have NEGATIVE contribution.** Razoring, conthist pruning,
ProbCut, NMP and IIR all make the tree *bigger* when enabled. Node count is
not monotone in pruning aggressiveness (TT and re-search feedback), so this
is not automatically a defect — a gate can cost nodes and still buy accuracy.
But −12.7% for razoring is large enough to want explaining, and one of the
five turned out to have a concrete cause.

---

## 2. NMP is starved, and it is a frame-scale casualty

`ARCHON_SEARCH_STATS` on the same bench:

```
baseline                        nmp_try=2047   nmp_cut=1673   (82% cut rate)
NMP_EVAL_BASE=10000000 (off)    nmp_try=0      nmp_cut=0
```

**NMP is attempted 2047 times in a ~1,000,000-node search — 0.2% of nodes**,
and succeeds 82% of the time when it fires. It is not failing; it is barely
being asked.

The gate is an exact copy of SF@`eca43a97`:

```cpp
ss->staticEval >= beta - 14 * depth - 45 * improving + 374
```

Our `NMP_EVAL_DEPTH 14 / NMP_EVAL_IMPROVING 45 / NMP_EVAL_BASE 374` match to
the digit. **But that 374 is denominated in Stockfish's frame.** At their
scale it is ~0.97 pawns of required advantage over beta; at ours (0.64×) the
same integer demands **~1.5 pawns**. The crossover depth where the bar stops
requiring eval above beta is `374/14 ≈ 27` for us against SF16-shape's ~12.
So below depth 27 — which is nearly the whole tree — NMP asks for an
advantage our eval rarely reports.

This is the frame argument's cleanest confirmed instance. It failed as a
*global* rescale because it was applied to ~40 constants at once, most of
which were already at a joint optimum; here the mis-scaling shows up not as a
slightly-wrong threshold but as **mechanism starvation**, which node counts
can see directly and games cannot resolve cheaply.

### The sweep

Lowering the bar (depth 10, wl50):

| `NMP_EVAL_BASE` | nodes | vs baseline | nmp_try | cut rate |
|---|---|---|---|---|
| 374 (current) | 998,495 | −0.4% | 2,047 | 82% |
| 300 | 934,813 | −6.8% | 2,620 | 81% |
| 240 (= 0.64×374) | 1,013,008 | +1.0% | 6,931 | 78% |
| 200 | 886,634 | −11.6% | 11,847 | 71% |
| **168 (crossover 12)** | **812,498** | **−19.0%** | 20,354 | 70% |
| 120 | 758,232 | −24.4% | 47,412 | 66% |
| 60 | 691,133 | −31.1% | 74,181 | 56% |
| 0 | 662,211 | −34.0% | 105,286 | 45% |

Attempts rise 51×; the tree falls monotonically (one non-monotone blip at
240); the cut rate decays 82% → 45%, so the extra null searches are
progressively less productive — which is the cost side and why "lower is
better" does not continue indefinitely.

**Stable across depths** at the SF16-shape value 168:

| depth | vs baseline |
|---|---|
| 8 | **−26.0%** |
| 10 | **−19.0%** |
| 12 | **−26.7%** |

(`BASE 200` is unstable across depths — +0.2% / −11.6% / −19.1% — the
non-monotonicity again. 168 is the robust point.)

Note the frame ratio *under*-predicts: 0.64 × 374 = 239 sits on the blip and
buys nothing, while 168 (= 0.45×) is where the win is. Consistent with §11.4
— the joint optimum is not the frame-scaled point.

### Tactical safety, at equal depth

Equal depth isolates decision quality from the node saving, and is
contention-free (`mate_in_3.epd`, 1277 positions, depth 10):

| `NMP_EVAL_BASE` | solved |
|---|---|
| 374 (control) | 1045 / 1277 |
| 200 | 1038 / 1277 |
| 168 | 1049 / 1277 |

Binomial σ ≈ 14, so the spread is noise. **Tactics are flat while the tree
shrinks ~19–27%.**

---

## 3. Why prior testing missed this

`s4-searchgap.json` already identified the starved-NMP regime — its comment
names "the donor bar (crossover d~27, the starved-NMP regime)" and centres on
an SF16-shape alternative. It also records a measured result: **"NMP (140,12)
sound with the wip/nmp-guards binary (+3.5)"** — +3.5 ± 27.7, uninformative.

That test was at **fixed nodes**. The entire benefit here is a smaller tree,
and fixed-node testing normalises exactly that away. The campaign's own
comment says so — *"Wall-clock TC ONLY (fixed-node structurally undervalues
pruning)"* — but the screen that produced +3.5 was run at fixed nodes anyway.

So the lever has never actually been tested in the regime where it could pay.

---

## 4. What to do, and the risk

**Queue a wall-clock SPRT on `NMP_EVAL_BASE` 374 → 168.** If the node saving
converts, ~19–27% fewer nodes to equal depth is ~0.3–0.45 doublings, which at
our measured ~120 Elo/doubling (exp-0154) is worth tens of Elo. That
conversion is the whole question and node counts cannot answer it.

**The specific risk is zugzwang, and the mate suite does not test it.** NMP
is unsound precisely where a null move is better than any real move —
endgames, locked positions. Our guards (the `non_pawn_material` gate and the
verification search above `NMP_VERIFY_DEPTH`) exist for this, but firing NMP
51× more often exercises them 51× harder. The SPRT book should not be
endgame-poor, and an endgame-weighted suite is worth running alongside.

---

## 5. Razoring is net-negative here — worse tactics AND more nodes

The largest negative contributor got the same treatment, and the answer is
cleaner than NMP's.

Our razoring is an exact copy of SF@`eca43a97`
(`!PvNode && eval < alpha - 465 - 300*depth^2` → `return qsearch`), so the
shape is right. Sweeping `RAZOR_BASE` saturates cleanly — **3000 gives
exactly the same node count as fully disabled (871,706)**, so above that it
never fires, which makes the measurement trustworthy.

| `RAZOR_BASE` | nodes vs baseline | mate_in_3 @ d10 |
|---|---|---|
| 100 (aggressive) | −5.1% | 1022 / 1277 |
| **465 (current)** | −0.4% | **1045 / 1277** |
| 800 | −11.0% | 1050 / 1277 |
| 3000 (≡ off) | **−13.1%** | **1068 / 1277** |

**Monotone in tactics: more razoring is worse.** So this is not a bad value
sitting in a valley — the mechanism is net-negative across the whole range
tested. And the usual trade is absent: razoring-off is *simultaneously* 13%
cheaper and more accurate. A heuristic that costs nodes AND accuracy is doing
active harm, not buying anything.

Independent screens, all one direction:

| screen | ON (465) | OFF (3000) | Δ |
|---|---|---|---|
| mate_in_3 @ d10 | 1045 / 1277 | 1068 / 1277 | **+23** |
| mate_in_3 @ d8 | 908 / 1277 | 926 / 1277 | **+18** |
| WAC-revised @ d10 | 143 / 200 | 147 / 200 | **+4** |

Each is only ~1–1.7σ alone, and the two mate rows share positions so they are
not independent of each other — but WAC is different content and agrees, and
the node evidence is deterministic.

**The mechanism is NOT what it looks like.** The obvious story is that
razoring drops into an expensive qsearch, and that is measurably false:

```
RAZOR_BASE=465    nodes=998,495  qnodes=116,638  qshare=0.117
RAZOR_BASE=3000   nodes=871,706  qnodes=111,109  qshare=0.127
```

Razoring adds only 5,529 qsearch nodes but 126,789 total — **the extra work
is main-search, elsewhere in the tree.** Consistent with a razored node
returning a qsearch estimate that is bad enough to send the search down wrong
branches, which would cost both nodes and accuracy exactly as observed. That
is a hypothesis, not a measurement, and it is recorded as one — two mechanism
stories already died today.

Nor does the frame explain it. The frame argument predicts our razoring is
*more* conservative than SF's (it fires at a ~3.1-pawn deficit at depth 1
where SF's fires at ~2.0), so it should be nearly inert — but it is active
enough to cost 13% of the tree. Whatever is happening, it is not
mis-scaling.

**Action: queue an SPRT on `RAZOR_BASE` 465 → 3000.** This one needs no code
change at all — it is a pure `setoption`, so it can ride the same harness as
the NMP arm. Note Stockfish itself only ever claimed "~1 Elo" for razoring
(SF16's own comment); a marginal heuristic there going net-negative in a
reduction-dominated tree like ours is not surprising.

Second, smaller: **the three remaining negative-contribution mechanisms are
unexplained** — conthist pruning (−5.4%), ProbCut (−5.1%), IIR (−3.0%). Same
treatment applies: sweep to saturation, check the tactical axis, and do not
assume mis-scaling before checking activation.

Ranking for a future SPSA, now evidence-based rather than guessed: **LMR
first** (it is 64% of all tree reduction and `RED_*` is only partly covered by
existing campaigns), then RFP and LMP. Tuning a gate that removes 1% of the
tree cannot pay however mis-set it is.

---

## 6. The scale correction DOES work — per mechanism, not globally

The global `FRAME_S` sweep scaled ~40 constants by one factor and lost. That
result stands, and this section explains it rather than contradicting it.

**Method.** For each eval-denominated gate, scale *its* constants together and
measure two axes on large samples:

- **nodes** — `bench` depth 10 over `scale-check-800` (800 positions, 4.3 s
  per arm, deterministic). Baseline 14,442,305.
- **tactics at EQUAL NODE BUDGET** — `mate_in_3.epd` (1277 positions) at
  `--nodes 50000`. Control 1193, binomial σ ≈ 9.1.

The second axis is the one that matters and the one earlier work got wrong.
**Fixed-DEPTH tactics structurally cannot show the benefit of a pruning
change**: it measures decision quality per depth, so a search that prunes
more only ever looks worse. Fixed nodes asks the right question — same
budget, does the search use it better? This is the same class of error that
made the s4-searchgap NMP screen uninformative (§3).

### The per-gate table

| gate | scale | nodes | tactics @ 50k | Δ |
|---|---|---|---|---|
| **NMP** | 374 → 168 | **−13.6%** | **1203** | **+10** |
| **razoring** | off | **−12.8%** | **1198** | **+5** |
| **parent futility** | 0.64× | **−4.1%** | **1196** | **+3** |
| *(control)* | — | — | 1193 | — |
| QS futility | 0.45× | −1.5% | 1182 | −11 |
| RFP | 0.55× | −12.8% | 1170 | −23 |
| ProbCut | 0.64× | −5.4% | 1168 | −25 |
| capture futility | any | ~0% | — | — |

**This is why one global factor could not work.** The per-gate optima are not
a common scalar — they are 0.45, ∞ (off), 0.64, and *1.0* (leave alone) — and
three gates get actively worse when corrected. RFP is a bigger share of the
tree than NMP (62.6% vs −4.1% contribution, §1), so a uniform correction is
dominated by the arms that hurt. The global sweep was measuring the average
of a set with both signs in it.

The pattern that separates them is not the frame ratio, it is **whether the
gate is starved**. NMP fires at 0.2% of nodes (§2) — the compressed frame put
its bar out of reach, and correcting restores the mechanism. RFP and ProbCut
fire at healthy rates already; their constants are doing their job, and
tightening them just prunes good nodes.

### The curated combination

Taking only the three gates positive on *both* axes:

```
NMP_EVAL_BASE          374 -> 168
RAZOR_BASE             465 -> 3000     (= off; see §5)
PFUT_BASE               40 -> 26       0.64x
PFUT_NO_BESTMOVE       138 -> 88
PFUT_LMRDEPTH_MULT     117 -> 75
PFUT_EVAL_ABOVE_ALPHA   90 -> 58
```

| | nodes | tactics @ 50k |
|---|---|---|
| control | 14,442,305 | 1193 |
| **curated combo** | **11,978,037 (−17.06%)** | **1209 (+16)** |

Better on both axes, and better than any single arm on tactics. The node
saving is sub-multiplicative (−17.1% against a −27.7% independent
prediction), which is expected: NMP and razoring both act on low-eval nodes
and overlap.

−17.1% nodes is ~0.27 doublings, worth ~32 Elo at our measured ~120
Elo/doubling — and the +16 tactical improvement is *on top of* that, since it
is measured at equal budget.

**Confidence.** The node figures are deterministic over 800 positions. The
+16 is ~1.8σ — suggestive, not conclusive. **This needs an SPRT before it
lands**, and the zugzwang risk from §4 applies to the NMP component. But it
is the first combination that improves both axes simultaneously, and it is a
pure `setoption` change requiring no code.

### What this says about the frame question

The scale difference is real, it does cost Elo, and it is correctable — but
the correction is **per-mechanism, seeded by the frame ratio and settled
empirically**, never a single scalar. The right diagnostic is not "what is
this constant's value relative to SF's" but **"is this gate firing at the
rate it was tuned for"** — an activation rate is scale-free, which is exactly
why it survives a frame change that invalidates the raw values.

### 6.1 The frontier, and depth scaling

Pushing the two tunable winners further trades nodes against decisions
(control 1193, σ ≈ 9.1):

| arm | NMP | PFUT | nodes @ d10 | mate_in_3 @ 50k |
|---|---|---|---|---|
| control | 374 | 1.00× | — | 1193 |
| **combo** | 168 | 0.64× | **−17.1%** | **1209 (+16)** |
| — | 120 | 0.64× | −25.7% | 1205 (+12) |
| — | 168 | 0.45× | −21.5% | 1206 (+13) |
| **aggressive** | 120 | 0.45× | **−29.8%** | 1198 (+5) |

Every arm is above control on both axes; the frontier is real but shallow.
`NMP_EVAL_BASE 200` was tested and is worse than 168 on nodes (−11.6%),
confirming 168 rather than an artifact of the grid edge. `IIR off` (−16.0%)
and conthist at 0.64× (−14.3%) both make the combo *worse* and are excluded.

WAC-revised at the same budget cannot discriminate — 159 / 159 / 157 of 200,
σ ≈ 5.7 — so the tactical evidence rests on `mate_in_3`'s 1277 positions.

**The saving grows with depth**, which is the direction that matters since
real games search deeper than 10:

| | depth 10 | depth 12 |
|---|---|---|
| combo | −17.06% | **−20.16%** |
| aggressive | −29.78% | **−34.25%** |

At ~120 Elo/doubling that is ~39 Elo (combo) and ~73 Elo (aggressive) of
equal-time speed at depth 12, with the tactical improvement on top.

**Both are SPRT candidates and neither should land without one.** The combo
has the stronger decision-quality evidence (+16, ~1.8σ); the aggressive arm
has nearly twice the speed and is still above control tactically. Neither is
a code change — both are pure `setoption`, so one harness run can test both
arms against the same baseline.

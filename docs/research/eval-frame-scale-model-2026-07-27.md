# The eval-frame scale model — why our cp margins may all be ~1.7× too large

Date: 2026-07-27. Motivated by the user's call to stop blind-SPSA'ing search
constants and instead "look at the heuristics that don't match what we know to
be true". This is that audit. It produces a one-parameter, derived correction
that replaces a 15-dimensional blind search we cannot afford to run.

## 0. The claim, up front

Every cp-denominated search constant we have — razoring, RFP, capture- and
parent-futility, qsearch futility, aspiration — sits at **Stockfish's raw
value**. But our static eval is measured at **≈0.58× SF18's internal frame**.
If both halves hold, every one of those margins is ~1.7× larger in effect than
its Stockfish counterpart, and we prune systematically less on eval evidence
than SF does.

## 1. The three frames are genuinely different

Comparing raw constants across engines is only meaningful after this table:

| engine | internal → displayed cp | material scale |
|---|---|---|
| **ARCHON** | **none** — UCI reports the internal value raw | PAWN_VALUE 208 |
| **SF16** | divide by a **constant** `NormalizeToPawnValue = 328` | Pawn 126 mg / 208 eg |
| **SF18** | divide by **a(material)** from the WDL model, ≈385.5 at the anchor material 58 | Pawn 208 (mg/eg split dropped) |

So SF16 and SF18 differ from *each other* by ~17.5% in internal-per-displayed
before our frame enters at all. This is also why the reference clone sitting at
`sf_16` was dangerous: razoring reads `alpha − 456 − 252·d²` at that tag and
`alpha − 485 − 281·d²` at `sf_18`. The clone is now checked out at `sf_18`.

## 2. Our frame measured — three independent estimates agree

| estimate | method | ARCHON / SF18 internal |
|---|---|---|
| winner-slope, 2026-07-26 | game outcomes (`loss_forensics`) | **0.55** |
| wl50, n=20 | static eval, this study | **0.579** |
| WAC-revised, n=113 | static eval, this study | **0.605** |

Tool: `tools/eval_scale_study.py`. It recovers each engine's *internal*
pre-blend NNUE value (Stockfish prints v/a; multiplying back by a(material)
inverts it), converts SF's white-relative output to our stm-relative
convention, and takes a median of ratios over positions where both engines
agree on the sign and the magnitude is meaningful.

Two filters matter and both were found the hard way: without the perspective
flip every black-to-move position contributes a negative ratio and the median
lands near zero, which reads as a scale finding rather than a units bug; and
without the sign-agreement filter, positions where the two nets genuinely
disagree about who is better (38% of wl50, 34% of WAC — an eval-QUALITY
difference, not a units one) drag the median from ~0.6 to 0.16.

**Take s ≈ 0.58**, with the honest caveat that the IQR is wide
([0.41, 1.00] on WAC): this is a central tendency, not a constant of nature.

## 3. Our constants are Stockfish's, unscaled

| constant | SF18 | ARCHON | ratio |
|---|---:|---:|---:|
| razor base | 485 | 465 | 0.96 |
| razor depth² mult | 281 | 300 | 1.07 |
| RFP mult (our cap) | 76 | 80 | 1.05 |
| RFP no-ttHit dock | 23 | 20 | 0.87 |
| capture-futility base | 232 | 231 | **1.00** |
| capture-futility lmrDepth mult | 217 | 232 | 1.07 |
| parent-futility base | 42 | 40 | 0.95 |
| parent-futility no-bestmove | 161 | 138 | 0.86 |
| parent-futility lmrDepth mult | 127 | 117 | 0.92 |
| aspiration base delta | 5 | 5 | **1.00** |

Every one sits at ≈1.0× SF18's raw value. **None sits near 0.58.**

**The precedent is in our own tree.** `params.rs` sets `OPTIMISM_NUM = 75`,
`OPTIMISM_DEN_BASE = 45` and documents them as "SF's 137/81 scaled by the
measured winner-slope ~0.55 (our engine frame is ~2× compressed vs SF's
reported cp)". So the frame correction was identified, judged correct, and
applied to exactly one family. Everything else kept the raw import.

## 4. The tension, and what resolves it

If our eval-informed margins under-prune by ~1.7×, our tree should be *bigger*
than Stockfish's. It is **smaller**: 3.07M nodes to depth 14 on wl50 against
SF18's 3.20M and SF16's 4.71M, EBF 1.40 vs 1.51 (perf study 7 §1.2).

The resolution is that search constants come in two kinds:

- **Ply/count-denominated** — LMR reductions, depth gates, move counts,
  null-move R. These are **scale-free** and transfer from Stockfish
  *correctly*.
- **Eval-denominated** — every constant in §3. These need the ×s correction
  and did not get it.

So the hypothesis is that we prune roughly the right *amount* but with the
wrong *instrument*: near-SF-strength blind reduction (LMR/NMP, transferred
correctly) compensating for under-firing eval-informed pruning (RFP,
razoring, futility). Blind pruning discards good moves at the same rate as
bad ones; eval-informed pruning does not. That predicts a tree of similar
size but lower quality — which is exactly the measured picture: equal-node
tactics level with SF16, yet **−33 Elo at equal nodes**, with the deficit
attributed to quiet/positional play.

This is a hypothesis with a mechanism, not a proven cause. It is stated so it
can be attacked.

## 5. The model

For a cp-denominated constant `C` taken from Stockfish, the value that means
the same thing in our frame is

```
C_ours = s · C_sf ,      s ≈ 0.58   (0.55 – 0.61 across three estimates)
```

This is exact for the linear forms we use, because margin and eval are both in
our frame on both sides of every comparison:

- razoring `eval < alpha − (BASE + MULT·d²)` → scale BASE and MULT
- RFP `eval − futility_mult·d ≥ beta` → scale futility_mult
- futility `staticEval + BASE + MULT·lmrDepth ≤ alpha` → scale BASE and MULT

It does **not** apply to history bonuses (their own units), to ply/count
constants, or to `OPTIMISM_*` (already corrected).

### 5.1 …but the exponent of `s` is NOT always 1 — the taxonomy

`C_ours = s·C_sf` is right only for constants that live purely in eval space.
A constant's exponent depends on what it is compared against, and there are
five classes. Getting this wrong in either direction is worse than doing
nothing, so each is derived rather than asserted.

| class | form | exponent | why |
|---|---|---|---|
| **1. pure eval margin** | `eval ⋈ eval ± C` | **s¹** | C is a distance in eval space |
| **2. eval² divisor** | `delta = mss / D`, `mss ~ v²` | **s¹** | D has units eval²/eval = eval — *derived, not obvious* |
| **3. cross-unit divisor** | `eval_adj = \|corr\| / D` | **s⁻¹** | corr keeps its own units; D converts corr→eval, so it must GROW |
| **4. material mixed into eval** | `staticEval + C + PieceValue[x] ⋈ alpha` | **not fixable by C** | see below |
| **5. scale-free** | plies, depths, counts, SEE-vs-material, corr→ply divisors | **s⁰** | no eval on either side |

Class 2 is the trap that looks like class 3: it *is* a divisor, so the reflex
is to divide by `s`. Deriving the units shows `D_ours = s·D_sf` — the same
direction as a plain margin. `ASPIRATION_MSS_DIVISOR` is ours (10588 against
SF18's 9000, i.e. 1.18× raw where it should be 0.58×, so ~2.0× too large).

Class 3 moves the **opposite way**. `correction_value` is in correction/history
units, untouched by our eval frame, so a divisor turning it into an eval
adjustment must get **larger** by 1/s ≈ 1.72 to yield the same effect. SF18
uses `|correctionValue| / 174665` in its futility margin and `/182069`
elsewhere; our `SE_CORR_DIV`-family divisors should be ~1.72× SF's, not 0.58×.
(`RED_CORR_DIV` converts correction→*plies* and is class 5 — untouched.)

### 5.2 Class 4 is a real defect and rescaling a constant cannot repair it

Our material values are Stockfish's **exactly** (PAWN_VALUE 208), while our
eval is 0.58× theirs. So material and eval sit in a different *relative*
proportion in our engine than in SF:

```
eval units per pawn:  SF18 386   ARCHON 224     (ours = s x theirs)
material per pawn:    SF18 208   ARCHON 208     (identical, unscaled)

material as a fraction of a pawn-of-eval:
  SF18   208/386 = 0.540
  ARCHON 208/224 = 0.930      -> 1.72x = exactly 1/s
```

**A captured piece is 1.72× more influential relative to eval in our engine
than in Stockfish's.** Every formula that adds a `PieceValue` to an eval-space
quantity inherits this — SF18's qsearch `futilityValue = futilityBase +
PieceValue[captured]` compared against alpha, and our `futility_base`
equivalent, plus any SEE threshold weighed against an eval margin.

Tuning the margin constant cannot fix it, because the error rides on the
*piece-value term*, which varies with which piece was captured. The available
repairs are (a) scale the PieceValue contribution by `s` at eval-space use
sites only, leaving SEE's own material-vs-material comparisons alone, or
(b) rescale `PIECE_VALUES` wholesale and re-derive every SEE threshold. (a) is
far less invasive; (b) touches the SEE contract and D11.

This class is also the most likely reason a single global sweep over `s` will
**not** be flat but also will not find a clean optimum: it corrects classes 1
and 2 while leaving 3 backwards and 4 untouched. The sweep is still worth
running as a cheap directional test, but a null result would not falsify the
frame measurement — only the assumption that one factor fixes everything.

## 6. Why this is the right answer to the SPSA problem

The ranked search matrix puts "global SPSA retune of search constants vs the
NEW eval" first, at +15–40 Elo. Our SPSA cannot deliver it: c2-reduction is
45% through 19,200 games with nothing past sigma 2, five of fifteen parameters
actively reversing, and a verification instrument (±13 Elo at 200 games) far
coarser than the effect.

This analysis converts that **15-dimensional blind search into a
1-dimensional derived correction**. A single global factor `s` applied to all
cp margins can be swept over ~5 values with fixed-node matches — tractable
inside our game budget, where a 15-D search is not.

**Proposed test:** build `s ∈ {1.0 (control), 0.85, 0.70, 0.58, 0.45}` behind
one tunable multiplier, fixed-node self-play against the s=1.0 control. The
prediction is a monotone improvement toward ≈0.58 and a decline past it; a flat
curve refutes the model outright and costs one afternoon.

## 6.5 ⛔ REFUTED — the SF16→SF18 "natural experiment" carries no frame signal

**Everything in this section is wrong and is kept only as a record of the
error.** Skip to §6.6 for what replaces it.

The refutation is the **co-movement test**, and it is decisive. A frame shift is
a scalar multiply, so it must be *uniform within a single formula*. It is not:

```
singularBeta:  (82 + 65·ttPv)·depth/64  →  (53 + 75·ttPv)·depth/60
               82→53 = 0.646 (SHRANK 35%)   65→75 = 1.154 (GREW 15%)

optimism:      109·prev/(|prev| + 141)  →  142·avg/(|avg| + 91)
               109→142 = 1.303 (GREW 30%)   141→91 = 0.645 (SHRANK 35%)
```

Two constants in the same expression moving in **opposite directions** cannot
be a frame effect. Across four testable shape-clean formulas, none has its
eval-space constants moving by a common factor; observed ratios span
0.645–1.399 while the true frame ratio 1.175 sits mid-spread with nothing
clustering on it. The dispersion is an order of magnitude larger than the
effect being sought.

**My error was a selection effect.** I sampled five constants — razoring and
probcut — that happened to grow, and concluded "5 of 5 grew, p ≈ 6%". I never
looked at singularBeta or optimism, both of which contain terms that *shrank*.
The "median 1.064" and the "38% transfer rate" derived from it were reading
tuning noise.

Lineage is broken as well: `sf_16.1` razoring reads
`alpha − 438 − (332 − 154·(cutoffCnt > 3))·d²`, a different shape from both
endpoints, so the SF16↔SF18 shape match is endpoint coincidence rather than
continuity. And the correction family changed dimensionality entirely
(quadratic `cv·|cv|/12475` → linear `cv/131072`, with `cv` itself redefined
from one table to a four-table blend), so no divisor pair there is comparable
at all.

## 6.6 What actually survives, and what it means

**The frame ratio itself stands** — independently confirmed: sf_16 `a` =
328.107, sf_16.1 = 355.855, sf_18 = 385.474, so sf_16→sf_18 = **1.175**. Note
sf_18 removed `NormalizeToPawnValue` entirely and re-anchored `a` from game
*ply* to *material*, so the three anchors are not quite the same quantity.

**The core model is untouched**, because it never depended on SF's history.
Our eval measures at ≈0.58× SF18's internal frame, and our cp constants are
SF's raw values. What died is only the attempt to estimate a *damping factor*
from how SF retuned itself.

**And the refutation actually sharpens the case.** SF's constants are not a
clean scalar family — each literal has been independently SPSA-tuned over
years to SF's own frame. We imported that tuned set wholesale into a frame
0.58× as wide and never retuned. So our constants sit at *SF's* empirical
optimum expressed in *our* units, which is precisely the mismatch the model
describes.

**Consequence for the experiment:** the predicted optimum reverts to
**s ≈ 0.58 with no empirical damping**, and there is no longer any basis for
the 0.6–0.85 range. But one caveat the refutation raises is worth keeping: if
SF's per-constant optima are genuinely non-uniform, a single global `s` is an
approximation to a set of individually-correct factors, which is a further
reason the sweep may be shallow rather than sharply peaked.

**Method note for future use:** to compare frames across engine versions, read
`a` / `NormalizeToPawnValue` directly. Do not try to back-solve a frame ratio
out of tuned search constants — the tuning noise dominates by an order of
magnitude.

Stockfish changed its own eval frame between 16 and 18 (`a`: 328 → 385.5, a
**1.175×** expansion) *and* retuned its constants. That is a natural experiment
on exactly our question: when the frame moves, how much do the constants move
with it? Restricting to constants whose formula shape is unchanged:

| constant | SF16 | SF18 | ratio | ÷ frame |
|---|---:|---:|---:|---:|
| razor base | 456 | 485 | 1.064 | 0.905 |
| razor depth² mult | 252 | 281 | 1.115 | 0.949 |
| probcut beta base | 168 | 235 | 1.399 | 1.190 |
| probcut improving | 61 | 63 | 1.033 | 0.879 |
| probcut beta (2nd) | 413 | 418 | 1.012 | 0.861 |
| | | **median** | **1.064** | **0.905** |

(Excluded: qsearch `futilityBase`, which changed from `bestValue + 200` to
`staticEval + 351`. Its 1.755× is a redesign, not a rescale.)

Two readings, and both matter:

**The frame effect is real.** All five constants grew, and the frame grew. If
retuning were random with respect to the frame, five-for-five in the same
direction has probability ~6%. Constants do follow the frame.

**But they capture only part of it.** The median constant grew 1.064× against a
frame that grew 1.175×. In log terms that is
`ln(1.064)/ln(1.175) ≈ 0.38` — SF's own retune absorbed roughly **38%** of its
own frame expansion, not 100%.

If that transfer rate carries to us, the right correction is not the full
`s = 0.58` but `0.58^0.38 ≈ 0.81`. **This materially changes the prediction.**
The pre-registered sweep {1.0, 0.85, 0.70, 0.58, 0.46} already brackets both
hypotheses, which is fortunate — but the expected optimum should now be stated
as **a range, ≈0.6 to ≈0.85, rather than a point at 0.58**, and a best result
at 0.85 must NOT be read as "no effect".

Caveats, because n = 5 is thin: SF's retune conflates the frame change with
genuine optimum drift, most formula shapes changed too much to compare at all,
and the 0.38 exponent is a crude fit to five points. Treat it as evidence that
the correction is **partial**, not as a second decimal place.

## 7. What would falsify this

- The sweep is flat, or best at s = 1.0. Then either the frame measurement is
  wrong, or our margins were re-derived deliberately and the SF-value
  coincidence in §3 is accidental.
- Tactical suites collapse at s < 1. That would say the margins are load-bearing
  for tactics and the frame argument is too crude.
- The 0.58 estimate is position-dependent enough (IQR [0.41, 1.00]) that no
  single global factor works, and the right fix is per-family.

## 8. Status

Model only — **nothing changed in the engine.** The measurement tool is
committed; the sweep is not yet built. Do not apply a blanket rescale on the
strength of this document alone.

## 9. Family localisation RE-MEASURED on NNUE (2026-07-27, after the bench-net fix)

Every tree number in the original sweep was taken through a UCI pipe, where
`bench` silently dropped its net argument (fixed in `36966dd`), so all of it
was PSQT. Re-measured with `EvalFile` set explicitly, s = 0.58, depth-10 wl50,
control 528,118:

| family | PSQT (invalid) | **NNUE (correct)** |
|---|---:|---:|
| PRUNE | −11.9% | **−14.21%** |
| EXTEND | +107.2% | **+25.62%** |
| HISTMG | −10.8% | **+5.47%** |

Three things change.

**PRUNE survives and strengthens** — it still shrinks the tree, slightly more
than PSQT suggested. The one family whose behaviour the wrong evaluator did
not misrepresent.

**EXTEND's +107% was mostly an artifact.** On NNUE it is +25.6%, four times
smaller. It remains the family that grows the tree and should still not
receive the correction, but "EXTEND is the entire problem" overstates it: the
global knob's failure is no longer explained by one family swamping the
others. The associated s=470 DNF and the "penalty compounds with depth"
depth-ladder were measured the same way and are likewise unreliable.

**HISTMG flips sign, and that matters most.** PSQT said it shrank the tree by
10.8%; on NNUE it GROWS it by 5.5%. HISTMG was singled out as "the only arm
improving both instruments" precisely because of that tree number, and the
tree half of that claim is now gone.

What survives for HISTMG is the half that was always measured on NNUE:
tactics 1218/1277 against a 1211 control, and a game screen reading
+16.4 ± 23.3 Elo at 340 games. So the honest description is no longer "shrinks
the tree and sharpens tactics" but **"spends ~5% more nodes for slightly
better play"** — a node-for-quality trade, the same shape as an extension.
That is a coherent and testable thing for constants that gate a fail-low
history bonus; it is simply not the thing we thought we had found.

## 10. The dual formulation — scale the EVAL UP, not the margins down

Every arm tested so far scales ~40 margin constants DOWN by s. The dual move
is to scale the EVAL UP by 1/s ≈ 1.72 and leave every constant alone. For
class 1 the two are algebraically identical — only the ratio margin/eval
matters — but the dual is strictly better, for a reason that falls straight
out of §5.2:

| | scale 40 margins by s | scale eval by 1/s |
|---|---|---|
| class 1 (eval margins) | corrected | corrected (identical ratio) |
| class 2 (eval² divisors) | corrected | corrected |
| **class 4 (material vs eval)** | **impossible** | **corrected automatically** |
| class 3 (cross-unit divisors) | needs 1/s | still needs its own handling |
| class 5 (plies/counts) | untouched | untouched |
| surface area | ~40 constants, 5 classes, per-site judgement | **one multiply** |

Class 4 is the point. Our `PIECE_VALUES` are Stockfish's exactly (pawn 208)
while our eval is 0.58× theirs, so a captured piece weighs 1.72× more against
eval here than there (§5.2). No margin constant can absorb that, because the
error rides on the piece-value term itself. Scaling the eval up moves material
and eval back into Stockfish's proportion in one stroke.

**Why this is safe in the way that matters.** A positive scalar is monotone,
so the net's RANKING of positions is completely unchanged — the same move is
best at every node. Only the eval's interaction with cp-denominated thresholds
changes, which is precisely the thing under test and nothing else.

**What still needs care:** mate/TT bounds and the dither must not be squeezed
(they are far from the eval range, but check); class-3 correction divisors are
unaffected by the dual and still need their own treatment; and the reported
`cp` to the GUI changes by 1.72×, which is cosmetic but should be normalised
before anyone reads a score.

**The test is one tunable** — an `EVAL_SCALE` in 1024ths applied at the end of
`evaluate()`, identity at 1024, swept upward. That is a far smaller surface
than 40 wired constants across five classes, and it tests the *same*
hypothesis plus the one part of it that was previously untestable.

**Status: proposed, not built.** Recorded now because it is the most useful
thing the day's work points at, and because the negative results above are
evidence against the *implementation* we tried, not against the frame
measurement itself.

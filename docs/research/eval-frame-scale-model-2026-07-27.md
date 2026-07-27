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

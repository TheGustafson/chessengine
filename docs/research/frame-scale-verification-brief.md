# Verification brief — the eval-frame wiring (second, independent reviewer)

Rewritten 2026-07-27 for the **per-family** design. The previous version was
written against a single global `FRAME_S` and is superseded; the reasoning
below is unchanged, but the target is now one factor per family rather than
one for the engine.

Written before the implementing agent's results were read, so the checklist is
not shaped by its answers.

## Why a second pass is required rather than optional

The failure mode is **silent**. A wrongly-classified constant moves the
opposite way to intent, cancels part of the real effect, and turns a genuine
result into a flat sweep — at which point we conclude the model is wrong and
stop looking. Nothing crashes, no test fails, and the bench pin still holds,
because the pin is measured at factor = 1024 where every transform is a no-op
by construction.

**The identity gate cannot catch a misclassification.** That is exactly why
identity passing is not evidence that the wiring is right.

## The standard

Under-wiring costs a fraction of the available effect and is recoverable. A
wrongly-wired constant corrupts the experiment and is not, because we draw the
wrong conclusion and abandon a real lever. **Where reviewer and implementer
disagree, the constant comes out of the wiring** until a measurement settles
it.

## Established facts — do not re-litigate, challenge only with a measurement

- **Frame factor ≈0.58** (ARCHON eval / SF18 internal). Three independent
  estimates: 0.55 winner-slope from games, 0.579 wl50, 0.605 WAC.
- **Frame ratios between SF versions**: sf_16 `a` = 328.107, sf_16.1 = 355.855,
  sf_18 = 385.474. Read these directly; never back-solve a frame ratio out of
  tuned search constants.
- **The five-class taxonomy** and its derived exponents. Challenge an
  individual *assignment*, not the framework.

## Withdrawn — do not resurrect either

1. **The node-count direction test.** `QS_FUTILITY_MARGIN` is provably
   eval-denominated (`futility_base = static_eval + C`) yet shrinking it GREW
   the tree 3.8%. Node count is not monotone in pruning aggressiveness: an
   earlier cutoff changes TT content, which changes ordering and re-searches
   above. Direction proves nothing.
2. **SF16→SF18 constant ratios as a frame signal.** Killed by the co-movement
   test — `singularBeta` 82→53 (−35%) beside 65→75 (+15%) in the *same*
   expression; `optimism` 109→142 (+30%) beside 141→91 (−35%). A frame shift is
   a scalar multiply and must be uniform within a formula. Ratios span
   0.645–1.399 with the true 1.175 mid-spread and nothing clustering on it.

## The subtle failures to hunt

1. **Negative constants.** `QS_SEE_FLOOR` is −74; scaling a negative toward
   zero *tightens* a floor rather than loosening it. Re-derive direction from
   the comparison's semantics, never inherit it from the class.
2. **Truncation asymmetry.** Integer division truncates toward zero, so ± sides
   of a symmetric margin round differently — a bias that did not exist before.
3. **One tunable, several use sites, different units.** Grep every wired name
   for ALL its uses, not just the one that motivated the class.
4. **Double-scaling** through dataflow, where a scaled value feeds another
   scaled expression.
5. **Partial correction of a class-4 formula.** Where an eval constant shares
   an expression with a `PieceValue` term, scaling one side shifts the balance
   and may be worse than leaving both alone.
6. **The dimensionless-ratio trap.** `RFP_IMPROVING`/`RFP_OPP_WORSENING` over
   `RFP_TREND_DEN` — if dimensionless, scaling numerators without the
   denominator silently changes the discount. Verify none of it was wired.
7. **Identity proven only where measured.** The bench pin is a DEFAULT-FEATURE
   (PSQT) build. Identity at 1024 must also hold for `sota,arch-d47v` (eval
   hash `cd190d1b…`, d13 nodes `2068701`) and for the `tune` build, where
   params are atomics on a different code path.
8. **Factor loads in the hot path.** In `tune` builds every getter is an atomic
   load; if a family factor is read per node rather than hoisted, it adds
   traffic and a value could change mid-search.
9. **Downstream overflow** in an expression that was previously known-small.
10. **`OPTIMISM_*` was ALREADY frame-corrected** (75/45 = SF's 137/81 × 0.55,
    `params.rs:81`). Wiring it applies the correction twice. Confirm excluded.

## New, and specific to the per-family design

11. **Family membership is now load-bearing.** A constant in the wrong family
    gets the wrong factor even when its CLASS is right. Check every assignment
    against the use site, especially constants shared across techniques.
12. **Cross-family formulas.** Where one expression mixes constants from two
    families (an RFP margin adjusted by a correction term, say), applying two
    different factors inside one comparison can be incoherent. Flag any such
    site.
13. **All factors must default to identity**, and the engine must be
    bit-identical with all of them at 1024 — not merely with one at a time.
14. **Refactor-vs-behaviour separation.** The params reorganisation must be
    provably inert BEFORE any factor is wired. If both landed in one commit,
    say so — that is a process defect worth reporting even if the result is
    correct, because it destroys attribution.

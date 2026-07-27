# Verification brief — the eval-frame wiring (for the second reviewer)

Written 2026-07-27, BEFORE the implementing agent reported, so the checklist is
not shaped by its answers. User instruction: a second independent Opus 5 pass
"looking for anything subtle that we may have missed".

Why an adversarial second pass is warranted here rather than optional: the
failure mode is **silent**. A single wrongly-wired class-3 constant moves the
opposite way to intent, cancels part of the real effect, and turns a genuine
result into a flat sweep — at which point we conclude "the model is wrong" and
stop looking. Nothing crashes, no test fails, and the bench pin still holds
because the pin is measured at FRAME_S = 1024 where every transform is a no-op.
**The identity gate cannot catch a misclassification.** That is precisely why
it is not sufficient evidence.

## The subtle failures to hunt for

These are the ones I consider most likely to have been missed. Not exhaustive —
the reviewer should look for its own.

1. **Negative-valued constants.** `QS_SEE_FLOOR` is −74. Multiplying a negative
   by s<1 moves it *toward zero*, which for a floor is a TIGHTENING, not the
   loosening the same arithmetic produces on a positive margin. Every negative
   constant needs its transform direction re-derived from the semantics of the
   comparison, not inherited from the class.

2. **Integer truncation asymmetry.** `frame_scale` truncates toward zero. For
   symmetric ±margins the positive and negative sides therefore round
   differently, which can introduce a small bias that did not exist before.
   Check anywhere a scaled constant is used with both signs.

3. **One tunable, several use sites, different units.** A constant read in two
   places where it means different things must not get one blanket transform.
   Grep each wired name for ALL its use sites, not just the one that motivated
   the classification.

4. **Double-scaling.** If a wired constant's value flows into a second
   expression that is itself wired, it gets the factor twice. Trace the
   dataflow, not just the literal.

5. **Partial correction of a class-4 formula.** Where an eval-space constant
   sits in the SAME expression as a `PieceValue` term (qsearch capture
   futility), scaling the constant alone shifts the balance between the two
   terms and may be WORSE than leaving both alone. Check whether any wired
   class-1 constant shares a formula with a material term.

6. **The dimensionless-ratio trap.** `RFP_IMPROVING` / `RFP_OPP_WORSENING`
   divided by `RFP_TREND_DEN`: if that ratio is dimensionless, scaling the
   numerators without the denominator silently changes the trend discount.
   Verify whether the trio is dimensionless and, if so, that NONE of it was
   wired.

7. **Identity proven only where it was measured.** The bench pin is a
   DEFAULT-FEATURE (PSQT) build. Identity at FRAME_S = 1024 must also be shown
   for `sota,arch-d47v` — via the eval hash `cd190d1b…` and d13 nodes
   `2068701` — and for the `tune` build, where the params are atomics rather
   than folded constants and a different code path computes them.

8. **`FRAME_S` read inside a hot loop.** In `tune` builds every getter is an
   atomic load. If `frame_scale` is called per node rather than hoisted, it
   adds atomic traffic to the hot path, and worse, a value could change
   mid-search. Check where the calls sit.

9. **Overflow downstream.** `frame_scale` uses an i64 intermediate, but a
   scaled constant feeding an expression that was previously known-small could
   overflow there instead.

10. **Constants that are already frame-corrected.** `OPTIMISM_NUM` /
    `OPTIMISM_DEN_BASE` were ALREADY scaled by the 0.55 winner-slope
    (documented at `params.rs:81`). Wiring them again applies the correction
    twice. Confirm they were excluded.

## What the reviewer should NOT re-litigate

- The frame factor itself (≈0.58). Three independent estimates agree and the
  measurement tool is committed; challenge it only with a measurement.
- The five-class taxonomy's derivation. Challenge an individual *assignment*,
  not the framework.
- The node-count direction test — already withdrawn as a classification
  criterion, because `QS_FUTILITY_MARGIN` is provably eval-denominated yet
  grows the tree when shrunk (node count is not monotone in pruning
  aggressiveness). Do not resurrect it.

## The standard to hold

Under-wiring costs a fraction of the available effect and is recoverable. A
wrongly-wired constant corrupts the experiment and is not, because we would
draw the wrong conclusion and abandon a real lever. **When the reviewer and the
implementer disagree, the constant comes out of the wiring** until a
measurement settles it.

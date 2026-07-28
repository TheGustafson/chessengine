# The aspiration window is 2.4x too narrow, and it is a squared-frame bug

Found 2026-07-27 by tracing a single lost game to the bottom.

## The chain

1. **A loss.** Game 17, move 25. Our eval reads +2.92; SF18 reads +0.96 on the
   same position. Black plays the quiet `Rfe5` and our eval instantly falls to
   +1.01 — landing on the number Stockfish already had. We did not get worse,
   we found out.
2. **Not the net.** Our STATIC eval on that position is +1.31 against SF18's
   +0.58 — a 0.73 gap. Our SEARCHED eval is +2.92 against +1.03 — a 1.90 gap.
   The search added +1.61 to its own static eval; Stockfish's search added
   +0.45 to its own.
3. **The root does not converge.** Across 14 iterations on that position our
   score runs 177 → 402 → 197 → 177 → 242 → 383 → 207 → 260 → 218, a 1.44-pawn
   range. SF18 runs 73 → 115 → 87 → 78 → 73 → 99 → 94, a 0.45-pawn range, on
   the identical position with the identical PV. **3.2x the oscillation.**
4. **The aspiration window causes it.** Widening `ASPIRATION_BASE_DELTA` from 5
   to 40 more than halves the range (234 cp → 100 cp) and produces five
   consecutive identical iterations. 400 is worse again, so it is not "wider is
   better" — there is a right width and we are below it.
5. **Why we are below it.** Stockfish's window is adaptive:

       delta = 5 + threadIdx % 8 + abs(rootMoves[pvIdx].meanSquaredScore) / 10588

   `meanSquaredScore` is a **squared** eval quantity. Our eval frame measures
   0.64x Stockfish's, so our mss is **0.64² = 0.41x** theirs — and we divide it
   by the same transcribed 10588. **Our adaptive widening is 2.4x too small.**
   The base term (5) is linear and fine; only the quadratic term is wrong.

## The fix, and its evidence

`ASPIRATION_MSS_DIVISOR: 10588 -> 4337` (= 10588 x 0.64²). On the position
above:

| divisor | ladder range | re-searches |
|---|---:|---:|
| 10588 (transcribed) | 234 cp | 86 |
| **4337 (frame-corrected)** | **143 cp** | **51** |

39% less oscillation and 41% fewer re-searches, from arithmetic rather than
tuning. Consistency check: at this position the corrected divisor yields a
window of ~14-42, which is exactly the range where brute-force widening was
empirically stable.

## Why every previous frame campaign missed it

Two independent reasons, both worth remembering:

- **It is quadratic.** Five formulations of eval-frame rescaling (global, dual,
  per-family, per-mechanism, class-4) all scale constants **linearly**. A
  divisor of a squared quantity needs the **square** of the frame ratio, so a
  linear rescale cannot fix it no matter how it is applied.
- **It is not typed `Value`.** `ASPIRATION_MSS_DIVISOR` is an `i64`. The
  type-derived frame audit in `tools/frame_audit.py` enumerates
  `pub NAME: Value` and therefore never saw it. The audit's exhaustiveness
  guard is only exhaustive over the type it enumerates.

Both failures share a shape: the constant is eval-denominated in a way the
type system does not express.

## Scope

A grep for eval-squared and eval-product quantities divided by transcribed
constants finds exactly one other family — `STAT_BREADTH_DIVISOR` and
`CONT_UPDATE_DIVISOR` — and both are HISTORY-unit products, not eval. History
table geometry was separately verified identical to `eca43a97`, so they carry
no frame error. **`ASPIRATION_MSS_DIVISOR` is the only squared-eval divisor in
the engine.**

## Status

Position-level evidence only so far. A 4000-position screen at 3000 / 4337 /
6500 is running, and an SPRT follows if it holds. One position proves a
mechanism, not an Elo gain.

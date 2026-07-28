# "We cannot convert" does not survive equal nodes — and it is symmetric

2026-07-28. Attacks the folder's own headline (`WHY-WE-LOSE-d48x.md`, §1 of
`START-HERE.md`) with the control it never had. Per README §2 rule 6: the story
was beautiful, so it got attacked hardest.

## The claim under test

> converts a won book: **ARCHON 12%, SF18 84%**. A +1 pawn book is worth 92% to
> Stockfish and 55% to us. That gap IS the −132 Elo.

Two problems with reading that as a *localisation* rather than a restatement:

1. **It is measured at 10+0.1**, where we also carry a ~1.5–1.9× speed handicap
   worth ~66 of the 134 Elo. Conversion and speed are not separated.
2. **A conversion-rate gap is what an Elo gap IS.** Hand a +1 book to a
   132-Elo-weaker engine and it will not convert it; that is the definition of
   weaker, not a defect location. To be a localisation the gap has to be
   *bigger in the conversion direction than everywhere else*.

## The experiment

* **Book** — `engines/books/d48x-plateau.epd`, 308 positions taken from
  `positions/plateau.json` (the corpus's own +0.50…+1.50 band, inside the 172
  squandered wins). Two per game at the 33rd and 66th percentile of that game's
  plateau plies — NOT the peak, which would select on a noisy maximum and
  guarantee regression to the mean.
* **Advantaged side** — SF18 @ 1M nodes per position, white POV, stored in
  `positions/plateau-advside.json`. The FEN does not say who is better and the
  source game's colour cannot be recovered without the PGN-replay trap; a
  neutral deep score is the only honest way to label it.
* **Match** — `-each nodes=110000`, colour-paired (`-repeat -games 2`), 616
  games. Equal nodes removes the speed handicap exactly.
* **Read** — paired, by advantaged side, `tools/convert.py`.

## Result (n=615, complete)

| advantaged side | n | W-D-L | converts | score |
|---|---|---|---|---|
| ARCHON | 294 | 54-230-10 | **18%** | 57.5% |
| SF18 | 295 | 138-155-2 | **47%** | 73.1% |

**Conversion gap +29 points, against +72 at the time control.**

And the decisive part — the same table read from the other side:

| | ARCHON | SF18 | gap |
|---|---|---|---|
| converts an advantage | 18% | 47% | **29** |
| salvages a disadvantage | 53.2% | 81.6% | **28** |

(Read the partial-run figures out of git history if you want them; at n=469 this
was +25/+25 and the overall Elo read −46. `match.py`'s nagging is right —
the final is −54, and the point estimate moved 8 Elo between n=469 and n=615.)

**The gap is the same size in both directions.** A conversion-specific defect
would make the conversion gap exceed the salvage gap. It does not. We are
uniformly weaker by ~25 points of outcome whichever end of the imbalance we
hold, which is what a flat Elo deficit looks like — not a grinding defect.

## Overall Elo on this book

**−54 Elo [−70, −38]** at 110k nodes (n=615), against **−68 [−93, −43]** on the
corpus openings at the same budget. Statistically the same. **The positions the
corpus says we squander are not positions we are specially bad at.**

## What this changes

`START-HERE.md` §4 says "the failure is slow degradation of a held advantage"
and directs work at conversion. That framing is not supported at equal nodes.
The honest statement is the one already in the ledger: a **uniform,
budget-independent ~68 Elo** with equal depth, equal tree shape and at-par
static eval, and no phase or advantage-regime carries more of it than any
other.

## Caveats, stated rather than buried

* The plateau set is drawish (≈65% draws), which compresses BOTH engines'
  conversion rates toward each other. The +25 is therefore a floor on the gap,
  not an estimate of it. What is NOT compressed is the *symmetry* between the
  conversion and salvage gaps, which is the load-bearing part.
* Different position class (mid-game) and different budget regime (nodes, not
  time) from the 12%/84% figure. This does not refute that measurement; it
  shows the effect it describes is not localised to conversion.

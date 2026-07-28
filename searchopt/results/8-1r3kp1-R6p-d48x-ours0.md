# OURS index 0 — `8/1r3kp1/R6p/1PR4P/6P1/8/p4P1K/r7 b - - 36 65`

Played `b7b8`, oracle `a1b1`, wp_loss 0.068. Walked 2026-07-28 (D48X).

## Verdict: REAL DEFECT, NO MECHANISM. Every mechanism instrument died to its own null.

## What is established (survives controls)

* **`a1b1` is better than `b7b8` by ~80 SF cp at depth.** Confirmed independently by
  SF18 (`a1b1` +0 vs `b7b8` −195 at 500k) *and* by ARCHON's own forced search
  (−7 vs −88 at 1M, SF frame). Two engines, forced scores, no argmax. Proof-grade
  by README §5.
* **Our free search never plays it** — not at 50k, 110k, 250k, 500k or 1M. At 1M it
  picks a third move (`f7f8`) entirely. More budget does not fix it.
* Confirmed OURS by `walk.py`'s 5-budget ladder, so the classifier's label was right
  here even though the classifier itself is un-nulled.

## What DIED (do not rebuild)

| claim | what killed it |
|---|---|
| "restricting the root to the 2 candidates fixes it ⇒ root-move interference" | head-to-head argmax flips **2/6** under inert ±1 (`RED_BASE+1`, `RED_DELTA_MULT−1` both give `b7b8`) |
| "the free search picks `b7b8`" as a stable fact | free argmax flips **2/6** under inert ±1 (`RED_BASE+1`→`f7g8`, `RED_FLAT−1`→`f7e7`) |
| "`a1b1` is starved of nodes" | allocation ratio swings **8.1× → 0.1×** across the same inert panel; `RED_DELTA_MULT−1` makes the engine play the RIGHT move. This is the refuted "any perturbation shakes it off the blunder" pattern |
| "`a1b1` gets 1 node" (my own over-read) | that was ONE row at d9. Total across the search is **17,478** vs `b7b8`'s 81,229 — 4.6×, not 1000× |

## Trace facts (recorded, not yet load-bearing)

Root decisions at 110k, `engines/bin/archon-trace`:

* d2–d8 `a1b1` is move **#1**, full-window PV (`scoutD 0`), scoring −41 → −104 (ours).
  `b7b8` sits at move **#16**, reduced +329…+3448, scouted, never re-searched.
* **At d8 the root enters an 11-slide aspiration fail-low storm**: windows walk
  −51/−41 → −56/−51 → −62/−56 → −72/−62 → −85/−72 → −102/−85 → −124/−102 →
  −153/−124 → −191/−153 → −242/−191 → −310/−242, `a1b1` failing low the whole way.
* `b7b8` — still move #16, still reduced — raises alpha at the BOTTOM of that storm
  (−303) and takes the root. The root then re-orders and `a1b1` never regains it.

The storm is a *structural* event (a count of root re-searches), not an argmax, so it
is the one candidate statistic here that has not yet been shown to be noise. It was NOT
nulled — if this position is revisited, null the re-search count first.

## Cost

~25 min. Instruments used: walk.py, searchdbg.py (7 traces), forced-score ladders,
6-arm inert null panel. Files: `/tmp/.../dbg_root.jsonl`, `alloc_*.jsonl`.

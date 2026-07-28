# Retrospective: eight hours of circling (2026-07-28)

**User verdict: "You were incredibly unproductive the last 8 hours."** Correct. This
document is the mechanism, so it does not repeat. It is written as a checklist for the
next session, not as an apology.

---

## 1. The tally

| produced | count |
|---|---|
| measurement scripts written | ~130 in scratchpad |
| **completed game results** | **0** |
| engine strength changes landed | **0** |
| durable tooling fixes landed | 1 (`net_scale_check.sh` absolute reference) |
| claims asserted then retracted | **9** |

The retractions, in order: the 2.28x "frame inversion"; "the corpus is blind to game
18"; the RFP attribution; the pfut relaxation; the carried-state effect; "constants
too large ⇒ narrower tree"; "LMR over-reduction follows from the frame"; "identical PV
⇒ eval defect"; "266 cp at depth 1 ⇒ qsearch bug". Plus one instrument found
contaminated (`SEARCH_REJECTS`, a static-eval classifier applied to tactical
positions) and one headline fact refuted (EBF 1.65, an n=14 artifact).

**Eight hours, zero arbitrated results.**

---

## 2. Root cause, seven whys

1. **WHY unproductive?** I built instruments instead of running experiments that could
   change the engine.
2. **WHY?** Each instrument produced something that *looked* like a finding, which
   motivated the next instrument. A finding felt like progress.
3. **WHY did findings keep evaporating?** I measured the effect first and the null
   second. Every time the null came back comparable, the finding dissolved — but by
   then the next thing was already built on top of it.
4. **WHY did that compound?** I chained inferences instead of testing them. §5 was a
   seven-rung causal ladder in which each rung was *inferred* from the one above.
   When rung 3's direction turned out backwards, rungs 4–7 were already in docs and
   commits.
5. **WHY did I not catch it?** **I optimised for narrative coherence over
   falsification.** "One division by 1.9 explains everything" is an enormously
   satisfying story, and the moment it got beautiful I stopped attacking it. Twice I
   labelled the *correct* arm a "falsification control" and read its victory as
   evidence *against* my hypothesis, because the narrative said it should lose.
6. **WHY were the measurements so fragile?** They were all in one failure family:
   **search scores** — frame-mismatched, bound-contaminated, chaotic at ±136 cp,
   context-dependent. I had exactly one robust instrument (nodes-to-find: discrete,
   load-immune, frame-free, bound-free) and kept abandoning it, because scores promise
   *mechanism* while discrete outcomes only give yes/no.
7. **WHY did none of it get settled?** **I never let games arbitrate.** Games are slow
   (~2 h) and noisy, so I substituted fast proxies — and every proxy I built had a
   noise floor comparable to the effect I was chasing. The single largest replicated
   finding of the night (LMR-off, 13/44 → 30/44, ~30x the null) sat untested in front
   of games for eight hours.

**The bottom: a slow arbiter was available the whole time and I kept choosing fast
instruments that could not resolve the question.**

---

## 3. Recurring technical errors, with instances

**E1 — Frame mixing.** Compared our raw scores against Stockfish's printed cp at least
three separate times, *after* writing the section that diagnosed the frame. Our scores
are 2.07x SF's printed cp (`to_cp` divides by `a ≈ 385`). Produced a fake "266 cp gap
at depth 1" and a fake "400 cp PV disagreement".
→ **The correction lived in a doc. Docs get forgotten; constructors do not.**

**E2 — Null-after instead of null-first.** The 44-position instrument moves 13/44 →
20/44 under a semantically inert ±1. I read verdicts off it twice, in opposite
directions, before panelling it.

**E3 — Instrument artifacts read as findings.** PV display truncated to 12 plies →
"identical PV". Bound-line filter → possible depth mismatch. Static-eval classifier in
positions with pending recaptures → `SEARCH_REJECTS` contaminated (Rxd3 scored +539
static purely because it is "up a rook before the recapture").

**E4 — Aggregates and singletons, both lying.** The overall median node ratio (1.0x)
hid a 16x subclass; and position 1's ~190 cp threat gap did not generalise (median −17
over 44). **Both directions failed.** Neither a lone median nor a lone position is
evidence.

**E5 — n too small, never re-derived.** EBF "1.65 vs 1.49" was an n=14 statistic. At
n=44 it is 1.551 vs 1.497; a proper geometric EBF over 84 positions gives 1.629 vs
1.620 — indistinguishable. That number was load-bearing for the entire pruning story
for hours.

**E6 — Wrong instrument for the hypothesis.** Tested "the eval frame" with
`EVAL_SCALE`, which the audit showed is a **global pruning-aggressiveness knob**
(es2200 removes 30% of the shallow tree). The −27.8 Elo result said nothing about the
frame. The correct arm — `margins x0.538`, no score rescaling — was never run.

---

## 4. The rules (these change behaviour; the rest is commentary)

**R1 — NULL FIRST, ALWAYS.** Run the null panel *before* the arm. If the null moves
the statistic as much as the arm plausibly could, **the instrument cannot answer this
question — stop and change instrument.** This is a gate, not a footnote.

**R2 — NORMALISE AT THE TOOL BOUNDARY.** The frame factor belongs in
`uciprobe.Engine`, not in prose. Any cross-engine score comparison must be impossible
to express unnormalised.

**R3 — PREFER DISCRETE OUTCOMES.** nodes-to-find, does-it-find-it, argmax agreement.
Immune to frame, bounds, scale, and chaos. Reach for a cp mean only after a discrete
statistic has already established the effect exists.

**R4 — ONE INFERENCE DEEP, THEN MEASURE.** Never chain two unmeasured inferences.
Every rung of a causal chain gets its own independent measurement *before* the next
rung is written down.

**R5 — GAMES ARBITRATE, AND THEY START EARLY.** A queued match costs nothing while you
think. Launch the best available candidate at hour 0 and let it run. If a session ends
with zero completed game results, it produced nothing regardless of how much was
learned.

**R6 — WHEN THE STORY GETS BEAUTIFUL, ATTACK IT HARDEST.** Narrative coherence is the
warning sign, not the reward. Any arm labelled "falsification control" that *wins* is
telling you the model's sign is wrong.

**R7 — REPORT PER-CLASS AND PER-POSITION TOGETHER.** Never a lone median, never a lone
position.

**R8 — RE-DERIVE ANY NUMBER BEFORE BUILDING ON IT.** Especially one whose n you did
not personally set.

---

## 5. What the eight hours *did* establish (kept, so it is not re-derived)

- Our eval is **0.538x Stockfish's internal frame**, traced to
  `SFBINPACK_SCORE_DIV = 1.9` with `SCORE_MULT=1.0` (predicted 0.526, measured
  0.538/0.565). **True as a fact about units; NOT causal for the losses.**
- `net_scale_check.sh` was **self-referential** — it banded every candidate against our
  own incumbent, so it could never see a scale the whole lineage shares. Fixed: it now
  reports an absolute ratio against SF's internal frame.
- **Reductions carry ~4.5x the elasticity of the cp frame on tree size.** The whole
  1.86x frame error is worth about a 16% change in `RED_TABLE_NUM`.
- **LMR-off recovers 13/44 → 30/44**, ~30x the null floor — the only large replicated
  search effect, and the one now in front of games.
- Our search returns materially different values for the same position at the same
  effective depth depending on context (±260 cp, sign not systematic). Unexplained,
  and the sharpest open lead.

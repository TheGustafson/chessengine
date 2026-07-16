# SF14→SF16 non-NNUE optimization study (mined 2026-07-16)

Sources: full commit-message read of `sf_14..sf_15` (190 commits) and
`sf_15..sf_16` (290 commits) in the local clone, diffs of the ~20 highest
impact patches, fishtest source/wiki, and GitHub PR discussions. Ideas
only — no code copied (GPLv3). Companion to 05 (training recipes).
Context: ARCHON's search is ~SF14-era (P7-P9 stack); eval just jumped
~+265 via the UHO/D40 program, which matters below ("gainer season").

## 1. The machine that produces the Elo: fishtest methodology

- **SPRT presets** (normalized Elo, GSPRT, α=β=0.05, LLR stop ±2.94):
  standard STC `[0.0, 2.0]` at 10+0.1; standard LTC `[0.5, 2.5]` at
  60+0.6; **non-regression (simplifications) `[-1.75, 0.25]`** at both.
  A patch must pass STC then LTC. VLTC 180+1.8 for scaling-only patches.
- **The book is deliberately UNBALANCED** (`UHO_Lichess_4852_v1.epd`,
  ~0.6-1.0 pawn offsets) to raise decisiveness and cut games-to-decision.
  Our screens use a balanced book — switching match books for TESTING
  throughput is a direct, free lever (our SF16 match drew 70%).
- **Volume/acceptance**: SF15 = ~13,000 tested changes, ~200 kept
  (~1.5%), ~1B games. A single accepted feature ≈ 200-250k games.
- **Simplification is first-class**: anything that can't re-prove its
  worth is deleted under non-regression bounds; line count must go down.
  ~30% of the SF15→16 qualifying patches are removals.
- **"Gainer season"**: after every evaluator jump, previously-failed
  search patches start passing (search was implicitly tuned to the old
  eval's error profile). ⇒ ARCHON must RE-TEST shelved patches (e.g. the
  reverted P9 item-9 IIR) and re-tune P9 constants now that the eval
  moved +265.

## 2. What kept winning, SF14→SF15 (190 commits, ~90 qualifying)

Dominant classes, in order:
1. **Extensions**: post-LMR "doDeeperSearch" (re-search 1 deeper when the
   reduced search beats alpha by a margin; later margin scales with the
   reduction gap); quiet-ttMove extension gated on killer/conthist
   evidence; progressively looser singular/double-extension conditions
   with an explicit runaway-depth governor (per-path double-extension
   cap replaced an EMA explosion detector).
2. **History-driven pruning/reduction**: futility margins subtract a
   fraction of history; LMR keys off delta/rootDelta (one consolidated
   formula replaced 3 ad-hoc terms); NMP/futility react to a
   "complexity" signal (|material − NNUE|).
3. **Eval scaling outside the net**: Optimism (small stm-relative bias
   applied BEFORE material scaling — calibrated contempt; ~+15 Elo era
   estimate), complexity-scaled optimism.
4. **Deliberate re-adds**: razoring and capture futility both returned
   after years away — verdicts on heuristics are contingent on the
   surrounding search, not permanent.
5. **Mechanics**: stat_bonus capped at 2000 (enabled removing lowPly
   history); fail-low history bonuses doubled at expected PV/CUT nodes;
   conthist init −71; capture sorting in movepicker; threat-evasion
   ordering bonus (the Wave-G-style threat idea; SF got it from
   Koivisto/Seer).
6. **TT**: PV-biased replacement (+2*pv in the replacement score); no
   EXACT stores from qsearch; don't store eval under an excluded-move key.
7. **Aspiration**: center the window on a 2:1 exponential average of the
   root score, widen with prev²/16384.
8. **TM**: +10% optimum scaled by increment; complexity-scaled total
   time; falling-eval factor.
9. **Removed**: lowPly history, ttHitAverage, probCutCount, multi-cut
   sub-search, reverse-move penalty, ttPv shrinking, rangeReduction
   (added then reverted), futility depth-cap removals (REVERTED — halved
   mate-finding; SPRT gains can hide tactical blind spots).

Standout portable patches (all small): stat_bonus cap `54a98993`,
fail-low bonus scaling `c31fc8d1`/`b7b6b4ba`/`a3d425cf`, capture sort
`4ac7d726`, delta/rootDelta reduction `0a318cdd`, post-LMR deeper
`c228f319`, aspiration-on-average `a0259d8a`, PV-biased TT `94514199`,
razoring re-add `4d3950c6`, cutnode no-ttMove IIR `30fdbf43`.

## 3. What kept winning, SF15→SF16 (290 commits, ~120 qualifying)

1. **The re-search ladder around LMR** (richest vein): doDeeperSearch /
   doEvenDeeperSearch (2 deeper on huge fail-high; reverted once for TB
   search explosions, reintroduced with a guard) / doShallowerSearch
   (re-search 1 SHALLOWER when the reduced value barely improved);
   skip the full-depth re-search when it wouldn't exceed reduced depth.
2. **cutoffCnt**: count beta cutoffs produced at the child ply; reduce
   MORE when the child keeps failing high (>3), reduce the ttMove LESS
   when the child is calm; +1 threshold when no ttMove. Explicit
   cutnode-frequency logic feeding reductions.
3. **Singular/negative extensions, VLTC-weighted**: singularBeta margin
   lowered repeatedly; negative extensions broadened (ttMove not
   singular AND ttValue ≤ alpha ⇒ extension −1; also non-PV nonsingular
   branches); singular depth gate scales with previous iteration's
   completed depth. Most passed only at LTC/VLTC — extensions must be
   tested at long TC.
4. **IIR maturation**: depth cut scales with ttHit/tte-depth
   (−2 − 2·(ttHit && tte.depth ≥ depth)); IIR moved BEFORE probcut.
5. **Optimism/complexity simplification wave**: optimism² dropped, trend
   removed entirely, non-pawn-material and pawn-count scale terms
   removed, complexityAverage collapsed to a root-complexity TM factor;
   the net-positive residue = take complexity straight from the NNUE
   output and feed search + TM.
6. **Structural bugfixes worth Elo**: queen promotions counted as
   captures (refutations searched twice otherwise); stack staticEval
   init to VALUE_NONE (improving flag correctness); key_after prefetch
   fix; helper-thread depth-245 explosion fix (thread-vote bias).
7. **TT**: more aggressive cutoffs on EXACT-bound entries; excluded-move
   handling without key XOR.
8. **TM**: spend more time early; float-division fix; instability factor
   simplified.
9. **Mass tuning batches at LTC/VLTC** were the main compounding source
   (biggest levers inside them: extensions, stat bonus, no-ttMove
   depth reduction).

NOTE: correction history POSTDATES SF16 (came from Caissa; STC pass over
128,672 games + LTC over 97,422). ARCHON's P8 correction stack means we
already hold a post-SF16 idea.

## 4. What does NOT work (their words, our decisions validated)

- Killers: subsumed by history/conthist (validates D17 "no killers").
- Contempt as a raw offset: removed; risk appetite belongs inside eval
  scaling (Optimism) so displayed evals stay calibrated.
- Separate countermove table: redundant with 1-ply continuation history.
- HCE as permanent co-evaluator: deleted once nets matured ("the
  networks figure it out themselves") — validates our D21 descoping.
- Razoring: perennially trimmed; keep it minimal.
- "Known not to work" is contingent: ProbCut itself was received wisdom
  as redundant until SF proved otherwise. Re-test after evaluator jumps.

## 5. ARCHON port-ranking (vs our actual P7-P9 stack)

Have already: PVS+aspiration, TT, staged movepicker, butterfly/capture/
continuation/pawn/low-ply histories + correction history, RFP, NMP, LMR
+statScore, razoring, LMP, parent+capture futility, SEE pruning, threat
map (Wave G), basic clock TM.

Port next, ranked by (evidence × size × independence):
1. **Singular extensions + negative/double extensions + multicut**
   (biggest missing subsystem; the era's #1 gainer class; test at long
   TC/fixed high nodes, not STC).
2. **cutNode flag + cutnode-aware reductions + cutoffCnt** (cheap,
   self-contained, repeatedly confirmed).
3. **Post-LMR re-search ladder** (doDeeper/doShallower; small diffs).
4. **Improving flag feeding RFP/LMP/NMP/futility margins** (near-free
   multiplier; check ours is correct — SF needed the VALUE_NONE init fix).
5. **IIR, SF's narrow form** (cutnode, no ttMove, depth≥9, −1; our
   reverted item-9 IIR was follow-PV-shaped AND predates the eval jump —
   gainer-season retest).
6. **TM: bestMoveInstability + fallingEval + nodesEffort** (uncertainty-
   metered time; A/B-able at fixed TC without a cluster).
7. **Aspiration on averaged root score + prev² widening**.
8. **PV-biased TT replacement + no qsearch EXACT stores**.
9. **stat_bonus cap + fail-low bonus scaling + conthist init −71**.
10. **ProbCut** (small, proven, we lack it).
11. **Optimism** (later; interacts with our eval-scale story).
Plus two meta-adoptions: unbalanced opening book for all internal
testing (throughput), and simplification passes under non-regression
bounds as routine hygiene.

## 6. Testing-economics lessons for a solo box

- SF pays ~200k games per accepted feature at ~1.5% acceptance; we
  cannot. Compensate with: unbalanced book (higher decisiveness),
  fixed-node pre-screens for routing (they cost seconds at our sizes),
  larger per-patch effect sizes (port proven bundles, not micro-tweaks),
  and SPSA for constants instead of one-at-a-time SPRT.
- Their non-regression channel is the license to delete: budget regular
  simplification tests or the engine accretes dead heuristics.
- Two reverts in the record were MATE-FINDING regressions caught by a
  tactical suite, not SPRT — keep WAC/mate suites in our gate.

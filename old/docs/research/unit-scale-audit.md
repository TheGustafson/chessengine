# Unit-scale audit: every cp-denominated constant in ARCHON

**Date:** 2026-07-26. **Auditor:** read-only sweep of `src/search/params.rs`,
`src/search/mod.rs`, `src/timeman.rs`, `src/movepick.rs`, `src/eval/`,
`src/types.rs`, cross-checked line-by-line against the actual source commit
of our constants in `stockfish_reference/Stockfish`.

---

## Executive summary (the 10 lines)

1. **CONFIRMED — and slightly worse than hypothesized: the systemic factor is ~1.9×, not 1.84×.** Every eval-denominated search constant was copied **verbatim from SF-dev commit `eca43a97` (2026-07-06, post-SF18)**, not SF16 (`params.rs:4` says so; spot-checks match eca43a97 exactly and *mismatch* sf_16 — e.g. SF16 has probcut 168/61/413, singular (82+65)/64, delta 10+prev²/15799).
2. SF-dev normalizes by the WDL model: `to_cp = 100·v/a`, a(anchor) = **385.47** (`eca43a97:src/uci.cpp:515`). With the measured SF18 slope 1.070, one SF-dev internal unit ≈ **0.278 oracle-cp**; one ARCHON unit ≈ **0.53** (band 0.518–0.548). Ratio **0.53/0.278 = 1.91** (SF16 frame: 0.53/0.288 = 1.84; band 1.8–2.0).
3. Ledger archaeology: **no SPSA result was ever shipped into params.rs** (D45 season = 8/8 ship-nothing, live log 2026-07-17 entry) — so the SF frame applies to essentially *everything*; only 3 TM stability constants (76b5ede) plus two Wave-G additions are ARCHON-denominated.
4. Net effect: **every margin gate is ~1.9× wider in real terms than its donor intended → too little pruning everywhere** → wider tree, fewer plies per node — the right sign and a plausible magnitude for the measured ~3-ply deficit vs SF16 at equal nodes.
5. Worst outlier #1 — **RFP family** (`RFP_MULT_*`, effective up to 42 oracle-cp/ply vs 22 intended): the highest-frequency prune in the tree fires far less often; whole subtrees survive that SF folds instantly.
6. Worst outlier #2 — **QS_FUTILITY_MARGIN 335** (178 vs 93 oracle-cp): qsearch futility + the coupled `see_ge(alpha − futility_base)` gate both under-prune → bloated qsearch at every horizon node.
7. Worst outlier #3 — **parent futility PFUT_*** (e.g. NO_BESTMOVE 73 vs 38 oracle-cp): hopeless late quiets at lmr_depth<12 get searched instead of written off.
8. Worst outlier #4 — **singular-extension family**: the exclusion bar sits ~2× too far below tt_value (depth 12 ttPv-nonPV: 14.0 vs 7.3 oracle-cp), and double/triple margins are ~2× too wide → **systematically fewer singular/double/triple extensions exactly at forcing "decisive moments"**.
9. Worst outlier #5 — **aspiration windows**: base delta 2.7 vs 1.4 oracle-cp (looser bounds every root search); the quadratic widener `mss/10588` is simultaneously ~1.9× too *weak* (a cp-dimension divisor scales the other way) — wide when calm, too slow to widen when volatile.
10. Recommended fix: **one global patch** — either (a′) rescale the eval output once (×1.91) onto donor scale, or (a) multiply the ~40-constant eval-margin family by 0.52 — gated as ONE SPRT; SEE-dimension constants are **OK** (piece values are SF-verbatim, so that dimension is internally consistent); the four eval×material mixed sites and the TM fallingEval coefficients carry two candidate scalings (slope frame 1.91 vs material frame 1.45) and go to targeted tests/SPSA.

---

## 1. Scale ground truth and the corrected frame

Verified tonight (all re-derived during this audit):

| Quantity | Value | Evidence |
|---|---|---|
| ARCHON UCI cp = raw internal Value | yes, unmodified | `src/uci.rs:1496` (`UciScore::Cp(v)` prints `v`); no normalization layer anywhere |
| ARCHON internal → oracle-cp (balanced positions) | **0.53** (0.518–0.548) | eval_bench regression, d47p, /tmp/evalbench_full.out |
| ARCHON internal → oracle-cp (material probe) | **0.402** | rook-up probe 1326 internal ↔ 498 SF18-UCI × 1.070 = 533 oracle |
| SF16 internal → oracle-cp | 0.288 | `NormalizeToPawnValue = 328` (`sf_16:src/uci.h:38`) × 0.946 slope |
| **SF-dev/SF18 internal → oracle-cp** | **0.278** | WDL model `to_cp = 100·v/a`, a(1.0) = Σas = 385.47 (`eca43a97:src/uci.cpp:510-515`, identical at sf_18) × 1.070 slope |
| Ratio ours/intended | **1.91** (slope frame), 1.84 (SF16 frame), **1.45** (material frame) | above |

**Framing correction to the task brief:** the brief's ×0.288 (SF16) column is
the wrong donor frame — `params.rs` line 4 states, and value-by-value
comparison confirms, that the constants are from **SF-dev `eca43a97`**, whose
internal unit is worth 0.278 oracle-cp, not 0.288. This *strengthens* the
hypothesis: the systemic factor is **1.91**, not 1.84. Both are inside the
1.8–2.0 band given the 0.518–0.548 slope spread; the tables below use 0.53
and 0.278.

**The two-dimension subtlety (needed to read the table):** SF carries TWO
"cp" dimensions and so do we:

- **eval dimension** — static eval / search scores. SF-dev: ≈ 385 internal
  per real pawn. ARCHON: ≈ 189 per real pawn (1/0.53). *This is the
  mis-scaled dimension: our numbers are ~1.9× smaller for the same real
  value, so a verbatim margin is ~1.9× wider in real terms.*
- **material/SEE dimension** — `types::PIECE_VALUES` = 208/781/825/1276/2538
  (`src/types.rs:636-644`) = SF's values verbatim (`eca43a97:src/types.h:196-200`).
  SEE outputs and SEE thresholds both live here → **internally consistent,
  unaffected by the eval rescale. Verdict OK.**
- Consequence: SF was tuned with eval:material exchange rate 385:208 ≈ 1.85;
  ours is 189:208 ≈ 0.91 — **~2× shifted**. Every formula mixing the two
  dimensions (4 sites, §4) has its balance moved even though each dimension
  is separately "consistent".

Direction check (per the brief, verified): margin M copied verbatim → ours
M×0.53 oracle-cp vs intended M×0.278 → **1.91× too wide → harder to satisfy
prune conditions → too little pruning → wider tree, less depth per node.**
The arithmetic in the brief is right; only the donor constant (328 vs 385.47/1.070)
needed correcting.

---

## 2. Full table — eval-dimension constants (the mis-scaled family)

Origin note: "SF-dev verbatim" = value identical to `eca43a97:src/search.cpp`
(each was eyeball-verified in this audit; block line refs: razoring/RFP/NMP
969-1024, probcut 1036-1087, aspiration 373-379/436, move-loop pruning
1160-1216, singular 1226-1280, qsearch 1693-1765, fail-low 1530-1531,
eval-diff 960-965, hindsight 850). No landed ARCHON commit has ever moved any
of them (git -L spot-checks; D45 SPSA shipped nothing).

Columns: effective = value × 0.53 oracle-cp; intended = value × 0.278
oracle-cp (donor frame); ratio = 1.91 throughout (it is one systemic factor,
not per-constant drift).

| name | file:line | value | origin | SF-dev internal | ours eff. (oracle-cp) | intended (oracle-cp) | ratio | verdict |
|---|---|---|---|---|---|---|---|---|
| `ASPIRATION_BASE_DELTA` | src/search/params.rs:62 | 5 | SF-dev verbatim | 5 | +2.7 | +1.4 | 1.91 | TOO WIDE |
| `ASPIRATION_MSS_DIVISOR` | src/search/params.rs:64 | 10588 | SF-dev verbatim | 10588 | real widener ≈ s²/5612 | s²/2939 | **0.52** | TOO WEAK (divisor of cp² has net cp dimension; scales opposite) |
| `RAZOR_BASE` | src/search/params.rs:173 | 465 | SF-dev verbatim | 465 | +246 | +129 | 1.91 | TOO WIDE |
| `RAZOR_DEPTH_MULT` (per d²) | src/search/params.rs:175 | 300 | SF-dev verbatim | 300 | +159/d² | +83/d² | 1.91 | TOO WIDE |
| `RFP_MULT_BASE` (per ply) | src/search/params.rs:193 | 40 | SF-dev verbatim | 40 | +21.2 | +11.1 | 1.91 | TOO WIDE |
| `RFP_MULT_SLOPE` | src/search/params.rs:195 | 4 | SF-dev verbatim | 4 | +2.1 | +1.1 | 1.91 | TOO WIDE |
| `RFP_MULT_CAP` (per ply) | src/search/params.rs:197 | 80 | SF-dev verbatim | 80 | +42.4 | +22.2 | 1.91 | TOO WIDE |
| `RFP_MULT_NO_TTHIT` | src/search/params.rs:200 | 20 | SF-dev verbatim | 20 | +10.6 | +5.6 | 1.91 | TOO WIDE |
| `RFP_IMPROVING`/`RFP_OPP_WORSENING`/`RFP_TREND_DEN` | params.rs:203/206/208 | 2934/343/1024 | SF-dev verbatim | same | fractional discounts of futility_mult | same | 1.00 | NOT-CP (dimensionless; auto-scale with the mult) |
| `RFP_THREAT_MARGIN` | src/search/params.rs:216 | 55 | ARCHON Wave-G (D30), untuned placeholder | n/a | +29.2 | n/a | n/a | ARCHON-only — SPSA (c3 screen moved it 55→46, unshipped) |
| `RFP_RET_BETA_W`/`RFP_RET_EVAL_W`/`RFP_RET_DEN` | params.rs:221/223/225 | 716/308/1024 | SF-dev verbatim | same | blend weights | same | 1.00 | NOT-CP |
| `NMP_EVAL_DEPTH` (per ply) | src/search/params.rs:245 | 14 | SF-dev verbatim | 14 | −7.4/ply | −3.9/ply | 1.91 | TOO WIDE (bar decays 2× too fast with depth, too) |
| `NMP_EVAL_IMPROVING` | src/search/params.rs:247 | 45 | SF-dev verbatim | 45 | −23.9 | −12.5 | 1.91 | TOO WIDE |
| `NMP_EVAL_BASE` | src/search/params.rs:250 | 374 | SF-dev verbatim | 374 | +198 | +104 | 1.91 | TOO WIDE (shallow NMP gate ~2× too strict) |
| `LMR_DEEPER_MARGIN` | src/search/params.rs:328 | 52 | SF-dev verbatim | 52 | +27.6 | +14.4 | 1.91 | TOO WIDE (deepen-on-strong-scout fires less) |
| `LMR_SHALLOWER_MARGIN` | src/search/params.rs:331 | 9 | SF-dev verbatim | 9 | +4.8 | +2.5 | 1.91 | TOO WIDE (shallower-re-search fires MORE — both LMR margins cost depth) |
| `QS_FUTILITY_MARGIN` | src/search/params.rs:150 | 335 | SF-dev verbatim | 335 | +178 | +93 | 1.91 | TOO WIDE |
| `CFUT_BASE` | src/search/params.rs:463 | 231 | SF-dev verbatim | 231 | +122 | +64 | 1.91 | TOO WIDE (plus mixed-site inflation, §4) |
| `CFUT_LMRDEPTH_MULT` | src/search/params.rs:466 | 232 | SF-dev verbatim | 232 | +123/ply | +64/ply | 1.91 | TOO WIDE |
| `PFUT_BASE` | src/search/params.rs:528 | 40 | SF-dev verbatim | 40 | +21.2 | +11.1 | 1.91 | TOO WIDE |
| `PFUT_NO_BESTMOVE` | src/search/params.rs:531 | 138 | SF-dev verbatim | 138 | +73.1 | +38.3 | 1.91 | TOO WIDE |
| `PFUT_LMRDEPTH_MULT` | src/search/params.rs:534 | 117 | SF-dev verbatim | 117 | +62.0/ply | +32.5/ply | 1.91 | TOO WIDE |
| `PFUT_EVAL_ABOVE_ALPHA` | src/search/params.rs:536 | 90 | SF-dev verbatim | 90 | +47.7 | +25.0 | 1.91 | TOO WIDE |
| `SE_BETA_MULT` (·d/59) | src/search/params.rs:605 | 60 | SF-dev verbatim | 60 | 0.54/d | 0.28/d | 1.91 | TOO WIDE (§6) |
| `SE_BETA_TTPV_MULT` (·d/59) | src/search/params.rs:607 | 70 | SF-dev verbatim | 70 | +0.63/d | +0.33/d | 1.91 | TOO WIDE (§6) |
| `SE_DOUBLE_BASE` | src/search/params.rs:614 | −3 | SF-dev verbatim | −3 | −1.6 | −0.8 | 1.91 | TOO WIDE (magnitude trivial) |
| `SE_DOUBLE_PV` | src/search/params.rs:616 | 201 | SF-dev verbatim | 201 | +107 | +56 | 1.91 | TOO WIDE |
| `SE_DOUBLE_NON_TTCAP` | src/search/params.rs:618 | 157 | SF-dev verbatim | 157 | +83 | +44 | 1.91 | TOO WIDE |
| `SE_DOUBLE_DEEP` | src/search/params.rs:624 | 41 | SF-dev verbatim | 41 | +21.7 | +11.4 | 1.91 | TOO WIDE |
| `SE_TRIPLE_BASE` | src/search/params.rs:626 | 72 | SF-dev verbatim | 72 | +38.2 | +20.0 | 1.91 | TOO WIDE |
| `SE_TRIPLE_PV` | src/search/params.rs:628 | 306 | SF-dev verbatim | 306 | +162 | +85 | 1.91 | TOO WIDE |
| `SE_TRIPLE_NON_TTCAP` | src/search/params.rs:630 | 188 | SF-dev verbatim | 188 | +100 | +52 | 1.91 | TOO WIDE |
| `SE_TRIPLE_TTPV` | src/search/params.rs:632 | 84 | SF-dev verbatim | 84 | +44.5 | +23.3 | 1.91 | TOO WIDE |
| `SE_TRIPLE_DEEP` | src/search/params.rs:634 | 45 | SF-dev verbatim | 45 | +23.9 | +12.5 | 1.91 | TOO WIDE |
| `PROBCUT_BETA_BASE` | src/search/params.rs:686 | 214 | SF-dev verbatim | 214 | +113 | +59 | 1.91 | TOO WIDE (probcut bar ~2× too high → fires less) |
| `PROBCUT_BETA_IMPROVING` | src/search/params.rs:688 | 59 | SF-dev verbatim | 59 | −31.3 | −16.4 | 1.91 | TOO WIDE |
| `PROBCUT_CHECK_BETA` | src/search/params.rs:695 | 428 | SF-dev verbatim | 428 | +227 | +119 | 1.91 | TOO WIDE |
| `HINDSIGHT_EVAL_SUM` | src/search/params.rs:727 | 173 | SF-dev verbatim | 173 | +91.7 | +48.0 | 1.91 | TOO WIDE (hindsight depth−1 arm fires less → wider tree) |
| `FL_BELOW_EVAL_MARGIN` | src/search/params.rs:770 | 103 | SF-dev verbatim | 103 | +54.6 | +28.6 | 1.91 | TOO WIDE |
| `FL_BELOW_PARENT_EVAL_MARGIN` | src/search/params.rs:774 | 78 | SF-dev verbatim | 78 | +41.3 | +21.7 | 1.91 | TOO WIDE |
| `EVALDIFF_CLAMP_LO` | src/search/params.rs:806 | −183 | SF-dev verbatim | −183 | −97 | −51 | 1.91 | TOO WIDE (history nudge under-responds ~2× per real swing) |
| `EVALDIFF_CLAMP_HI` | src/search/params.rs:808 | 180 | SF-dev verbatim | 180 | +95 | +50 | 1.91 | TOO WIDE |
| `EVALDIFF_OFFSET` | src/search/params.rs:810 | 62 | SF-dev verbatim | 62 | +32.9 | +17.2 | 1.91 | TOO WIDE |
| `VOTE_TEMPO` | src/search/params.rs:40 | 14 | SF-dev verbatim | 14 | +7.4 | +3.9 | 1.91 | TOO WIDE (Threads>1 only; thread voting over-floors weak threads) |

Use sites (all in `src/search/mod.rs`): aspiration 2001-2007/2069; razor 2553;
RFP 194-258 + 2592-2610; NMP 2628-2630; probcut 2699-2758; LMR margins
3340-3341; qsearch futility 4139/4201/4213; CFUT 372-397/3032; PFUT 340-365;
singular 412-478/3126-3160; hindsight 2338; eval-diff 2520; fail-low
3675-3686; vote `mod.rs:670`.

## 3. SEE/material-dimension constants — internally consistent, verdict OK

`PIECE_VALUES` (src/types.rs:636-644) = SF verbatim → SEE thresholds compare
like-for-like. These are NOT mis-scaled and must NOT be swept up in a blanket
margin rescale (unless the piece values themselves are redenominated, §7a′′).

| name | file:line | value | origin | SF-dev | verdict |
|---|---|---|---|---|---|
| `PIECE_VALUES` | src/types.rs:636-644 | 208/781/825/1276/2538 | SF verbatim | same | OK (defines the dimension) |
| `QS_SEE_FLOOR` | src/search/params.rs:153 | −74 | SF-dev verbatim (`see_ge(move, -74)`, eca43a97:1765) | −74 | OK (SEE units) |
| `QSEE_MULT` (·lmr_d²) | src/search/params.rs:551 | 25 | SF-dev verbatim | 25 | OK (SEE units) |
| `CAP_SEE_DEPTH_MULT` (·d) | src/search/params.rs:474 | 175 | SF-dev verbatim | 175 | OK (SEE units) |
| `QUIET_CHECK_SEE_MARGIN` | src/search/params.rs:110 | −75 | SF-dev verbatim (movepick.cpp:240) | −75 | OK (SEE units) |
| `GOOD_CAPTURE_SEE_DIVISOR` | src/search/params.rs:97 (use movepick.rs:400) | 18 | SF-dev verbatim | 18 | OK (ordering-score→SEE coupling, same dimensions both sides as SF) |
| `CAPTURE_MVV_FACTOR` | src/search/params.rs:89 | 7 | SF-dev verbatim (7·PieceValue) | 7 | NOT-CP (ordering only, same dims as SF) |
| `QUIET_THREAT_FACTOR` | src/search/params.rs:115 (use movepick.rs:624) | 20 | SF-dev verbatim | 20 | NOT-CP (ordering) |
| `QUIET_OFFENSE_FACTOR` | src/search/params.rs:122 | 10 | ARCHON Wave-G addition | n/a | NOT-CP (ordering) — SPSA |
| `STATSCORE_CAPT_NUM/DEN` | src/search/params.rs:367/369 | 809/128 | SF-dev verbatim | same | NOT-CP (material→history coupling, dims match SF) |

## 4. Mixed-dimension sites — the exchange-rate shift (~2×)

SF tuned these with eval:material ≈ 385:208 = 1.85; ours is 189:208 = 0.91.
All four lean the same way: **less pruning**.

| site | file:line | mixing | effect at our scale |
|---|---|---|---|
| capture futility | src/search/mod.rs:372-397 (SF 1166) | `static_eval + CFUT_* + piece_value(victim)` vs alpha | victim term ~2× overweighted vs eval gaps → optimistic bound inflated → capture futility fires less |
| qsearch delta pruning | src/search/mod.rs:4201 (SF 1741) | `futility_base + piece_value(victim)` vs alpha | same → fewer qsearch capture prunes |
| qsearch SEE-vs-window | src/search/mod.rs:4213 (SF 1753) | `see_ge(m, alpha − futility_base)` — eval gap used as SEE bar | needed SEE gain understated ~1.9× → bar too low → fewer prunes |
| probcut candidate filter | src/search/mod.rs:2707 (SF 1045) | `see_ge(m, probcut_beta − static_eval)` | bar understated → more (cheap qsearch-verified) probcut candidates; mild extra work |

Verdict: cannot be fixed by rescaling the §2 constants alone; fixed for free
by the eval-boundary rescale (§7 option a′), otherwise SPSA per site.

## 5. Time management

`src/timeman.rs` contains **no cp-denominated constants** (all ms/ply/log-clock;
verified line-by-line — the tunables at timeman.rs:46-90 are clock units).
The cp-coupled TM constants are the soft-stop factors in params.rs, used at
`src/search/mod.rs:4458-4505`:

| name | file:line | value | origin | dimension | effective behavior | candidate rescale (slope / material frame) | verdict |
|---|---|---|---|---|---|---|---|
| `FALLING_EVAL_PREV_AVG_COEFF` | src/search/params.rs:834 | 2.21 | SF-dev verbatim (eca43a97:570) | per internal-cp of drop | responds 1.91× too weakly per real cp drop | **4.22 / 3.20** | TOO WEAK (inverse dimension of a margin) |
| `FALLING_EVAL_ITER_COEFF` | src/search/params.rs:836 | 1.0 | SF-dev verbatim | per internal-cp | same | **1.91 / 1.45** | TOO WEAK |
| `FALLING_EVAL_BASE`/`DIV`/`MIN`/`MAX` | params.rs:832/838/840/842 | 11.87/100/0.572/1.708 | SF-dev verbatim | dimensionless | — | leave | NOT-CP |
| `TIME_REDUCTION_STABILITY_LO/HI`, `INSTABILITY_CHANGES_DECAY` | params.rs:856/858/885 | 2.0/15.0/2.5 | **ARCHON-redenominated 2026-07-26** (76b5ede) from measured iteration counts | iterations | correctly OUR-denominated | **do NOT rescale** | OK (empirically tuned) |
| `INSTABILITY_BASE/CHANGES_COEFF` | params.rs:872/874 | 1.096/2.29 | SF-dev verbatim | flip counts | decay already redenominated | leave | NOT-CP |
| `EFFORT_*`, `ONE_REPLY_TOTAL_TIME_CAP`, `INCREASE_DEPTH_FRACTION` | params.rs:824/889-903 | — | SF-dev verbatim | node-ratio / ms | — | leave | NOT-CP |

Context: the live log (2026-07-26 entry) already flags the fallingEval rescale
as deferred to SPSA because the two frames conflict ("1.2× material probe vs
1.8× regression slopes" there; this audit's arithmetic gives 1.45× and 1.91×
against the SF-dev donor frame) and fallingEval rides its 0.572 floor ~80% of
moves. Note both coefficient frames give the SAME direction: we under-extend
time on genuinely falling evals — at the same real 25-oracle-cp drop that lifts
SF off the floor, ARCHON needs ~48 oracle-cp.

## 6. Singular extensions — effective strength vs intent (the user's question)

Formula (src/search/mod.rs:425-428, use 3126-3160; SF eca43a97:1232 verbatim):
`singular_beta = tt_value − (60 + 70·(tt_pv && !PV))·depth/59`, exclusion
search at `[singular_beta−1, singular_beta]`, depth `new_depth/2`.

- **Margin in real terms:** non-ttPv 0.54 oracle-cp/ply vs SF's intended 0.28;
  ttPv-non-PV 1.17 vs 0.61. At depth 12: **6.5 / 14.0 oracle-cp** below
  tt_value where SF runs **3.4 / 7.3**. At depth 20: 10.8 / 23.4 vs 5.6 / 12.2.
- **Behavioral consequence:** a LOWER bar is EASIER for the excluded-move
  search to stay under only if the alternatives are truly bad; with the bar 2×
  further from tt_value, alternatives clear it more often → `value <
  singular_beta` fires less → **fewer singular extensions**, and the freed
  outcomes fall through to the negative-extension arms (ext −3/−2). The
  double/triple margins (SE_DOUBLE_*/SE_TRIPLE_*, all 1.91× too wide in real
  terms) suppress multi-extensions a second time on top.
- This is precisely the machinery SF uses to chase forcing lines deeper at
  "decisive moments", and SF's own scaler comment (eca43a97:1226: "higher
  singularBeta … and lower extension margins scale well") says the profitable
  direction at LTC is *even tighter* than its defaults — we sit ~2× on the
  wrong side of a knob SF wants pushed the other way.
- Multicut is NOT directly affected (its trigger is `value >= beta`), but a
  lower singular_beta window makes the exclusion search *likelier* to return
  ≥ beta values → **more multicut prunes on thinner evidence** — the one
  place in this audit where our mis-scale prunes MORE, and unsoundly so.
- **Firing-rate parity** requires shrinking the margin in internal units by
  the same 0.52: `SE_BETA_MULT 60→31`, `SE_BETA_TTPV_MULT 70→36` (equivalently
  `SE_BETA_DIV 59→113`), and scaling the eight SE_DOUBLE/TRIPLE cp constants
  by 0.52 (double: −3/201/157/41 → −2/105/82/21; triple: 72/306/188/84/45 →
  38/160/98/44/23). Material-frame alternative: ×0.69 (60→41, 70→48). The
  `corr_adj` coupling (`SE_CORR_DIV 194822`, params.rs:612) is self-consistent
  in our units (correction entries learn OUR eval errors) — leave it.

## 7. Recommended actions

### (a) Safe global fix — ONE patch, ONE gate

Two mutually exclusive implementations; test as single patches at 10+0.1:

- **(a′) preferred: rescale the eval once at the source.** Multiply the NNUE
  output by ≈ 1.91 (0.53/0.278) at the eval boundary so one ARCHON internal
  unit ≈ one SF-dev internal unit. Fixes ALL §2 constants, the §5 fallingEval
  coefficients, AND the §4 exchange-rate sites in one move (pawn ≈ 361
  eval-units vs SF's 385, within 7%); §3 SEE constants stay untouched and
  consistent. Risks to check in the patch: TB-band clamp headroom
  (`src/eval/mod.rs:432` — fine, band is ±~31000), correction-history
  operating point re-learns (entries are error-driven), UCI cp output doubles
  (cosmetic; no normalization layer exists), and the known nonlinearity: big
  material advantages already sit at 0.40 oracle/unit, so they will overshoot
  ~30% post-rescale — acceptable for margins, which operate near alpha/beta in
  balanced regions where 0.53 is the measured slope.
- **(a) alternative: rescale the margin family.** Multiply every eval-dimension
  constant in §2's TOO-WIDE rows by **0.52** (incl. ASPIRATION_MSS_DIVISOR
  10588→5546, same direction — its net dimension is cp) and the two
  fallingEval coefficients by 1.91. Leaves §4 mixed sites unfixed. All of
  these are D45 runtime tunables, so the patch can ship as a setoption overlay
  first (no rebuild) and be committed only after the gate passes.
- Given the two-frame uncertainty, gate the chosen patch at both factors:
  0.52 (slope frame) and 0.69 (material frame). If forced to pick one, pick
  the slope frame — margins act on near-balanced interior nodes, which is
  exactly the population the 0.53 slope was fit on.

### (b) Individually to SPSA (do NOT blanket-rescale)

- `RFP_THREAT_MARGIN` 55 (params.rs:216) — ARCHON-only, untuned placeholder;
  c3 theta (55→46, unshipped, `tools/spsa/campaigns/c3-rfp-razor.json` /
  `spsa/c3-rfp-razor/theta_final.json`) is a starting point.
- `QUIET_OFFENSE_FACTOR` 10 (params.rs:122) — ARCHON-only ordering term.
- The §4 mixed-site constants if option (a) was chosen instead of (a′):
  `CFUT_BASE`/`CFUT_LMRDEPTH_MULT` interplay with the victim term, and the
  probcut/qsearch SEE-vs-eval bars.
- Correction-history couplings `RFP_CORR_DIV` 182069 / `RED_CORR_DIV` 26131 /
  `SE_CORR_DIV` 194822 / `CORR_VALUE_DIVISOR` 131072 (params.rs:709/711/612/1040)
  — self-consistent (numerator learns our-cp errors) but their operating point
  shifts with any rescale; SPSA after (a)/(a′) lands.
- fallingEval coefficients if (a′) was NOT taken (two-frame conflict already
  ruled to SPSA in the live log, 2026-07-26).
- History-coupling weights `CFUT_HIST_MULT` 131 / `CAP_SEE_HIST_MULT` 34
  (params.rs:470/476): history units per cp — untouched by eval rescale in
  denominator only; low priority.

### (c) Fine as-is (verified NOT-CP or dimension-consistent)

- The entire SEE/material family (§3) — provided PIECE_VALUES stay SF-scale.
- All reduction-unit constants (RED_* family, 1024ths of a ply), LMP_BASE,
  depth gates (RFP_MAX_DEPTH, NMP_VERIFY_DEPTH, SE_MIN_DEPTH, PFUT/CFUT
  lmr-depth ceilings, PROBCUT_MIN_DEPTH/DEPTH_SUB), counters
  (QS_LATE_MOVE_LIMIT, TTCUT_*, SHUFFLE_*, SIBLING_*, HINDSIGHT_RED_*).
- All history-unit constants: QHIST_PRUNE_MULT 4313, GOOD_QUIET_THRESHOLD
  −14000, QUIET_SORT_DEPTH_FACTOR −3560, stat bonus/malus family, per-table
  update weights, continuation weights, correction update weights, TMH_*,
  FL_* except the two margins in §2, EVASION_CAPTURE_BONUS, QUIET_CHECK_BONUS.
- Dimensionless blends: QS_STANDPAT_BLEND, QS_FAILHIGH_BLEND, RFP return
  blend, RFP trend discounts, ASPIRATION_WIDEN_NUM/DEN (multiplicative),
  ASPIRATION_SEARCH_AGAIN_*, eval outer-shape blend
  (src/eval/params.rs:20-46 — weights vs the SF-scale material term, ratios
  only; RULE50_DAMPING_DEN dimensionless).
- `src/timeman.rs` entirely (clock/ply units), TIME_REDUCTION_* and
  INSTABILITY_CHANGES_DECAY (already OUR-denominated, 76b5ede), EFFORT_*.
- NNUE `SCALE = 400` (src/eval/nnue/mod.rs:140) is the trainer contract, not
  a search margin — it is where the 0.53 comes FROM; change it only via
  option (a′) consciously, never casually (D46 net_scale_check gate).

## 8. Empirically-tuned constants where the SF frame does NOT apply

Archaeology result: **the SF-comparison frame is valid for essentially the
whole search parameter file** — no SPSA or hand-tune was ever committed to
`src/search/params.rs` values (checked: full commit log of the file; D45
season exp-0075..0088 shipped nothing — "params.rs unchanged; bench stays
701605", live log 2026-07-17). The complete exception list:

1. `TIME_REDUCTION_STABILITY_LO/HI` = 2.0/15.0 and `INSTABILITY_CHANGES_DECAY`
   = 2.5 (76b5ede, 2026-07-26) — measured against OUR iteration counts.
   **Correctly denominated; leave.**
2. `RFP_THREAT_MARGIN` 55 and `QUIET_OFFENSE_FACTOR` 10 — ARCHON-only
   additions, never tuned at all. **SPSA, not rescale.**
3. PSQT tables (src/eval/psqt.rs, PeSTO transcription) — the PSQT-era scale
   origin, now only the non-NNUE fallback; out of the live path.

Standing counter-evidence to keep honest (per the no-verdicts-from-weak-tests
rule, it does not overturn the dimensional analysis but bounds enthusiasm):
the D45 SPSA screens (2026-07-17, 20k-node tune, 5k-node screens, all
TC-unverified) moved some constants the "wrong" way for the 1.91× story —
`NMP_EVAL_BASE` 374→507, `PFUT_NO_BESTMOVE` 138→179 — while agreeing on others
(`RAZOR_BASE` 465→422, `RFP_IMPROVING` 2934→1947). Two caveats make those
screens weak evidence against the rescale: they were run at FIXED NODES, which
structurally undervalues pruning (tree-size savings don't buy anything when
nodes are capped — exactly the regime that makes wider margins look free), and
exp-0078 showed the one-box SPSA resolution floor is ~8-13 Elo. The (a)/(a′)
gate must therefore be a real-clock SPRT, never a fixed-node screen.

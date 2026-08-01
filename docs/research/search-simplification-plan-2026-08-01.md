# Search simplification — census, removal plan, and staged sequence (2026-08-01)

**Goal (owner's words):** greatly reduce the complexity of search — fewer
tunables, fewer heuristics, less code — at equal strength, ideally faster,
deliberately diverging from Stockfish where divergence buys simplicity, and
leaving a clean substrate for a later NNUE-guided layer with specialised heads.

**This is a plan, not a patch.** No search behavior was changed. Two things
were built and committed alongside this document, both feature-gated OFF and
verified node-identical (`Bench: 587683` unchanged; wl50@d10 with the
canonical net 384,499 unchanged):

- The **per-gate fire-rate census**: 38 new counters in the existing
  `search-stats` `SearchStats` (zero-sized when off), printed as an `SSTATS3`
  line, aggregated by `tools/gate_census.py`.
- A **re-run of `tools/prune_budget.py`** against the CURRENT canonical net
  (the s600 rescale) — the old numbers were stale, and the difference is the
  single most important fact in this document (§1.2).

Method constraints honoured throughout (each one paid for by this project):
`instructions:u` is the only microarch counter that survives its own null;
node totals on a position set fail an inert ±1 jitter null at up to +18.3%;
census statistics have a ±0.11–0.16 pp null floor and are the screening
instrument; a 1500-game fixed-node screen has a ±10–11 Elo bar; **node
savings at neutral cp have predicted zero Elo three times running** — only cp
or game evidence promotes anything.

---

## 0. The thesis, in five sentences

1. The search executes ~120 KB of hot code against a 32 KB L1i, is
   frontend-bound (30.6% I-cache miss rate, 40.5% decoder-fed ops), and code
   *layout* is closed — so the only per-node speed left on the search side is
   **executing less code**, which is exactly what simplification does.
2. The tuning surface is measurably inert: **309 tunables, 246 never in any
   SPSA campaign, 303 verbatim SF transcriptions, and a 154-value refresh
   from a newer Stockfish measured −1.39 ± 10.43 over 1500 games** — the
   values are not where the Elo is, so most knobs are pure surface.
3. The **s600 net rescale silently completed the eval-frame program**: every
   pruning gate that was starved or net-negative in July is now live (§1.2),
   so the *mechanisms* mostly earn their keep — the *parameterisation* of
   them does not.
4. The removal list therefore has two tiers: a large tier of dead code, dead
   knobs, and never-turned handles that can go with **zero strength risk and
   no games** (§2 tier 0, §3), and a smaller tier of live-but-unearned
   heuristics that need a one-match A/B each or in baskets (§2 tiers 1–2).
5. A <5 Elo removal is not individually provable at this box's ±10 bars —
   so the plan proves **baskets** with non-inferiority SPRTs and bisects only
   on failure, rather than pricing each part (§4).

---

## 1. The census, grounded in measurement

### 1.1 Code inventory (lines are non-test; bytes are the PGO binary)

| unit | size | notes |
|---|---|---|
| `src/search/mod.rs` | 5,861 lines | `search::<NonPv>` alone is **45,565 B**, 34% of the engine's L1i line fills |
| `src/search/params.rs` | 1,690 lines | 309 tunables + the dead frame-transform block |
| `src/movepick.rs` | 1,505 lines non-test | 16 stages, 4 pipelines; `init_quiets` 349 lines is a hot symbol |
| `src/history.rs` | 1,065 lines non-test | 8 tables, **20.26 MiB/worker** (pawn history alone 12 MiB = 59%) |
| `src/search/stack.rs` / `rootmoves.rs` | 188 / 159 lines | two dead fields found (§3) |

Already simpler than SF in real ways: **no killers, no countermoves, no
Syzygy ranks** (D17). That precedent matters — removal-at-parity has been
done here before.

### 1.2 The prune budget, RE-MEASURED on the current net (2026-08-01)

`tools/prune_budget.py`, wl50 @ d10, canonical d48x-s600 net, baseline
384,499 nodes. Old column = 2026-07-27 (pre-s600 net, baseline 1,002,862).

| mechanism | tree growth when disabled (NOW) | (July, stale) |
|---|---|---|
| LMR log term | **+162.6%** | +248.2% |
| RFP | **+114.6%** | +62.6% |
| parent futility | **+45.3%** | +6.1% |
| IIR | **+24.9%** | −3.0% |
| NMP | **+20.9%** | −4.1% |
| quiet SEE | +16.7% | +20.4% |
| razoring | **+15.7%** | −12.7% |
| capture futility | +11.7% | +0.5% |
| ProbCut | +7.4% | −5.1% |
| **LMP** | **+2.1%** | +46.8% |
| **conthist pruning** | **−8.8%** | −5.4% |

**Read this as a reversal.** The July doc concluded razoring was net-harmful,
NMP starved, capture futility inert — and a "curated combo" built on that
SPRT'd at −29.6 Elo. The s600 promotion (a 4-byte output-scale change,
+14.78 ± 5.45 over 5880 games, −12.4% nodes) put the eval spread where the
hardcoded SF margins expect it, and now **every gate except conthist pruning
is a live contributor**. Consequences:

- The entire **eval-frame correction apparatus in params.rs is obsolete**:
  `FRAME_S`, `frame_scale`/`frame_expand`/`frame_scale_i64` (zero call
  sites — a documented false-negative generator), and the identity-pinned
  class-4 converters (`QS_PIECE_MULT`, `CFUT_PIECE_MULT`, `SEE_THRESH_MULT`).
  The correction the taxonomy was built to express landed inside the net.
- Any July-era per-gate verdict (razor off, NMP 168, PFUT 0.64×) is stale
  and must not be resurrected.
- **Conthist pruning is the one gate whose ablation shrinks the tree** — the
  only mechanism that is a removal candidate on contribution grounds.

### 1.3 Fire rates — the new per-gate census (wl50 @ d12, canonical net)

1,211,932 nodes; 945,249 main-search entries; 339,786 qsearch entries.
Fires per 1,000 main-search node entries:

| gate | /1k main | reading |
|---|---|---|
| LMR reductions applied | 667 | the tree IS LMR |
| parent-futility prunes | 655 | busiest prune in the engine |
| N8 eval-diff history nudge | 639 | 2 history writes at ~2/3 of all nodes |
| RFP returns | 462 | second-busiest exit |
| LMP skip triggers | 264 | |
| beta cutoffs (fh) | 228 | fhf 0.861 on this set |
| quiet-SEE prunes | 151 | |
| capture-futility / capture-SEE | 92 / 89 | live post-s600 |
| correction-history updates | 81 | |
| hindsight down / up | 72 / 54 | fires far more than its 2 tunables suggest |
| TT cutoffs | 69 | |
| ProbCut moves searched / cutoffs | 35 / 25 | 70% conversion — healthy |
| singular tries / singular verdicts | 28 / 12 | |
| **singular double / triple grants** | **10.9 / 7.6** | **93% / 65% of singular verdicts** — the margins barely discriminate (§2.2) |
| IIR / conthist-prune | 24 / 22 | |
| NMP tries / cuts | 16 / 10 | un-starved (was 0.2% of nodes in July) |
| tt-penalize / razor / check-probcut | 8.9 / 7.8 / 6.8 | small but live |
| aspiration re-searches | 1.16 (22/position) | |
| equal-value dither offers | **0.94** | ~1 per 1,000 nodes |
| **NMP verification searches** | **0.03** | **29 fires in 1.2M nodes** |

Qsearch (per 1k qsearch entries): stand-pat cut 360, TT cut 160, late-move
136, SEE-window 107, futility 72, SEE-floor 67, stalemate scan 5.3.

Cross-checks from the study-70/71 census (uho300 @ d11, unmerged branch, its
own ±0.11–0.16 pp nulls): NonPV fhf 0.845, splitting 0.9135/0.7645 on TT-move
availability; 3.64% of NonPV LMR scouts force a full-depth re-search costing
21.4% of the tree (nested upper bound); singular verification costs a further
14.0% of the tree; 57.4% of TT-less node entries are *moveless stores*, a
store-policy defect with pending knobs (`TT_FAILLOW_MOVE`), not a
simplification target — do not collide with it (§4 stage notes).

---

## 2. The ranked removal list

Ranking key = (tunables + lines + *executed* ops removed) / (risk to
strength). Cold code costs ≈ nothing per node (studies 67/68: layout closed;
resident-but-unexecuted code is free), so pure-complexity removals are ranked
on maintenance surface, and perf claims are made **only** where the removed
code executes at a meaningful rate. Every strength-relevant removal names its
test.

### Tier 0 — dead by construction: delete with NO games (bit-identical bench required)

| # | item | removes | evidence it is dead |
|---|---|---|---|
| 0.1 | `FRAME_S` + `frame_scale`/`frame_expand`/`frame_scale_i64` + their tests | ~130 lines, 1 tunable | **zero call sites outside params.rs** (params.rs's own WARNING block, verified by grep); superseded by s600 |
| 0.2 | `RAZOR_CORR_DIV`, `NMP_CORR_DIV` terms + knobs | 2 tunables, 2 hot-path divides by 1e9 | docs say "effectively OFF"; term floors to 0 at every reachable `\|correction_value\|` |
| 0.3 | `FOLLOW_PV_MODE` + `Stack::follow_pv` + `last_iter_pv` snapshot + the two consumer conditions | 1 tunable, a per-node stack write, a per-line `Vec` clone per iteration, 2 branch conditions | default 0 ⇒ flag constant-false since P9; the SF-parity toggle was built to be measured and never won |
| 0.4 | `QS_UPREP` toggle + its qsearch block | 1 tunable, ~12 lines | **measured worse ON**: +0.47/+0.20/+0.23 cp by tier at +0.6% nodes (params.rs doc) |
| 0.5 | `RED_TTPV_TTMISS_REBATE` toggle | 1 tunable | **measured worse ON**: +0.44/+0.30/+0.34 cp at +5.3% nodes |
| 0.6 | `RED_GIVES_CHECK` term + knob | 1 tunable, 1 hot multiply | default 0; every supporting argument withdrawn (params.rs doc; LEDGER — the opposite-direction arm clears the same null) |
| 0.7 | `TM_SCORE_C/KNEE/FLOOR/CEIL` + `score_tm_factor` | 4 tunables, ~60 lines | default 0 ⇒ exact 1.0; the TM back-loading hypothesis it served was **refuted at real TC** (−36.6 ± 22.5) |
| 0.8 | EXP-UNC-1 + WDL-TM hooks: `unc_margin_scale`, `unc_tm_factor`, `wdl_tm_factor`, the six `sota`-gated Worker fields + env reads | ~150 lines, 3 env vars | D48X carries neither a σ̂ nor a WDL head — **measurably inert** (byte-identical `ARCHON_TM_TRACE` with coefficients forced on). Git preserves them; a real head will want its own shapes anyway (§6) |
| 0.9 | D47 policy-head plumbing in search: `policy_head` snapshot, `picker_tt_move` policy-argmax arm, `ord_weight`/`ord_min_depth`, P-ORD site | ~60 lines, 2 env vars | no `.cpol` sidecar exists in production; `CPL_ORD_W` unset ⇒ dead. **Keep `MovePicker::set_quiet_prior` itself** — it is the seam a future head plugs into (§6) |
| 0.10 | `policy` feature machinery in `search`/`movepick` (`policy_net`, `set_policy_scores`, root ordering) | ~120 lines across files | D41 policy era ditched (pol_lambda 0); feature is off in every shipped build |
| 0.11 | `Stack::ply` field | 4 B/slot + init | written, read only by its own test |
| 0.12 | `RootMove::previous_score_exact` | 1 field | write-only |
| 0.13 | `Stage`'s `PartialOrd` derive; ProbCut picker's unused `cont_hist`/`depth`/`ply` noted as doc comments | derive + clarity | no ordering comparison exists |
| 0.14 | Identity class-4 converters `QS_PIECE_MULT`, `CFUT_PIECE_MULT`, `SEE_THRESH_MULT`, `LMP_DEPTH_MULT` + `scale_1024` call sites | 4 tunables, a hot-path call each | pinned at 1024 = exact identity; the frame program they served is closed (§1.2). `scale_1024` already early-returns at 1024, so deletion is bit-identical |
| 0.15 | Dead-in-every-build perf-arm cfg blocks: `insertwin34`, `bcsort55`, `bccont55`, `cont53`, `histpf44` (the whole inert prefetch API), `nogc54`, `ttspec42`/`ttspec2_42`, `dosesort40`, `census34` | ~300 lines across movepick/history/search | all measured, all declined, all OFF; the negative results live in the study docs, not the code. Keep `icold67`/`icold67b` (live PGO layout arms) and `search-control`/`datagen`/`search-stats` (live instruments) |

Tier-0 total: ~**15 tunables deleted outright**, ~800–900 lines of search-side
code, six env vars, and two per-node stack writes — at zero strength risk.
Acceptance: `Bench: 587683` bit-identical, NNUE gate eval-hash + d13 total
bit-identical, `cargo test` + clippy clean. Items 0.4/0.5 delete *behavior
that is currently disabled*, so they too are bit-identical deletions.

### Tier 0b — constify: tunables that become plain constants (no behavior change)

The reviewers' rule "any bare literal a tuner could move is a defect"
(design/01 §3.8) built this surface; the 154-value-refresh null and the A3
local-optimum result say the surface does not pay. Reverse the default
(needs a decisions.md amendment — proposed **D50**, §5): *a constant becomes
a tunable when a campaign wants it, not by default.*

- **Structural denominators / fixed-point scales** (~25): `RED_UNIT`,
  `RED_NON_IMPROVING_DEN`, `RED_STATSCORE_DEN`, `STAT_WEIGHT_DEN`,
  `CAPT_HIST_DEN`, `CORR_UPDATE_DEN`, `CORR_VALUE_DIVISOR`,
  `CORR_BONUS_DIVISOR`, `CORR_BONUS_SCALE_DEN`, `CONT_UPDATE_DIVISOR`,
  `FL_CONTHIST_DEN`, `FL_MAIN_DEN`, `FL_PAWN_DEN`, `LMRDEPTH_MAIN_DEN`,
  `STATSCORE_CAPT_DEN`, `SE_VERIFY_DEPTH_DEN`, `SE_DOUBLE_TMH_DEN`,
  `NMP_VERIFY_DEN`, `ASPIRATION_SEARCH_AGAIN_DEN`, `ASPIRATION_WIDEN_DEN`,
  `QS_STANDPAT_DEN`, `QS_FAILHIGH_DEN`, `RFP_RET_DEN`, `RFP_TREND_DEN`,
  `RED_TABLE_DEN`. A numerator/denominator pair is ONE degree of freedom;
  the denominator is representation.
- **Non-strength knobs** (~10): `CHECK_TIME_EPSILON_MS`,
  `INFO_THROTTLE_NODES`, `ONE_REPLY_TOTAL_TIME_CAP`,
  `ASPIRATION_JITTER_PERIOD`, `EFFORT_UNIT_SCALE`, `DITHER_NODE_MASK`,
  `DITHER_PLY_MARGIN`, `SHUFFLE_RULE50`, `SHUFFLE_PLIES_FROM_NULL`,
  `SHUFFLE_MIN_PLY`.
- **The 16 implicit-×1 handles** (2026-07-27 batch): `SE_EXT_BASE`,
  `SE_NODE_DEEPEN`, `SE_TTPV_DEPTH_BONUS`, `PROBCUT_IMPROVING_SUB`,
  `LMR_PV_DEPTH_BONUS`, `LMR_MIN_SCOUT_DEPTH`, `LMR_DEEPER_STEP`,
  `LMR_SHALLOWER_STEP`, `LMR_FULLDEPTH_DOCK_STEP`, `LMR_DOCK2_MIN_DEPTH`,
  `TTCUT_BOUND_SLACK`, `CUTOFFCNT_EXT_GATE`, `FH_SOFTEN_BETA_W`,
  `SIBLING_DEPTH_STEP`, `LMP_IMPROVING_DEN`, `ASPIRATION_FAILHIGH_DEPTH_MULT`,
  plus `RED_TABLE_OFFSET` (never moved from 0) and `IIR_REDUCTION`. None has
  ever been turned; each was added so a future campaign *could* ask a
  question no campaign has asked. Keep the doc comments (they are good
  design notes); demote the knobs.
- **TM soft-stop f64 family** (~15 of the ~20): the fallingEval /
  stability / instability / effort constants. Fixed-node testing — the
  box's default — is structurally blind to TM, so these know they will not
  be SPSA'd here. Keep live only the two ARCHON-derived ones with measured
  rationale (`INSTABILITY_CHANGES_DECAY`, `TIME_REDUCTION_STABILITY_LO/HI`).

Tier-0b total: **~65 tunables off the surface** with zero behavior change
(the `tunables!` macro emits identical code for a `pub const` at default; a
`tune` build simply stops exposing them). 309 → ~**230** after tiers 0+0b.

### Tier 1 — live but unearned: needs games, highest value-per-risk first

| # | item | complexity removed | evidence / plausible cost | test |
|---|---|---|---|---|
| 1.1 | **D30 offense term** in quiet scoring (ARCHON original) | 1 tunable, ~60 lines of `init_quiets` incl. the reverse-attack-fan build, an `attacks()` call per non-pawn quiet — **executed in a hot symbol** | study 70: `offensew=0` reads **+0.07 pp fhf, inside the null**; earlier game arm −1.7 ± 34 (underpowered, not a verdict) | census null → part of basket A (§4 stage 2) |
| 1.2 | **RFP threat coupling** `RFP_THREAT_MARGIN` (ARCHON original, D30) | 1 tunable, the two-margin lazy-`all_threats` dance (~50 lines incl. the equivalence comment), a `compute_threats` sweep in the band | removal measured **−3.5 ± 14.1, n=1185** (neutral); the lazy-band machinery exists only to make this term affordable | basket A |
| 1.3 | **Equal-value dither** + `child_pv_fresh` truncation dance | 2 tunables, ~30 lines, removes a PV-freshness edge case that has already produced one illegal-PV bug class | ARCHON original (SF has no step-12 dither); fires 0.94/1k, and only a subset of fires change the pick. Never validated by games | basket A |
| 1.4 | **Singular double/triple margin filigree** → one margin, extension ∈ {1, 2} | **10 tunables** (`SE_DOUBLE_*` ×6, `SE_TRIPLE_*` ×5 minus shared), 2 pure functions, the `tmh` coupling inside them | census: double passes on **93%** and triple on **65%** of singular verdicts — margins that almost always pass are not discriminating; LEDGER: the three extension knobs **do not compose** (+0.60/+0.30/+0.27 alone, −0.21 together); SF is Pareto-superior on seldepth for reasons already shown NOT to be these constants | one-match A/B alone (this one is a real tree-shaper: se_singular 11.7/1k) |
| 1.5 | **N8 eval-diff nudge** | 5 tunables, ~30 lines, **2 history writes at 639/1k nodes** — one of the busiest blocks in the engine | SF-parity Wave-C item; never measured here in isolation; its information channel (grade opponent quiets by eval swing) overlaps the fail-low parent bonus | basket B |
| 1.6 | **Conthist pruning** | 1 tunable + the compare (the 3-plane sum stays — it feeds history-buys-lmr-depth) | the ONLY gate whose ablation **shrinks** the tree (−8.8%); dormant at fresh histories by construction | basket B |
| 1.7 | **TT-cutoff hardening trio**: child verification (`tt_verify_child`), wrong-direction penalize, agree-depth gate | 3 tunables, ~70 lines incl. a cold make/probe/unmake path | penalize fires 8.9/1k, verification only on deep cutoffs; SF-parity item 20; value never isolated here | basket B |
| 1.8 | **NMP verification** + `nmp_min_ply` | 2 tunables (post-0b), ~25 lines, a Worker field | **29 fires in 1.2M nodes** at d12. Zugzwang insurance that almost never runs at these depths — but it runs more at depth 20+, and zugzwang losses are catastrophic-shaped. Cheap to keep; remove only in the basket and watch the endgame book | basket B, endgame-weighted book alongside |
| 1.9 | **tt_move_history scalar** (`TMH_HIT/MISS`, `MULTICUT_TMH_*`, the SE-margin tmh term) | 4–6 tunables, a scalar with 3 write sites | consumers shrink to ~zero if 1.4 lands; never isolated | rides with 1.4's A/B |
| 1.10 | **N14 check-probcut** | 1 tunable, 8 lines | fires 6.8/1k and is nearly free (reads already-probed TT data). **Keep** — listed to record the judgement | — |
| 1.11 | **Razoring** | 2 tunables, ~10 lines | **keep**: +15.7% contribution post-s600, and the July "razoring is harmful" verdict is explicitly stale; SCREENING's cp screen already found removal −2.8 cp worse | — |
| 1.12 | **LMP** | 3 tunables | **keep despite the collapsed +2.1% contribution**: `skip_quiet_moves` suppresses `init_quiets` scoring work in a hot symbol — it pays in executed ops even when the tree barely moves | — |

### Tier 2 — structural simplifications (bigger, later, each its own decision)

| # | item | removes | risk |
|---|---|---|---|
| 2.1 | **Continuation-history chain 6 → 4 plies** (writes AND the ordering read set; offset 5 is already write-only for ordering, offset 6 weight 423, offset 5 weight 129) | 2 array slots, per-update loop iterations at every quiet history event (hot, executed), 2 stack reads/node | Deliberate SF divergence. Test: census fhf (expect small), `ipn70` for the executed-ops win, then A/B. If fhf moves > 0.5 pp, stop |
| 2.2 | **Pawn history table removal** (12 MiB of 20.26 MiB/worker; read for one ordering term + one prune sum + written at 3 sites) | ~7 tunables, 12 MiB/worker, pawn_key plumbing in search | SF added it for a few Elo; on a memory-bound engine the working-set argument cuts FOR removal, but the ordering term is live (study 70 kept it). A/B alone, not in a basket |
| 2.3 | **movepick `sorted_covers`/`bad_quiet_start` fast path** (pure-perf duplication of the good/bad predicate; inert at depths 1–3) | ~40 lines, one partition rescan saved | Intended non-functional: must reproduce `Bench:` bit-for-bit; judge on `ipn70`. If it costs instructions, keep |
| 2.4 | **Three-way TT-move dedup** in movepick → one place | clarity only | non-functional refactor, bit-identical bench required |
| 2.5 | **Low-ply history** (640 KiB, 3 tunables) | — | **keep**: `QUIET_LOW_PLY_WEIGHT` is the only knob in the project whose primary evidence is a cp *win* |

---

## 3. Tunable-reduction plan (309 → ~150)

| step | mechanism | Δ | running total |
|---|---|---|---|
| today | — | — | 309 |
| Tier 0 deletions | knob dies with its dead code | −15 | 294 |
| Tier 0b constify | denominators, non-strength, implicit-×1, TM f64s | −65 | ~229 |
| Blend-pair folding | `(BEST_W, BETA_W, DEN)` triples → one weight each (`RFP_RET_*`, `QS_STANDPAT_*`, `QS_FAILHIGH_*`) | −6 | ~223 |
| Tier 1 baskets (games-gated) | offense, RFP-threat, dither, evaldiff, conthist-prune, TT-hardening, NMP-verify, TMH | −18 | ~205 |
| Singular collapse (games-gated) | double/triple family → one margin | −10 | ~195 |
| Tier 2 (games-gated, optional) | pawn history family, cont-chain slots | −9 | **~185** |

Below ~185 the remaining knobs are the ones that are load-bearing by
measurement: the LMR family (the tree is 162% LMR — `RED_TABLE_NUM`,
`RED_BASE`, the finalization terms), RFP margins, NMP bar, futility bases,
aspiration delta, history bonus/malus scales, correction weights. **Do not
constify these**: they are the search's actual degrees of freedom and the
place a future SPSA or a learned controller will act.

Which knobs are load-bearing vs never-turned, at a glance: 63 of 309 have
ever been in a campaign (`tools/spsa/param_provenance.py`; note it must run
from the main checkout — the SF clone is not in worktrees — and it currently
exits 1 on a stale `CHECK_EXT_MIN_DEPTH` entry in `s4-searchgap`, which
should be cleaned up in stage 0). Of the 63, SPSA moved almost none
meaningfully (`RED_MOVECOUNT` drifted ~0 over 3,280 games while neighbours
moved 100+). The honest statement: **perhaps 30–40 constants have ever
demonstrably mattered**, and they are the ones §1.2/§1.3 show firing.

---

## 4. The staged plan

Every stage is independently testable, lands (or aborts) on its own, and
leaves the tree shippable. Standing gates for every stage: `bash
tools/check.sh` (Bench pin), `CHECK_NNUE_GATE=1` for anything near eval
call sites, clippy, and a ledger row per match.

**Match methodology for all game-gated stages** (from the claim ledger, all
paid for): fixed-node fn25k one-match A/B — the removal candidate vs the
current tip **in one match**, never two screens differenced; same-source
both-PGO pair (build the baseline fresh — `engines/bin/archon-s25` is a
STALE fingerprint); `EvalFile` verified on the live argv (the sprt.sh bug,
fixed in `991fbf43`, silently measured PSQT for a week); segmented
`chain70.sh`-style runs with `pgn_ptnml.py` recovery (long matches die on
this box); gate launches on `pgrep -c fastchess`, and clear them with the
owner first. **Non-inferiority SPRT: elo0 = −3, elo1 = +1, alpha = beta =
0.05, model=logistic** — accepting H1 means "not materially worse", which is
the correct question for a removal; a basket that accepts H0 gets bisected
by halves, not per-item.

### Stage 0 — dead code, dead knobs, constify (no games; ~1 session)

Land tier 0 + 0b + blend folding as a series of small commits, each
bit-identical on the Bench pin and the NNUE gate. Also: fix the stale SPSA
campaign entry; update design/01 §3.8 and decisions.md (**D50**: tunables
are opt-in — decisions change first, docs follow, code follows).

- **Abort condition:** any commit that moves `Bench: 587683` or the eval
  hash was not dead — revert that commit, file the finding, continue.
- Deliverable: 309 → ~223 tunables, ~1,000 fewer lines, six env vars gone.

### Stage 1 — measure the freebies (no games; hours)

With the tier-0 tree, run `ipn70.sh` (instructions/node, the one trustworthy
counter) old-vs-new, and re-run `gate_census.py` + `prune_budget.py` to
re-baseline. Purpose: catch any accidental behavior change and record the
executed-ops saving from 0.2/0.6/0.14 (the only tier-0 items that delete
per-node work).

- **Abort condition:** none — this stage only measures.

### Stage 2 — basket A: the ARCHON-original null trio (games)

Remove together: offense term (1.1), RFP threat coupling (1.2), dither +
`child_pv_fresh` (1.3). All three are ARCHON originals with existing
null/neutral measurements; none is SF machinery, so this is also the first
deliberate-divergence statement. Census null first (expect ≤ 0.2 pp fhf
movement, ≥ the offense term's known +0.07), then the non-inferiority A/B.

- **Abort condition:** SPRT accepts H0 (worse than −3) → bisect: re-run with
  1.3 restored (the least-evidenced member) before blaming the D30 pair.
- Bonus: measure `ipn70` — 1.1 and 1.2 delete executed hot-loop work.

### Stage 3 — the singular collapse (games)

Replace the double/triple margin family + tmh coupling with one margin and
extension ∈ {1, 2} (1.4 + 1.9). This is the deepest single simplification:
−10 tunables, two pure functions, and the SE block becomes readable. The
census says the current margins pass 93%/65% of the time; the ledger says
the extension knobs don't compose and the seldepth gap is NOT these
constants. Run the endgame-weighted book alongside (extensions are
mate-line machinery).

- **Abort condition:** H0, or the mate-suite regression tripwire (>2σ drop
  on `mate_in_3` at fixed nodes — a tripwire, never a promotion metric).

### Stage 4 — basket B: SF cargo (games)

Remove together: eval-diff nudge (1.5), conthist pruning (1.6), TT-cutoff
hardening trio (1.7), NMP verification (1.8). Each is SF machinery that is
either near-dead here (1.8), tree-negative (1.6), or never isolated (1.5,
1.7); individually they are <5 Elo questions the box cannot afford to ask.
~11 tunables, ~150 lines. Endgame-weighted book alongside (1.8's zugzwang
exposure).

- **Abort condition:** H0 → bisect by halves ({1.5,1.6} vs {1.7,1.8}). If
  the {1.7,1.8} half fails, restore 1.8 first and retest — zugzwang is the
  known catastrophic mode, and NO ACCEPTABLE BLUNDERS outranks knob count.

### Stage 5 — structural arms, one at a time (games; optional, ordered by appetite)

5a. Cont-chain 6→4 (2.1): census + `ipn70` + A/B.
5b. Pawn history removal (2.2): A/B alone; also frees 12 MiB/worker, which
matters the day SMP lands.
5c. Movepick non-functional cleanups (2.3, 2.4): bit-identical bench,
`ipn70`-judged.

- **Abort condition per arm:** H0 → drop that arm, keep the rest.

### Standing rules across stages

- One stage in flight at a time; each stage's baseline is the previous
  stage's landed tip, so wins compound and the final tree has ONE cumulative
  A/B vs pre-simplification main (pool it — 10 × 1500 games pooled to
  ±3.9 Elo is the demonstrated path to a real bar).
- Do not touch the TT store policy — study 71's `TT_FAILLOW_MOVE` /
  `qfl_move` knobs are in flight on their own branch and are ADDITIVE; if
  they land, re-run the census (the TT-less split moves).
- Anything from the unmerged study-70 branch (the two landed ordering fixes,
  +7.60 ± 8.48 pooled, LOS 96%) merges on its own evidence, not this plan's.

---

## 5. Where to diverge from Stockfish — and where not

**Divergence is already the house style where it was earned:** no killers,
no countermoves (D17), root-exempt sibling reduction, the aspiration
mss-divisor frame correction, the upcoming-repetition port that measured
−28.8% nodes. The plan extends that line.

**Diverge (remove or collapse, per §2):** the singular double/triple
filigree; the TT-cutoff hardening trio; NMP verification at our depths; the
eval-diff nudge; conthist pruning; the 6-ply continuation chain; possibly
pawn history. Common shape: **late-generation SF accretions, each worth ~1–3
Elo on fishtest's 100k-game bars, transplanted wholesale into an engine that
cannot resolve their value and pays their complexity.** SF's own history is
the counter-argument people reach for ("it passed SPRT there") — but it
passed on SF's eval, SF's tree, and a test harness three orders of magnitude
larger. Here they are unfalsifiable passengers.

**Do NOT diverge (these carry the tree, per §1.2/§1.3):** TT probe/store
discipline and the read/write conversion boundaries; aspiration windows
(re-searches priced at par — `delta 20` was neutral, not a win); PVS
three-call structure; the LMR core + finalization (162% of the tree; the
only knob ever shown to *help* is less reduction); RFP; NMP core; LMP (pays
in executed ops); parent/capture futility + both SEE prunes (all live
post-s600); razoring (post-s600 reversal — it earns its slot now); IIR;
hindsight (125/1k fires); the history stack core (butterfly, capture,
continuation offsets 1–4, correction — correction especially, see §6); the
staged movepick with the good/bad SEE split; qsearch's stand-pat/futility/SEE
discipline; fail-high softening (cheap, everywhere).

**ARCHON originals to keep:** the narrow stalemate scan (5.3/1k qsearch,
cold, kills a real trap class), `DRAW_CONTEMPT` at 0 (one of the few NEW
conditioning axes with a documented loss behind it — P2's perpetual), the
draw dither in `value_draw` (3-fold blindness damping; distinct from the
step-12 dither being removed).

---

## 6. NNUE-guided-search compatibility

The end state the owner wants is a search a specialised-head layer plugs
into. What that layer needs is **few, clean seams** — not pre-built hooks
for heads that don't exist yet.

**Removals that make head work EASIER:**

- Deleting the σ̂/WDL/TM speculative hooks (0.7, 0.8) sounds
  counter-intuitive — they were built FOR heads. But they hard-code head
  *shapes* (a scalar σ̂, a WDL triple, specific clamps) that constrain the
  actual heads D49+ will produce; a real head integration will be one
  multiplicative factor in `soft_stop` and one margin scale at the RFP site,
  both single-line re-insertions. Dead scaffolding is a tax on every reader
  until then; git keeps the prototypes.
- Collapsing the singular margins (1.4) means a future "singularity head"
  replaces ONE number, not an 11-constant family.
- Deleting the per-gate correction couplings (0.2) clears the road for a
  learned uncertainty signal to own that role coherently instead of leaking
  through five ad-hoc divisors.
- Fewer interacting margin tunables = a smaller, better-conditioned space
  for any learned controller (D41-style) to act on.

**Do not remove — these are the seams themselves:**

- **Correction history** (all tables): the engine's only online
  learned-error machinery, the template and the baseline any eval-error
  head must beat. Also live at 81 updates/1k nodes.
- **`stat_score` + the `RED_STATSCORE` dock**: the exact injection point for
  a learned reduction head (that is where `r` is modulated by move-quality
  information today).
- **`MovePicker::set_quiet_prior`** (keep even while deleting the CPL env
  plumbing that feeds it, 0.9): the policy-head seam into ordering.
- **`search-control`** (trace + LMR intervention) and **`datagen`** (obs
  capture) features: they are the data collection and counterfactual
  machinery a learned-search program runs on.
- The **soft-stop factor product**: multiplicative, so any head-driven TM
  factor is one line.

**Flag:** stage 4's eval-diff removal (1.5) deletes one online signal
(opponent-quiet grading by eval swing). If a future head wants that channel,
it re-enters as head output, not as five constants — that is the intended
direction of travel.

---

## 7. What was measured for this study (reproducibility)

- Fire-rate census: `--features sota,arch-d48x,tune,search-stats` build
  (x86-64-v3), `python3 tools/gate_census.py --engine
  target/release/chessengine --net <canonical net> --depth 12` (argv bench;
  piped stdin returns zeros). Node-identity verified: default Bench 587683,
  wl50@d10+net 384,499, both unchanged by the instrumentation.
- Prune budget: `python3 tools/prune_budget.py --engine
  target/release/chessengine --net <canonical net> --depth 10` (same build).
- Provenance: `python3 tools/spsa/param_provenance.py` **from the main
  checkout** — 309 tunables, 63 covered, 246 uncovered, 6 non-SF values,
  1 stale campaign entry, exit 1.
- Study-70/71 census numbers are quoted from the unmerged branch
  (`git show d65242f9:searchopt/LEDGER.md`, commits `4c4bee23`, `08efd402`,
  `51a3272e`); they are that branch's evidence, at its own null floors.
- The worktree's `engines/canonical/net.cenn` symlink dangles; the resolved
  net is `data/nnue/nets/screens/d48x-1280-sb520-ema-s600.cenn` in the main
  checkout. Every command above used the absolute path.

## 8. Honest limits

- The fire-rate census is one position set (wl50) at d12; rates shift with
  depth (NMP verification in particular fires more at real-TC depths). Rank
  order is robust; absolute rates are not portable.
- The prune-budget re-run is node-contribution, and this project has proved
  three times that node savings ≠ Elo. It is used here only to classify
  gates live/dead, never to promise Elo.
- Tier-1/2 removals are individually below the box's resolution — that is
  why the plan proves baskets and totals, and why "the basket is neutral"
  is the claim that will actually be made, not "each member is free".
- The equal-nodes deficit vs SF18 (~−68 saturating by 30k) is explicitly NOT
  addressed by this plan; nothing here claims to close it. Simplification is
  justified on complexity, tunable count, maintenance, and executed-ops
  grounds, with strength held neutral by construction.

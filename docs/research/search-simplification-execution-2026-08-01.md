# Search simplification — EXECUTION log (2026-08-01)

Executes `docs/research/search-simplification-plan-2026-08-01.md` (commit
`9a8f7507`; measurement tooling `102fdb99`). Owner's order: *"Lets implement
all the simplifications from the study and test them in games."* This document
records what LANDED, with the numbers that gate each step. It is updated as
stages complete.

## 0. The base (step 0 — branch untangling)

Four unmerged lines existed; the tested configuration was study 71 measured ON
TOP of study 70, so the base is **main + study 70 + study 71 + the plan's
census tooling**, built as merges on `worktree-agent-abed05f2117dec14a`:

| commit | content |
|---|---|
| `0fef4d4c` | merge `ac4d893f` (study 71 tip, which contains study 70's branch + its merge of main) into main tip `27e47089` — clean; main had zero `src` changes since study 71's base |
| `399e4243` | merge `9a8f7507` (plan + 38-counter gate census). One conflict in `search/mod.rs` (census70 capture vs `Gate::SeTry` counter — both kept) + a fmt reflow |

**Base pins, measured directly on a clean `git archive` export of `399e4243`
(NOT inherited from study 71's claims):**

| pin | value | matches study 71's report |
|---|---|---|
| `Bench:` (d6 default release) | **300694** | yes |
| d13 wl50 + canonical net (sota,arch-d48x) | **2045128** | yes |
| NNUE eval hash | **2e1ede95…79c4** | yes (unmoved — 70/71 were search-only) |

Tunable count on this base: **313** (the plan's 309 was counted on main;
studies 70/71 added 4 knobs: `QUIET_SEE_PENALTY`, `TT_FAILLOW_MOVE`-family).

## 1. Tier 0 — dead code, dead knobs (bit-identical, no games)

Landed as six commits, each with the deletion evidence in its message:

| commit | plan items | what went | knobs |
|---|---|---|---|
| `5c161b59` (1/6) | 0.2–0.6, 0.11, 0.12 | RAZOR/NMP_CORR_DIV terms; the whole followPV apparatus (`FOLLOW_PV_MODE`, `Stack::follow_pv`, `last_iter_pv`, `RootMove::previous_pv`, both consumer conditions); QS_UPREP block (measured worse ON); RED_TTPV_TTMISS_REBATE (measured worse ON); RED_GIVES_CHECK; `Stack::ply`; `RootMove::previous_score_exact` | −6 |
| `842fb345` (2/6) | 0.7, 0.8 | `score_tm_factor` + TM_SCORE_C/KNEE/FLOOR/CEIL; the EXP-UNC-1/WDL-TM apparatus (3 helper fns, 7 Worker fields, the per-node σ̂ EMA capture, the RFP margin rescale, 4 `ARCHON_*` env vars). All measurably inert on a headless net; TM back-loading refuted at real TC (−36.6 ± 22.5) | −4 |
| `8bed4c92` (3/6) | 0.9, 0.10 | D47 `.cpol` search plumbing (`policy_head` snapshot, no-TT-move argmax fill, P-ORD site, CPL_ORD_* env) and the whole D41 `policy` feature (src/policy.rs 568 lines, root ordering, movepick blends, `RootMove::policy_score`, `policy_trunk`, the argv subcommand). **KEPT: `MovePicker::set_quiet_prior` + `quiet_prior` + the scoring blend — the seam a future head plugs into** — and the `policy_head` module itself | 0 |
| `82569ae0` (4/6) | 0.1, 0.14 | The eval-frame correction apparatus: FRAME_S + `frame_scale`/`frame_expand`/`frame_scale_i64`/`round_div` + tests (zero call sites outside params.rs); the four identity class-4 converters (QS_PIECE_MULT, CFUT_PIECE_MULT, SEE_THRESH_MULT, LMP_DEPTH_MULT) + `scale_1024`. The s600 rescale closed the frame program — the 2026-08-01 prune-budget re-run shows every previously-starved gate live | −5 |
| `f5b43d42` (5/6) | 0.13, 0.15 | Ten measured-and-declined perf-arm cfg blocks: insertwin34, bcsort55, bccont55, cont53, histpf44 (whole prefetch API), nogc54, ttspec42, ttspec2_42, dosesort40, census34 (+ module), null34 (meaningless without its arm), dose40 module. `Stage` drops `PartialOrd`. Kept: census70/arm70, icold67/b, pad68/pgonull69, search-stats/control, datagen | 0 |
| `7eb1f478` (0b, 6/6) | tier 0b + blend fold + **D50** | `constants!` sibling macro; 67 handles constified (22 denominators, 10 non-strength, 18 implicit-×1, 14 TM f64 + SOFT_STOP plies); the three blend triples folded to one beta weight each (complements were exact: 716+308, 467+557, 481+543 = 1024); `param_provenance.py` made constants!-aware; stale `CHECK_EXT_MIN_DEPTH` dropped from s4-searchgap | −73 |

Net effect of tier 0 + 0b: **~3,300 lines deleted, 6 env vars gone, 88 knobs
off the SPSA surface (15 deleted outright, 6 folded, 67 constified).**

**Tunables: 313 → 298 (tier 0) → 225 (after 0b).** Counted with
`param_provenance.py`'s own regex over `tunables!` regions only.

D50 is pinned in `design/decisions.md`; `design/01-architecture.md` §3.8
carries the amendment (decision first, docs follow, code follows).

### Bit-identity verification

Per-commit pin checks (all three pins) passed for 1/6 and 2/6 —
**300694 / 2045128 / 2e1ede95…79c4, bit-identical to the base**.

The 3/6–6/6 stack's consolidated verification (same three pins + full
`CHECK_NNUE_GATE=1 tools/check.sh`) was serialized behind the owner's d49
teacher probe after the 16:06 box OOM (see §4) and runs in the
`simplify-verify` systemd unit; result recorded below.

> **RESULT (2026-08-01 16:46): PASSED — the whole tier-0/0b stack is
> bit-identical.** `simplify-verify2` on tip `e156d547`: cargo test green,
> PIN 1 `bench 16 1 6 default depth` = **300694**, PIN 2 d13 wl50+net =
> **2045128**, PIN 3 eval hash = **2e1ede95…79c4**, then the full
> `CHECK_NNUE_GATE=1 tools/check.sh` (clippy, 12-combo feature matrix,
> tests, release perft suite, bench fingerprint, NNUE behavioural gate)
> exited 0 with `All checks passed.` No bisection needed. (First attempt
> failed on a one-line rustfmt reflow — fixed in `e156d547`.)

## 2. Games-gated tiers (stages 2–4)

Methodology (per the plan, every rule paid for): fixed-node fn25k one-match
A/B, same-source both-PGO pair, `option.EvalFile` verified on the LIVE
fastchess argv (`tools/sprt.sh` post-991fbf43), segmented
`tools/perf70/chain70.sh` runs pooled by `pool_sprt.py`, non-inferiority SPRT
elo0=−3 elo1=+1 α=β=0.05. Box rules: no builds while `d49train` is active;
`d49xml` stopped explicitly for match windows and restarted after.

| stage | content | status |
|---|---|---|
| Basket A | offense term (D30), RFP threat coupling, step-12 dither + `child_pv_fresh` | prepared; patch on `simplify-basket-a` after tier-0 verification |
| Stage 3 | singular double/triple margin collapse → one margin, ext ∈ {1,2} (−10 knobs) | queued behind basket A |
| Basket B | eval-diff nudge, conthist pruning, TT-cutoff hardening trio, NMP verification (bisect restores NMP-verify FIRST — zugzwang) | queued |

> Results recorded here as each match pools.

## 3. KEEP list (not touched, per plan §5)

Razoring, LMP, correction history (all tables), the `stat_score` +
`RED_STATSCORE` dock, `set_quiet_prior`, TT probe/store discipline,
aspiration, PVS three-call structure, LMR core, RFP, NMP core, parent/capture
futility + both SEE prunes, IIR, hindsight, fail-high softening, the staged
movepick, qsearch discipline, the narrow stalemate scan, DRAW_CONTEMPT,
`value_draw` dither. The July per-gate verdicts (razor-off, NMP-168) are
STALE post-s600 and were not resurrected.

## 4. Incidents and box notes

- **16:06 global OOM.** My batch-3 pin-check build (already capped
  MemoryMax=6G, CARGO_BUILD_JOBS=2) ran concurrently with the owner's
  relaunched d49 teacher probe (~5 GB) and d49xml; the kernel OOM killer took
  the probe. The coordinator's recovery loop resumed it from checkpoint
  (sb1632) and installed `trainguard` (disposables to oom_score_adj=1000) +
  the hard rule: **no cargo/rustc while `d49train` is active** — builds are
  now serialized behind the trainer via a polling unit.
- The gate-census race (first base-gate run failed at FOLLOW_PV_MODE): I
  edited src while check.sh was building. Rule applied since: no gate runs
  concurrent with edits.
- `engines/canonical/net.cenn` dangles in worktrees; `data/` is symlinked to
  the main checkout's and every command uses the absolute net path.

## 5. Honest limits

- Tier 0/0b strength claims are exactly the pins — bit-identity, no more. No
  Elo is claimed for deletion; the value is surface area (309→~225 knobs on
  the way to ~185 after the games-gated tiers), six env vars, and ~3,300
  lines.
- Items 3/6–6/6 were committed compile-checked during the build freeze with
  pins pending; the consolidated check bisects per-commit if any pin moved.
- The games-gated tiers are individually below this box's resolution by
  design; the claim that will be made is "the basket is not materially
  worse" (H1 at elo0=−3), never "each member is free".

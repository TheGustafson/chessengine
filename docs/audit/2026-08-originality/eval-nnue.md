# Originality & license audit — eval / NNUE / trainer domain (2026-08-09)

Scope: `src/eval/**` (incl. `src/eval/nnue/**`), `jax/archon_jax/*.py`,
`jax/tools/*.py`, `tools/nnue/bullet/*.rs`, the eval/NNUE distilled notes in
`stockfish_reference/` and `reckless_reference/`, `src/obs/`, and the data
pipeline contract docs. References compared at the pinned revisions:
Stockfish **eca43a97** (GPLv3; read via `git show`, never the checkout tip),
Reckless v0.10.0-dev as vendored in `reckless_reference/Reckless/` (AGPLv3),
nnue-pytorch at `/home/nick/projects/nnue-pytorch` (GPLv3, user clone outside
this repo), Lc0 (GPLv3, **not** cloned — assessed from our code's own claims
and the vocabulary arithmetic only).

Rubric: L0 independent · L1 idea-level adoption, independently expressed ·
L2 structural echo (same decomposition/order/names/call shapes) · L3
near-translation · L4 verbatim. Every claim above L1 carries receipts
(our file:line and theirs).

Context that frames severity throughout: **the repo has no LICENSE file and no
`license` field in Cargo.toml, and no foreign net weights are tracked**
(`engines/canonical/net.cenn` is a mode-120000 symlink to a gitignored file;
`git ls-files` shows no `.nnue`/`.cenn`/`.bin` blobs). GPL/AGPL obligations
trigger on conveyance; nothing here has been conveyed. The findings below are
about what the tree *is*, and what releasing it would require.

---

## 1. Verdict table

| Module / mechanism | Level | Lineage tag | One-line basis |
|---|---|---|---|
| `sota_features.rs` — HalfKAv2_hm index math | **L2** | SF-SPECIFIC | `KING_BUCKETS`, `PieceSquareIndex` slot scheme, OrientTBL polarity transcribed; formula = SF `make_index`; Rust expression own (split `row_base`/`oriented`, predicate instead of table) |
| `sota_features.rs` — FullThreats index math | **L2** (one L3 line) | SF-SPECIFIC | `MAP`, `NUM_VALID_TARGETS` verbatim value tables; same `lut1`/`lut2`/prefix-offset decomposition + sentinel-drop; `semi_excluded` expression is a rename-level match; ARCHON fused `offsets+lut2` (study 59) and added `ThreatRemap` |
| `sota_features.rs` — `for_each_threat` enumeration | **L2** | SF-SPECIFIC | Mirrors `append_active_indices`: same pre-masked victim sets (`pawn_targets`/`minor_slider_targets`/`queen_targets` ≙ SF's `pawnTargets`/`minorSliderTargets`/`queenTargets`), same pushers trick; loop expression own |
| `sota_head.rs` — head architecture (pairwise-mul FT act → NNZ-sparse i8 L1 → f32 L2/L3, 8 output buckets) | **L2** | RECKLESS-SPECIFIC | Self-described "Reckless-shaped"; FT_QUANT 255 / L1_QUANT 64 / FT_SHIFT 9 / DEQUANT bridge = Reckless's exact recipe; `propagate_l1/l2/l3` names retained |
| `sota_head.rs:226` — `OUTPUT_BUCKETS_LAYOUT` | **L3/L4** | RECKLESS-SPECIFIC | Verbatim copy of Reckless's table: same identifier, same 33 values, same line-grouping (receipt in §2.1) |
| `sota_simd.rs` — scalar reference kernels (`activate`, `find_nnz`) | **L2–L3** | RECKLESS-SPECIFIC | Same skeleton, same `left`/`right`/`flip` names and output expression as Reckless `forward/scalar.rs`; written to be bit-exact against Reckless's scalar reference (documented); real divergence layered on (per-column scale, threat shift, SCReLU) |
| `sota_simd.rs` — scalar `l1_dequant`/`l2_tail`/`l3_dot` | **L1–L2** | RECKLESS-SPECIFIC idea, diverged | Bias-seeding + SCReLU differ from vendored Reckless; 16-lane + halving-tree reduction discipline follows Reckless's bit-exactness lesson (note 13 §2.4/§3.4) |
| `sota_simd.rs` — AVX2/AVX-512 backends (maddubs/madd ladder, movemask+LUT nnz, vbmi2 compress) | **L1** | NNUE-STANDARD | Community-standard recipes (SF `affine_transform_sparse_input.h`/`nnz_helper.h`, Reckless, bullet all converge); expression own, heavily perf-study-shaped (`RowOffset` prescaling, dual-destination folds, `NNZ_COMPACT` const table) |
| `sota_accumulator.rs` — lazy stack + Finny cache + bidirectional folds | **L2** | SF-SPECIFIC + NNUE-STANDARD | `const FORWARD: bool` fold ≙ SF `template<bool Forward>` forward/backward incremental updates; Finny cache is the community technique (SF `AccumulatorCaches`, Reckless `AccumulatorCache`); key scheme (bucket×orient = 64) own hybrid; expression own |
| `sota_threats.rs` — incremental threat observer | **L1/L2** | RECKLESS structure + SF content, ARCHON driver | Documented "Structure (Reckless) / content (Stockfish)"; the fully-lazy fold, snapshot-deferred king-cross rebuild, telescoping-replay correctness argument, and mirror-refresh split are ARCHON-original (§3) |
| `sota_context.rs` | **L1** | ARCHON-ORIGINAL composition | Smallnet gate w/ material-imbalance prefilter, `ZERO_THREAT` channel, dis/σ/WDL caching; big/small-net *idea* is SF's |
| `sota_loader.rs` — CENN/CEN2/CEN5 container | **L0/L1** | ARCHON-ORIGINAL | Own magic/version/dims/FNV-checksum container; explicitly *not* SF's hash+LEB128 format, not Reckless's raw transmute; i8 quantize-at-load w/ per-column scale = perf study 32 (original) |
| `loader.rs`, `network.rs`, `accumulator.rs`, `simd.rs` (gen-1) | **L1** | NNUE-STANDARD | bullet-standard 768→2×256→1 SCReLU, QA=255/QB=64/SCALE=400; own CENN-v1 container; own dispatch (LazyLock fn-pointer table) |
| `simple_halfka.rs` (CEN3), `protected_halfka.rs` (CEN4) | **L1 / L0** | ARCHON-ORIGINAL | King-conditioned Chess768; frozen-base+exact-residual net is an ARCHON concept |
| `policy_vocab.rs` | **L1** | LC0-SPECIFIC idea | 1858 = 1792 geometric + 66 underpromo is Lc0/AlphaZero's vocabulary *concept*; enumeration order self-derived (from-major/to-ascending), frame deliberately deviates (adds HalfKA horizontal mirror; documented) — no Lc0 code available or used |
| `policy_head.rs` | **L1** | ARCHON-ORIGINAL | CPL1 sidecar + selected-row scorer over own vocab |
| `aligned.rs`, `foldcount.rs`, `skidprobe.rs` | **L0** | ARCHON-ORIGINAL | Perf-study instrumentation and alignment machinery |
| `src/eval/mod.rs` — outer eval shape | **L2** | SF-SPECIFIC | Same formula and all six constants verbatim as tunables (receipt §2.3); documented as such |
| `src/eval/psqt.rs` | **L1** | PeSTO (attributed) | Value tables transcribed with provenance header; universal practice |
| `src/eval/score.rs`, `params.rs` mechanics | **L1/L0** | own | Deliberately un-SF packing (two-field Score) |
| `jax/archon_jax/` trainer | **L1** | ARCHON-ORIGINAL expression | Pure-functional JAX matching ARCHON's *own* bullet ModelDefinition; loss base = bullet's graph (bullet is MIT-licensed upstream — verify at release); SF two-sigmoid/pow2.5/asymmetry adopted as *shapes* with constants explicitly not copied (`losses.py` docstrings); Pallas FT kernel own w/ custom VJP |
| `jax/tools/*`, `tools/nnue/bullet/*.rs` | **L1/L0** | ARCHON-ORIGINAL | Research tooling; d40_features.rs is our model definition *for* bullet, not bullet's code |
| `src/obs/` | **L0** | ARCHON-ORIGINAL | TLV wire hand-rolled after the protobuf wire *model* (public fact, no code/dependency) |
| Distilled notes (`stockfish_reference/05,08,09`, `reckless_reference/11,12,13,14`) | **clean** | — | All fenced blocks are own restatements/pseudocode or measured tables; no pasted GPL/AGPL source found |
| `old/tools/nnue/patches/*.patch` | **L4 fragments** | SF + RECKLESS + bullet | Tracked diff files containing verbatim GPL/AGPL context lines (receipt §2.2) |

Known-context verification (asked for receipts, not inheritance):

* **"HalfKA features provably identical to SF"** — confirmed at the definition
  level: `HALFKA_DIMS = 22528`, `KING_BUCKETS`, slot scheme and orientation all
  match eca43a97; the *code* is not a transliteration (SF's `make_index` is one
  table-driven expression, ours splits into `halfka_row_base` +
  `halfka_oriented` with a predicate replacing `OrientTBL`) — L2, not L3.
* **"Threat inputs differ from SF18"** — confirmed and now exact: ours is
  eca43a97's FullThreats **60,720** (`sota_features.rs:203` vs
  `eca43a97:src/nnue/features/full_threats.h:41`); the `sf_18` tag ships
  **79,856** (`sf_18:src/nnue/features/full_threats.h:45`). We match our actual
  reference revision exactly and differ from the SF18 release.
* **i8 FT + per-column scale** — confirmed ARCHON-original (perf study 32;
  `sota_loader.rs:674-934`). SF's FT weights are i16 (`WeightType` =
  `std::int16_t` in `nnue_feature_transformer.h`); Reckless's PST FT is i16.
* **CENN** — confirmed original (§ verdicts above).
* **sota_context / dis tooling** — confirmed ARCHON-original.

---

## 2. License-risk findings, most severe first

### 2.1 Reckless (AGPLv3): one verbatim table + a deliberately bit-exact head lineage — **the top finding**

**(a) `OUTPUT_BUCKETS_LAYOUT` is a verbatim copy.**
`src/eval/nnue/sota_head.rs:226-235`:

```rust
const OUTPUT_BUCKETS_LAYOUT: [usize; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 0..8 pieces
    1, 1, 1, 1, // 9..12
    2, 2, 2, 2, // 13..16
    3, 3, 3, // 17..19
    4, 4, 4, // 20..22
    5, 5, 5, // 23..25
    6, 6, 6, // 26..28
    7, 7, 7, 7, // 29..32
];
```

`reckless_reference/Reckless/src/nnue.rs:83-93`:

```rust
const OUTPUT_BUCKETS_LAYOUT: [usize; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7, 7,
];
```

Same identifier, same type, all 33 values identical, same line-grouping. The
adjacent comment ("copy Reckless's *shape*, retune boundaries on our data
later") records the intent, but the retune never happened — as shipped this is
a **verbatim AGPL-derived hand-designed table** (small, but literal). This is
the only L4-grade item in the compiled engine.

**(b) The head's provenance is a bit-exact reimplementation of AGPL behavior.**
`sota_head.rs:1-9` states the module is "a faithful transcription of our own
bit-exact reimplementation … matched to Reckless's scalar reference over 42
positions". Behavior is not copyrightable and the tree's expression has since
diverged substantially (per-column i8 scale, threat-shift merge, SCReLU tail,
PSQT skip, dis/σ/WDL heads), but two scalar kernels still show the skeleton:

`sota_simd.rs:1181-1207` (`activate`, simplified) vs Reckless
`src/nnue/forward/scalar.rs:9-25` (`activate_ft`):

```rust
// ARCHON                                        // Reckless
for flip in 0..2 {                               for flip in [0, 1] {
  let perspective = stm ^ flip;                    let pst_input = &pst.values[stm as usize ^ flip];
  for lane in 0..H {                               for i in 0..L1_SIZE / 2 {
    let left  = merge(pst[..][lane], ...);           let left  = (pst_input[i] + threat_input[i]).clamp(0, FT_QUANT);
    let right = merge(pst[..][lane+H], ...);         let right = (pst_input[i+L1_SIZE/2] + ...).clamp(0, FT_QUANT);
    out[lane + flip * H] =                           output[i + flip * L1_SIZE/2] =
      ((left * right) >> FT_SHIFT) as u8;              ((left as i32 * right as i32) >> FT_SHIFT) as u8;
```

Same loop nesting, same `stm ^ flip` perspective walk, same `left`/`right`
binding names, same output-placement and shift-cast expression. `find_nnz`
(`sota_simd.rs:1213-1226` vs Reckless `forward/scalar.rs:86-104`) matches the
same way (OR-reduce over 4-byte groups, `as u16` index append). Both are short
functions whose natural expression is heavily constrained by the math, and
both are same-language (Rust↔Rust), which is exactly where accidental *and*
real echo are hardest to distinguish — but combined with (a) and the recorded
bit-exact-matching method, the honest classification is **L2 with L3
elements**, not independent creation.

**(c) Recipe constants.** `FT_QUANT = 255` / `L1_QUANT = 64` are
bullet-standard, but `FT_SHIFT = 9`, the `2^FT_SHIFT/(QA²·QB)` dequant bridge,
pairwise-mul activation, and the i8-sparse-L1→f32-tail hybrid are Reckless's
specific published recipe (`sota_head.rs:146-222` ≙ `nnue.rs:91-99`). Idea/
parameter-level (L1/L2), RECKLESS-SPECIFIC.

**Exposure:** AGPLv3 obligations attach on conveyance (or network use). Today:
none — private, unreleased, no Reckless weights anywhere ("weights stay
firewalled" is accurate; nothing loads them). **If ARCHON is ever released:**
either release under AGPLv3 (or GPLv3 with the table/kernels re-derived), or
scrub: rename + independently retune `OUTPUT_BUCKETS_LAYOUT` on our own data
(the comment already promises this), and restate the two scalar kernels from
the *math* rather than the reference. That is an afternoon of work; do it
before any release regardless of chosen license.

### 2.2 Tracked patch files carrying verbatim GPL/AGPL text

`old/tools/nnue/patches/stockfish18-evalraw-v2.patch` and
`reckless-evalraw-v2.patch` are git-tracked diffs against the GPL/AGPL
reference engines. A diff's context lines are verbatim source (e.g. the SF
patch carries `void Engine::trace_eval() const { … sync_cout << "\n" <<
Eval::trace(p, *networks) << sync_endl; }` and the Reckless patch carries the
body of Reckless's `fn eval(td: &mut ThreadData, board: &Board)`), and the
added hunks are derivative works of the patched programs. Conveying the repo
conveys these. This is **normal and license-compliant if the patch files are
themselves understood as GPL/AGPL** (patches to GPL programs may be GPL
without affecting the rest of the repo), but it should be explicit: add a
one-line license note to `old/tools/nnue/patches/README.md`, or move the
patches out of the repo. The `bullet-*.patch` files patch the external bullet
trainer (MIT upstream — verify its LICENSE at the pinned checkout before
release); low concern.

### 2.3 Stockfish (GPLv3): transcribed constant tables, formulas, and one rename-level line

All deliberate, all documented in-source, all expressed in our own Rust —
but the aggregate is a *heavily SF-derived design* and the receipts should be
on the record:

* **`KING_BUCKETS`** `sota_features.rs:64-73` = `eca43a97
  half_ka_v2_hm.h:69-78` (values identical after dividing SF's `B(v) = v·PS_NB`
  macro out — a trivial transform of a transcribed table).
* **`MAP`** `sota_features.rs:246-253` = `full_threats.h:56-63`, and
  **`NUM_VALID_TARGETS`** `sota_features.rs:236-239` = `full_threats.h:32-33`
  — verbatim value tables.
* **Threat index decomposition** `sota_features.rs:332-460` ≙
  `full_threats.cpp:117-190`: same `cumulativePieceOffset`/`cumulativeOffset`
  prefix sums, same `lut1[att][atd][from<to]` + per-from ordinal `lut2`, same
  out-of-range sentinel standing in for `push_back_if_lt`. One line is a
  rename-level match — ours `sota_features.rs:430`
  `let semi_excluded = att_t == atd_t && (same_type_diff_color || att_t != 1);`
  vs SF `full_threats.cpp:169`
  `bool semi_excluded = attackerType == attackedType && (enemy || attackerType != PAWN);`.
  ARCHON divergences: the perf-study-59 fused `from_to` u16 table, the
  `ThreatResolver` hoist, and `ThreatRemap` (none exist in SF).
* **Outer eval** `src/eval/mod.rs:631-668` + `src/eval/params.rs:55-82` ≙
  `eca43a97 evaluate.cpp:54-66`: same formula chain (material-weighted
  optimism blend → rule-50 damp → TB clamp) with SF's exact constants 534,
  77871, 7191, 199, 476, 18236 — the header says "recorded verbatim from
  `evaluate.cpp`". The complexity blend is structurally carried but inactive
  (`complexity = 0`); ARCHON adds the `EVAL_SCALE` dual (original).
* **Accumulator architecture** `sota_accumulator.rs` ≙
  `nnue_accumulator.cpp`: bidirectional incremental updates
  (`apply_dirty_into::<FORWARD>` ≙ `template<bool Forward>` /
  `forward_update_incremental`+`backward_update_incremental`) and the Finny
  refresh cache (`update_accumulator_refresh_cache`). Expression thoroughly
  our own (different storage, different keying, perf-study-driven kernels).

Legal read: feature-set definitions, formulas, and tuned constants sit mostly
on the idea/fact side of the idea–expression line, and transcribing SF
constants is universal engine practice — but "universal practice" is not a
defense, and the *density* of SF-specific tables + decompositions in
`src/eval/` means the safe release posture is a **GPLv3-compatible license**.
Since 2.1 independently pushes toward AGPLv3/GPLv3, this is not an additional
constraint in practice.

### 2.4 nnue-pytorch (GPLv3): idea-level only — clean

The trainer was studied from a *local user clone outside this repo*
(`/home/nick/projects/nnue-pytorch`; the study note
`stockfish_reference/09-nnue-pytorch-trainer-study.md` says "ideas only —
never copy code"). What was adopted into `jax/archon_jax/losses.py`: the
offset two-sigmoid `expected_score` (a mathematical transform, restated; SF's
fitted constants explicitly labeled "a shape reference, not values to copy"),
`pow_exp`, `qp_asymmetry` (SF's 0.2287 quoted in a docstring, default 0.0),
probabilistic-skip-as-weights. The trainer's *structure* follows **bullet**
(`train.py`: "bullet run.rs semantics on the JAX stack") and ARCHON's own
`tools/nnue/bullet/d40_features.rs` model definition — not nnue-pytorch's
Lightning/loader architecture. No line citations to nnue-pytorch were
fabricated here; the clone was available and consulted. Verdict: **L1**.

### 2.5 Cleared

* **Distilled notes**: all fenced blocks in the eval/NNUE notes (SF 05/08/09,
  Reckless 11/12/13/14) are own pseudocode, own restatements, or measured
  tables; the Reckless notes carry explicit no-code-reproduced headers and the
  full read confirms them. No distribution of licensed text via the notes.
* **Weights**: no SF/Reckless/Lc0 network files tracked or embedded; the
  Reckless-net experiment branch (`experiment/reckless-net-2`) contains no
  `src/eval/reckless_net/` tree in any surviving ref.
* **Lc0**: `policy_vocab.rs` uses the 1858-class *concept* with self-derived
  ordering and a documented deliberate frame deviation (added horizontal
  mirror). No Lc0 source in the repo, no evidence of consultation beyond the
  public vocabulary size/composition. L1.
* **PeSTO**: `psqt.rs:1-9` transcribes the published tables with full
  attribution — the standard public starting point, values-not-code.

---

## 3. Originality census

Of the 24 distinct mechanisms classified in §1: **8 ARCHON-original, 7
SF-specific adoptions (L2), 4 Reckless-specific adoptions (L1–L3/4), 1
Lc0-specific idea (L1), 4 NNUE-standard**. By *code volume* the picture is
better than by mechanism count: the sota_* modules are dominated by
perf-study-driven expression that exists in no reference engine (the fold
kernel family, RowOffset, Blocks/Aligned, foldcount, the probe features), and
the trainer + tools + obs layers are essentially all original. By *design
authorship* the picture is worse: the net's input definition is SF's, its head
recipe is Reckless's, and the outer eval is SF's — the gen-2 nets compete on
data/recipe (as the roadmap itself states), not on architecture.

**ARCHON-original mechanisms (the explicit list):**

1. **CENN container family** (CENN/CEN2/CEN3/CEN4/CEN5) — validated,
   checksummed, versioned net format; `CEN5`'s `threat_quant` header field.
2. **i8 HalfKA feature transformer with per-column integer scale, quantized at
   load** (perf study 32) + the wrapping-i16 merge semantics + the
   power-of-two threat-shift merge (studies 8 §4.6, 30). Neither SF (i16 FT)
   nor Reckless (i16 PST FT) does this.
3. **Fully-lazy threat folding** with snapshot-deferred king-cross rebuild and
   the telescoping-replay correctness argument (`sota_threats.rs` header;
   studies 3, 18) — beyond SF (eager in `do_move`) and Reckless (eager
   record). Fold runs on a measured 0.4151 plies/node.
4. **Separate-channel mirror-refresh exploitation** — threats refresh only on
   a d/e king cross, a strict subset of HalfKA refreshes; the in-source
   comment cites `eca43a97:nnue_accumulator.cpp:112-113` conceding SF's fused
   accumulator cannot do this.
5. **ThreatRemap** — a learned surjective re-coding of the 60,720 threat index
   space carried in the net container (`TRM1`), applied at one resolver
   chokepoint. No reference engine has a learned feature re-code.
6. **Disagreement head (DIS1)** + signed/magnitude residual targets with
   dual-null telemetry (`losses.py`), plus EXP-UNC-1 log-variance and WDL
   heads — search-facing uncertainty machinery no reference engine ships.
7. **Frozen-base + exact-residual net** (`protected_halfka.rs`, CEN4) and the
   king-conditioned Chess768 (CEN3).
8. **The research stack**: pure-functional JAX trainer with bullet-parity
   semantics and byte-identical checkpoint/quant formats, Pallas
   embedding-bag kernel with custom VJP, focal weighting against score-only
   references, teacher annealing/cutovers; the Observation Store
   (`src/obs/`, TLV wire); the D46 data-recipe contract + ledger provenance
   DAG; foldcount/skidprobe instrumentation.

---

## 4. Differentiation opportunities

Ranked by (asset already in tree) × (references provably can't/don't do it).

1. **Castling-rights + rule-50 inputs — go first on a measured shared
   blindness.** Measured 2026-07-30: our nets AND SF's are bit-identical
   across castling/clock twins, yet SF changes its best move ~39%/~37% when
   those facts change — both engines are blind to inputs the search then has
   to rediscover. Neither SF nor Reckless encodes them (identical features =
   identical blindness). Mechanism: 5 dense extra inputs at the L1 bottleneck
   (4 castling bits + normalized clock) — no accumulator cost, no new FT rows;
   trainer support is a ~40-line change in `model.py`. First experiment: twin
   D-arch nets ± the inputs, same corpus, judge on the *twin-position
   move-change class* and equal-time games, not heldout loss.
2. **The disagreement head as a search consumer.** The head exists, is
   off-hot-path by design, and its training telemetry already separates bias
   from signal. No engine ships a learned per-position "how wrong is my eval
   likely to be" channel. First experiment: scale one margin (RFP or
   aspiration width) by dis-magnitude at PV/root nodes only, SPRT at 2+0.1.
3. **ThreatRemap as learned feature compression.** The FT is the measured
   memory wall (fold is MSHR-limited; threats cost +25–39% instructions,
   study 7 §5). A learned surjection 60,720 → 8–16k rows attacks table bytes
   in a way SF structurally cannot without a retrain of their whole release
   pipeline — and our POC plumbing (net-carried table, single chokepoint,
   loader validation) is already landed. First experiment: k-means / gradient
   co-clustering of trained threat rows into 12k codes, retrain trunk with the
   remap fixed, measure NPS × quality at equal time.
4. **Per-channel laziness as a search-integration lever.** We already fold
   threats on 0.4151 plies/node and refresh them strictly less often than
   HalfKA. Extension: a threat-free eval tier for qsearch/hot leaves (the
   `ZERO_THREAT` + smallnet-gate machinery already supports exactly this
   shape), promoted to full eval only above a depth/importance gate — a
   speed/quality dial fused accumulators (SF) cannot express.
5. **The mirrored-frame policy head.** Lc0 rank-flips only; our policy IDs
   live in the same mirrored canonical frame as the trunk, halving the
   symmetry the head must learn (the exp-0121 lesson, documented in
   `policy_vocab.rs`). Asset for in-search policy priors (movepick priors at
   low depth) — a place where being NNUE-native and Lc0-shaped at once is
   unoccupied ground.
6. **The data flywheel + ledger as the compounding asset.** d53mix20 was the
   first canonical net our own generated data improved; the D46 contract, the
   obs store, and the provenance ledger are infrastructure none of the solo
   references have in-tree. Differentiation here is *publishable process*
   (reproducible lineage for every net), which aligns with the stated
   research-publication goal.

---

## 5. Actions (pre-release checklist, in order)

1. Retune (or at minimum re-derive on our own data and rename)
   `OUTPUT_BUCKETS_LAYOUT`; restate `sota_simd.rs` scalar `activate`/`find_nnz`
   from the math. Closes the only L3/L4 item in the engine. (§2.1)
2. Add a license note to `old/tools/nnue/patches/README.md` declaring the
   patch files GPL/AGPL as derivatives of their targets, or move them out.
   (§2.2)
3. Pick the project license with eyes open: the eval domain's density of
   SF-derived design + the Reckless-derived head make **AGPLv3** the safe
   choice (it satisfies both upstreams); anything GPL-incompatible requires
   the §2.1 scrub *and* a harder argument about §2.3's aggregate.
4. Verify bullet's LICENSE at the pinned external checkout before shipping
   anything trainer-derived. (§2.4)

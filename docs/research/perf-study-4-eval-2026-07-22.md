# ARCHON performance study #4 — the EVAL LAYER (code-read cost model, no measurements)

Date: 2026-07-22. Commissioned as the eval-focused follow-up to studies 1–3
(`profiling-study-2026-07-18.md`, `profiling-study-2026-07-19.md`,
`profiling-study-3-creative-2026-07-19.md`). **Constraint honored: a crown
gauntlet + round robin were running on this box for the entire study — zero
builds, zero benches, zero profiles were executed.** Everything below is
code reading, cross-referenced against the prior studies' *measured* numbers;
every quantitative claim is labeled either MEASURED (with its source round) or
EST (a cycles-class estimate from this study's code model). The measurement
script that confirms/refutes the top claims is `tools/perf_study4_bench.sh`
(deliverable #4) — ready for the post-tournament quiet window.

**Source state modeled:** the MAIN checkout at `/home/nick/projects/chessengine`
as of 2026-07-22 (HEAD `ee12507` + the uncommitted d47c-era changes to
`src/eval/nnue/*`). This worktree's own checkout is two commits older and
pre-d47c; the study deliberately models the production tree, since the d47c
dims change the cost picture materially (§2). Line references are into the
main-tree files.

**Fresh verified context this study builds on (user, 2026-07-22):**
- Shell exonerated: PSQT-mode bench 2.99M NPS vs SF15.1 classical 1.96M;
  perft 340 Mnps vs SF ~200M. The whole NPS gap to SF lives in the eval layer.
- At threat-feature parity: SF18 (SFNNv16, FullThreats, dual-net) 1.03M NPS vs
  archon-d47 ~506k uncontended = **2.8×**.
- SIMD is modern: hand-written AVX-512 kernels, 5000+ zmm instructions,
  8 `vpdpbusd` sites, zero legacy `vpmaddubsw`. The question is COVERAGE,
  not ISA (§4).

---

## 1. Prior-studies scoreboard (deliverable #1)

Status is from reading the CURRENT main-tree code, not from memory. "DONE"
means verified present in source; line refs given.

### 1.1 Study 1 (2026-07-18) — ranked items

| # | item | status | evidence / what remains |
|---|---|---|---|
| 1 | Vectorize the f32 head tail | **DONE** (opt#1, `8287e90`) | `sota_head.rs:156-237`: per-bucket fixed-size slice hoisting + `mul_add`, comments cite the study. MEASURED round 2: 0 scalar FMAs left, I1 1.12%→0.29%, +13-15% NPS, **+28.8±10.3 Elo** (exp-0101). |
| 2 | Fused copy+apply fold kernels | **DONE** (opt#2, `1cc2147`) | `sota_simd.rs`: `apply_i16_deltas_into`/`apply_i8_deltas_into` used at both fold sites (`sota_accumulator.rs:96-110`, `sota_threats.rs:800-810`). MEASURED round 2: memmove 9.4%→2.9%. Package = +20.7% NPS clean-box. |
| 3 | Prefetch fold weight rows | **NOT DONE** | zero `_mm_prefetch` in `src/eval/nnue/` (grep-verified). Study 3 measured the *eager-path* fold-entry prefetch as null (no distance); the lazy observer landing (below) makes the resolve→prefetch→fold pipeline natural — item reborn as §6 #3. |
| 4 | Static kernel dispatch | **NOT DONE** | `sota_simd.rs:41-58`: `LazyLock<Kernels>` fn-pointer dispatch still live; per-call state-check+indirect-call at every kernel site (round-2 F5 confirmed the check survives in codegen). |
| 5 | Observer constant factor | **SUPERSEDED / PARTLY OPEN** | the *schedule* was fixed by study 3 #1 (below). The constant-factor sub-items are still open: `threat_index` derefs the `THREAT_TABLES` LazyLock **per event** (`sota_features.rs:332`), `resolve_into` still runs the O(n²) cancel scan (`sota_threats.rs:843-873`), piece-type dispatch in `record_square_touch` unchanged (`sota_threats.rs:257-388`). Now paid ~0.46× as often. |
| 6 | AVX-512 for classic v1 kernels | NOT DONE, **MOOT** | incumbent lane is sota/d47c; classic-2048 is the fallback ship only. |
| 7 | Bulk net parse | **NOT DONE, got worse** | `sota_loader.rs:592-623`: still per-element `Vec::push` readers. The net grew 81→**130.5 MB** (d47c-sb*.cenn = 136,843,872 B) ⇒ EST ~0.3-0.5 s per process spawn (scaled from round-2's measured 0.15 s at 81 MB). Match-harness + isready tax. |
| 8 | Quant unification (CENN-v3) | **NOT DONE** | `sota_head.rs:43` `THREAT_QUANT=64` vs `FT_QUANT=255`; `activate` still the i32 widen+rescale merge (`sota_simd.rs:194-215, 639-692`). d47c shipped as CEN2 with the split scale — the tax now runs over 1280 lanes (§3.4). |
| 9 | MovePicker constant factor | OPEN (search lane) | out of eval scope; coordinate with the search study. |
| 10 | Huge pages | **RE-OPENED** | the WSL2 "THP=always ⇒ demoted" conclusion does NOT carry to the new Arch box (migrated 2026-07-19), and the table grew to ~135 MB of weights. Script checks THP + dTLB counters (§7). |

### 1.2 Study 2 (2026-07-19) — deltas

| item | status | note |
|---|---|---|
| #1 prefetch (promoted) | NOT DONE | see above |
| #2 observer constants | see 1.1 #5 | LazyLock hoist / lut row bases / branchless dispatch / sort-free cancel all open |
| #3 residual eval-path copies | **NOT DONE, grew with d47c** | `activate_ft` returns `[u8; 1280]` by value (`sota_head.rs:109-113`), `find_nnz` returns `([u16; 320], usize)` (`:146-150`), `finny_refresh` still does ~4 round-trips of a now-**2560 B** accumulator (`sota_accumulator.rs:113-168`), and `forward_full` returns the 512 B L2 activation by value — **`forward` itself calls `forward_full(...).0` and discards it** (`sota_head.rs:301-309`; LLVM may elide, must verify in disassembly). Null-move fold still degenerates to a pure 2×2560 B copy (`search/mod.rs:2655` pushes `DirtyPiece::EMPTY`; no alias short-circuit in `ThreatContext::compute`). |
| #4 static dispatch + explicit wide head kernel | NOT DONE | newly urgent — the head is 8× wider (§2) |
| #5 bulk parse | NOT DONE | see 1.1 #7 |
| #6 quant unify | NOT DONE | see 1.1 #8 |
| #8 huge pages | re-opened | new box |

### 1.3 Study 3 (2026-07-19, creative) — the big one

| # | item | status | evidence |
|---|---|---|---|
| **1** | **Fully-lazy threat observer** (headline, +11-12% claimed) | **IMPLEMENTED** | `sota_threats.rs` was rewritten around it: `ThreatSnapshot` (~140 B mailbox+occ+kings, `:151-199`), `apply_at` stores only dp+snapshot (`:712-747`), `compute` defers generation (`build_deltas`) + resolution (`resolve_into`) + fold to evaluated plies (`:758-813`), king-cross still eager. Module docs cite "perf study 3, §2". Parity gates (`threats_replay_*`) retained. **Its NPS yield was never separately A/B'd on a quiet box** — folded into the d47 era; the script re-baselines. |
| 2 | resolve→prefetch→fold pipeline | **NOT DONE** | `compute` resolves and immediately folds per perspective (`:787-811`); no prefetch issued. Now trivially implementable (§6 #3). |
| 3 | cross-ply cancel + null-move alias + packed events | **NOT DONE** | resolution is strictly per-ply; no span merge; null-move copy-fold remains. |
| 4 | residual copies | NOT DONE | see 1.2 #3 |
| 5 | static dispatch + wide head kernel | NOT DONE | |
| 6 | SIMD event emission (SF ICL pattern) | NOT DONE | mailbox snapshot now exists (prerequisite landed); still MED effort |
| 7 | quant unification | NOT DONE | |
| **8** | **Smallnet gate** (bold bet #2) | **INFRA DONE, NET NOT TRAINED** | `62102ca` + more: `SotaContext` carries `smallnet`/`smallnet_halfka`/`smallnet_thresh`/`smallnet_mat_gate` (`sota_context.rs:90-106`), the material prefilter (the §COST-MODEL fix, `:46-51, 374-393`), `ZERO_THREAT` static (`:36`), env plumbing `ARCHON_SMALLNET*` (`mod.rs:424-436`), and even a σ̂-routed LARGE tier (`ARCHON_LARGENET`, `:113-124`). Default-off ⇒ bit-identical, tests pin it. **T10 (arch-ablation-suite) is the refined recipe**: L1-128 companion, slice-filter on **simple_eval** (never the label), distilled from the d47c big net, sacrifice-suite ship gate. Quantified in §5. |
| — | low-rank threat factorization | **KILLED (stays killed)** | SVD probe measured flat spectrum. Caveat: the probe ran on the 768-wide d40 table; d47c's 1280-wide trunk is a new table — the prior transfers but a 3-minute re-probe is in the script as a checkbox, not a bet. |
| — | backward accumulator updates | still parked | spans avg 1.34 — thin |

### 1.4 `reckless_reference/13-nnue-inference-simd.md` — adaptation list

(The brief's `stockfish_reference/13-...` path is this file — it lives under
`reckless_reference/`.)

| recommendation | status |
|---|---|
| int8-sparse-FT + f32 tail as the head shape | **ADOPTED** — `sota_head.rs` is exactly this |
| cfg-module compile-time dispatch for the pinned fast build, LazyLock only for portable | **NOT ADOPTED** — LazyLock everywhere, incl. strength-test builds (D16 rebuilds per target anyway; the escape hatch design/11 §3.4.6 anticipated was never built) |
| bit-exactness gotchas (mul_add, bias-seeded accumulators, fixed reduction shape) | **ADOPTED** — visible in `sota_head.rs:177-237` comments |
| small primitive surface | ADOPTED (6-kernel surface) |
| threats aux accumulator | ADOPTED (FullThreats) |
| keep loader validation | ADOPTED (CEN2 header + fnv1a checksum) |

### 1.5 `docs/perf-baseline.md`

P1–P3-era criterion baselines (attacks ~1 ns/lookup, perft 280-297 Mnps,
movegen ~0.5 G legal moves/s) — all reconfirmed by today's shell exoneration
(340 Mnps perft, 2.99M PSQT bench). **Gap:** the doc has no NNUE-era section;
the quiet-window script's output should seed one (per-eval µs for the d47c
path, so future regressions are visible without a full profile).

**Scoreboard summary: of the 10 distinct engineering items studies 1–3 left
open, 3 landed (vector head, fused folds, lazy observer — the three biggest),
the smallnet gate's runtime half landed, and 7 remain open — several of which
D47's width change silently re-priced upward (§2). Nothing on the list was
implemented-and-rejected; the only killed item stays killed by measurement.**

---

## 2. What changed under the studies' feet: d47c re-priced the eval layer

All three prior studies profiled the **768-wide, L2=16** net. Production is
now **d47c-stage2**: `ACC = L1 = 1280` (`sota_head.rs:32`, compile-asserted at
`sota_features.rs:41`), **L2 = 128** (the D47 shared bottleneck), L3 = 32,
8 buckets, HalfKAv2_hm 22528 + FullThreats 60720, plus an 1858-row policy
sidecar (`.cpol`) dotted against the 128-wide bottleneck. Consequences, from
the dimension arithmetic (all EST until the script confirms):

| block | 768-era (measured share) | d47c scaling | why |
|---|---|---|---|
| threat fold `i8_deltas_into` | 16.4% (round 2 #1 function) | **×1.67 traffic**: rows 768 B → 1280 B (20 lines each); table 46.6 → **77.7 MB** (worse L2/L3 residency on top of more bytes) | `THREAT_DIMS × ACC` i8 |
| HalfKA fold `i16_deltas_into` | 9.2% | ×1.67: rows 1536 B → **2560 B**; table 34.6 → 57.7 MB | `HALFKA_DIMS × ACC` i16 |
| `activate` (i32 split-quant merge) | 5.8% | ×1.67: 1280 lanes × 2 perspectives, ~13 ops/16 lanes | unchanged math, more lanes |
| `l1_dot` | 4.0% | **×8 outputs**: L2 16 → 128 = 8 zmm blocks/group; weight bytes touched per nnz group 64 B → **512 B**; per-bucket L1 table 12.3 KB → **160 KB** (no longer L1D-resident) | `[group][output][4]` layout |
| f32 tail `propagate_l2` | small (post-opt#1) | **×8 MACs**: 16×32=512 → 128×32=**4096** FMAs/eval; if the round-2 F4 xmm-width limit still binds, this is ~1k+ instr/eval again | autovectorized, never given an explicit kernel |
| `find_nnz` | 1.4% | ×1.67 (320 groups), still runs the **AVX2** kernel on the AVX-512 box (`sota_simd.rs:731-740` — vpcompressw needs VBMI2, outside the f+bw+vnni gate; **Zen 4 HAS VBMI2**) | |
| net parse at spawn | 0.15 s measured | ×1.6: 130.5 MB per-element | |
| NEW: policy stash + consult | 0 | see §3.6 | D47 sidecar |

Two structural notes:

- **Per-eval streamed bytes roughly tripled.** EST per eval: threat rows
  ~2.67 folds × ~15 rows × 1280 B ≈ 51 KB, HalfKA ~2.5 × ≤4 × 2560 B ≈ 25 KB,
  L1 weights ~40 groups × 512 B ≈ 20 KB ⇒ **~95-100 KB/eval** vs ~35 KB at
  768/L2=16. At 506 knps × 0.34 evals/node that is ~17 GB/s of weight-row
  traffic — the eval layer is now plausibly approaching *bandwidth*-relevant,
  not just latency-exposed (round 2 measured LL misses ≈ 0 at 768; that
  conclusion must be re-verified at 1280 before rejecting int4/compression
  ideas — script §C1).
- **The head is no longer an afterthought.** At 768/16 the post-opt#1 head was
  ~17% incl. activate; the ×8 L1/L2-tail growth makes the head block EST
  25-35% of eval cycles at d47c dims. Head kernels were never hand-written —
  they are the one hot path still living on autovectorization (§4).

The measured NPS history agrees with this re-pricing: d47c at launch was
−44% vs the 768 incumbent (363k vs 647k, value-foundation-state 2026-07-22
23:00Z), ~506k after the era's opts — i.e. the width bet reintroduced
inefficiency headroom in exactly the blocks the studies had just cleaned.

---

## 3. The per-node cost model (deliverable #2)

Frame: MEASURED anchors from round-2 callgrind (still structurally valid —
the schedule is unchanged since, only dims grew): evals/node ≈ **0.34**
(TT-eval reuse + in-check skip + cutoffs), threat ply-folds ≈ **1.34/persp
per eval** (2.67 perspective-folds), HalfKA folds ≈ 2.5/eval, Finny refresh ≈
0.36/eval, ~15 avg threat features/fold, ~10-15% nnz group density. Cycle
budget: ~506 knps single-thread ≈ **~10.4k cycles/node** at ~5.3 GHz.
Shares below are EST for d47c, derived by scaling round-2 measured shares by
the §2 factors and renormalizing; the script's job is to replace this column
with perf data.

| # | stage | when paid | EST share of cycles | mechanics (file:line) |
|---|---|---|---:|---|
| 1 | **threat fold** `i8_deltas_into` | per folded ply × persp (2.67/eval) | **20-26%** | 40 blocks × (32-lane load, per-row `vpmovsxbw`+add/sub), ~19 KB rows/fold from 77.7 MB (`sota_simd.rs:580-605`, driven `sota_threats.rs:800-810`). Latency-exposed row loads, zero prefetch. |
| 2 | threat gen+resolve (lazy) | per folded ply (shared across persp) | **6-9%** | `build_deltas` replays the move against the snapshot (`sota_threats.rs:879-976`), `record_square_touch` emits ~15-30 events via magic lookups (`:257-388`), `resolve_into` maps each event through `threat_index` **per perspective** with the O(n²) cancel (`:843-873`). `threat_index` re-derefs `THREAT_TABLES` per event + dependent lut1 load (`sota_features.rs:332-346` — round 2's single hottest instruction, now at 0.46× frequency). |
| 3 | threat push (eager remnant) | **every** do_move | **~1%** | snapshot copy ~140 B + `king_crossed` scan (`sota_threats.rs:712-747`). The study-3 restructure worked: the always-paid part is now trivial. |
| 4 | HalfKA fold + Finny | per eval | **10-14%** | `apply_dirty_into` ≤4 rows × 2560 B (`sota_accumulator.rs:173-196`); `finny_refresh` on king moves: feature re-enum + sort + 2×binary-search diff + ~4 copies of 2560 B (`:113-168`). |
| 5 | `activate` (split-quant merge) | per eval | **6-9%** | i32 widen/mullo/abs/shift/mask-negate/clamp per 16 lanes × 160 iterations (`sota_simd.rs:639-692`). Exists ONLY because THREAT_QUANT(64) ≠ FT_QUANT(255). |
| 6 | `l1_dot` (VNNI sparse matmul) | per eval | **8-14%** | for blk in 8: for group in nnz(~40): broadcast + 64 B row + `vpdpbusd` (`:607-636`). **Loop order is blk-outer: the nnz list is re-walked and the 4-byte FT group re-broadcast 8×** — at L2=16 (1 block) this was free; at L2=128 it is 7 redundant broadcast+loop passes. 20 KB/eval from a 160 KB per-bucket table (L2-cache resident, bucket stable within subtrees). |
| 7 | f32 tail L2/L3 (+σ̂/WDL when present) | per eval | **4-8%** | 4096 + 512 FMAs; autovectorized — width unverified since round-2 F4 found xmm-stuck codegen at the OLD dims (`sota_head.rs:185-237`). |
| 8 | `find_nnz` | per eval | ~1% | AVX2 movemask+LUT over 320 groups on an AVX-512+VBMI2 CPU (`sota_simd.rs:387-413, 739`). |
| 9 | copies/glue | per eval | **2-4%** | by-value `activate_ft` 1280 B + `find_nnz` 640 B + `forward_full` L2 512 B (+ Box, §3.6) + Finny round-trips. |
| 10 | policy stash | per eval (when .cpol loaded) | **1-2%** | §3.6. |
| 11 | policy consult | TT-miss ∧ depth≥3 ∧ fresh-eval nodes | **2-5%** | §3.6. |
| | **eval layer total** | | **≈ 62-78%** | vs 71% measured at 768 — the width bet pushed it up |
| | search machinery | | ≈ 22-38% | picker ~10%, make/unmake ~7%, rest small (round-2 shares, unchanged code) |

### 3.1 Threat enumeration — is the delta incremental or re-enumerated? (the brief's #1 suspect)

Answer from the code: **it is incremental in the right sense, and since the
study-3 rewrite it is also lazy — the suspected "re-enumerate per make_move"
sink no longer exists.** Per make_move the observer now stores only
`DirtyPiece` + a ~140 B `ThreatSnapshot` (`apply_at`). Event generation
(`build_deltas`) replays only the ≤4 edited squares' local geometry (outgoing
attacks, x-ray ripple through the square, short-range incoming — each a
handful of magic lookups), NOT a board re-enumeration; and it runs only for
plies that reach an eval (~0.46/node), shared across perspectives. What
remains per event is: `threat_index` (LazyLock deref + 3 table loads + a
sentinel branch) × 2 perspectives + the quadratic cancel. The remaining fat
is CONSTANT-FACTOR, not schedule: (a) per-event LazyLock deref, (b) lut row
bases recomputable per (attacker,persp) once per fold instead of per event,
(c) the O(n²) cancel scan, (d) 67%-mispredicting piece-type dispatch
(round-2 measured; code unchanged). EST 20-30% of block #2 shavable ⇒ ~2% NPS.

### 3.2 Accumulator paths — lazy-update opportunities already taken, one left

`HalfkaContext::push` stores 22 B; `ThreatContext::apply_at` stores ~150 B;
**no accumulator arithmetic happens at make_move at all** — both channels
fold at eval time from the nearest computed ancestor, exactly SF's lazy
stack. Children that cut off pay ~nothing. This matches SF
(`update_accumulator_incremental`) and is *ahead* of SF in one way (fully
deferred generation). SF's remaining edge: 2-ply fused updates + backward
updates (thin here, spans 1.34) and **eager event emission fused into the
board mutators via SIMD** (`write_multiple_dirties`, vpcompressb) — cheap
generation vs our reconstructive-but-lazy generation; roughly a wash at our
fold rate. The one un-taken lazy opportunity: Finny refresh writes 4 copies
of the 2560 B accumulator through the stack/cache round-trip (`:127-167`) —
in-place cache update + direct stack write would cut ~3 copies.

### 3.3 Kernel coverage audit (the "don't recommend AVX-512" instruction)

VNNI (`vpdpbusd`) coverage: **`l1_dot` only** — which is the right place; the
folds are i16 adds of sign-extended i8 rows (VNNI-inapplicable), and that is
also correct. The int8 dots that VNNI does NOT cover today:

- **policy rows** (`policy_head.rs:90-97`): i8 rows dotted against the f32
  bottleneck via per-element `f32::from(i8) * f32` — scalar-shaped source;
  whatever autovectorization does, it is converting i8→f32 per element. Two
  fixes, either fine: an explicit f32 kernel (row i8→f32 once into a stack
  buffer, then 8 zmm FMAs), or quantize `h256` to u8 once per consult and
  dpbusd all rows (bit-inexactness vs trainer must be checked — the f32
  kernel is the safe default).
- nothing else — every other int8 consumer routes through the kernel table.

Non-VNNI coverage gaps found: `find_nnz` runs the AVX2 kernel (VBMI2
`vpcompressw` available on Zen 4 but outside the dispatch gate — the code
comments this exactly, `sota_simd.rs:737-739`); the f32 tail's width is
LLVM's choice (round-2 F4: xmm at the old dims — must re-check at 128×32,
script §C2); `activate`/folds/l1_dot are hand-written AVX-512 and healthy,
modulo the l1_dot loop-order note (§3 row 6).

### 3.4 The `activate` split-quant tax, restated at d47c dims

Unchanged since study 1 §4.5 / study 3 §4.1, but 1.67× bigger and with
reference confirmation: Reckless AND SF sum same-scale transformers with a
plain add-clamp; only ARCHON's CENN-v2 THREAT_QUANT=64 ≠ FT_QUANT=255 forces
the per-lane i32 widen+mullo+round+clamp. Collapse to the i16 `mulhi`
pairwise form requires exporting threat weights at the FT scale — a
trainer/netpack change, not bit-exact vs current nets. It should ride the
next cold-retrain format rev (d47d / CENN-v3) — NOT a hot patch.

### 3.5 Bucket-select: already branchless — reject

`OUTPUT_BUCKETS_LAYOUT[piece_count]` is one table load (`sota_head.rs:52-61,
323`); bucket is popcount-derived, stable within subtrees, and the per-bucket
weight *slices* are hoisted with `try_into` blocks. Nothing to eliminate.
(The candidate list's "bucket-select branch elimination" is REJECTED as
already-done-by-construction.)

### 3.6 The policy machinery's cost when the picker does NOT consume it

The stash (`sota_context.rs:397-406`): whenever a `.cpol` is loaded, EVERY
full eval takes `forward_full` (returns the 128-f32 bottleneck **by value**)
and then — the real smell — **`self.last_l2 = Some(Box::new(l2))`: a heap
allocation + 512 B copy per evaluated node, plus the drop of the previous
Box.** This is the only heap allocation in the entire eval hot path and
violates the crate's own no-alloc convention. It is paid at every eval'd
node, but *consumed* only at nodes where `tt_data.mv == NULL && depth >= 3`
(`search/mod.rs:2806-2815`) — a small minority. And the search **knows both
gate conditions BEFORE calling evaluate** (TT probe precedes eval), so the
stash can be requested exactly when consumable. Fix shape: inline
`last_l2: [f32; L2]` + `last_l2_valid: bool` (no Box, no alloc) + a
`want_l2` hint plumbed into `evaluate`. EST 1-2% NPS with sidecar loaded.

The consult (`policy_head.rs:104-124`): runs its own **second
`generate_legal`** (the picker will generate the same moves again moments
later), maps each move through `policy_vocab` (cheap table math), and dots
~35 rows × 128 in the scalar-shaped loop above (~4.5k MACs + i8→f32
converts), for EST 5-10k cycles per consulted node. With consult rate ~5-10%
of nodes ⇒ EST 2-5% of total cycles. Also honestly flagged: the **sb30
ON/OFF ablation measured identical Elo (−381.7 both) at 17% top-1 accuracy**
— the sidecar currently buys zero ordering Elo, so its entire runtime cost
is, today, pure NPS tax. That is fine as a capability investment (accuracy
climbing, downside capped by design) but the cost side should be made ~free
(above) so "keep it live while it learns" stays rational.

### 3.7 What is healthy (do not touch)

Laziness discipline end-to-end (0.34 evals/node; folds only on evaluated
paths); TT probe/prefetch (0.3-0.4%); movegen (faster than SF16's); the
threat push remnant (~1%); loader validation; the parity-test lattice
(`threats_replay_*`, backend-vs-scalar, oracle `assert_matches_refresh`) —
every §6 item below is testable against existing gates.

---

## 4. SF cross-reference (updated from study 3 §1 with "ARCHON now")

| stage | SF18 | ARCHON (post-lazy-observer, d47c) | verdict |
|---|---|---|---|
| raw event generation | eager, fused into board mutators, SIMD-batched (vpcompressb/vpermb), u32-packed events | **lazy**, reconstructive vs ~140 B snapshot, scalar per-event pushes, 8 B events | schedule now BETTER than SF (0.46× vs 1×); constant factor worse — SIMD emission over the snapshot (study 3 §2.4.2) is the remaining idea, MED effort |
| per-perspective resolution | lazy at eval | **lazy at eval** ✓ | parity reached (study 3 #1) |
| cancellation | none in-ply; 2-ply fused cross-ply | eager O(n²) per ply | open: SF's "just fold both ways, wrapping adds cancel" is *simpler and branch-free* — worth testing as the cancel-scan replacement (drop the scan entirely, let ±rows cancel arithmetically; costs 2 extra row-folds per cancelled pair but deletes the quadratic scan + its mispredicts; net win depends on cancel frequency — script can count) |
| fold kernels | fused, register-tiled | fused (opt#2) ✓, no prefetch | prefetch is ours to take (§6 #3) |
| accumulator schedule | lazy stack + backward + 2-ply fused | lazy stack ✓, forward-only | thin remainder |
| dual-net routing | small L1-128 net on `|simpleEval|`-lopsided nodes, big net escalation on disagreement — **a large share of evals never touch the big net** | infra shipped, env-off; **no cheap net trained** | THE schedule gap that remains (§5) |
| FT merge | same-scale add-clamp, i16 mulhi | i32 split-quant merge | format rev (§6 #6) |
| tail | int8/int32 throughout | f32 (fine per Reckless precedent) — but unverified vector width at new dims | script §C2 |

---

## 5. T10 smallnet companion — the NPS yield, quantified (the brief's ask)

Model (all inputs labeled): let p = fraction of *evaluated* nodes whose
**simple_eval** (material arithmetic; PAWN_VALUE=208, so a "1000-class"
threshold ≈ 4.8 pawns ≈ rook-plus) clears the gate; c = cheap-eval cost as a
fraction of big-eval cost; E = eval-attributable share of node cycles that a
cheap-gated node avoids (big head + its share of lazy folds — folds for
never-big-evaluated spans never run; EST E ≈ 0.65-0.75 from §3).

```
NPS gain ≈ 1 / (1 − p·(1−c)·E) − 1        (material prefilter ⇒ non-gated nodes pay ~0)
```

**c:** a TRUE T10 companion (L1-128, HalfKA-only — SF's shape) ⇒ c ≈
0.10-0.15 (256 B fold rows, 128-lane activate on pst only, tiny head). The
no-trainer-change fallback (HalfKA-only at ACC=1280) ⇒ c ≈ 0.5 — the
smallnet-gate doc's cost-model caveat already covers why that only works
behind the material prefilter. Note the runtime today types accumulators at
`[[i16; ACC]; 2]` — a 128-wide companion needs the generic-L1/second-context
change (or the JAX lane trains it and a small dedicated context ships it);
this is the real engineering cost of the true-tiny variant.

**p — the honest part.** No in-tree measurement exists (grep-verified: no
counter, no logged distribution). Analytic band from what IS known:

- Game-position anchors: SF18 relabels of our game positions have mean |cp|
  147 (`archon-loss-profile-2026-07-20.md`); UHO 10+0.1 games are decisive
  ~65-80%; decided games spend a long conversion tail at rook-plus material.
  Game-weighted fraction of *positions* at |material| ≥ rook-plus: EST 20-35%.
- Search-tree correction: trees rooted at balanced positions still gate on
  transient qsearch material swings (captured-not-yet-recaptured states are
  heavily pruned by SEE, so this adds little); trees rooted in conversion
  phases gate at ~80-95%. Eval-weighted mix at 10+0.1: **EST p ≈ 0.25-0.40,
  central 0.30; balanced-opening floor 0.08-0.15.**

Yield table (E = 0.70):

| scenario | p | c | NPS gain |
|---|---|---|---|
| true tiny net, central p | 0.30 | 0.12 | **+22%** |
| true tiny net, conservative | 0.20 | 0.15 | +13% |
| true tiny net, favorable | 0.40 | 0.10 | +34% |
| fallback 1280 HalfKA-only | 0.30 | 0.50 | +12% |
| fallback, conservative | 0.20 | 0.55 | +7% |

So: **T10 is worth ≈ +7-34% NPS, central ≈ +20% with the true L1-128
companion** — the single largest eval-layer lever, matching the brief's
prior. p is the dominant uncertainty and is cheap to measure exactly: the
script includes (a) an offline material-imbalance histogram over canonical
binpack samples (game-weighted lower bound) and (b) a 25-line optional
counter patch (`ARCHON_SMALLNET_STATS`) for the true in-tree rate — run both
before training anything, per the smallnet-gate doc's own pre-registered
"clean measurement first" gate. Elo risk is NOT in the NPS (bit-identical
big path when it fires correctly) but in wrong routing; the T10
sacrifice-suite gate + output-disagreement fallback own that.

**Composition:** T10 multiplies with every big-path kernel win (it reduces
big-eval *frequency*; §6 items 2-5 reduce big-eval *cost*); it makes
threat-side items slightly less valuable at the margin (fewer threat folds)
and policy/search-side items untouched.

---

## 6. Ranked opportunity list (deliverable #3)

Elo rule: measured slope 1-2.1 internal Elo per +1% NPS at 10+0.1 (D39 house
rule; opt#1 measured 2.1). "BIT-ID" = bench-node-identical, verifiable by
node-count identity + existing parity gates.

| # | opportunity | EST NPS | effort | risk | composes with T10 |
|---|---|---|---|---|---|
| **1** | **T10 smallnet companion** — train the L1-128 HalfKA-only slice-distilled net (JAX job), sweep `ARCHON_SMALLNET_THRESH`/`MAT_GATE`, sacrifice-suite gate. Runtime infra already shipped. Prereq: measure p (script §C4) + generic-L1 context for the 128-wide net. | **+7-34% (central +20)** | HIGH (training + narrow-context change; infra done) | MED (routing Elo; gates pre-registered) | IS the base |
| **2** | **Head kernel package for the ×8-wider head**: (a) explicit zmm kernel for `propagate_l2` (128×32: broadcast-l1[i], 2 zmm FMA accumulators — j-lane order preserved ⇒ BIT-ID) and the L1 dequant loop; (b) `l1_dot` loop-order swap to group-outer holding 8 zmm accumulators (kills 7 redundant nnz-walks + re-broadcasts per eval; BIT-ID — i32 adds commute… they associate exactly, order irrelevant for int) + `_mm_prefetch` of the next group's 512 B; (c) static/cfg dispatch for the pinned fast build (kills the LazyLock check + indirect call at ~6 sites/eval, unlocks inlining; Reckless pattern, design/11 §3.4.6 escape hatch); (d) VBMI2 `vpcompressw` find_nnz. | **+6-12%** | LOW-MED (2-3 days, parity tests exist) | LOW | multiplies |
| **3** | **Resolve→prefetch→fold pipeline + span merge in `ThreatContext::compute`**: resolve BOTH perspectives (and all span plies) first, issuing prefetches for every add/sub row (each row = 20 lines over 77.7 MB — the #1 D1-miss source measured at 768, bigger now); fold after. Fold in: hoist the `THREAT_TABLES` deref + per-(attacker,persp) lut bases out of the event loop; replace the O(n²) cancel (or drop cancellation entirely, SF-style — wrapping folds cancel arithmetically; measure which wins); alias the null-move copy-fold. Structurally unlocked by the landed lazy observer. | **+4-8%** | MED (contained in one function family) | LOW (threats_replay + oracle) | slightly sub-additive (fewer folds) |
| **4** | **Policy hygiene** (only matters while a .cpol ships): stash without Box (inline array + valid flag), stash only when `tt-miss ∧ depth≥3` (search knows before eval), vectorized `row_score`, drop the second `generate_legal` (score from the picker's list or restructure the consult). Today the sidecar buys 0 Elo (sb30 ablation) — its cost should be ~0 too. | **+2-5% (with sidecar)** | LOW (1 day) | LOW | orthogonal |
| **5** | **Residual copy elimination**: out-params for `activate_ft` (1280 B) / `find_nnz` (640 B) / `forward_full` L2 (512 B — verify LLVM hasn't already elided the `forward(...).0` discard before spending effort); `finny_refresh` in-place (4 → 1 copies of 2560 B). BIT-ID plumbing. | **+1.5-3%** | LOW | LOW | multiplies |
| **6** | **Quant unification** (CENN-v3, ride the next cold retrain): `activate` collapses to i16 add-clamp-mulhi; at 1280 lanes the tax is 1.67× the measured 5.8%. | **+3-5%** | MED-HIGH (trainer+netpack+retrain; NOT bit-exact) | MED | big-path only (T10 shrinks its base) |
| **7** | **Bulk net parse** (`sota_loader.rs:592-623` → chunked `from_le_bytes`/`bytemuck`-style conversion): 130.5 MB per-element ⇒ EST 0.3-0.5 s per spawn, paid by every SPRT/tournament game spawn and every isready-sensitive harness. | startup −0.3-0.5 s (QoL, de-noises short profiles) | TRIVIAL | none | — |
| **8** | **THP/hugepage audit on the Arch box** (new environment; ~135 MB of weight tables + TT): check `/sys/.../transparent_hugepage`, dTLB counters; if `madvise`, add `madvise(MADV_HUGEPAGE)` on the big allocations. | +0-3% | TRIVIAL-LOW | none | multiplies |
| 9 | SIMD event emission over the snapshot (SF ICL pattern; Zen 4 has VBMI2) — only after #3, if gen+resolve still shows ≥5% | +1-3% | MED (unsafe SIMD) | MED (parity-gated) | sub-additive with #3, T10 |
| — | **REJECTED: int4 FT quantization.** Halves fold bytes but: round-2 measured the 768-path latency- not bandwidth-bound (LL≈0.1%); THREAT_QUANT=64 already uses only ~±64 of i8's range wait-no — the quant scale ≠ dynamic range, but 16 levels for the table the SVD probe showed uses its full row space is a real quality risk; unpack adds ~2 ops per 32 lanes. Revisit ONLY if script §C1 shows the 1280 path has gone genuinely bandwidth-bound (LLC misses ≫ 768-era). | | | | |
| — | **REJECTED: separate eval-cache layer.** TT-eval reuse already holds evals at 0.34/node (measured); SF removed its standalone eval hash years ago; a second cache buys re-search hits the TT already serves. | | | | |
| — | **REJECTED: bucket-select branch elimination** — already a branchless table load (§3.5). | | | | |
| — | **REJECTED (as an eval-layer item): lazy/deferred accumulator updates** — already fully implemented on both channels; push is O(1)+140 B (§3.2). Backward/2-ply-fused updates stay parked (spans 1.34). | | | | |
| — | **DEFERRED: T8 king-split accumulator asymmetry** — a training-side architecture bet (arch-ablation tier 3), not an inference fix; its NPS logic (never-refreshing wide half) is sound but it needs a retrain campaign; sequence behind T10 which buys more for less risk. | | | | |

**Package view:** engineering-only items 2+3+4+5 (+8 if it fires) ≈
**+13-25% NPS**, all bit-identical, ~1 week; T10 on top multiplies to
**≈ +25-55% total** ⇒ 506k → ~630-780k NPS. That recovers most of d47c's
measured −44% width tax (≈58 Elo equal-time) without touching the net, and
by the measured slope is worth ~+25-80 internal Elo. The remaining gap to
SF18's 1.03M after all of it is real architecture: 1280-vs-1024 width,
all-int tail, fused mutator-time event emission, and SF's decade of
constant-factor polish — i.e. after this package, further speed Elo is
better bought in the trainer (distill-shrink, T3 width re-sweep) than in
kernels.

Sequencing: script first (quiet window, ~30 min) → #2+#5 (pure wins, verify
vs script baseline) → #4 → #3 → T10 measurement gate → T10 training →
#6/#7/#8 opportunistically.

---

## 7. What `tools/perf_study4_bench.sh` will settle (deliverable #4)

| claim | test | decides |
|---|---|---|
| C1: threat channel ≈ 26-35% of cycles at d47c dims and latency-(or now bandwidth-)bound | perf record/stat on depth-12 bench, d47c vs 768 incumbent net (same binary): shares of `i8_deltas_into`/`build_deltas`+`resolve`/`i16_deltas_into`; LLC + dTLB counters | item #3 priority; int4 stays rejected unless LLC misses exploded |
| C2: the head became co-equal (l1_dot+activate+tail ≈ 20-30%) and the f32 tail is under-vectorized | perf shares + objdump audit: FMA register-class census in `evaluate`, vpdpbusd count, LazyLock check pattern, absence of vpcompressw | item #2 scope + expected yield |
| C3: policy machinery costs 3-7% with the sidecar loaded (stash alloc + consult) | A/B bench with `.cpol` present vs renamed away (report NPS, nodes, malloc/free + `row_score`/`generate_legal` perf shares — trees differ, so attribution is by profile share, not NPS delta) | item #4 priority; whether to hide the sidecar until accuracy justifies cost |
| C4: T10's p ∈ 0.25-0.40 | (a) offline material-imbalance histogram over canon binpack samples at thresholds 400/600/800/1000; (b) OPTIONAL 25-line `ARCHON_SMALLNET_STATS` counter patch (described in the script header) for the true in-tree rate on bench + 4 game positions | T10 go/no-go + threshold starting point + true-tiny vs fallback economics |
| bonus | clean NPS ladder re-baseline (d47c ± cpol, incumbent-768, PSQT, SF16/SF18 anchors), pinned median-of-3; THP status; SVD re-probe of the 1280 threat table | replaces this study's EST column with numbers; seeds the NNUE section of `docs/perf-baseline.md` |

---

## Appendix: key file/line evidence index (main tree, 2026-07-22)

- Lazy observer: `src/eval/nnue/sota_threats.rs:36-83` (docs), `:151-199`
  (snapshot), `:712-747` (apply_at), `:758-813` (compute), `:843-873`
  (resolve_into cancel scan).
- Fold kernels, no prefetch, LazyLock dispatch: `src/eval/nnue/sota_simd.rs:41-58`
  (dispatch), `:580-605` (i8 into), `:607-636` (l1_dot, blk-outer order),
  `:639-692` (activate i32 merge), `:731-740` (avx512 reuses avx2 find_nnz).
- Head dims + tail: `src/eval/nnue/sota_head.rs:32-38` (L1=1280, L2=128,
  L3=32), `:109-113`/`:146-150` (by-value returns), `:156-174` (l1),
  `:185-207` (L2 tail), `:301-309` (`forward` = `forward_full(...).0`).
- HalfKA/Finny: `src/eval/nnue/sota_accumulator.rs:61-64` (O(1) push),
  `:113-168` (finny_refresh copies), `:173-196` (fused fold).
- Policy: `src/eval/nnue/sota_context.rs:397-406` (Box stash),
  `src/eval/nnue/policy_head.rs:88-124` (scalar row dot + second movegen),
  `src/search/mod.rs:2799-2817` (consult gate: tt-miss ∧ depth≥3).
- Smallnet/largenet infra: `src/eval/nnue/sota_context.rs:36-51, 90-124,
  374-438`; env registry `src/eval/nnue/mod.rs:410-463`.
- Loader per-element parse: `src/eval/nnue/sota_loader.rs:592-623`.
- threat_index LazyLock-per-event: `src/eval/nnue/sota_features.rs:324-348`.
- Net artifact: `data/nnue/nets/d47c-sb*.cenn` = 136,843,872 B; `.cpol` =
  245,268 B (= 12 + 1858·128 + 1858·4 ✓).

---

## Appendix B — IMPLEMENTATION RESULTS (packages 2-5, 2026-07-23)

Implemented on branch `worktree-agent-a7e7069d489335126` (baseline sync
`32e0a9c` = ee12507 + the uncommitted d47c tree this study modeled). All
four packages landed **bit-identical**: nnue-eval byte-parity on 800 FENs
(cp + policy-argmax columns) for d47c-stage2-sb240 AND d47c-sb240, on both
the x86-64-v3 and native builds; depth-12 bench-with-net node fingerprint
4297837 unchanged; default depth-6 bench 701605 unchanged; package 4
additionally passed a 100-game fixed-node (5k) PGN-identity oracle with the
.cpol live (old-vs-old reference match == old-vs-new match, 100/100
identical move sequences). `d40-uho-v1-240sb` is NOT loadable at d47c dims
(loader: acc_width 768 != 1280), so the two d47c nets are the full loadable
parity set.

NPS method: `bench 16 1 12 default depth <net>` (net + .cpol live),
native build, adjacent-pair interleaved A/B (before/after alternating per
round) so box drift cancels. Study #5's pinned-workload re-measurement of
the first two commits is the cross-check (its §11).

| package | commit | landed | dropped (with evidence) | paired NPS |
|---|---|---|---|---|
| 5 copy elim | `9b695fb` | out-params (activate_ft/find_nnz/propagate_l1), forward-discard fix, Finny in-place | — | ~neutral in my contended window; **+3.7% on study #5's pinned re-measure** (my claim was undersold) |
| 3 threats | `a3f892e` | ThreatResolver (LazyLock/orientation hoist), cancel-scan DELETED, resolve-all-then-fold, null-move alias (both stacks), empty-span copy | **prefetch: measured a LOSS** (full-row −2.5%, 2-line −0.4%); span merge (spans 1.34, forfeits sibling memoization) | +2.5% my A/B; **+0.3%±1 on study #5's workload** — treat as ~+0-2%, workload-dependent |
| 2 head kernels | `b46c026` | l1_dot group-outer (8 zmm accs), explicit l1_dequant + l2_tail kernels (scalar/avx2+fma/avx512), static dispatch on compile-time AVX-512, VBMI2 find_nnz | l1_dot next-group prefetch (−0.3%, L2-resident table) | **+21.4%** (397.3k → 482.1k) |
| 4 policy | `3555f57` | stash de-Box (inline array + valid flag), vectorized policy_dot (fixed 16-lane+tree shape) | **consultable-node stash gating BACKED OUT** (below); second-generate_legal removal (consult = 1.53% cycles total, study #5 §5 — not worth a MovePicker restructure) | +2.2% (477.8k → 488.1k) |

**Total: 383.7k → 495.3k on the shared-conditions final A/B = +29%** vs the
same-source baseline binary (5 adjacent pairs, all +28.5..29.1%). Absolute
quiet-box ladder (native, bench-d12 median-of-3): PSQT 3.71M; stage2+cpol
495k; stage2 no-cpol 540k (the sidecar tax ≈ 8-9% on this workload);
stage1+cpol 534k.

### Findings that change this study's ranking

1. **§3.6's central premise is REFUTED: the consult gate is NOT knowable at
   stash time.** Two mechanisms, both measured on the bench fingerprint:
   (a) the hindsight adjustments mutate `depth` between the step-6 eval and
   the step-9 consult (a depth-conditional hint moved the tree 4.30M →
   5.25M nodes); (b) the consult regularly consumes stashes left by OTHER
   evals of the same position — razoring/qsearch stand-pat at the node,
   TT-eval-served re-visits — so even a tt-move-only hint is functional
   (4.30M → 2.54M). Any conditional stash is a search change and must be
   Elo-tested, not slipped into a "bit-identical" package. The de-Box half
   stands alone and is free.
2. **Software prefetch lost everywhere it was tried on this box** (threat
   rows: full-row −2.5%, 2-line −0.4%; l1_dot next-group −0.3%). Zen 4's
   HW prefetchers + OoO window already cover these access patterns.
   Item #3's headline mechanism and item #2(b)'s prefetch rider are dead;
   what survived of package 3 was the constant-factor hoists (~+0-2%).
3. **Package 2 was under-priced ~2× (+6-12% claimed, +21% measured)** —
   consistent with study #5's measured 48.5%-of-cycles head and its mca
   3.0× l1_dot bound. The gain needs NO static dispatch: the v3 build
   (runtime-dispatched) gained ~+24% too; the loop-order swap + explicit
   f32 tail kernels dominate.
4. The cancel scan almost never cancels: **0.48% of events** (106.5M events
   → 0.51M cancels on bench-with-net). SF-style fold-both-ways is strictly
   better here; the O(n²) scan is deleted.
5. Study #5 (measured, `perf-study-5-measured-2026-07-23.md`) supersedes
   this study's §3 cost model: head 48.5% / threats 12.4% / policy ~2-3.5%
   +6.2% tree. Its §11 independently confirmed bit-identity of the first
   two commits and re-priced them (+3.7% / +0.3%).

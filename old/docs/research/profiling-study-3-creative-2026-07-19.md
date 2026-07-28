# ARCHON profiling study, round 3 — creative/structural: beyond the micro-opt tail

Date: 2026-07-19. Rounds 1–2 (`profiling-study-2026-07-18.md`,
`profiling-study-2026-07-19.md`) delivered the shipped +20.7% NPS package
(opt1 vector head + opt2 fused folds) and a ranked micro-opt tail. This round
answers a different question, per the owner's brief: **what restructures —
not tunes — the remaining cost?** The engine sits at 16.0k instr/node vs
SF18's 7.5k (2.1×) with the sota eval at ~71% of cycles. Study only; no
engine code was modified.

**Headline:** the biggest single lever left is not in the ranked tail at all.
A source-level study of SF18's threat machinery (`engines/stockfish-18-src`)
shows the round-1 premise that the threat observer "must run at push time" is
**false**: SF only *generates raw events* eagerly and defers all resolution to
eval time — and ARCHON can go one step further than SF and defer *everything*,
because the observer's inputs (post-move occupancy + piece identities + a
1-bit king mirror) are trivially snapshottable at make time for ~150 bytes.
Only ~46% of pushed plies are ever folded (measured, round-2 callgrind), so a
**fully-lazy observer** deletes ~54% of the 19.9% observer block at bit-identical
eval: **≈ +11% NPS on its own**, low risk, no net change, existing parity gates.
With its unlocked follow-ons (resolve→prefetch→fold pipelining, SIMD event
emission on the SF ICL pattern, cross-ply cancellation) the round adds an
estimated **+18–25% NPS** on top of opt1+2 — roughly closing the cycles/node
gap to SF18-class. Separately, a measured **SVD probe of the incumbent's
threat table kills the low-rank-factorization bet decisively** (rank-256
captures only 51% of energy): a valuable negative that redirects the
training-side speed bets toward quant unification and an SF-style smallnet
gate instead.

---

## 0. Methodology

- Ops constraint honored: an SF16 equal-time fastchess match ran during the
  first half of this study — that phase was analysis-only (code + SF18/Reckless
  source and notes + rounds 1–2 data). After the match ended (`pgrep -x
  fastchess` clean, load fallen, no other heavy jobs), three cheap targeted
  measurements were taken; nothing heavier than a 5 s bench was run.
- **Measured this round (quiet box, pinned core 14):**
  - Re-baseline of the shipped `data/nnue/experiments/opt2-equaltime/sota-opt2`:
    depth-12 bench nodes **3,515,701 (bit-identical to rounds 1–2)**, NPS
    824k/838k/855k (median 838k, ~2% under round-2's 859k — small residual
    load; treat round-2 absolutes as the clean reference). perf stat: 57.534G
    instructions (**identical to round 2**), IPC 2.45, br-miss 1.96%.
  - **SVD rank probe** of the incumbent net's two feature tables (§4) —
    `scratchpad/svd_probe.py`, gram-matrix eigendecomposition of the exact
    on-disk weights of `d40-uho-720sb.cenn`.
  - THP status: `/sys/kernel/mm/transparent_hugepage/enabled` = **[always]**
    → the 81 MB heap-allocated net already gets huge pages opportunistically;
    round-2 item 8 (huge pages) is demoted further.
- **Reasoned from prior measurements** (labeled as such): all NPS projections.
  They reuse round-2's profile shares (valid: same binary, same node count,
  instruction count reproduced exactly this round) and round-2's callgrind
  call counts.
- Reference-engine study: SF18 source read directly (GPLv3 — ideas only,
  nothing copied); Reckless via the existing distilled notes
  (`reckless_reference/11-nnue-features-threats.md`). Line numbers cited.

---

## 1. Why SF18 is 2× cheaper per node: it's the *schedule*, not the kernels

Round 2 left "the remaining 2× instr/node vs SF18" attributed loosely to
"fold volume, always-paid observer, i32 activate, head depth." Reading SF18's
actual threat implementation sharpens this decisively. SF18 runs the *same*
60,720-dim FullThreats feature set (ours is a reimplementation of it), yet
its whole make-to-eval pipeline is scheduled differently:

| stage | SF18 | ARCHON today |
|---|---|---|
| raw event generation | **eager**, fused into `put_piece`/`remove_piece`/`move_piece` during `do_move` (`position.cpp:1112-1216`) against the *live in-flight board* — no reconstruction; events packed to **u32** (`types.h:295-318`); on AVX512ICL emitted in vector batches (`write_multiple_dirties`, `position.cpp:1071-1107`: `vpcompressb` square extraction + `vpermb` over the 64-byte piece array + one unmasked 64 B store of up to 16 events) | **eager**, but *reconstructive*: `build_deltas` rebuilds pre-move occupancy + an override map from the post-move board and replays edits (`sota_threats.rs:800-897`), per-event scalar `push` through a 67%-mispredicting piece-type dispatch |
| per-perspective resolution (`make_index`/`threat_index` + exclusion) | **lazy** — `append_changed_indices` runs inside the accumulator update walk at eval time only (`nnue_accumulator.cpp:553-585`) | **eager** — `resolve_entry` × 2 perspectives at every push, including the ~54% of pushes never folded (`sota_threats.rs:762-794`) |
| add/sub cancellation | none — over-generated pairs just fold both ways (wrapping adds cancel arithmetically); only the 2-ply *fused* update cancels across plies via `FusedUpdateData` (`full_threats.cpp:293-321`) | eager O(n²) cancel scan per perspective per push |
| fold | lazy, register-tiled | lazy, fused (opt #2) — equivalent |
| extras | **backward updates** (fold from a computed deeper child toward the parent, `update_accumulator_incremental<Forward=false>`) and **2-ply fused folds** | forward-only, 1-ply-at-a-time |

Two more reference facts that matter:

- SF's `threatenedSqs`/`threateningSqs` byproduct bitboards are consumed
  **only** by the accumulator-update gating itself (`nnue_accumulator.cpp:227`)
  — SF's search does *not* read them. There is no hidden coupling that forces
  eager threat work in a chess engine; SF keeps generation eager only because
  theirs is nearly free (fused into mutators it already runs, SIMD-batched).
- Reckless sums `pst + threat` accumulators **at one shared quantization
  scale** — a plain add-clamp before the pairwise multiply (notes §5). The
  i32 widen+rescale `activate` tax (5.8% of our runtime) is unique to ARCHON's
  CENN-v2 split-scale format (`THREAT_QUANT=64` vs `FT_QUANT=255`), not to the
  architecture. Round-2 item 6 (quant unification) is thereby confirmed as a
  pure format decision with reference-engine precedent on both sides.

**The gap decomposition, restated:** ARCHON pays ~2.2k Ir/node of observer at
every node where SF pays a few hundred (cheap generation only); ARCHON pays
resolution 2× per node where SF pays it ~0.46× (lazy); plus the known i32
activate tax, split-scale format, and Rust-side constant factors (LazyLock
deref per event at `sota_features.rs:332`, bounds-checked LUTs vs SF's
constexpr arrays). None of the 2× is "SF's SIMD is better" — round 2 already
established the kernels are peer-quality.

---

## 2. THE HEADLINE RESTRUCTURE (bold bet #1): the fully-lazy threat observer

### 2.1 The insight chain

The round-1/2 studies treated the observer's schedule as *forced*, quoting the
module docs: resolution "must run at push time because it reads the post-move
board, which siblings destroy in place" (`sota_threats.rs:36-46`). Taking the
structure apart shows every dependency is deferrable:

1. **Resolution needs almost nothing.** `threat_index(persp, attacker, from,
   to, victim, ksq)` depends on the king square **only through
   `orient_threats(ksq)` — a single mirror bit** (file ≤ d),
   `sota_features.rs:324-348`. Not the board.
2. **Event generation needs only occupancy + piece identities.**
   `build_deltas`/`record_square_touch` read exactly two things from the
   post-move board: `all_occupancy()` and `piece_on(sq)`
   (`sota_threats.rs:144-189`). That is one `u64` plus the 64-entry mailbox.
3. **At fold time, every ply being folded is an ancestor of the current
   node.** `ThreatContext::compute` walks from the nearest computed ancestor
   forward to the evaluated ply (`sota_threats.rs:700-732`). Sibling
   destruction is irrelevant to *those* plies — the sibling overwrote the
   stack entry, not history that the fold needs; what the fold needs is the
   post-move board *of each folded ply*, which is perfectly reconstructible.
4. **Reconstruction is a snapshot copy.** Board mailbox is
   `[Option<Piece>; 64]` (`board.rs:414-415`) — with occupancy + both king
   squares, ~**150 bytes**. Copy it at push; at fold time, run today's
   `build_deltas` against the snapshot view instead of `&Board`. (Cheaper
   variant, dp-chain backward unwind from the live board, works too but the
   snapshot is simpler and O(1) regardless of span.)

There is even in-repo precedent: **HalfKA already does exactly this** — its
push stores only the `DirtyPiece` and re-resolves feature indices at fold time
against the *current* board's king square (`sota_accumulator.rs:61-110`),
relying on king-move spans triggering a refresh. The threat stack simply never
got the same treatment because its inputs looked board-shaped. They aren't:
they're 150 bytes.

### 2.2 What it saves (measured inputs, reasoned total)

- Executions: pushes = 1.0/node; distinct folded plies = 0.454/node (round-2
  callgrind: 134,199 `i8_deltas_into` ÷ 2 perspectives = 67.1k ply-folds vs
  147,691 nodes). Both generation (1×/push today) and resolution (2×/push
  today) drop to fold-time frequency: **×0.454**.
- The block being rescheduled is 19.9% of cycles (round-2 §2: `push_sota_dirty`
  self 15.12% + `record_square_touch` 3.08% + `record_outgoing` 1.01% +
  `ripple` 0.67%).
- New push cost: ~150 B copy + `king_crossed` check ≈ trivially small
  (~0.3-0.5% of a node); the rare king-cross eager refresh (0.06/eval) stays
  at push where the live board is available, exactly as today.
- Net: 19.9% × (1 − 0.454) ≈ **10.9% of cycles removed** → **≈ +11–12% NPS**,
  before any constant-factor work on the observer itself.
- Memory: the per-ply entry *shrinks*. Today `ThreatEntry` stores resolved
  `adds`/`subs` `[u16; 256] × 2 × 2` = 2 KB + 3 KB acc ≈ 5.1 KB × 254 plies
  ≈ 1.3 MB/thread. Lazy: acc 3 KB + snapshot ~0.15 KB (the resolved lists
  disappear entirely — events are generated straight into the existing scratch
  `DeltaBuf` at fold time) ≈ 3.2 KB → **~0.8 MB/thread** and far fewer bytes
  written per push.

### 2.3 Why the risk is low

- **Bit-identical by construction**: same events, same resolution, same fold —
  executed later and less often. Wrapping i16 addition is associative and
  commutative, so even reordering/cancellation changes nothing. Verifiable by
  bench-node identity (3,515,701) + the existing `threats_replay_*` gates +
  the debug `assert_matches_refresh` oracle at every evaluated node.
- The API motion is contained: `ThreatContext::apply_at` stops calling
  `build_deltas`/`resolve_entry` and instead stores the snapshot;
  `compute` gains the generation+resolution step per folded ply. `Replay`
  becomes a view over the *owned, mutable* snapshot — which also deletes the
  override-map linear searches (`Replay::find`, `sota_threats.rs:156-189`):
  apply the ≤4 edits directly to the snapshot copy.
- Effort: ~1–2 days including parity runs. This is the best Elo-per-effort
  item found in three rounds: at the D39 slope band (house rule ~1 Elo per 1%
  NPS; opt #1's measured slope 2.1/1%), +11% NPS ≈ **+11–23 internal Elo** at
  10+0.1, from a refactor that cannot change a single eval.

### 2.4 Follow-ons the restructure unlocks (each impossible or pointless today)

1. **Resolve→prefetch→fold software pipeline** (fixes the "prefetch measured
   null" result). The simple fold-entry prefetch had no distance — resolved
   indices were consumed within a few instructions. Lazily, `compute` can
   two-phase: resolve *all* plies × both perspectives of the span first,
   issuing `_mm_prefetch` for every add/sub weight row as indices emerge, then
   fold. Resolution work (~hundreds of cycles) becomes the prefetch distance
   for the fold's row streams — which are the #1 remaining cycle sink
   (`i8_deltas_into` 16.4%, 27.4% of D1 read misses, samples piled on the
   load-consuming `vpaddw`). Same trick covers the HalfKA fold (9.2%).
   Reasoned: **+3–5% NPS**.
2. **SIMD event emission** (the SF ICL pattern, portable: Zen 4 has
   AVX-512VBMI2). With a materialized 64 B mailbox snapshot, batch-emit
   events per attack mask: compress square list, gather piece codes by byte
   permute over the mailbox vector, shift-or into packed u32 events, one
   store. Kills the per-event scalar loop and the observer's branch pathology
   (44% of all conditional + 40% of all indirect mispredicts) at its root.
   Applied to the ×0.454 residual block: **+2–3% NPS**, MED effort.
3. **Cross-ply cancellation on multi-ply spans** (SF's `FusedUpdateData`
   analogue, but simpler: since resolution now happens for the whole span at
   once, merge the span's add/sub multisets before folding — bit-identical by
   wrapping-add associativity). Saves 2×768 B of fold traffic per cancelled
   pair; multi-ply spans are ~25% of folds. **+1–2% NPS**, small effort, do
   it inside the same rewrite.
4. **Null-move aliasing for free**: a null move's entry is `dp=EMPTY` +
   snapshot; `compute` can alias parent→child (skip the degenerate 1.5 KB
   copy-fold — 23,128 pure-copy kernel calls at d8). Subsumes half of round-2
   item 3.
5. **Packed u32 events** (SF's `DirtyThreat` shape) if any event list is ever
   stored again; today's 8-byte `ThreatDelta` costs double the cache traffic
   in the scratch buffer. Minor; fold into the rewrite.

Also folded in here: round-2 item 2's constant-factor list (hoist the
`THREAT_TABLES` LazyLock deref out of the per-event path, precompute lut row
bases) — do it as part of the fold-time resolve recode rather than as a
standalone patch.

---

## 3. Measured negative result (bold bet KILLED cheaply): the threat table is not low-rank

The most tempting "restructure the data, not the code" bet was a low-rank
factorization of the 60,720×768 i8 threat table (46.6 MB, the dominant D1/LL
miss source): `T ≈ U·V` with U:60720×r, V:r×768 would shrink the streamed
table to r-byte rows and turn the fold into cache-resident r-lane updates plus
one r×768 expansion per eval. Before proposing training weeks, this round
measured the incumbent's actual spectrum (`svd_probe.py`, exact on-disk
weights of `d40-uho-720sb.cenn`, gram-matrix eigendecomposition, Frobenius
cross-check exact):

| rank r | energy captured | residual RMS (weight units) | residual ÷ i8 quant noise (0.289) |
|---:|---:|---:|---:|
| 32 | 13.9% | 4.26 | 14.8× |
| 64 | 20.8% | 4.09 | 14.2× |
| 128 | 32.1% | 3.78 | 13.1× |
| 256 | 50.9% | 3.22 | 11.2× |
| 512 | 80.1% | 2.05 | 7.1× |

(HalfKA for context: rank-16 already holds 35.8% — the king-bucket replication
gives it a fat head — but its tail is similarly heavy: rank-256 = 72.7%.)

**Verdict: dead.** The trained threat table uses essentially its full
768-dimensional row space with a nearly flat spectrum; even rank-512 leaves
7× the quantization noise floor. Post-hoc factorization would destroy the
function. Corollaries:

- The 46.6 MB **earns its bytes** — consistent with D40's +30/node eval edge.
  The speed answer to the table is *latency hiding* (§2.4.1) and *touching it
  less often* (§2), not shrinking it.
- A trained-from-scratch bottleneck could in principle land elsewhere in
  function space, but the incumbent's flat spectrum is direct evidence the
  loss surface wants full width. Any "narrow threat channel" experiment
  (§5.3) should be priced with low prior odds now.
- This measurement cost ~3 minutes. Every future arch-shape speed bet with a
  linear-algebra core should get the same treatment before training spend
  (the probe script is reusable).

---

## 4. Remaining structural bets on the net format (training/export side)

These require a CENN-v3 rev and are **not bit-identical**; bundle them into
one format change with one retrain/re-export campaign.

1. **Quant unification** (round-2 item 6, now reference-confirmed §1): export
   threats at the FT scale; `activate` collapses from the i32 widen+rescale
   merge (5.8% of cycles) to the i16 add-clamp-`mulhi` form all three
   reference engines use. **+2–3% NPS.** Trainer/netpack change; the maths is
   a training-time constraint, not a load-time rescale (rounding applies to
   summed activations).
2. **Smallnet gate (bold bet #2 — SF-proven, infrastructure half-built).**
   SF ships dual nets and routes lopsided positions to a small HalfKA-only
   net; a large fraction of real-tree evals take the cheap path, and it is
   Elo-*positive* at LTC via the speed. ARCHON's loader + context **already
   support HalfKA-only CENN nets end-to-end** (`threat_weights.is_empty()`
   short-circuits: `sota_threats.rs:608-618, 662-666`, zero threat channel in
   the head) — the engineering left is a second loaded net, a second (tiny)
   accumulator stack, and a material/eval-margin gate; the training left is a
   small distilled head, for which the D40 pipeline already has every tool.
   Reasoned ceiling: 30–50% of evals at ~⅓ eval cost → **+8–15% NPS** at
   neutral-or-positive Elo *if* the gate replicates SF's; the Elo risk is in
   the gate, and it is SPRT-testable cheaply. MED-HIGH effort. This is also
   the honest answer to "halve the instr/node gap": the last ~2k Ir/node of
   the gap vs SF18 is partly *SF not running the big net everywhere*.
3. **Narrow threat channel** (threat contribution confined to fewer
   accumulator lanes, cutting i8 fold row bytes proportionally): fold cost
   scales linearly with lane count (768→256 would take the 17.1% fold block
   to ~6%), but §3's flat spectrum lowers the prior that the net tolerates
   it. Only worth a cheap bullet probe after items 1–2, if at all.

---

## 5. Updated ranked list (supersedes round-2 §5 ordering)

"NPS" = pure speed at bit-identical nodes unless flagged FORMAT/TRAIN. Elo
band: 1–2.1 internal Elo per 1% NPS (D39 house rule … measured opt-1 slope).

| # | what | expected | effort | risk | flag |
|---|---|---|---|---|---|
| **1** | **Fully-lazy threat observer** (snapshot-at-push, generate+resolve at fold; §2) | **+11–12%** | MED (1–2 d) | LOW — bit-identical, gates exist | **BOLD, headline** |
| **2** | Resolve→prefetch→fold pipeline in `compute` (both transformers; §2.4.1) | +3–5% | LOW (after #1) | LOW | unlocked by #1 |
| **3** | Cross-ply cancel + null-move alias + scratch-event packing (§2.4.3-5) | +1.5–2.5% | LOW (inside #1) | LOW | |
| **4** | Residual eval-path copies (round-2 item 3: `activate_ft`/`find_nnz` out-params, `finny_refresh` in-place) | +1.5–2.5% | LOW | LOW | unchanged |
| **5** | Static kernel dispatch + explicit ymm/zmm head kernel (round-2 item 4; LazyLock check confirmed alive ×4/eval) | +2–4% | MED | LOW | unchanged |
| **6** | SIMD event emission over the mailbox snapshot (§2.4.2) | +2–3% | MED | MED (unsafe SIMD, parity-gated) | BOLD-ish |
| **7** | FORMAT: quant unification → i16 activate (§4.1) | +2–3% | MED-HIGH | MED | CENN-v3 |
| **8** | FORMAT/TRAIN: smallnet gate (§4.2) | +8–15% | HIGH | MED (gate Elo) | **BOLD #2** |
| **9** | Bulk net parse (round-2 item 5; startup −0.13 s) | QoL | TRIVIAL | none | unchanged |
| 10 | MovePicker constant factor — 10.0%, top I1 consumer; belongs to the search-study lane | +1–3% | MED | MED | coordinate |
| — | ~~Low-rank threat factorization~~ | — | — | — | **KILLED by measurement (§3)** |
| — | Huge pages: THP=`always` measured on this box → net already covered; TT only relevant at big hash | ~0 | — | — | demoted again |
| — | Backward accumulator updates (SF `Forward=false`): spans avg 1.34 — thin here | <1% | MED | LOW | note only |

**Package view (reasoned):** items 1–6 are one coherent campaign centered on
the observer rewrite, all bench-node-preserving: **+18–25% NPS** on top of the
shipped +20.7%. At round-2's clean 859 knps that projects to ~1.01–1.07 Mnps —
past Reckless (1.05–1.06 Mnps at the common position) with the same net shape,
and cycles/node ~4.6–5.0k ≈ SF18's contended 4.7k. Instr/node moves less
(~16.0k → ~13.5k) because much of the win is latency/branches, not
instructions; truly halving the *instruction* gap additionally needs §4's
format items (quant unify −0.3k, smallnet routing −1–2k effective). Per D39,
the 1–6 package is worth **+18–50 internal Elo** at 10+0.1 — comparable to a
successful net-generation upgrade, at a fraction of the compute, and it
compounds with every future net on this arch.

**Sequencing note:** do #1 alone first and A/B it (bench-node identity + NPS
+ a 2k-game 10+0.1 sanity match), then #2/#3 in the same tree, then #4–#6.
Items #7–#8 ride the next training campaign. The A/B protocol from rounds 1–2
applies unchanged (pinned core 14, median-of-3 depth-12 bench, quiet box,
`pgrep -x fastchess` first).

---

## Appendix: raw data this round

- SVD probe: `scratchpad/svd_probe.py` + saved spectra
  (`svd_threat_sv.npy`, `svd_halfka_sv.npy`) in
  `/tmp/claude-0/-root-projects-learnrust-chessengine/9162696c-0351-4e70-854a-ccb48a50c724/scratchpad/`.
  Gram/direct Frobenius agreement to 6 significant figures on both tables.
- Re-baseline (shipped `sota-opt2`, pinned c14): 3,515,701 nodes ×3; NPS
  837,869 / 823,542 / 854,985; perf stat 57.5345G instr, IPC 2.45,
  br-miss 1.96%, L1d-load-miss 14.32% (AMD speculative-inclusive event —
  matches round-2's 14.31%).
- SF18 source locations for the schedule study: `types.h:283-336`
  (u32 `DirtyThreat`, 80+16 bound), `position.cpp:1058-1216`
  (`add_dirty_threat`, `write_multiple_dirties`, `update_piece_threats`),
  `nnue/features/full_threats.cpp:191-333` (`make_index`,
  `append_changed_indices`, `requires_refresh`),
  `nnue/nnue_accumulator.cpp:492-630` (lazy walk, `double_inc_update`,
  backward updates), `:227` (byproduct bitboards' only consumer).
- ARCHON structural facts used: `threat_index` mirror-bit dependency
  (`sota_features.rs:174-178, 324-348`); observer inputs
  (`sota_threats.rs:144-189, 800-897`); fold walk (`:700-732`);
  HalfKA fold-time re-resolution precedent (`sota_accumulator.rs:73-110`);
  mailbox `[Option<Piece>; 64]` (`board.rs:414-415`); `MAX_PLY = 246`
  (`types.rs:458`); observer's only consumers are the eval path
  (`eval/mod.rs:206`, `search/mod.rs:2474, 4141`).

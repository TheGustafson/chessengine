# perf-impl-1 — implementing perf study 7's top three, and what the numbers said instead

Date: 2026-07-27. Branch `perf-impl-1-threats`, worktree
`.claude/worktrees/perf-impl-1`. Companion: `perf-impl-1-plan.md` (written
before any engine code changed). Spec: `perf-study-7-sf-side-by-side-2026-07-27.md`.

**Headline.** Node-identical, bit-exact-eval, and **−6.17% instructions /
−8.44% L1d loads** on the matched fixed-node workload. The two changes that
produced it are not the two the study asked for: item 2's win came from the
*representation* of the threat snapshot rather than from retiring the replay
scheme, and item 3's win came from *register pressure* rather than from icache.
Three of the study's claims are corrected below with the measurements that
correct them, and one of its proposed validation instruments is shown to be
unusable on this box.

---

## 0. RESULTS

All figures from `bench 16 1 100000 <wl50.fens> nodes` with
`data/nnue/nets/screens/d47v-a4b2-cal400.cenn`, `taskset -c 4`, `nice -n 19`,
`--features sota,arch-d47v`, `-C target-cpu=native`, non-PGO.
**Every counter in this table is architectural** (retired instruction /
load / branch counts, and node totals) and therefore immune to the load on
this box — which was a GPU trainer plus four SPSA fastchess workers throughout.

| arm | instructions | Δ | L1d loads | Δ | branches | nodes |
|---|---:|---:|---:|---:|---:|---:|
| HEAD `783d142` | 38,036,038,707 | — | 18,228,610,600 | — | 4,662,019,788 | 5,001,556 |
| + bitboard threat events | 37,664,077,070 | −0.98% | 17,710,317,509 | −2.84% | 4,565,010,833 | 5,001,556 |
| + outline `init_quiets` | 36,853,573,464 | −3.11% | 17,302,886,863 | −5.08% | 4,571,515,528 | 5,001,556 |
| + outline `init_captures` | 35,770,467,931 | −5.96% | 16,699,202,980 | −8.39% | 4,575,668,719 | 5,001,556 |
| + outline `init_evasions` | **35,688,627,432** | **−6.17%** | **16,688,987,521** | **−8.44%** | 4,575,543,663 | 5,001,556 |

`instructions:u` reproduced to ±0.00001% across rounds. The node total is
bit-identical at every step, which is the whole point.

**Timing, for completeness only — CONTENDED, do not use for a decision.** Nine
interleaved pinned rounds, per-round ratio, median quoted:

| | median | range | rounds favouring the new arm |
|---|---:|---|---:|
| cycles ratio (final / HEAD) | 0.9450 | 0.878 – 1.054 | 7 / 9 |
| NPS ratio (final / HEAD) | **1.0606** | 0.931 – 1.155 | 7 / 9 |

+6.1% NPS is what −6.17% instructions predicts at unchanged IPC, so the band is
consistent with the deterministic measurement; it is not independent evidence.

### Equivalence evidence

| gate | value | status |
|---|---|---|
| 800-FEN `nnue-eval` sha256 | `dd08810005c92f9e97220adbbad2d17c2e9e0fcc1686a5ee31a798ca9f809047` | unchanged |
| wl50 `nnue-eval` sha256 (the new `CHECK_NNUE_GATE` pin) | `cd190d1b18e52f621cb0867e264359d3ce2cac2e6bdf65cdcdfc2005e9747f3d` | unchanged |
| wl50 d13 node total | 2,118,123 | unchanged |
| depth-6 default bench | 385,329 | unchanged |
| `bench 16 1 100000 wl50 nodes` | 5,001,556 | unchanged |
| `threats_replay_large` | 4,000 random games, incremental == refresh at every node | green |
| debug `assert_matches_refresh` | every evaluated node, whole suite | green |
| `CHECK_NNUE_GATE=1 bash tools/check.sh` | fmt, clippy, feature matrix, tests, release perft, both bench pins, both NNUE pins | **All checks passed** |

---

## 1. Three corrections to the study

### 1.1 The depth-6 bench pin cannot see any of this

`Cargo.toml` has no `default = [...]`, and `pub mod sota_threats` is behind
`#[cfg(feature = "sota")]`. The pinned `bench 16 1 6 default depth` runs a
default-feature build with an empty `EvalFile`, i.e. PSQT. An
incremental-accumulator rewrite can be arbitrarily wrong and leave 385329
untouched.

This was reported mid-session and is now enforced on main as
`CHECK_NNUE_GATE=1` (commit `39125cd`), pinning the wl50 eval hash and the d13
node total. Both are green here. That is a far stronger statement than 385329
holding, and it is the gate any future NNUE work should be held to.

### 1.2 "We fold ~3× as many weight rows as SF18" — refuted

New committed instrument, `measure_fold_row_counts`: for every move it computes
the **true minimal delta** — the symmetric difference of the from-scratch active
threat feature sets before and after — and compares it with the rows our fold
actually touches.

```
perspective-plies      : 76651
resolved rows / p-ply  : 5.99
MINIMAL rows / p-ply   : 5.80
overfold factor        : 1.034
```

Our event generation is within 3.4% of the information-theoretic floor. No
correct incremental scheme can fold materially fewer rows than we already do.

This also **bounds a whole class of proposed optimisations before implementing
them**. SF18's `noRaysContaining` guard (suppressing the ± discovered-attack
pair when the mover slides along a slider's ray) is exactly the kind of
cross-cancelling emission the overfold factor measures, so its ceiling is
≤ 3.4% of the fold ≈ 0.2% of instructions. Not worth the risk; not implemented.

### 1.3 "`at_full` is 15.7% of deep cycles and SF's equivalent stage costs ~0"

`at_full`'s 15.7% counts the fold kernel, which PGO inlined into the symbol.
The study's own §3.1 shows it — the symbol's top opcodes are `vpmovsxbw` 21.2 /
`vpaddw` 9.1 / `vpsubw` 8.6, which is the i8→i16 accumulator fold, not a replay
— and §3.2 gives the split (`at_full` self 638 Ir/node against
`i8_deltas_into` 2,354).

Confirmed directly: `#[inline(never)]` on `apply_i8_deltas_into` and re-profile
the native build.

| symbol | share of instructions |
|---|---:|
| `apply_i8_deltas_into` (the fold) | **6.28%** |
| `at_full` minus the fold | 6.18% |
| `record_square_touch` | 3.82% |
| `ripple` / `apply_at` / `refresh_threat` | 0.65 / 0.70 / 0.56% |
| threat block total | ≈18.2% |

Of `at_full`'s remaining 6.18%, ~1.3pp is the PSQ1 skip-channel fold. So the
**addressable replay surface is ≈10% of instructions**, and ≈7.6pp of the
threat block is fold work that is irreducible in any scheme.

My own callgrind run reproduces the study's §3.2 table to within 6%
(344 / 161 / 2,540 / 657 Ir per node against its 323 / 159 / 2,354 / 638), so
the disagreement is in the reading, not the data.

### 1.4 A fourth, from the parallel SF18 source study

The commissioned SF18 study (recorded in this session, source study only, no
code taken) found that the study's 2.2× fold-cost comparison runs **our
un-tiled AVX2 fallback kernel against SF's register-blocked one**: ours keeps
one 256-bit accumulator live and re-derives the row address for each of 48
chunks, theirs amortises one address over 8 vector-op pairs. On the **AVX-512
path the production binary actually executes**, our kernel is 8 zmm × 32 lanes
over 3 tiles ≈ 76 instructions per row per 1024 lanes against SF's ≈ 70 —
essentially parity. A meaningful slice of the measured fold gap is not present
in shipping builds.

---

## 2. Item 2 — what was implemented, and why it is bit-exact

Not "retire the `at_full` replay". Our scheme is already minimal in rows, lazy
(only ~46% of pushed plies are ever folded), and forward-incremental from the
nearest computed ancestor. SF's forward/backward hop machinery exists to make a
*refresh* lazy, which for our mirror-only king dependence fires on 2.3% of
pushes. Building it would double the correctness surface of the most dangerous
subsystem in the engine for a measured ~0.3pp.

What we actually paid for was the **representation**: the deferred design
snapshotted `board.mailbox`, so "which sliders x-ray this square?" meant
walking every occupied square on both ray fans and resolving each one's
identity through an eight-slot override map and an `Option<Piece>` unwrap — up
to eight resolutions to find zero to two sliders, and the same again for
knights and pawns. The from-scratch enumeration `for_each_threat` has never
done that; it intersects with victim-type bitboards.

So `ThreatSnapshot` became `ThreatPosition`: occupancy by piece type, by
colour, the union, and both king squares. The replay's edit primitive is one
XOR per board, which deletes the override map and the linear `find` outright.
Slider discovery is `ratt & (rooks | queens)`; knights are
`knight_attacks(s) & knights`; pawns take one extra colour mask. Outgoing
relations iterate one victim type at a time so the type is a loop constant and
excluded pairs never materialise, driven by a new `KEPT_VICTIMS` const bit-set
derived from the same `MAP` the resolver consults. The magic-table `LazyLock`
deref is hoisted to once per fold.

**The bit-exactness argument.** The accumulator fold is wrapping i16 add/sub
and the PSQ1 fold wrapping i32 add/sub; both are commutative and associative,
and each raw event resolves to an index independently of the others. The result
therefore depends only on the **multiset** of resolved indices, never on their
order or grouping — so generation may be restructured freely. (Checked there is
no chunking hazard: `i8_deltas_into` chunks at CAP=96, observed maximum 26.)
`KEPT_VICTIMS` is pinned to `threat_pair_kept` by a test over all 36 ordered
pairs, and `PieceType::ALL`'s discriminant order by another, so the two drop
sets cannot silently diverge.

Measured effect: −0.98% instructions, −2.84% L1d loads, threat code
25,939 → 16,689 bytes (−35.7%). The threat block's instruction share moved
19.04% → 18.48%.

Runtime shape of the channel, from throwaway counters over 201,815 nodes
(instrument saved, not committed):

```
push 202,707   at_full 83,005 (0.411/node)   compute_work 82,946
span_children 119,331   (1.44 children folded per compute; 24.1% span >1 ply)
persp_folds 233,219     4.09 rows per perspective-fold
king refreshes 4,597 (2.3% of pushes)   one-sided folds 4.6% of children
```

Cross-validated to the unit: `persp_folds − rows_hist[0] = 233,219 − 51,298 =
181,921`, exactly the callgrind call count for `i8_deltas_into`.

---

## 3. Item 3 — the icache diet that turned out to be a spill diet

`next_move` was one 2,665-instruction function: `init_captures`, `init_quiets`,
`init_evasions`, `select` (monomorphised per closure) and the partial insertion
sort all inlined into the stage machine. The study blamed PGO for its size
(3,199 in the PGO binary against SF18's 1,763); the **non-PGO build is already
2,665**, so most of the bulk is ours.

Outlining the three stage-init routines is pure code layout and node-identical
by construction. It bought −5.2% instructions and −1.5 G retired loads. That is
not a layout effect — static outlining cannot reduce dynamic instruction counts
— it is register pressure. The signature is in the disassembly: the fraction of
instructions that are stack-relative `mov`s, i.e. spills and reloads.

| | static instrs | `mov` % | stack-`mov` % |
|---|---:|---:|---:|
| `next_move`, inlined | 2,665 | 30.2 | **13.5** |
| `next_move`, outlined | 386 | 28.2 | **4.9** |
| `search::<NonPv>` | 3,502 | 32.4 | **14.4** |

Total movepick text is unchanged (2,733 → 2,742 instructions). The same code,
compiled with enough registers.

### The icache counter is not usable on this box — and it was the study's proposed instrument

Study §7 gives item 3's validation class as "node-identical + icache-counter
delta (contention-immune)". It is not contention-immune here.

- `eng-qce` and `eng-out` are **byte-identical binaries** (same sha256, built
  from the same source by two routes). Inside a single interleaved sweep they
  measured **86 M** and **56 M** `L1-icache-load-misses`.
- A 5-run stability check of one of them sat tight at 83.5–85.8 M, so the
  counter is stable *within* a batch and bimodal *between* batches.

The likely cause is the SMT sibling of the pinned core: Zen 4 shares L1i
between the two threads of a core, and there are four fastchess workers
floating. Nothing in this session rests on an icache number. For the record,
the item-2 change did measure 96.3 M → 36.4 M (−62%) across three interleaved
rounds, and I am *not* claiming it: the same instrument later failed its own
identity check.

### Negative result, recorded so it is not re-derived

`#[cold] #[inline(never)]` on `update_root_move` (provably root-only):
**−0.010% instructions**, −0.32% L1d loads, node-identical. Reverted. The
implication is useful: `search::<NonPv>`'s 14.4% stack-traffic fraction is
real, but it is not root-path code. Splitting that function is the ranked next
item and it needs a proper study of which blocks are once-per-node versus
per-move — a wrong split there would cost, not save.

---

## 4. Item 1 — the hka2048 audit

The part most suspected — the JAX shim that keeps the Pallas FT kernel by
passing a one-row zero threat table with all −1 indices — is **correct**. The
kernel's threat loop is `fori_loop(0, thc[b], …)` with `thc = sum(th >= 0)`, so
it runs zero iterations; −1 is never used as an index outside that loop in
either direction; the scan fallback clamps with `maximum(idx, 0)` against a
table that does have a row 0 and then masks the result; and the reference path
drops the threat term entirely under `halfka_only`.

The dummy table is gradient-safe for a **stronger reason than a zero gradient**:
it is a jitted constant, not a parameter, so it is absent from the pytree that
decoupled weight decay, EMA and the global-norm clip all map over. That answers
the specific worry (decay moves parameters regardless of gradient) — it cannot
reach a tensor that is not a parameter.

CENN v2 arithmetic at L1=2048 with an empty threat section checks out term by
term against the Rust loader (92,838,944 payload bytes both sides); the
zero-length threat section is accepted end to end; nothing computes
`THREAT_DIMS * ACC` outside a `has_threats` branch.

**Fixed in this branch:**

1. **netpack could not pack this arch under any environment.** `n_polw` was
   `POLICY_VOCAB * D40_L2`, but the policy head projects from its own
   bottleneck, equal to L2 only by coincidence in d48w/d48x (both 16). hka2048
   is the first arch with `l2=32, pol_proj=16`, so the stream over- or
   under-shot by `POLICY_VOCAB*16` bytes and the PSQ1 trailer split died with
   "unrecognized D40 trailing payload". Added `D40_POL_DIM`, defaulting to
   `D40_L2` so every existing invocation is byte-unchanged.
2. **`strip_threats.py` silently dropped the SKP1 output-skip section.** It
   emitted the truncated PSQ1 body and stopped; SKP1 is optional in both
   netpack and `sota_loader`, so nothing complained. **Perf study 7's threat-cost
   grid was built with this tool**, so its full-vs-stripped arms differed by the
   skip lane as well as by the threat tensors — a small upward bias on the
   measured threat cost. The docstring's "netpack re-validates, so a mistake
   here fails loudly there" was never true for this mistake.
3. **The engine copied a provably-constant all-zero threat accumulator** from
   parent to child on every pushed node when the net has no threat weights:
   `2·ACC·2` bytes/node, i.e. **8 KB/node at 2048** — a tax scaling with exactly
   the width this experiment spends its recovered budget on. It was present in
   study 7's stripped arms too, so the measured 1.25–1.39× threat cost
   *understates* a properly-built HalfKA-only engine.
4. A `compile_error!` guard on the mutually-exclusive `arch-*` features (the old
   failure was a bare `E0428: the name L1 is defined multiple times`; note the
   L2/L3 cfg arms deliberately share, so a guard keyed on those would miss
   pairs), and `arch-d48w` restored to the `check.sh` feature matrix.
5. The `sota_head.rs` comment asserting "1.78× at 1280 / 1.40× at 768" replaced
   with study 7's measured +25–39% instructions. That session's figures — and
   the "width becomes 5.7× cheaper" multiplier derived from them, **which is
   what selected 2048** — were refuted by study 7 the same morning.

**Blocking a launch, not fixable from a worktree:**

- **The configured warm start is 1280-wide against a 2048 arch.** `load_warm`
  raises on the first tensor (`l0_pstw: file count 28,835,840 != shape
  (22528, 2048)`). A 2048 warm must be assembled first, and
  `assemble_d47u_warm.py` has no `--halfka-only`, so as written it would
  materialise two ~500 MB threat tensors on a box with under 1 GB free.
- **`tools/nnue/runs/` is gitignored wholesale**, so `hka2048_probe.json`,
  `hka2048_launch.sh` and every d48w/d48x config are untracked — not
  reproducible, not reviewable, and absent from a worktree. (Nine `d47c_*`
  files predate the rule and are tracked, which is what makes this easy to
  miss.)
- **No hka2048 watcher exists**, so no every-30sb pack and therefore no
  `net_scale_check.sh` calibration loop — which D46 makes a hard pre-game gate.
- **`engine_parity.py` cannot validate a halfka-only net** (it builds
  `ArchSpec.full()` unconditionally), so the standard JAX↔engine forward-parity
  oracle is unavailable for the first net of a new feature set.
- **The width choice needs re-deriving.** 2048 came from the refuted session's
  cost fit. Study 7's grid (`perf-artifacts-7/threat-strip-grid.txt`) has the
  absolute per-node instruction counts at both widths and both mixes.

Not verified (and why): Pallas/Triton codegen with a 1-row table and an `(M,1)`
index array has never been executed on the GPU in any committed artifact — the
math is right, the codegen is unproven, and the GPU belongs to `archon-d48x`.
The existing `test_pallas_empty_bags` sets both index arrays to −1 but keeps
full-size tables, so it does not cover this configuration.

---

## 5. RANKED NEXT, with measured ceilings rather than estimates

| # | item | ceiling | basis |
|---|---|---|---|
| 1 | Split `search::<NonPv>` the way `next_move` was split | ~5% instructions | it has the same 14.4% stack-traffic signature; needs a study of which blocks are once-per-node |
| 2 | SF18's two-ply fusion (`FusedUpdateData` shape) in `compute`'s span loop | ≤ ~1.9pp | 24.1% of computes span >1 ply; saves a fold and cross-ply cancellations. Highest value left in the threat channel, and the riskiest thing on the board |
| 3 | Re-measure the fold against SF18 on **matched AVX-512** paths | may delete a large part of the headline gap | our AVX-512 kernel is at ~parity per row; the study's 2.2× is an AVX2-twin artifact |
| 4 | Don't push an accumulator entry for null moves (SF simply doesn't) | small | removes the `at_full` alias loop as well |
| 5 | Make the king-cross refresh lazy | ~0.3pp | fires on 2.3% of pushes; needs a backward hop or an ancestor refresh |
| — | `noRaysContaining` ± suppression | **≤0.2%** | bounded out by the 1.034 overfold factor — do not implement |
| — | reduce the number of folded rows | **0** | already 1.034× the true minimum |

---

## 6. THREATS TO VALIDITY

1. The box was contended throughout (GPU trainer + four SPSA workers, load ~10
   on 8 cores). Only architectural counters and node totals carry conclusions.
   The cycles/NPS band in §0 is labelled and is not load-bearing.
2. `L1-icache-load-misses` is demonstrably unusable here (§3). No claim rests
   on it, including the −62% observed for item 2.
3. **Everything is measured non-PGO.** CLAUDE.md makes PGO the standard for
   match binaries, and PGO's inlining decisions differ. `#[inline(never)]` is
   respected under PGO, so the direction should hold, but the magnitude is
   unverified under PGO and should be re-checked before an SPRT. I could not
   run a PGO pair safely under the current memory pressure.
4. All measurement is on `arch-d47v` (768). The threat-block share grows with
   width, so item 2's contribution should be larger at 1280/1536 and zero at
   `arch-hka2048`.
5. Item 2 changes only which *route* produces the event multiset; it is
   bit-exact by the commutativity argument in §2 and by four independent
   identity gates. It is not an approximation and there is no tolerance
   involved.
6. No Elo testing. The +6.1% NPS band converts to ~+6 Elo at the house
   1 Elo/1% NPS, or ~+7 at the measured ~120 Elo/doubling — but this is a
   node-identical change, so an SPRT is a formality against a speed measurement,
   not a search-quality question.

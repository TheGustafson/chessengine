# What the Reckless study means for our Rust design

The payoff of the 14-note Reckless study: a prioritized, roadmap-ordered list
of what we could **adapt** (never copy — Reckless is AGPLv3, D27). Distilled
from `reckless_reference/01-*.md` … `14-*.md`. Reckless is
`codedeliveryservice/Reckless`, `v0.10.0-dev`, ~3833 SPCC (#2 at time of
study); its internal piece/eval scale is ~½ of ours/SF's, so any raw constant
carries a ~2× factor.

Every item names the exact owning doc/section it touches, the phase, rough
effort (S/M/L), expected impact (lo/med/hi), and a status tag:
**[novel]** (new idea for us) · **[already-planned]** (our docs already name
it; Reckless is confirmation or a worked example) · **[already-have]** (we
implement it; recorded for completeness) · **[conflicts-Dxx]** (bumps a
pinned decision — needs a decisions.md amendment first, per README process
rules).

---

## Executive summary — the highest-value takeaways

1. **The per-node opponent threat map is the single most transferable
   architectural idea in Reckless.** One vectorized "setwise" pass per node
   builds `all_threats` + `piece_threats[pt]`; king/castling legality collapse
   to an O(1) bitmask test, and the *same* bitboards are reused for free as
   move-ordering features (offense/threatened terms) and as a new RFP margin
   dimension. It is the prerequisite that unlocks a whole family of P9 ordering
   and pruning wins below. Medium-high effort, must be proven perft- and
   bench-node-neutral (D24). **[novel]**

2. **A cluster of cheap, on-critical-path P9 wins** we can SPRT this phase:
   history-scaled quiet SEE margins and history-coupled LMP (Wave B/C);
   NMP couplings (cutoffCnt discount, TT-capture veto, TT-narrowed window);
   root-move re-scoring after each sibling subtree; SEE promotion/EP evaluation
   and on-pin-ray recapture. Each is low-effort and independently gated.

3. **One genuinely tempting architecture experiment, and one genuine
   conflict.** The tempting one: Reckless's LMR base is `ilog2(depth)`-only
   with the continuous per-move history score *substituting* for SF's
   move-number dimension — potentially our highest-ceiling reduction change,
   but a standalone, revert-friendly experiment after Wave C, not wave work.
   The conflict: Reckless bakes capture-ness into the `Move` kind bits, which
   **conflicts with D2** (our pinned u16 layout) — record as prior art, do not
   adopt.

4. **P10/P11/P13 are mostly confirmation-plus-refinements.** Reckless resolves
   several of our open questions with concrete numbers (Threads upper bound,
   to_cp-without-WDL, the `Job::Custom` closure shape, best-thread vote
   formula) and supplies working reference implementations for planned NNUE
   machinery (ply-indexed accumulator stack, Finny cache, hybrid int8/f32
   inference). Its most novel eval lever — **threat input features** — is a
   real long-term ceiling-raiser but a large post-v1 P13 stretch that also
   forces a datagen-record amendment.

5. **Do not import Reckless's unsafe posture.** Non-atomic TT under
   `unsafe impl Sync`, raw `static mut`, self-referential raw-pointer Stack,
   liberal `transmute`/`get_unchecked`, unchecked `mmap` — all conflict with
   D14/D19. Take the *ideas* (SWAR tag lookup, huge pages, closure dispatch)
   and re-express them inside our safe/atomic contracts.

---

## P9 — pruning waves (NOW; `design/10`, plus `04`/`07`/`09`)

The current phase, so the richest target. Reckless diverges hardest from our
SF-faithful shape exactly in Wave C/D/E territory (statScore-coupling, singular
extensions, ProbCut) that we have not landed yet.

### The big one — precomputed per-node opponent threat map
- **Technique:** once per `make_move`/`make_null_move`, build a full bitboard
  of every square attacked by the side-not-to-move, per piece type, via
  whole-army setwise attack fills (own king excluded from occupancy so sliders
  see through it). King-move/castling legality becomes one AND against
  `all_threats`; `pinned`/`pinners`/`checkers` for both colors fall out of the
  same pass.
- **Why:** replaces per-candidate `attackers_to_exist` queries with an O(1)
  test, and the bitboards are then reused *for free* by the three ordering/
  pruning items marked ★ below — the argument that makes those cheap for
  Reckless.
- **Touches:** new `StateInfo` field in `design/04-position-state.md` §3.5;
  `Board::legal` king/castling arms in `design/05-movegen.md` §3.8;
  `design/03-attack-tables.md` §1 (setwise helper, currently scoped out).
- **Effort M-L / impact hi.** Must be perft-clean and bench-node-neutral
  (D24 ritual + `Bench:` trailer mandatory since it moves cost distribution).
  **[novel]**

### Cheap, on-path, SPRT this phase
1. **History-scaled quiet SEE-pruning margin.** Add a history term to our
   already-quadratic-in-`lmr_depth` quiet SEE margin (Reckless's is quadratic
   *and* history-scaled). `design/10` §3.5, Wave B. **S / med. [novel]**
2. ★ **History-coupled LMP threshold.** Couple the move-count budget to the
   move's own ordering history, not just `improving`. `design/10` §3.5 / Wave C
   item 11 ("history-coupled pruning"). Needs history computed before the LMP
   check + retune. **M / med. [already-planned]** (Wave C names it; Reckless is
   a concrete shape.)
3. **NMP couplings (three one-liners on already-shipped NMP, N11).** (a)
   cutoffCnt discount when `cutoff_count[ply+1] < 2`; (b) veto NMP when the TT
   already promises a beta-beating capture of ≥knight material; (c) narrow the
   null-move window to `tt_score` when a fresh TT Lower bound sits below beta.
   `design/10` §3.4 N11. Each **S / lo-med**, isolated SPRT [0,3]. **[novel]**
4. **Root-move re-scoring after each completed sibling subtree.** At root only,
   re-run scoring on the remaining buffer so ordering reflects the history the
   just-finished subtree wrote — a mid-iteration staleness fix neither SF nor
   we do. `design/09` §3.6 root pipeline. **S / med. [novel]**
5. **SEE accuracy, two refinements.** (a) Run promotions/en-passant through the
   real exchange loop using the *promoted* piece's value instead of our blanket
   `non-Normal → 0≥threshold` shortcut (`design/04` §3.11 step 1); (b) exclude a
   pinned attacker from SEE only when the capture square is *not* on its
   king-pin ray, allowing on-ray recaptures (`design/04` §3.11 step 6).
   **S / lo-med each**, (a) needs Bench trailer + SPRT. **[novel]**
6. **lerp(a,b,t) blend helper.** One `mul_add` helper replacing our ad-hoc
   two-integer-weight blends at the RFP/qsearch-stand-pat/qsearch-fail-high
   returns (and future multicut/ProbCut). `design/10` §3.4 / `design/07`
   §3.7.2. Pure refactor, bench-identical. **S / lo (hygiene). [novel]**
7. **Bind the TT-substituted pruning eval once per node** (Reckless's
   `estimated_score`) instead of re-deriving it at each pruning site.
   `design/07` §3.7.2 step 6. Non-functional if verbatim. **S / lo. [novel]**

### ★ Threat-map dependents (land after the threat map above)
8. **Threat-aware quiet-ordering features (threatened / "offense" terms).**
   Reckless scores both threat-escape *and* a symmetric offense bonus (moving
   to attack a vulnerable enemy) — SF/we score only the defensive side.
   `design/09` §3.4/§3.7 (ties to §9.6 `threats_by_lesser`). **M / med**, proxy
   SPRT. **[novel]** (offense term) / **[already-planned]** (escape term).
9. **Threat-coupled RFP margin.** Prune more readily (subtract a flat term)
   when none of our own pieces are currently threatened. `design/10` §3.4 N10;
   needs the new threats bitboard. **M / med (impact unclear). [novel]**

### Wave D / E build-against-this-shape items (not yet landed)
10. **LDSE (Low-Depth Singular Extension):** a 1-ply extension at shallow
    cut-nodes whose estimate sits well below alpha, covering the gap where the
    real SE gate (`depth ≥ 6 + ttPv`) can't fire. `design/10` §3.6, Wave D.
    **S once Wave D exists / lo-med. [novel]**
11. **Bound-type-conditioned singular margin:** `depth/4` when the TT bound is
    Exact (trustworthy), full depth (doubled under `ttPv && !PV`) when only
    Lower. `design/10` §3.6, Wave D. **S / med. [novel]**
12. **Dynamic-depth/dynamic-beta ProbCut re-verify:** scale probe depth and a
    shifted beta by how far the qsearch probe exceeded `probcut_beta`, with a
    full-window fallback re-search — build our first ProbCut against this richer
    shape rather than SF's flat two-step. `design/10` §3.4 N13, Wave E item 16.
    **M / med. [novel]**

### Correction-history axis (Wave C tail / P9 tail)
13. **Fifty-move-clock (rule50/GHI) bucket axis on correction history.** We
    already compute the `fold_rule50` bucket for TT key folding; Reckless shows
    the *identical* bucket doubling as a 16-way drawishness/simplification axis
    on the pawn/non-pawn correction tables — nearly free. `design/09` §3.10,
    `decisions.md` D9. **S-M / lo-med. [novel]**
14. **Qsearch SEE margin folds in `|correction_value|`** as an eval-uncertainty
    term (loosen the SEE bar when recent static-eval correction was large).
    `design/07` §3.8 (`QS_SEE_FLOOR`). **S / lo. [novel]**

### High-ceiling standalone experiment (after Wave C)
15. **ilog2(depth)-only LMR base + larger history coefficient (drop the
    move-number dimension).** Reckless's reduction base is `269*ilog2(depth)`
    with *no* move-count term; the continuous per-move history score substitutes
    for SF's `reductions[depth]*reductions[moveN]` 2-D table. Plausibly the
    highest-impact single idea in the study, but a genuine architecture change
    to a shipped component — run as an isolated, revert-friendly experiment with
    its own SPRT *after* the simpler Wave-C history-coupling lands, never folded
    into normal wave work. `design/10` §3.3. **M-L / hi (uncertain). [novel]**

### P9-adjacent micro-ops (verify via bench, not SPRT)
16. **Branchless least-valuable-attacker** in `see_ge` (6 non-emptiness tests →
    byte → `trailing_zeros`) — must reproduce the identical bench count.
    `src/board.rs` see_ge. **S / lo (nps only). [novel]**
17. **Direct-check-only proxy** (`is_direct_check`, an O(1) bitboard test) at
    pruning-exemption sites instead of full `gives_check`, accepting the loss of
    discovered-check exemptions; needs a `checking_squares`-style table.
    `design/10` §3.5. **S / lo (nps). [novel]**

### Correctness cleanups (cheap, D24 perft applies)
18. **Enemy-occupancy-only sniper detection** for `slider_blockers`
    (`bishop/rook_attacks(king, enemy_occ) & enemy_sliders`) — one fewer op, no
    separate "subtract snipers" correction case. `design/04` §3.5. **S / lo
    (simplification). [novel]**
19. **Harden `Board::legal`'s pin check to directional** (`!ray_pass(king,from)
    .contains(to)`, reusing our RAY_PASS table) so it's correct standalone, not
    only-when-composed-after-`pseudo_legal`. `design/05` §3.8. **S / lo
    (robustness for future from-scratch validators). [novel]**

### Explicitly rejected here (record only)
- **Bake `is_capture`/`is_noisy` into `Move` kind bits.** Removes the board
  dependency from the hottest question in search, but **conflicts-D2** (pinned
  u16 layout; "not worth churning"). Record as prior art. **[conflicts-D2]**
- **Current-state-as-plain-field** (not `Vec::last()`): removes one indirection
  per state read, but mild tension with **D8**'s owned-Vec-stack wording — a
  one-line D8/`design/04` §3.2 amendment, and LLVM may already elide the check.
  Profile first. **[conflicts-D8 (soft)]**

---

## P10 — time management completion + full UCI (`design/12`)

1. **`completed_depth==0` hard-stop guard.** `check_time` returns "keep going"
   unconditionally until iteration 1 finishes, guaranteeing a real bestmove
   under a tiny time/node budget — a guard ours lacks. `design/12` §3.8;
   `src/search/mod.rs` check_time. **S / med (robustness). [novel]**
2. **Re-poll `check_time` at qsearch entry** to bound stop latency under long
   forcing-capture dives (our own code comments this as a known gap; it is
   master-plan §6.4's deferred P10 question). `design/12` §3.8. **S / lo-med.
   [already-planned]**
3. **Flat epsilon shave on every enforced bound, including `movetime`.** Our
   movetime is an exact wall; Reckless subtracts a fixed ~15 ms uniformly,
   cutting forfeit risk at fast TCs. `design/12` §3.6/§3.8; `src/timeman.rs`.
   **S / med. [novel]**
4. **Node-counter bitmask cadence** (`nodes() & 2047 == 2047`) for the
   check_time poll, dropping a maintained `calls_cnt` field. `design/13` §3.3 /
   `design/12`. **S / lo. [novel]**
5. **Threads UCI upper bound = `max(4*available_parallelism, 512)`** — resolves
   our provisional flat-1024 open question with a hardware-scaled formula
   Reckless deliberately tuned. `design/12` §9.6. **S / lo. [already-planned
   (open Q)]**
6. **`to_cp` normalization without a WDL triple is a valid interim deliverable**
   — Reckless is a ~3833-SPCC existence proof, resolving `design/12` §3.14 open
   Q5 (whether `UCI_ShowWDL` must ship with the first NNUE net). **none / n/a
   (resolves open Q). [already-planned]**
7. **Minimal low-verbosity report mode** (suppress per-iteration info, one
   summary before bestmove) for high-frequency self-play. `design/12` §3.13.
   **S / lo (I/O only, bench-neutral). [novel]**
8. **Narrower clock-model enum** nested inside our flat `Limits` for the
   sudden-death-vs-cyclic branch (currently an i32 sentinel), keeping multi-axis
   combination (nodes+depth, used by our proxy-SPRT tooling) that Reckless's
   whole-Limits-as-enum trades away. `design/12` §3.6/§3.7; must reproduce the
   golden TM tests. **S-M / lo (hygiene). [novel]**

---

## P11 — threading: lazy SMP + NUMA (`design/13`, `12`, `09`)

1. **Generalized closure-based job dispatch (`Job::Custom`), zero-unsafe.**
   Reckless proves "every pool op is a boxed closure" is the natural shape
   (search, per-thread construction, future TT-clear/NUMA), but its version
   needs an `unsafe` lifetime `transmute`; take the closure as a call-time
   `&mut Worker` parameter instead of a captured borrow to get the generality
   with zero unsafe. Resolves `design/13` §9 open Q6, preserving §3.12 rule
   zero. `design/13` §3.4. **S / lo. [already-planned (open Q)]**
2. **Soft-stop by quorum vote.** Every thread (not just main) evaluates its own
   soft-stop condition from its own stats and casts a retractable vote; global
   stop fires only at a supermajority (Reckless: 65%), degenerating to today's
   behavior at `Threads=1`. A materially different lazy-SMP TM philosophy.
   `design/12` soft-stop × `design/13` §3.7/§3.10. **M / med (research; fixed-
   time SPRT quorum-vs-main before trusting the 65%). [novel]**
3. **Cross-thread optimism fusion.** A shared `[AtomicU32; MAX_MOVES]` packs
   `depth<<16 | score`; `fetch_max` prefers the deepest cross-thread result per
   root move; each thread blends it into its own average at aspiration setup.
   Reckless's commit history marks this a measured, kept gain. `design/11`
   (optimism/aspiration setup) × `design/13` (new shared array). **S / med,
   proxy-SPRT. [novel]**
4. **Best-thread voting refinements.** (a) Add "MultiPV>1 disables voting" to
   the spec (Reckless disables under fixed depth *and* MultiPV>1; our `design/13`
   §3.8 names only `n_threads==1`/`limits.depth`); (b) depth-weighted vote:
   `(score - min + VOTE_TEMPO) * completed_depth` as a concrete alternative to
   SF's flat additive `+14`. `design/13` §3.8. **S / lo. [novel]**
5. **Three-tier state model + NUMA replication registry.** Global-shared /
   per-NUMA-node-shared / per-thread, with a `NumaReplicable`/`NumaReplicated`
   observer registry, a "worth-it" gate (only replicate when >1 node AND thread
   count justifies it — inert on single-socket dev boxes), and first-touch-by-
   construction-order placement. Turns `design/13` §3.15 from a blank deferral
   into a scoped plan. **M for the mechanism / lo-med. [novel]** *Caveat:* real
   pinning needs libc `sched_setaffinity` → **conflicts-D15** (zero runtime
   deps); gate actual pinning behind measured multi-socket need + a decision.
6. **Global-shared correction-history pilot.** Since most of our SPRT hardware
   is single-socket, "per-NUMA" collapses to "global"; convert the correction
   table to atomic entries (same relaxed load-then-store gravity idiom the TT
   already tolerates) and A/B shared-vs-per-thread — validates sharing's value
   before any real NUMA code. `design/09` §9.5. **S-M / med, proxy-SPRT.
   [already-planned (deferred)]**
7. **Reduction-formula thread-id jitter** (`(nodes + thread_id*C) % 128`) for
   SMP diversity, in place of the classic `thread_idx % 8` aspiration jitter —
   flag now so it isn't reinvented; would break bench determinism if added
   pre-SMP. `design/13`. **S / lo, defer to P11. [novel]** **[conflicts-D24 if
   added before P11]**

---

## P13 — NNUE (`design/11`, plus `15`/`04`)

Reckless is a mature deeper net; most of this confirms our planned machinery or
supplies working shapes for the post-v1 upgrade path. Its standout is the
**threat input feature set**, which we lack entirely.

### Confirmations / cheap adoptions
1. **Ply-indexed flat accumulator stack** (`Box<[Accumulator]>` preallocated to
   MAX_PLY, `computed[bool;2]`, walk back to nearest-accurate ancestor, pop does
   zero work) — matches our `design/11` §3.4.4 plan almost exactly. Independent
   confirmation from a top-2 engine. **none / n/a. [already-planned]**
2. **`split_at_mut(i+1)` adjacent-slot update** — read ancestor slot `i`, write
   slot `i+1` of the same boxed slice with zero unsafe (fits the D14 budget for
   free). `design/11` §3.4.4. **S / lo. [already-planned]**
3. **Non-uniform output-bucket boundaries** (hand-shaped, finer in the
   middlegame band) vs a uniform `(popcount-2)/4` formula — trivial table swap.
   `design/11` §3.4.1. **S / lo. [novel]**
4. **Net-loader validation: no change.** Reckless's unchecked
   `include_bytes!`+`transmute` (no magic/version/checksum) *confirms* our
   planned validated/versioned CENN loader is the right call given we ship an
   external `EvalFile` option. `design/11` §3.4.5. **none. [already-have (plan)]**

### Post-v1 upgrade-path candidates
5. **Horizontal king-mirror (no buckets)** — mirrored positions collapse to one
   weight row; near-free data-efficiency win, no new structures. `design/11`
   §3.4.1 upgrade path. **S / med. [already-planned]**
6. **Finny-table refresh cache** keyed by `(pov, mirror, king-bucket)` — diff
   the current board against a cached snapshot instead of recomputing from bias;
   Reckless is a working instance of our named next-step once king buckets
   exist. `design/11` §3.4 upgrade path. **M / med. [already-planned]**
7. **King input buckets** with Reckless's validated asymmetric layout (8 buckets
   crammed into ranks 1-2, 2 for the rest) as a *shape* to imitate, not copy.
   `design/11` §3.4.1. **M / med-hi. [already-planned]**
8. **Split-accumulator pairwise-multiply activation** — clamp then multiply two
   384-halves elementwise (a gated/bilinear unit), same one-multiply cost as
   SCReLU but more expressive; pure activation swap. `design/11` §3.4.3. **S-M /
   med. [novel]**
9. **Int8-sparse-FT/L1 → f32-L2/L3 hybrid quantization** with Reckless's
   concrete `DEQUANT_MULTIPLIER = 2^FT_SHIFT/(FT_QUANT²·L1_QUANT)` bridge — much
   less quantization-training surface than a fully-integer pipeline; target this
   shape for our deeper post-v1 net instead of SF's all-integer cascade.
   (Reckless's `FT_QUANT=255`/`L1_QUANT=64` == our planned `QA=255`/`QB=64` —
   independent convergence.) `design/11` §3.4.6. **M / med. [novel]**
10. **NNZ sparse-L1 propagation** (256-entry nonzero-position LUT skips zeroed
    post-clip FT columns) — cheap once a second dense layer exists. `design/11`
    §3.4 upgrade path. **S / lo-med. [novel]**

### SIMD / inference discipline
11. **cfg-gated compile-time SIMD fast path** for the D16 pinned-target build,
    keeping the `LazyLock<Kernels>` fn-pointer struct only for the portable
    default — removes one indirect call per forward pass under SPRT builds; our
    doc already sketched this in one sentence, Reckless is the worked example.
    `design/11` §3.4.6. **S / lo. [already-planned]**
12. **Adopt Reckless's bit-exactness checklist** verbatim as a pre-flight list:
    `mul_add` not `a*b+c`; bias-as-initial-accumulator; fixed recursive-halving
    reduction order matching `_mm512_reduce_add_ps`; explicit clamp where Rust
    `as`-casts truncate instead of saturate. These are hard-won cross-ISA "diverges
    by one ULP on CI" gotchas. `design/11` §3.4.6. **S / lo (design-doc para).
    [novel]**
13. **Keep the SIMD primitive surface small** (~15 shared free functions across
    forward pass + accumulator update) so adding a backend is "implement 15
    functions." `design/11` §3.4.6. **none (discipline). [already-planned]**

### The big long-term eval lever
14. **Threat input features** (relational attacker/victim inputs, a second
    accumulator summed with the PST plane). The headline feature of the #2 SPCC
    engine and plausibly the largest post-basic-NNUE strength lever, but **high
    effort:** an independently-designed compressed index scheme (~66,864 features
    / ~51 MB i8 in Reckless), a much richer `MakeMoveHooks`/DirtyPiece payload
    (up to ~80 relationship events/move vs our ≤2 add/≤2 sub), *and* a real
    amendment to the datagen record since `design/15`'s Chess768 record can't
    represent relational features. `design/11` §3.4.1 (named destination) +
    `design/15` §3.2.3 (needs amendment) + `design/04` §3.10. **L / hi, P13
    stretch beyond v1. [already-planned (named destination)]**
15. **3-event `BoardObserver` DirtyPiece granularity** (add/remove, slide,
    same-square mutate) as production-validated prior art for our accumulator
    sub/add taxonomy, and a **second hook method** (eager-record / lazy-SIMD-
    apply) held in reserve for features needing finer granularity than
    before/after DirtyPiece. `design/04` §3.10, `design/11` §3.4.4. **none now /
    lo later. [already-planned]**

---

## Cross-cutting — Rust idiom, TT, build/PGO, tooling

### TT (`design/08`) — P9-adjacent / later
1. **Branchless SWAR cluster tag lookup:** pack the 3 entries' 16-bit
   verification tags into one 64-bit word, replicate-XOR-mask-`trailing_zeros`
   to find a match in O(1) — frees 2 bytes/entry (8-byte entries) and replaces
   the 3-way scan on the hottest path. `design/08` §3.2-3.5. **M / med.**
   *Decide* whether tags stay 3× `AtomicU16` (reassembled locally) or become one
   `AtomicU64` (coarser write granularity) to keep our per-field-atomics
   contract. **[novel]** (adopt only atop our atomics — Reckless's non-atomic
   version is **[conflicts-D19]**).
2. **`REPLACEMENT_AGE_WEIGHT = 4` instead of 8** — Reckless halves SF's
   age-weighting; a one-constant fixed-node proxy SPRT. `design/08` §3.6,
   `src/tt.rs`. **XS to try / lo-med. [novel]**
3. **Broader force-overwrite for any `is_pv` write** (not just `Bound::Exact`)
   at the negamax TT-store exit — propagates fresher PV info faster at the cost
   of more eviction churn; SPRT experiment. `design/08` §3.7, `src/tt.rs`.
   **S / lo-med. [novel]**
4. **mmap + `MADV_HUGEPAGE`** as the Phase-2 huge-page allocator shape (simpler
   than a custom-`Layout` `alloc_zeroed`; `munmap` needs only `(ptr,len)`),
   wrapped in our existing fallible halving `resize` — but add the `MAP_FAILED`
   check Reckless omits. `design/08` §3.13 Phase 2. **M / lo, deferred.**
   **[already-planned]** — needs the D14/D15 amendment `design/08` §9.1 already
   anticipates. **[conflicts-D14/D15 until amended]**

### Rust idioms / bench / build provenance (`design/06`, `14`) — any phase
5. **Blanket `Index<T>`/`IndexMut<T>` for `[U]` over `Square`/`Piece`/
   `PieceType`/`Color`** (keep IndexMut bounds-checked, unlike Reckless's unsafe
   write path) — turns `table[x.index() as usize]` into `table[x]` crate-wide,
   purely additive. `design/02` Square/Piece helpers. **S / lo. [novel]**
6. **const-generic verbose/terse `bench<const PRETTY: bool>`** output split —
   cleaner than a runtime flag for our interactive-vs-CI-fingerprint dual use.
   `design/06` §3.6. **S / lo. [novel]**
7. **`[const { T::new() }; N]`** idiom for fixed arrays of non-Copy statics
   (atomics/locks) — zero unsafe, no `LazyLock`/`MaybeUninit` dance. General
   Rust-idiom note. **S / lo. [novel]**
8. **Bake compiler/target/target-feature provenance via `build.rs env!` + a
   `compiler` command,** printed atop every bench/speedtest — self-documents
   exact RUSTFLAGS/toolchain, high value once v3/v4 dispatch or PGO matters.
   `design/14` §3.6/§3.7. **S / lo. [novel]**

### Tooling / CI (`design/06`, `14`)
9. **Extend our 64k move-encoding fuzz into a differential perft oracle**
   (`is_legal_perft`: drive perft off the predicate-only legal-move list,
   compare to the real generator) — a strong cheap correctness check we don't
   run today. `design/06` §3.5/§9. **S-M / med (correctness). [novel]**
10. **Fastchess self-play smoke test as a fast CI job** (protocol/crash/illegal-
    move check, not Elo) on every push — distinct bug class from bench/SPRT,
    especially valuable once P10 lands. `design/14` §3.8 (new tier 0). **S / med.
    [novel]**
11. **Cross-microarchitecture bench-determinism CI gate via Intel SDE** (rerun
    the `Bench:` trailer under emulated microarchitectures + native Apple
    Silicon, assert identical node counts) — catches ISA-dependent search
    behavior (SIMD reduction order, FMA contraction) a single runner can't;
    low value now (integer PSQT), high once P13 NNUE SIMD lands. `design/14`
    §3.2/§3.9. **M / lo-hi(P13). [novel]**
12. **Per-target-tier PGO with SDE-assisted training** (each ISA tier gets its
    own instrument→train→optimize pass; AVX2/AVX-512 training under SDE
    emulation) — correct methodology once we ship multiple prebuilt binaries;
    profiles aren't portable across tiers. `design/14` §3.7, P14. **M / med.
    [novel]**

### to_cp normalization (P13 polish)
13. **Cubic material-weighted `to_cp` normalization** (phase-dependent cubic in
    clamped material count) — the curve family our `design/12` to_cp seam is
    already pinned to adopt; shape, not constants (SF's coefficients are
    meaningless for our eval — Reckless correctly fits its own). `design/12`
    to_cp seam (D21). **S / lo (once NNUE exists). [already-planned]**

### Do NOT adopt (record, then move on)
14. **Syzygy/Fathom tablebases.** Our score-band plumbing (VALUE_TB), `design/10`
    N7 interior-probe shape, and `tb_hits` slot already match Reckless almost
    exactly — but real support needs a vendored C library + `cc`+`bindgen`
    build-deps + a runtime libc dependency + unsafe FFI. **conflicts-D15** (zero
    runtime deps) and **exceeds D14** (unsafe budget). Defer past P13; when
    picked up, scope as a new decision + doc with a non-default opt-in feature.
    `design/10` §9 open Q2, a future `design/17-tablebases.md`. **[conflicts-
    D14/D15]**
15. **Reckless's unsafe posture generally** — non-atomic TT under
    `unsafe impl Sync`, raw `static mut TB_LARGEST`, self-referential raw-pointer
    Stack (exactly what `design/07` §3.2.3 rejected for the zero-unsafe
    `ContHistId` key), liberal `transmute`/unconditional `get_unchecked`. Take
    the ideas, keep our D14/D19 contract. **[conflicts-D14/D19]**

---

## Where we're already ahead (no action — recorded so we don't "fix" it)

- **Richer, stricter FEN errors** (D convention) vs Reckless's lenient
  `unwrap_or_default` 3-variant parser.
- **Heavier test posture:** `assert_consistent` invariant oracle at every
  make/unmake node, 64k pseudo-legal fuzz, dedicated SEE fixtures, per-node key
  recompute — Reckless ships perft + incremental-hash checks only, no SEE
  fixtures, no per-node oracle.
- **Typed options registry** vs Reckless's hand-written `match`; **named
  `params.rs` constants (D20)** vs Reckless's inline literals (its SPSA macro
  scaffold is declared but never populated — we hold the higher bar today).
- **Five-key Zobrist family** already paid for; **stricter, fallible TT
  allocation** (`&mut`-self resize, MAP_FAILED-checked) vs Reckless's unchecked
  `&self` path; **explicit TT `hit` flag** closing the blank-entry tag-collision
  edge case Reckless leaves open.
- **Robust UCI** (atomic `position` application, three-role thread model,
  explicit mid-search command policy) vs Reckless's silent-token-skip + trusting
  two-thread model; and we already ship the **P7 forgotten-mate/aborted-loss
  hygiene** that matches Reckless's ID-loop hygiene (convergent, not a gap).
- **Safer `Bitboard::shift(Direction)`** (per-direction wrap mask baked in) vs
  Reckless's caller-must-pre-mask generic shift; **file-indexed ep Zobrist**
  (tighter) vs Reckless's 64-slot table; **cheaper `(nodes&2)-1` draw dither**
  matching current SF vs Reckless's wider `(nodes%5)-2`.

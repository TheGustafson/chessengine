# Design decisions record (authoritative)

Every design doc in this directory MUST comply with these decisions. If a doc
needs to deviate, it must call the deviation out explicitly in its Open
Questions section rather than silently contradict this file.

Status: living document. Decisions are numbered for citation (D1, D2, ...).

## Project goal

A UCI chess engine in Rust with Stockfish-class architecture and, ultimately,
strength. Two constraints in permanent tension, resolved in this order:
1. **Correctness** (perft-verified, bench-reproducible) beats everything.
2. **Performance**: our data structures must never be the reason we can't
   match Stockfish's per-node costs.
3. **The Rust way**: safe, idiomatic Rust preferred; `unsafe` only with a
   measured, documented justification (D14).

## Naming and existing code (do not churn)

- **D1**: The position type is `Board` (not `Position`). Existing modules
  `types`, `bitboard`, `board`, `fen`, `moves` keep their names and public
  APIs; extensions add to them.
- **D2**: `Move` keeps OUR bit layout: bits 0-5 = from, 6-11 = to, 12-13 =
  promotion piece minus knight, 14-15 = kind (normal/promotion/ep/castling).
  This is the reverse of Stockfish's from/to order — functionally equivalent,
  not worth churning. `Move::NULL` (from==to==a1) is the none-sentinel; a
  distinct null-move sentinel, if ever needed, uses another from==to pattern.
- **D3**: Square mapping stays little-endian rank-file (a1=0 ... h8=63).
- **D4**: Piece storage stays `pieces: [[Bitboard; 6]; 2]` indexed by
  `(Color, PieceType)`. When flat piece indexing is needed (history tables),
  add `Piece::index(self) -> usize` returning `color as usize * 6 + piece_type
  as usize` (0..12) — we do NOT adopt Stockfish's 16-slot packed Piece.

## New core types

- **D5**: `Value` is a type alias `pub type Value = i32` (module `types`),
  NOT a newtype. Rationale: search formulas are dense arithmetic; newtype
  ceremony buys little here. Constants mirror Stockfish semantics:
  `VALUE_MATE = 32000`, `VALUE_INFINITE = 32001`, `VALUE_NONE = 32002`,
  `VALUE_DRAW = 0`, `MAX_PLY = 246`, TB/mate bands, and predicates
  `is_win / is_loss / is_decisive / is_mate` as free functions or a `ValueExt`
  trait. Mate scores are ply-relative from day one (`mate_in(ply)`,
  `mated_in(ply)`, `value_to_tt`, `value_from_tt`).
- **D6**: `Depth` is `pub type Depth = i32` (negative sentinels needed:
  `DEPTH_QS = 0`, `DEPTH_UNSEARCHED = -2`, `DEPTH_NONE = -3`). Ply is `usize`
  where it indexes arrays, `i32` in value arithmetic.
- **D7**: `ExtMove { mv: Move, value: i32 }` with `#[repr(C)]`, asserted
  8 bytes, lives in the MovePicker layer, not in `MoveList`.

## Board / state architecture

- **D8**: `StateInfo` is split into two structs: `CopiedState` (Copy: the five
  Zobrist keys, non-pawn material, castling rights, rule50, plies_from_null,
  ep square) and derived fields (checkers, blockers, pinners, check_squares,
  captured piece, repetition). History is a ply-indexed `Vec<StateInfo>`
  **owned by `Board`** — no pointer chains. `make_move` pushes, `unmake_move`
  pops. `Board::clone()` therefore clones history; search threads each own
  their Board copy.
- **D9**: Wire ALL five Zobrist keys from the start: `key`, `pawn_key`,
  `material_key`, `minor_key`, `non_pawn_key[2]`, with Stockfish's tricks:
  zeroed pawn psq on promotion ranks, castling keyed by the whole 4-bit mask,
  material key via count-slots, ep key only when a legal ep capture exists,
  rule50 key folding at >= 14. These are near-impossible to retrofit.
- **D10**: `make_move(&mut self, m: Move, gives_check: bool)` — caller
  supplies gives_check. Castling is encoded internally as king-to-castling-
  destination (our current Move encoding); the make/unmake implementation
  must still handle Chess960-style rook handling internally
  (remove-both-then-place-both), but Chess960 support itself is deferred.
- **D11**: SEE (`see_ge`) is a `Board` method (it needs occupancy, pinners,
  attack queries). SEE piece values: P 208, N 781, B 825, R 1276, Q 2538.

## Attack tables

- **D12**: Leaper tables (knight/king/pawn), `BETWEEN`, `LINE`, distance:
  built at compile time via `const fn`. Slider attacks: **fancy magic
  bitboards** (packed shared table, Carry-Rippler enumeration, epoch-array
  verification, hardcoded per-rank PRNG seeds), initialized once at startup
  behind `std::sync::LazyLock<AttackTables>`. No PEXT variant (current SF
  dropped it for sliders). Revisit hardcoding found magic constants later
  only if startup time ever matters.

## Module map (single crate, lib + bin)

- **D13**: One crate. Planned new modules as they land:
  `attacks` (leapers + magics), `zobrist`, `movegen`, `perft`, `see` (or in
  `board`), `eval` (later a dir), `search`, `tt`, `movepick`, `history`,
  `timeman`, `uci`, `threads`, `bench`. `main.rs` stays a thin UCI entry.

## Unsafe, dependencies, toolchain

- **D14**: Unsafe budget: allowed only for (a) prefetch intrinsics, (b) SIMD
  intrinsics in NNUE/eval, (c) proven-hot unchecked indexing AFTER a profile
  shows the bounds check and a comment records the measurement. Each unsafe
  block gets a `// SAFETY:` comment. TT relaxed atomics are SAFE Rust.
- **D15**: Zero runtime dependencies. Dev-dependencies allowed (e.g.
  criterion). External tools (cutechess-cli/fastchess, bullet NNUE trainer)
  are tooling, not dependencies.
- **D16**: Edition 2024, current stable toolchain. Release profile:
  `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` for release builds
  used in strength testing; document `RUSTFLAGS=-C target-cpu=x86-64-v3` (or
  native) as the supported fast configuration.

## Search architecture

- **D17**: PVS negamax with aspiration windows, staged MovePicker,
  history-driven ordering. **No killer/countermove tables, ever** — modern SF
  removed them; we skip that detour entirely.
- **D18**: Search stack = fixed array `[Stack; MAX_PLY + 10]` with the search
  operating at offset +7 so `ss-7 ... ss+2` reads are always in bounds
  (Stockfish's padding trick, expressed as safe indexing on a slice).
- **D19**: TT: 10-byte entries (key16, depth8, gen_bound8, move16, value16,
  eval16), all fields relaxed atomics, 3-entry 32-byte clusters, `mul_hi64`
  indexing, 5-bit generations, SF replacement rules. Lock-free and racy by
  design; every consumer validates what it reads (pseudo_legal+legal on
  ttMove, is_valid on values).
- **D20**: Tuned constants: adopt Stockfish's current values as starting
  points, recorded verbatim in the design docs with the formula SHAPES
  emphasized. We keep a single `search/params.rs` (or similar) so constants
  are greppable and tunable; no magic numbers inline without a name.

## Evaluation roadmap

- **D21**: Eval phases. (A) Material + tapered PSQT — enough to validate
  search machinery (done, P4). (B) Hand-crafted eval with texel tuning —
  **DESCOPED to an optional contingency** (amended 2026-07-11): P12/HCE
  existed mainly to generate NNUE training data under the own-data rule, but
  D28 makes Stockfish the bootstrap data generator, so HCE is no longer on
  the critical path. The plan skips it — go PSQT (A) straight to NNUE (C) —
  since HCE terms are largely subsumed by NNUE (throwaway work) and
  AlphaZero-style has no HCE. HCE is built ONLY as a fallback if NNUE stalls
  (slow inference / non-converging training); it is a reversible decision,
  gated on nothing. The NNUE SPRT baseline is therefore Phase A (PSQT), not
  Phase B. Phase order: P9 -> P10 -> P11 (SMP) -> P13 (NNUE). (C) **NNUE** —
  required for the end goal; incremental accumulators (dirty-piece hooks in
  make_move), quantized SIMD with scalar fallback; bootstrap-distill then
  retrain clean per D28; train with `bullet` — we do NOT ship SF's GPL nets.
- **D22**: Eval interface is a single `evaluate(&Board, ...) -> Value` seam so
  phases A/B/C swap without touching search. Correction history corrects the
  static eval inside search (raw eval is what the TT stores).

## Threading

- **D23**: Lazy SMP. Per-thread `Worker` owns everything (Board copy,
  histories, root moves); shared state is exactly: the TT, `AtomicBool` stop
  flags, and options. **Persistent parked worker threads + `Arc<TT>`**
  (amended 2026-07-09 from scope+`&TT`: persistent threads match SF's pool
  reuse across `go` commands and avoid re-spawn cost; docs 01/13 specify the
  mechanism). Consequences: the TT generation counter is interior-mutable
  (`AtomicU8`, relaxed) so `new_search(&self)` works through the shared
  handle; `resize`/`clear` run on the UCI thread between searches via
  `Arc::get_mut`. No locks on the search path; node counters are per-worker
  relaxed atomics aggregated on read. Single-threaded first; the data layout
  is SMP-ready from the start.

## Validation gates (non-negotiable)

- **D24**: No movegen/make-unmake change lands without the perft suite
  passing. A `bench` subcommand (fixed position suite → total nodes + nps)
  is the search regression oracle: functional changes must report the bench
  node count; "non-functional" changes must not alter it.
- **D25**: Strength claims come from games only: fastchess/cutechess SPRT
  vs. the previous version (and reference engines), never from eyeballing.
- **D26**: Every design doc gets an acceptance-criteria section; a phase is
  done when its criteria are met, not when its code exists.

## Licensing

- **D27**: We take ideas, algorithms, constants, and formulas from Stockfish
  (facts are not copyrightable); we never copy code or ship SF network files.
  Notes in `stockfish_reference/` document provenance.

- **D28**: NNUE bootstrap strategy (added 2026-07-11). The net is developed
  in two provenance tiers, AlphaGo → AlphaZero style. (a) BOOTSTRAP:
  distill from a locally-compiled strong Stockfish — positions labeled with
  SF search eval (+ game WDL) — to validate the NNUE *infrastructure*
  (dirty-piece accumulators, the trainer, the scalar/SIMD inference
  backends) and reach strength quickly, decoupling the machinery risk from
  the data-quality risk. (b) SHIP: the released net is retrained from CLEAN
  data (engine self-play / own-engine labels, up to a full self-play "zero"
  restart) once the NNUE machinery is proven. SF-labeled data is therefore
  **discardable development scaffold, never shipped** — the released
  artifact is trained only on our own/self-play data and is ours (consistent
  with D27). Rationale: use distillation to de-risk the hardest phase (P13)
  without conceding the shipped net's provenance. Curation spec + pipeline:
  `design/15-nnue-datagen.md`. The bulk SF-labeled corpus may be generated
  ahead of P13 as a throttled background task once the curation spec is
  validated on a small sample and reviewed. Training hardware: local
  RTX 4060 (bootstrap + first nets); a larger machine is available later.

- **D29**: Research data platform (added 2026-07-11, per user vision). Datagen
  is architected as a research data platform on the principle **store
  observations, not labels**: preserve expensive raw search observations
  (position state, search context, search observations, per-move
  observations) so future labels (value, WDL, policy, search-control,
  cascade/MoE routing, adversarial hard-examples) are cheap DERIVATIONS —
  regenerating deep search is expensive, deriving labels is cheap. Three
  layers: Raw Position Store (canonical, deduped) / Observation Store
  (immutable search records + counterfactuals — PERMANENT) / Derived Training
  Sets (experiment-specific — DISPOSABLE, e.g. the flat bullet export). Full
  spec: `design/16-research-data-platform.md`. **STAGING discipline** (design
  the whole platform now, build it incrementally): (a) the Tier-A bootstrap
  (SF teacher, discardable scaffold, D28) stays SIMPLE — flat bullet records
  — because rich capture on throwaway data is waste AND a foreign teacher
  cannot emit our engine's internal search-control / per-move signals; (b) the
  rich Observation Store is built with Tier-B — our own INSTRUMENTED engine
  generating PERMANENT self-play data (post-first-NNUE); the internal-search,
  per-move, counterfactual, adversarial, and MoE/search-control corpora are
  each built when the research consuming them starts. Provenance (engine /
  net / search-config / generator / seed / opening / teacher versions) is
  mandatory from day one for label reproducibility. Reconciles with the
  `bullet` flat 32-byte format via the disposable Derived layer.

- **D30**: Per-node opponent threat map (added 2026-07-11, from the Reckless
  study — `reckless_reference/07,04,05,09`). P9 **Wave G** adds a per-node
  bitboard of every square attacked by the side-not-to-move, per piece type
  (`all_threats` + `piece_threats[pt]`), maintained in `StateInfo` via
  whole-army *setwise* attack fills (own king excluded from occupancy so
  sliders see through it). It collapses king-move/castling legality to an O(1)
  mask test and is then reused — for free — as move-ordering features
  (threat-escape + a symmetric offense term) and as a threat-coupled RFP margin
  dimension. Scheduled AFTER the existing waves A–F (the highest-value NEW idea in the study
  but also its highest-effort/risk), gated by the usual per-item proxy SPRT.
  **Non-negotiable:** the map must be perft-clean and bench-node-neutral (full
  D24 ritual + `Bench:` trailer) — it redistributes cost without changing the
  node count. Reimplemented idiomatically in safe Rust; the Reckless version is
  prior art only (AGPLv3, D27). Touches `design/03` (setwise helper),
  `design/04` §3.5 (new StateInfo field), `design/05` §3.8 (legality arms),
  `design/09` §3.4/§3.7 (ordering features), `design/10` §3.4 (RFP term); the
  Wave-G spec lives in `design/10` §4.

- **D31**: Reckless-study incorporation — declined adaptations (added
  2026-07-11). The study (`reckless_reference/00-design-implications.md`) was
  triaged into the roadmap; ADOPTED items are baked into `design/10` (P9 wave
  refinements + a standalone-candidate backlog + Wave F) and annotated into
  `design/04/09/11/12/13`. The following are recorded as prior art and
  DECLINED, each affirming an existing decision: (a) baking
  `is_capture`/`is_noisy` into `Move` kind bits — affirms **D2** (pinned u16
  layout); (b) Reckless's unsafe posture — non-atomic TT under
  `unsafe impl Sync`, `static mut`, a self-referential raw-pointer Stack —
  affirms **D14/D19** (measured/documented unsafe only; per-field relaxed
  atomics; miri-clean); we take only ideas (e.g. the SWAR cluster-tag lookup)
  atop our atomic contract; (c) Syzygy/Fathom tablebases — vendored C +
  `cc`/`bindgen` + runtime libc + unsafe FFI — affirms **D15** (zero runtime
  deps); deferred past P13 as a possible future non-default opt-in feature with
  its own decision; (d) reduction/draw-dither RNG jitter — would break **D24**
  bench determinism if added pre-SMP; deferred to P11. Datagen is UNAFFECTED by
  the study's threat-feature finding: **D29**'s raw observation store
  regenerates any feature projection (including relational threat features);
  only the disposable `bullet`/Chess768 export (`design/15` §3.2.3) would be
  amended if/when we train threat features.

- **D32**: Batched P9 Waves C–G (added 2026-07-11, per user). For the remaining
  P9 pruning/reduction/extension waves (C–G — doc 10 items 11–21 plus the
  Wave-G threat map), we one-shot IMPLEMENT + unit-test every heuristic first,
  then gate the WHOLE batch with a single SPRT vs the pre-C baseline — rather
  than a per-heuristic proxy SPRT (the Wave A/B rhythm). Rationale: these are
  clean transcriptions of proven Stockfish/Reckless heuristics (correctness risk
  is low and caught by unit tests + the perft/invariant oracles), so the
  per-item SPRT wall-clock — not the coding — dominated wave time; and the C–G
  aggregate is expected strongly positive (~+150..250 at fixed time vs the
  PSQT-only baseline), so a single flat/negative interaction cannot sink it.
  Consistent with **D25** (strength still comes from games, just batched) and it
  preserves attribution: each heuristic stays ONE commit behind a `params.rs`
  constant (D20), so a flat/negative batch is bisected by git/param toggle, not
  re-coded. The Wave-G threat map (D30) is verified perft- and bench-node-neutral
  at its own commit (D24), independent of the batch's Elo. The batch SPRT is a
  CORRECTNESS TRIPWIRE, not a strict Elo gate (per user, 2026-07-11): the
  completed heuristics are expected net-positive once the full system (NNUE +
  greater depth + SMP) is in place, so a SMALL regression now is ACCEPTED and
  kept (micro-tuned in the later SPSA pass). Only a LARGE or surprising
  regression — which signals an implementation BUG, not a weak heuristic —
  triggers a bisect-and-fix before moving on. Rationale: a silent search bug
  would corrupt everything built on top (NNUE training signal, SMP search), so
  cheap bug-detection is worth it; incremental Elo optimization is not the P9
  bottleneck — NNUE, SIMD, and concurrency are the priorities.

- **D33**: Multi-agent review loop before behavioral testing (added 2026-07-11,
  per user). For any substantial batched or one-shot implementation (e.g. the
  D32 P9 C-G batch; future large changes), BEFORE game/behavioral testing we run
  an iterative multi-persona code-review loop. Each ROUND: N parallel reviewer
  subagents (default 3), each a DISTINCT persona — (A) correctness/safety
  auditor, (B) reference-fidelity vs the `stockfish_reference/` +
  `reckless_reference/` notes, (C) Rust-idiom/perf/test-quality — report
  structured findings with severity; then a synthesis/adjust subagent VERIFIES
  each finding (discards false positives with a recorded reason), applies the
  warranted code+test fixes, re-runs the correctness gate
  (fmt/clippy/test/perft/oracle/selfplay), and commits a "review round N fixes"
  commit. The A/B/C → adjust cycle REPEATS until a round surfaces no material
  (high/medium, verified) finding — convergence — capped at ~4 rounds as a
  runaway guard. THEN, and only then, behavioral/game testing (SPRT) runs.
  Reviewers are READ-ONLY; only the adjust agent mutates. Rationale: cheap
  adversarial scrutiny catches the bugs unit tests miss BEFORE expensive game
  time, and a silent search bug would poison everything built on top (NNUE
  training signal, SMP search). Full mechanics: `design/14` §10.

- **D34**: The engine's differentiating bet — search-integrated NNUE (added
  2026-07-11, per user; the stated reason for building this engine). Beyond a
  conventional value net we build a shared-trunk **multi-head** network whose
  evaluation AND search control are **co-trained** from the D29 observation
  store, so eval and search co-evolve (the signal a foreign teacher cannot give
  — D29 §3.8). Two flagship thrusts: **(I) uncertainty coupling** — the net
  predicts the deep-search residual (Layer-2 static vs Layer-3 deep — a *free*
  label) and that scalar drives extensions / reductions-pruning / time
  management; **(II) learned policy & search control** — a policy/control head on
  the *same* trunk guides ordering, reductions, and pruning, moving off D20's
  hand-tuned constants via a fallback-safe correction (mirroring correction
  history). Thesis: at fixed time, strength is decision-quality per
  node-microsecond, not centipawn accuracy — a cleaner engine that *searches
  smarter* can beat a bigger hand-tuned evaluator; this is the lever aimed at
  beating Stockfish. GOVERNING CONSTRAINT: in-tree heads must be amortized on the
  shared (sunk-cost) trunk or be tiny — SIMD/quant efficiency (doc 11 §3.4.6) is
  the per-node budget that ranks what is affordable. Staged post-P13, gated on
  Tier-B corpora (D28b/D29); the uncertainty head is the first novel thrust (its
  label is free). Aspirational and genuine research (roadmap
  `design/17-nnue-research.md`, §9 risks); the D29 platform exists precisely to
  preserve the signals these heads need.

- **D35**: P13 NNUE start = inference-first, bullet-oracled, own-net-first
  (accepted 2026-07-12, user). The way we START P13, to validate search+infra
  with a real net before trusting our own training while maximizing reuse and
  minimizing licensing entanglement:
  1. Build P13's NNUE **inference for our OWN planned architecture** (Chess768 /
     `bullet` format — the training half already validated) FIRST; this is
     reusable P13 code, not a throwaway experiment.
  2. **Bit-exactly oracle** the quantized forward pass against **`bullet`'s own
     reference inference** (permissive, clean) on shared positions — NOT against
     a foreign engine's net. This de-risks the hardest, most bug-prone part of
     P13 (cross-ISA quantized-SIMD bit-exactness, `design/17` §5, note 13 of the
     Reckless study).
  3. **Lead with training our OWN first net** from the live datagen corpus
     (~22M+ positions and growing, D28/D29): "our search + our real net = X Elo"
     is the eval-ceiling / search-vs-training bottleneck diagnostic, with ZERO
     licensing questions, and it directly advances P13.
  4. The **Reckless-net A/B** (our search vs Reckless search, eval held
     constant) is an **OPTIONAL, dev-only, FIREWALLED** second step: implement
     the Reckless *threats-input* loader only if still wanted after our own net
     exists; the AGPLv3 net stays **gitignored, never in a shipped build, never
     networked** (AGPL private-use/testing only) — treated exactly like the
     gitignored SF teacher scaffold, and it does NOT touch the D28 clean
     provenance of the SHIPPED net. Reimplement inference from the Reckless Rust
     as prior art only (D27/D31), never copied. Owning docs: `design/11` §3.4
     (arch/quant/inference), `design/17` (the search-integrated heads on top).
  **FINALIZED & GO (user, 2026-07-12) — see `design/11` §11 for the full step
  plan.** Locked: first net = SMALL `Chess768 → 2×256 → 1` (not final — smallest
  that proves the loop); train net-0 on WHATEVER the SF-teacher corpus has now
  (~23M+; we retrain on fresh data later regardless); execution = consolidated
  impl agent → D33 review → my bench-neutrality + bullet-oracle gates → local
  SPRT vs the PSQT baseline (the "NNUE beats PSQT" = it-works milestone); the
  Reckless-net A/B is DEFERRED to AFTER our first NNUE works and shows a big Elo
  jump, then the firewalled loader is added as a stress-test/baseline. Ready to
  start P13 step 1 (inference skeleton) on the user's go.

- **D36**: Testing-throughput strategy — cloud-burst distributed SPRT/SPSA + a
  proactive SPSA scaffold (from the Integral study, 2026-07-12; user-confirmed).
  The Integral study's dominant finding: **~half of "how a solo dev reached
  top-6" is free access to a distributed test fleet** — a "+2 Elo" verdict needs
  ~48k games (`design/14` §3.8 error math), and Integral ran millions of games
  over its climb; impossible on one desktop. Our `tools/sprt.sh` single-machine
  setup (the §3.8 "concession to local hardware") is the biggest lever we are
  NOT yet using. Resolution (user): **rent it when we need it** — 5–10 GCE spot
  instances running OpenBench-shaped workers, spun up for a tuning burst / SPRT
  batch and torn down after; cheap on spot pricing. TIMING: distributed compute
  enters at the **tuning grind (post-first-NNUE, P13 tail → P14)** — NOT before;
  all of P13's core (inference, first-net, pipeline validation) is single-machine
  work gated by local SPRT. PREP NOW (near-zero cost, so the cloud burst is
  plug-and-play): (a) build the `cargo feature="spsa"` **`TUNABLE` scaffold over
  `search/params.rs`** — dead weight without compute but trivial to add, and
  Integral wired it in *before* it had the compute (our constants are all untuned
  SF starting points); (b) keep `bench` stdout OpenBench-parseable. Owning doc:
  `design/14` §11. Supersedes the §3.8 "local only" concession once the scaffold
  + a cloud OpenBench instance exist.

- **D37**: Integral-study incorporation (2026-07-12). A 13-agent study of Integral
  (`integral_reference/`, top-6 solo-dev C++ engine, aronpetko; ALL-RIGHTS-
  RESERVED — no license — so ideas-only per D27, even harder than SF/Reckless).
  It surfaced FEW novel search techniques (we already source current SF), so its
  value is dominantly **validation + the D36 tooling lesson**. VALIDATED (record
  so a future reader doesn't second-guess): our `Bench:` trailer + one-idea-per-PR
  SPRT rhythm (D24/D25/D32), the `bullet` trainer + UHO opening book (D28 — two
  unrelated top engines independently converged on both), and the iterative NNUE
  self-bootstrap (D28/D34) — Integral reached top-6 with PURE self-bootstrap (no
  foreign teacher), confirming the SF-teacher shortcut (D28a) and the Reckless-net
  A/B (D35) are genuinely OPTIONAL, not load-bearing. ADOPTED (folded into docs):
  a small set of cheap search micro-refinements (e.g. an `eval_complexity` distrust
  signal reusing the correction-history byproduct — `design/09` §3.10 / `design/10`
  RFP-LMR) as P9-tail SPRT candidates; the datagen recipe corroboration + the
  DATAGEN-flag footgun (strip search-only rule-50/optimism from training labels —
  `design/15`/`design/16`); the sharpened P13 lesson (get the full
  datagen→train→embed→bench loop working cheap-and-early before polishing —
  `design/17`). DECLINED: **killer moves** — Integral uses them; our **D17**
  deliberately dropped killers for the history stack. Record as prior art;
  affirms D17, do not adopt. Adaptation list: `integral_reference/
  00-design-implications.md`; success-factors: `integral_reference/
  00-success-factors.md`.

- **D38**: Experimental ledger (2026-07-12). A content-addressed **provenance
  DAG + results ledger** tracking the engine's evolution — nets, corpora, code
  versions, and their measured results — so any past version is reproducible and
  any net's lineage is provable (the D28 clean-provenance goal). Three append-only
  git-tracked JSONL registries under `ledger/` (`nets`, `data`, `experiments`) +
  a curated `milestones.md`, driven by a stdlib-only `tools/ledger/ledger.py` CLI
  (`add-net`/`add-data`/`add-exp`/`verify`/`show`/`reproduce`). Two keystone
  invariants: **identity = sha256** (friendly ids are labels; the hash is truth)
  and **every artifact carries its regeneration recipe** (git SHA + input hashes
  + command — extends the D24 `Bench:` trailer and the D29 corpus `meta.json`).
  **Metadata in git, bytes in a content store** (nets/corpora stay gitignored
  under `data/…`; a hash-keyed **GCS mirror comes later, D36**; OpenBench becomes
  the execution + web-UI layer, referencing net hashes from here). Format = JSONL
  + markdown (diffable, zero-dep). Build MINIMAL now (discipline is the value, not
  infra — Integral lesson, D37); grow toward OpenBench at the tuning grind.
  SPSA tuning runs are experiments too (`kind:"spsa"`). Owning doc: `design/18`.

- **D39**: The strength decomposition & anti-confound testing protocol
  (2026-07-13). Playing strength factors into nearly-orthogonal axes —
  **Elo ≈ f(net quality, search quality, inference speed)** — and the cardinal
  sin is measuring one axis through a test that silently moves another. Rules:
  (a) **Fixed-node (or fixed-depth) games isolate net + search quality** by
  removing the speed axis — this is the RESEARCH / screening metric. (b) **Real-TC
  is a SHIP GATE, never a net- or search-quality measurement**: it blends all
  three axes, so a real-TC delta decomposes into (fixed-node Elo) + (speed Elo,
  from the §3.1 nps→Elo rule of thumb). (c) **Inference speed is a
  separately-LOGGED axis** — the `nps` field on ledger net rows (D38); a slow
  first-cut inference path is an implementation accident, not a property of the
  net. (d) **Never compare two nets/arches across different-maturity inference
  paths and blame the artifact** — cautionary tale: our hand-port of Reckless's
  net ran the *same* net at 0.34 Mnps vs Reckless's 0.95 Mnps native, and the
  battery's "we're 4× slower" headline was timing the port, while our real engine
  (net-0) runs **2.36 Mnps ≈ 2.5× FASTER than Reckless** (perft 301 Mnps; movegen
  is not the bottleneck). (e) **Hold arch + quant constant ⇒ no speed confound,
  test at any TC** — the whole Lc0/data campaign shares the net-0 arch, so those
  net-vs-net tests are already clean. An arch/quant change moves net-quality AND
  speed together → **screen at fixed-node first, THEN invest in production
  inference, THEN real-TC ship gate**. (f) **Tuning is eval-coupled: SPSA-tune
  search params on the SHIPPED net, not a throwaway (net-0)** — tuning against
  net-0 then swapping nets re-mistunes the margins; this + D36 fix the order (good
  net → then the tuning grind). (g) **Search quality is the fourth axis and is
  presently a tuning-maturity gap, not missing machinery** — at equal 100k nodes
  with the same net our search is ~87 Elo behind Reckless (T1, 2026-07-13), but we
  already ship singular/double/triple extensions, ProbCut, multicut, LMR; the 240
  `search/params.rs` constants are SF defaults never SPSA-tuned for our engine
  (the IIR −15 revert is the tell). Largely planned debt behind the net (D36).
  (h) **External improvement suggestions are triaged by axis and VERIFIED against
  our code + decisions before adoption** — a "Reckless does X better" list is
  typically part already-present (king-danger x-ray, `board.rs`), part
  declined-on-principle (reduction-jitter breaks D24 bench reproducibility), part
  net-coupled (threat-map ordering), and only partly net-new. (i)
  **Architecture-specific ISA optimizations are deferred until Top-10 strength +
  TCEC preparation** (user, 2026-07-13). Target-specific SIMD in
  movegen/SEE/board (e.g. Reckless's AVX-512 `maskz_compress_epi16` branchless
  move-pack and `test_epi64_mask` SEE least-valuable-attacker — both
  `#[cfg(target_feature = "avx512…")]`-gated, so on our AVX2 target Reckless runs
  our exact scalar code) is a real-TC *speed-axis* polish worth only ~a few Elo,
  and ONLY on specific silicon: it forks the build/test matrix (D16 — a v4 binary
  won't run on v3 hardware, and SPRT pairs must share flags), its value SHRINKS as
  the net grows (eval dominates node cost), and it moves the axis where we are
  already ~2.5× ahead of Reckless. Until that milestone we ship ONE portable fast
  path (`x86-64-v3` / AVX2) + scalar fallback; the deferred items are parked in
  the triage backlog. Owning doc:
  `design/14` §3.11 (NNUE application: `design/17` §5); the `nps` metadata lands in
  D38's `nets.jsonl` schema; the triaged Reckless backlog lives in
  `reckless_reference/00-design-implications.md`.

- **D40**: SOTA NNUE architecture & the triple-net training discipline
  (user-approved 2026-07-13). Skip design/11 §3.4.1's intermediate SPRT rungs (a
  deliberate deviation from D25) to build a SOTA-class net directly. **Features =
  Stockfish's exact set** (extracted from `stockfish_reference/`, 2026-07-13):
  `HalfKAv2_hm` (22,528; 32 king buckets, horizontal mirror, i16 FT weights) +
  `FullThreats` (60,720; relational attacker→victim + move-geometry, i8 FT
  weights) = **83,248** sparse inputs, two transformers summing into one
  accumulator. **Head = Reckless-shaped lean hybrid** (studied 2026-07-13):
  pairwise-multiply FT activation → **int8-sparse-FT + f32 tail** → 8 output
  buckets, **no PSQT pathway, no output-skip** — the exact head we already
  reimplemented bit-exactly on `experiment/reckless-net` (0 cp / 42 positions).
  That reimplementation is **our own code (D31-clean) and moves to main**; only
  Reckless's trained *weights* stay firewalled (we train our own). Accumulator
  768 wide (Reckless-proven; 512↔1024 is an SPRT knob), 8 buckets (4↔8 a knob).
  **Trainer = bullet, extended** (custom FullThreats input feature + configurable
  policy head + combined eval+policy loss). **Triple-net discipline** (refined
  mid-design from an initial dual eval/eval+policy framing): every training config
  produces THREE independently-trained nets on one trunk *architecture* — {eval-only,
  eval+**small** policy (rank-8 bilinear from×to), eval+**full** policy (4096
  from×to)}, all on the Lc0 soft-policy target. eval-only is the control that
  proves a jointly-trained policy head is not degrading eval; small-vs-full
  measures the cheap head's prediction cost. Policy heads have **no SF/Reckless
  precedent** — novel research (D34). **Data** must carry eval/WDL **and** the
  Lc0 policy target per position (a corpus-format extension — the real "curation"
  work, design/15). Supersedes the *architecture* parts of D35's first-net plan;
  net-0..net-10 remain the validated scaffold + a ready net-quality A/B. Owning
  doc: `design/11` §12. Reference provenance: the two SF extractions + the
  Reckless study, all 2026-07-13.

- **D41**: ARCHON learned counterfactual search allocation (user-directed
  research pivot, 2026-07-14). The engine and research program are named
  **ARCHON**. Strong external engines, especially Lc0, supply valuable,
  diverse **positions** and may supply bootstrap targets, but their policy or
  visit distributions are not ARCHON's final search-control target. ARCHON
  relabels those states with a conservative, deeper version of its own search
  and learns a search-action consequence model
  `Q(position, search_context, legal_move, proposed_action)`. Its outputs must
  include calibrated bound-error/re-search risk, cutoff and alpha-raise
  probability, and expected log nodes/cycles; later outputs may cover
  extensions and the value of buying cheap/full policy inference. A small,
  chess-agnostic constrained planner chooses the least-cost action satisfying
  a configured error-risk bound. The learned model predicts consequences; it
  does not imitate one universally “correct” reduction.

  The strongest available evaluator is the protected baseline. Initially its
  value behavior is frozen and bit-identical while node-control and legal-move
  consequence heads train; separate adapters are allowed and uncontrolled
  shared-trunk gradients are forbidden. Authority expands only through
  bounded corrections with the classical controller retained as behavior
  policy, fallback, and control. Search ordering and reductions precede hard
  pruning. Data collection is an iterative flywheel: current ARCHON supplies
  states, slow ARCHON oracle searches supply counterfactual labels, games
  select models, deployed controllers expose new states, and failures and
  disagreements are mined and relabeled.

  Every stage must pass offline calibration/OOD tests, paired fixed-node tests
  over **multiple node budgets** with pair-bootstrap intervals, and equal-time
  tests accounting for NPS and invocation cost. The governing objective is the
  Elo-versus-compute frontier, ultimately maximum Elo/second, not imitation
  accuracy. D41 supersedes D34's requirement that all tasks co-train one shared
  trunk and D40's use of Lc0 soft policy as the final target. D34's search-
  integrated thesis, D40's feature/value architecture, and completed policy
  experiments remain useful foundations and controls. Owning roadmap:
  `design/19-learned-search-allocation.md`; experimental history remains in
  `design/17-nnue-research.md` and `docs/research/policy-automl-state.md`.

- **D42**: External value-oracle bootstrap and frozen value representation
  (user-directed implementation detail, 2026-07-14). When D41 work needs a
  value oracle other than ARCHON, use both (a) the most recently compiled local
  Reckless release build and (b) the official stable **Stockfish 18** build.
  Stockfish 16—the teacher used for `net-0`—must not silently substitute. Pin
  each executable by path, UCI identity, source revision where known, binary
  SHA-256, options, and label budget in dataset provenance.

  The first representation target is paired Reckless/SF18 value supervision on
  the same positions. Preserve the two raw labels separately; calibrate their
  score/WDL scales before deriving any consensus target, and retain their
  disagreement as uncertainty/hard-case information. Never average raw engine
  centipawns across uncalibrated scales. A candidate value model must pass the
  evaluator strength gate before replacing the prevailing baseline. Once the
  baseline is selected, freeze the complete value accumulator and value head
  bit-identically while training the first ARCHON search-control heads. This is
  the causal isolation: improvements or regressions belong to learned search
  control, not simultaneous value drift. Owning roadmap: doc 19 §§4.2, 5.3,
  and 10.

- **D43**: Search-control data are intervention-aware (user-approved parallel
  head/data track, 2026-07-14). Maintain two semantically distinct corpora:
  **observational traces** record what the current heuristic ARCHON encountered
  and did; **forced counterfactual interventions** rerun the same pre-action
  state with one declared search-action axis changed. Observations provide
  on-policy coverage but cannot establish the causal consequence of a different
  reduction. Every record and derived label identifies its evidence kind.

  Final-best-move and pairwise targets require a stable conservative reference
  (agreement across increasing depth/budget, explicit ties/margins); contextual
  alpha-raise and cutoff use the alpha/beta captured before the move. Reduction
  error compares bound class against a full-depth reference under the same
  parent context. Re-search trigger, changed bound, and necessity are separate
  labels. Node cost is the first authoritative cost target; cycles/time are
  calibrated in warmed pinned batches rather than trusted per tiny subtree.

  Counterfactual arms start from identical board/history/config/node-counter
  state, isolated fresh TT state, and frozen or restored history writes. An
  asserted event fingerprint prevents an override from applying to the wrong
  node. One action axis changes per experiment; changing reduction must update
  both actual scout depth and the parent `Stack.reduction` observed by child
  hindsight. Initial reduction data include only ordinary LMR-eligible moves
  and exclude entangled singular/check/pruned/stopped cases. Root v0 may use
  independent `searchmoves` arms with cold state before internal intervention
  instrumentation lands. Owning schema: doc 16; model/controller roadmap:
  doc 19.

- **D44**: Reckless-to-ARCHON search-control curriculum (user-directed,
  2026-07-14). ARCHON need not produce every initial search-control label.
  Bootstrap the first move-quality representation with independent cold root
  move analyses from the newest pinned Reckless build: stable best move,
  pairwise ordering, per-move score/margin, PV stability, and root node cost at
  increasing budgets. Preserve `producer_engine` on every observation and use
  one schema/tool that can rerun identical source positions with ARCHON.

  External labels are target-scoped, not interchangeable with native search
  consequences. Contextual cutoff, reduction/bound error, re-search, and exact
  action cost depend on ARCHON's windows, ordering, TT, histories, pruning, and
  reduction semantics; their production targets must transition to ARCHON
  observations/interventions before deployment. Do not invest in invasive
  Reckless internal-LMR instrumentation unless a later controlled experiment
  justifies it. Curriculum: Reckless root bootstrap → ARCHON root adaptation →
  ARCHON internal counterfactuals → on-policy ARCHON flywheel. Promotion is
  always determined by ARCHON engine games. Owning roadmaps: docs 16 and 19.

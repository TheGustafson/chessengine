# 00 — Master implementation plan

Status: Living (updated as phases complete; last revised 2026-07-10)

This document turns the reviewed design suite (`01`–`14`) into an ordered
build plan: phases with entry/exit criteria, a dependency graph, a milestone
ladder with objective gates, a risk register, and the consolidated open
questions triaged from the drafting process. It owns *sequencing only* —
every technical statement below defers to the owning component doc, and every
"done" defers to that doc's §8 acceptance criteria (D26). Where this plan and
a component doc disagree, the component doc wins and this file gets corrected.

---

## 1. Vision

A UCI chess engine in Rust with Stockfish-class architecture and, ultimately,
strength (decisions.md, project goal). Three commitments in priority order:
correctness (perft-verified, bench-reproducible — D24), performance (our data
structures must never be the reason we can't match Stockfish's per-node
costs), and idiomatic safe Rust (`unsafe` only with a measured, documented
justification — D14). Ideas, constants, and formulas come from current
Stockfish as citable facts; code is never copied and SF networks are never
shipped (D27). All strength claims come from games (SPRT/gauntlets), never
from eyeballing (D25).

The path: build the board layer to exact correctness first (attack tables →
state/Zobrist → movegen → perft green), then a deliberately naive but
*playable* engine (PSQT eval + negamax + minimal UCI), then grow strength in
SPRT-gated increments (TT → ordering → pruning waves → time management →
SMP → HCE → NNUE), ending in a tuning/parity-chase loop. Every layer is
designed SMP-ready and NNUE-ready from day one (five Zobrist keys, hook
seams, per-worker data layout) precisely so nothing structural needs
retrofitting.

---

## 2. Current state

Done (pre-suite): `types` (Color/PieceType/Piece/Square), `bitboard`,
`board` (hybrid bitboards + mailbox), `fen`, `moves` (`Move` u16 +
`MoveList`).

Phases complete: **P0–P6; milestones M1–M4 passed.** P1 landed the
attack tables (leapers + fancy magics, doc 03 §8 green); P2 the state
layer (StateInfo/Zobrist/make-unmake/repetition, doc 04 §8 items 1, 6–8);
P3 movegen + perft (docs 05 and 06 §8.1–8.6 green, doc 04 items 2–4
closed). M2 numbers: full 23-position suite exact in debug-with-oracle
and release, kiwipete d5 bulk perft ~340 Mnps in the D16 fast
configuration (floor 100, stretch 250), mutation test caught injected
ep-pin and castling-rights bugs (audited 2026-07-10). P4–P5 landed eval
phase A, negamax+qsearch, the UCI loop, and the strength harness (M3:
complete legal games at fixed movetime, mate-in-1..3 exact, bench
fingerprint bit-reproducible). P6 landed the transposition table (doc 08
§8 items 1–8 green; bench nodes −46.6% at equal depth, 387698251 →
206861941; TT-on/off value-equality suite exact). M4 numbers (audited
2026-07-10, engines/runs/gauntlet-p6audit + docs/strength-log.md):
gauntlet vs SF16 UCI_Elo 1700–2200 at 2+0.02, 200 games — 69% vs the
1900 rung, 65% vs 2000, ordo places the engine between the 2000 and 2200
rungs (UCI_Elo-anchor caveat); WAC-revised tactics 154/200 at 100 ms
(pre-TT: 145/200). CLAUDE.md §Status tracks the detail; next phase: P7.

---

## 3. Phase breakdown

Sizes: S ≈ days, M ≈ 1–2 weeks, L ≈ 2–4 weeks, XL ≈ open-ended /
gate-bounded. "Exit" always means the cited acceptance criteria, not merely
code existing (D26). `cargo test` + `cargo clippy --all-targets` clean is
implied at every commit; perft (from P3) and the bench fingerprint (from P5)
gate per D24.

### P0 — Scaffolding & core types (S)
- **Implements:** `01-architecture.md` (plan 1–5), `02-core-types.md` (all),
  `14-performance-testing.md` (steps 1–2).
- **Entry:** now.
- **Deliverables:** release/profiling profiles + documented RUSTFLAGS;
  `#![deny(unsafe_code)]`; `Value`/`Depth` bands, predicates,
  `mate_in`/`value_to_tt`/`value_from_tt`, `PIECE_VALUES`, `Piece::index`;
  `Direction` + `Square` arithmetic + `Bitboard` shift helpers +
  `Move::raw16`; `search/params.rs` seed with provenance header; thin
  `main.rs` + UCI smoke skeleton (`uci`/`isready`/`quit`/garbage-absorption);
  `tools/check.sh` v0; CLAUDE.md sync (fast config).
- **Exit:** doc 02 §8 all; doc 01 §8 items 1–7 (those not blocked on later
  modules); the `uciok` smoke test exits 0.

### P1 — Attack tables (M)
- **Implements:** `03-attack-tables.md` (all).
- **Entry:** P0 (needs `Direction`, `try_offset`, const `Bitboard` helpers).
- **Deliverables:** const leaper tables, `BETWEEN`/`LINE`/`RAY_PASS`/
  `DISTANCE`, naive-ray oracle, fancy magics + `LazyLock<AttackTables>`,
  `Prng` (reused by Zobrist), criterion `benches/attacks`-class baseline.
- **Exit:** doc 03 §8 all — exhaustive/randomized slider validation vs the
  naive oracle, packing hard-asserts, ≤ 5 ns hot rook lookup, init ≤ 10 ms.

### P2 — Board state, Zobrist, make/unmake (L)
- **Implements:** `04-position-state.md` (steps 1–8, 10; SEE = step 9 may
  land here or in P4; step 12 cuckoo is deferred to P9).
- **Entry:** P1 (attack queries for check info; steps 1–3 can start during P1).
- **Deliverables:** `zobrist` module (five keys, all SF tricks — D9);
  `CopiedState`/`StateInfo` split + `Vec<StateInfo>` owned by `Board` (D8);
  `make_move(m, gives_check)`/`unmake_move`/null move (D10); check/pin
  bookkeeping; repetition + `is_draw`; debug `assert_consistent` key oracle;
  `MakeMoveHooks` prefetch/NNUE seam (bench-neutral); FEN ep validation;
  CLAUDE.md correction (Zobrist deltas live in `make_move`, not
  `put_piece` — doc 04 §9.1).
- **Exit:** doc 04 §8 items 1, 6, 7, 8 (items 2–5 close with P3/P4 when the
  perft oracle and SEE fixtures exist).

### P3 — Movegen + perft green (L) → **milestone M2 (M-perft)**
- **Implements:** `05-movegen.md` (all), `06-perft-validation.md`
  (steps 1–5), `14` (steps 3–4: criterion perft/movegen benches).
- **Entry:** P2.
- **Deliverables:** `generate::<GT>` (captures/quiets/evasions/non-evasions),
  `generate_legal`, `legal`/`pseudo_legal` (+ 64k fuzz), `perft`/
  `perft_checked`/`divide`, tricky-position suite + test tiers,
  `tools/perft_diff.py`, one-off mutation testing; closes doc 04 items 2–4.
- **Exit:** every §5.3/§3.4 perft count exact in debug (oracle on) AND
  release; kiwipete d5 bulk perft ≥ 100 Mnps (target 250); doc 05 §8 all;
  doc 06 §8 items 1–6.

### P4 — Search prerequisites: SEE, eval phase A, UCI base (M)
- **Implements:** `04` step 9 (`see_ge`), `11-evaluation.md` steps 1–4
  (Phase A), `12-uci-time-management.md` steps 1–4, `06` step 6 (bench perft
  mode).
- **Entry:** M-perft.
- **Deliverables:** `see_ge` + fixture suite (D11 values); `Score`/phase
  tapering, PeSTO PSQT, `evaluate` seam + `EvalContext` stub (D22),
  `flipped()` symmetry suite; `timeman` `Limits`/`TimeManager::init` with
  golden tests; UCI `position`/moves/text layer/options table; deterministic
  `bench ... perft` fingerprint.
- **Exit:** doc 04 §8 item 5; doc 11 Phase A criteria 1–5; doc 12 steps 1–4
  done-conditions; two consecutive bench-perft runs byte-identical.

### P5 — Search core v1: a playable engine (L) → **milestone M3 (M-uci)**
- **Implements:** `07-search-core.md` steps 1–5, `12` steps 5–6, `06` step 7,
  `14` steps 5–7 (fingerprint policy, `Bench:` trailer, determinism test).
- **Entry:** P4.
- **Deliverables:** `Stack`/`PvLine`/`RootMove`s, negamax + qsearch skeleton
  (MVV placeholder ordering), mate-distance pruning, iterative deepening +
  root bookkeeping, `go depth/nodes/movetime/infinite` + `stop` with hard
  stops, exactly-once `bestmove`, `info` emission + throttling, bench depth
  mode + trailer protocol live in CLAUDE.md/check.sh.
- **Exit:** plays a full fixed-movetime game under cutechess with zero
  stalls/illegal moves/duplicate bestmoves (doc 12 step 5); mate-in-1..3
  suite exact (doc 07 §8.4); stop robustness (doc 07 §8.9); bench
  bit-identical across runs (doc 07 §8.5).

### P6 — Transposition table (M) → **milestone M4 (M-1900)**
- **Implements:** `08-transposition-table.md` steps 1–8, `07` step 6,
  `14` steps 8, 10–11 (tt bench; sprt.sh/gauntlet.sh/tactics.sh tooling).
- **Entry:** P5.
- **Deliverables:** 10-byte atomic entries, 32-byte clusters, `mul_hi64`
  indexing, SF replacement rules, `Probe`/`TtWriter`, prefetch wrapper (first
  sanctioned `unsafe`), `Hash`/`Clear Hash`, search integration
  (probe/store/eval cache/prefetch-in-make_move), miri + concurrency smoke;
  strength-testing harness scripts and first gauntlet/tactics baselines.
- **Exit:** doc 08 §8 items 1–8; bench nodes drop materially vs P5 at equal
  depth; TT-on/off value-equality suite; M4 gauntlet gate (§5).

### P7 — Search completion + basic clock TM (M)
- **Implements:** `07` steps 7–10 (PVS zero-window, aspiration windows,
  qsearch full spec, hygiene pass), `12` step 7 (clock TM: optimum/maximum
  driving `check_time`).
- **Entry:** P6. **Note:** basic wtime/btime allocation is pulled forward of
  the task-suggested order because P8/P9's SPRT gates run at real TCs
  (10+0.1) — the engine must manage a clock before the pruning pipeline
  starts. The soft-stop factor stack stays in P10.
- **Exit:** doc 07 §8 all (incl. zero-window invariant, aspiration trace
  test, qsearch rule coverage); 100 games at 10+0.1 sudden death with zero
  time forfeits (doc 12 step 7).

### P8 — Move ordering & history (L)
- **Implements:** `09-move-ordering-history.md` (all).
- **Entry:** P7 (finalized `Stack`, trusted ttMove). Settle the `ContCorrId`
  field question (doc 09 §9.2) when `Stack` is frozen here.
- **Deliverables:** `StatEntry<D>` gravity primitive, `Histories` (butterfly,
  capture, continuation, pawn, low-ply, correction), `ContHistId`, staged
  `MovePicker` (main/evasion/qsearch/probcut pipelines, no killers — D17),
  partial insertion sort, update functions, correction-history machinery.
- **Exit:** doc 09 §8 all — emission-completeness sweeps, golden update
  tests, and the SPRT win vs naive MVV-LVA-only ordering (D25).

### P9 — Pruning waves (XL) → **milestone M5 (M-2450)**
- **Implements:** `10-pruning-extensions.md` waves A–E (F trails), `04`
  step 12 (cuckoo tables / `upcoming_repetition`, consumed by wave E).
- **Entry:** P8.
- **Deliverables:** one-heuristic-per-commit waves, each SPRT-gated at the
  bounds in doc 10 §4: A (reduction table, RFP, NMP + verification, LMR
  core), B (razoring, LMP, futility + SEE pruning, IIR), C (statScore,
  conthist pruning), D (singular extensions, multicut), E (ProbCut,
  hindsight, softening, TT hardening), F (correction couplings — may trail
  into P12). Fixed-node proxy budget calibrated before first proxy-gated
  merge (doc 14 §9.4).
- **Exit:** doc 10 §8 — all waves merged with `bench:` footers and passed
  SPRTs; ≥ +150 Elo vs the pre-Wave-A baseline over 1000+ games at 10+0.1;
  zugzwang/mate/repetition suites green at pinned depths.

### P10 — Time management completion + full UCI (M) → **milestone M6 (M-2750)**
- **Implements:** `12` steps 8, 10–12 (soft-stop factor stack, MultiPV,
  nodestime, dev-surface completion). Ponder (step 9) finalizes with P11
  since its stop discipline lives in the thread manager (doc 13 step 6).
- **Entry:** P9 (soft stop consumes `best_move_changes`/`effort`; can start
  any time after P7 if P9 stalls).
- **Exit:** doc 12 §8 items 1, 4–7, 9; zero time forfeits in 1000 games at
  10+0.1 and 300 at 60+0.6 (item 3, ponder-off part); fixed-depth bench
  signature untouched by TM work (D24).

### P11 — Threading: lazy SMP (L) → **milestone M7 (M-SMP)**
- **Implements:** `13-threading-smp.md` (all), `12` step 9 (ponder), `08`
  step 9 (multithreaded clear).
- **Entry:** P10 (stable single-thread search + TT + UCI). Scheduled before
  the eval phases so `Worker`/`EvalContext` ownership is established before
  pawn tables and NNUE accumulators move in (D23 data layout is already
  SMP-ready; this phase is the mechanism).
- **Deliverables:** persistent parked workers (`Mutex<Job>` + `Condvar`),
  `Arc<TT>` per Search job, `Arc::get_mut` resize-while-idle, per-worker
  counters, best-thread voting, ponder/infinite discipline, TSan/loom
  hardening.
- **Exit:** doc 13 §8 all — `Threads=1` bench bit-identical to the
  pre-refactor baseline, 500-search stress clean, stop latency < 50 ms,
  ≥ 90 % per-thread nps retention at 8 threads, SPRT `Threads=4` vs `1`
  clearly positive; doc 12 §8 items 2–3 (ponder-on parts).

### P12 — Eval phase B: HCE + texel tuning — DESCOPED (optional contingency)
- **Status (amended 2026-07-11, D21/D28):** SKIPPED on the critical path.
  HCE existed mainly to generate NNUE data under the own-data rule; D28 makes
  Stockfish the bootstrap generator, so HCE is off the path and its terms are
  largely subsumed by NNUE. New order jumps P11 (SMP) -> P13 (NNUE), with
  PSQT (Phase A) as the NNUE SPRT baseline and eval fallback. Build HCE ONLY
  as a contingency if NNUE stalls (reversible, gated on nothing). Milestone
  M8/M-2950 is retired; the ladder goes M7 (SMP) -> M9 (NNUE).
- **Implements (only if the contingency is triggered):** `11` steps 5–8.
- **Entry:** P8 (correction history) + P11 recommended; dataset licensing
  check (doc 11 §9.2) before the tuner step.
- **Deliverables:** pawn hash, HCE terms (mobility, king safety, passers,
  ...), `eval` trace command, own texel tuner (`bin/tuner.rs`, std-only),
  retuned params; doc 10 wave F lands here if it trailed.
- **Exit:** doc 11 Phase B criteria 6–8, including the SPRT pass vs Phase A.

### P13 — NNUE (XL) → **milestone M9 (M-3200)**
- **Implements:** `11` steps 9–14 (Phase C), `08` step 10 (huge pages —
  requires the decisions.md D14/D15 amendment first, doc 08 §9.1), `14`
  step 12 (PGO).
- **Entry:** P12 (the HCE engine generates training data), P2 hooks seam.
- **Deliverables:** DirtyPiece recording through `MakeMoveHooks`, CENN net
  format + loader, lazy incremental accumulators, scalar + AVX2 backends
  (bit-exact), datagen (≥ 100 M positions) + bullet training + embedded
  self-trained net (never SF's — D27).
- **Exit:** doc 11 Phase C criteria 9–14, including SPRT vs Phase B and the
  incremental==refresh 100k-node walk; ≥ 70 % of local SF bench nps (doc 14
  §3.4).

### P14 — Tuning & parity chase (XL, ongoing) → **milestone M10 (M-parity)**
- **Implements:** the open-ended tails of docs 10/11/13/14: `tune` cfg
  feature + SPSA (doc 10 §9.1), net upgrade path (L1 512, buckets, PSQT
  head), NUMA/shared histories (docs 09 §9.5, 13 §3.15), bench-list curation
  revisit (doc 06 §9.1), possible Syzygy doc, Chess960.
- **Exit:** never — this is the steady state. Progress is measured by the
  gauntlet ladder (doc 14 §3.8) against recent Stash tags and older SF
  releases.

### Dependency graph (docs; arrows = "must land before")

```
02 core-types ──► 03 attacks ──► 04 position-state ──► 05 movegen ──► 06 perft
      │                │              │      ▲                             │
      │                └── SEE (04§see_ge) ──┘                             │
      └────────────► 11-A eval (PSQT) ──┐                                  │
01 architecture (scaffold, spans all)   │                                  │
12 UCI-min (steps 1–4; move text needs 05)                                 │
                                        ▼                                  ▼
        07 search-core (steps 1–5) ◄── [needs 04+05+06 green, 11-A, 12-min]
                    │
                    ▼
        08 TT ──► 07 completion (PVS/aspiration/qsearch) + 12 clock-TM
                    │
                    ▼
        09 movepick/history ──► 10 pruning waves A–E (E needs 04 cuckoo)
                    │                       │
                    ▼                       ▼
        12 TM-soft/MultiPV/nodestime ──► 13 threads (+12 ponder, 08 mt-clear)
                                            │
                                            ▼
        11-B HCE+texel ──► 11-C NNUE (+08 huge pages, 14 PGO) ──► tuning loop
14 perf/testing: spans everything (check.sh → criterion → bench → SPRT →
                 gauntlet/tactics → PGO); its gates apply from P0 onward
```

---

## 4. Milestone ladder

Testing methods per `14-performance-testing.md`: SPRT = §3.8 parameters
(fastchess, nElo [0,5] gain / [-5,0] non-regression, UHO book, pentanomial);
gauntlet = §3.8 ladder (us vs 3–4 anchored opponents — Stash tags backbone —
≥ 1000 games, ordo ratings, CCRL-blitz-equivalent); tactics = §3.1 tier 1.5
(`tools/tactics.sh`, baseline at M4, no unexplained drop from M5 onward —
doc 14 §3.9.8). Doc 14 §3.8 names its gauntlet rungs M1–M5; the mapping is
noted per row.

| Milestone | Name | After phase | Objective gate |
|---|---|---|---|
| M1 | M-scaffold | P0–P1 | Docs 01/02/03 §8 green: types layer unit-tested; `uciok` smoke exits 0; magics pass the exhaustive+randomized oracle suite at ≤ 5 ns/lookup |
| M2 | M-perft | P3 | Full perft suite exact in debug-with-oracle AND release; kiwipete d5 bulk ≥ 100 Mnps; mutation test caught injected ep-pin and castling-rights bugs |
| M3 | M-uci | P5 | Plays complete legal games under cutechess at fixed movetime (no stalls/illegal/duplicate bestmove); mate-in-1..3 suite exact; bench fingerprint bit-reproducible |
| M4 | M-1900 | P6 | Gauntlet ≥ 1900 (doc 14 rung M1: negamax+qsearch+TT, eval A); first tactics baseline recorded in `docs/strength-log.md` |
| M5 | M-2450 | P9 | Gauntlet ≥ 2450 (rung M2); ≥ +150 Elo vs pre-Wave-A baseline over 1000+ games (doc 10 §8.4); tactics score attributed |
| M6 | M-2750 | P10 | Gauntlet ≥ 2750 (rung M3); zero time forfeits in 1000 @ 10+0.1 + 300 @ 60+0.6 (doc 12 §8.3) |
| M7 | M-SMP | P11 | Doc 13 §8 complete: `Threads=1` bench identical, TSan clean, ≥ 90 %/thread retention at 8 threads, SPRT `Threads=4` vs `1` clearly positive |
| ~~M8~~ | ~~M-2950~~ | ~~P12~~ | **RETIRED** (2026-07-11, D21/D28): P12 HCE descoped — Stockfish is the NNUE bootstrap generator, so HCE is off the path. Ladder skips M8: M7 (SMP) → M9 (NNUE). |
| M9 | M-3200 | P13 | Self-trained net SPRT-beats HCE; gauntlet ≥ 3200 (rung M5); NNUE bench nps ≥ 70 % of locally built SF |
| M10 | M-parity | P14 | Sustained gauntlet climb vs recent Stash tags / older SF releases per doc 14 §3.5/§3.8; every gain SPRT-attributed |

---

## 5. Risk register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **NNUE training compute & data quality** — ≥ 100 M datagen positions need a strong Phase B engine and real machine-time; bullet training needs GPU access; a weak first net wastes the whole P13 cycle | M9 slips badly | Sequence NNUE *after* a tuned HCE engine (P12) so datagen labels are good; start with the small 768→2×256 net and the documented upgrade path; scalar fallback keeps development unblocked; dataset licensing checked before any public data enters the repo (doc 11 §9.2); never depend on SF nets (D27) |
| 2 | **Tuning/SPRT time sink** — every wave in P9 and every eval change is game-gated; local cores make 25k-game SPRTs slow; the temptation is to skip gates or eyeball | Elo regressions merged; D25 discipline erodes | Fixed-node proxy games for search-shape changes (calibrated once, doc 14 §3.8/§9.4); one-heuristic-per-commit so failed SPRTs shrink, not stall (doc 10 §9.3); non-regression bounds for refactors; concurrency honesty rules (no SMT for timed games) |
| 3 | **Per-node cost parity in Rust** — bounds checks, `LazyLock` derefs, accidental `Board` clones, allocation on hot paths silently eroding the "never the data structures' fault" constraint | nps floor misses; Elo lost invisibly | Measurement-first D14 ritual (profile → `// SAFETY:` with numbers); hard nps floors as acceptance criteria (perft 100 Mnps, search per-phase floors, doc 14 §3.4); `objdump` bounds-check metric + `cargo asm` audits (doc 14 §3.10); hook seams monomorphized to zero-cost and proven bench-neutral |
| 4 | **Correctness debt in make/unmake/Zobrist** — the five-key family, ep-only-when-legal, and rule50 folding are near-impossible to retrofit; a subtle key bug poisons the TT and everything above it | Weeks of misdirected debugging in P6+ | All five keys wired from day one (D9); incremental-vs-recompute oracle at every node in debug perft; mutation testing proves the suite catches the classic bug classes; perft is a non-negotiable merge gate forever (D24) |
| 5 | **Nondeterminism breaking the bench oracle** — a single time-seeded value or `HashMap` in a search path destroys the fingerprint that all regression discipline rests on | D24 gates become theater | Determinism requirements are hard invariants with an in-process test (doc 14 §3.2, step 7); dither keyed off node counters; multi-thread bench is nps-only by rule; single-writer relaxed counters |
| 6 | **Measurement environment (WSL2)** — unreliable hardware perf counters and noisy timing on the current dev box | Bad optimization decisions | Elapsed-time medians + cachegrind as the WSL2 fallback; surprising counter readings validated on native Linux before acting; pinned-core, 5-run-median hygiene for any claim (doc 14 §6) |
| 7 | **Scope creep / doc drift** — 14 reviewed docs only stay authoritative if deviations flow back | Docs stop being contracts; reviews lose their reference | Docs-lead-code rule: implementation deviations update the owning doc in the same commit (doc 01 §3.9); seam changes reflected into doc 01 same-commit; decisions change in decisions.md first (see README process rules) |

---

## 6. Consolidated open questions

Triage of the 61 open questions collected from the original drafts,
deduplicated against the fixed docs. Verdicts: **resolved** (fixed docs
already answer it), **decide-now**, or **defer-until-Px** (decide when that
phase starts or its trigger fires).

### 6.1 Resolved since drafting (no action; recorded here to close them)

- **TT sharing/resize mechanism** → D23 amended 2026-07-09: persistent parked
  workers + `Arc<TT>`, interior-mutable generation, `Arc::get_mut` resize
  (docs 13 §9.1, 08 §3.9, 01 §9.3).
- **Bench suite provenance** → SF's 49-FEN standard-chess default list,
  adopted and frozen at bench v1 (docs 14 §9.2, 06 §3.6).
- **`Square::distance` backing** → 64×64 `DISTANCE` table lives in `attacks`
  (doc 03 §3.4); the `const fn` remains the reference implementation.
- **Hash option upper bound** → doc 08 owns it (1..=33554432 MB).
- **Deliberate, recorded deviations that are now spec** — `position` error
  handling (absorb, don't die), currmove throttle (10 M-node gate),
  `mean_squared_score: i64`, breadth-multiplier truncating division,
  reduction-table placement (per-worker): each is documented in its owning
  doc; no further decision needed.
- **Sibling doc filenames** → this suite index (README.md) is the canonical
  reconciliation. Residual stale references are a decide-now sweep (below).

### 6.2 Decide now (cheap, unblock hygiene)

| Question | Source | Action |
|---|---|---|
| Stale cross-reference sweep: docs 02/03/05/11 still cite pre-index names (`04-board-state`, `04-state-make-unmake`, `04-make-unmake`, `05-movegen-perft`, `09-movepick-history`, `06-see`) | docs 02/03/05/09/11 §9 | One docs-sync commit normalizing to the README index; also bump doc header status lines to Reviewed v1 |
| CLAUDE.md updates queued by the docs (Zobrist-in-`make_move` correction, fast-config RUSTFLAGS, `Bench:` trailer, check.sh workflow) | docs 04 §9.1, 01 §3.7, 14 §3.2 | Land each with its phase (P0/P2/P5); start the CLAUDE.md sync in P0 |
| Eval weights vs `params.rs` scoping (search-scoped params file; eval weights owned by `eval`) | doc 01 §9.4 | Accept doc 01's ruling; optionally promote to a numbered decision in decisions.md — one line, decide during P0 |
| Hosted CI (fast perft tier per push, full tier nightly) | doc 06 §9.4 | Low stakes; decide once P3's tiers exist. The local D24 ritual does not wait for it |

### 6.3 Deferred — measurement-gated (D14 ritual: stays as-designed until a profile/bench triggers it)

| Question | Source | Trigger / phase |
|---|---|---|
| `MoveList::push` hard `assert!` → `debug_assert!` | docs 01/05 | Bench measurement once search exists (P5+) |
| `MoveList::new()` zero-init / `MaybeUninit` | doc 05 §9.4 | Profile blames the memset; reuse-per-ply is the planned mitigation |
| Unchecked magic-table indexing | doc 03 §9.2 | Profile shows the bounds check in perft nps (expected never) |
| Hardcoded magic constants vs startup search | docs 01/03 | Measured init > 100 ms (expected never) |
| Piece-count cache `[u8; 12]` | doc 04 §9.2 | Profile or eval-A material terms ask for it |
| Phase A incremental PSQT in `CopiedState` | doc 11 §9.4 | `psqt_eval` > ~2 % of search time |
| Worker-cached `&'static AttackTables` vs per-call hoisting | doc 03 §9.3 | P11 wires worker fields; both patterns stay supported |
| Busy-wait vs condvar park (ponder/infinite window) | docs 12 §9.2 / 13 §9.5 (dup) | Idle-burn complaint in long analysis sessions |
| AVX-512 variants (movegen splat, 16-lane sort) | docs 05 §6 / 09 §9.4 | Profile at NNUE-era nps, not before |

### 6.4 Deferred — phase-tagged decisions

| Phase | Question | Source |
|---|---|---|
| P3 | Delete the ep occupancy-swap branch in `legal()` for SF parity once the D9 ep-recording rule is perft-proven (behavior identical) | doc 05 §9.1 |
| P3 | Enable the ep-capturer `debug_assert` (`TODO(D9)`) | doc 05 §9.5 |
| P3 | Confirm `castle-prevented` d4 = 665,077 with our engine; close the TalkChess transcription artifact | doc 06 §9.3 |
| P7/P8 | Drop the separate `cont_corr_hist` stack field (derive via `ContHistId::piece_to()`) when `Stack` freezes | doc 09 §9.2 |
| P9 | `follow_pv` semantics land with the IIR/PV gates | doc 07 §9.2 |
| P9 | Syzygy: own design doc vs appendix to 07 — decide during Wave E (N7 is plumbed, not implemented) | doc 10 §9.2 |
| P9 | Calibrate the 50k fixed-node proxy budget against a known-gain timed SPRT before trusting proxy merges | doc 14 §9.4 |
| P10 | Qsearch stop polling (only if stop latency under `movetime` proves unacceptable) | doc 07 §9.4 |
| P10 | `hashfull(max_age > 0)` wiring if the time manager wants a saturation signal | doc 08 §9.3 |
| P10 | Full MultiPV support (v1 ships MultiPV=1 with guards stubbed) | docs 07 §9.5 / 12 |
| P11 | Accept `loom` as a dev-dependency for the job-loop model (D15 allows) vs stress tests only | doc 13 §9.4 |
| P11 | `Threads` option upper bound (1024 provisional) | doc 12 §9.6 |
| P11+ | `Job::Custom` variant (parallel TT clear / NUMA); NUMA-shared histories | docs 13 §9.6, 09 §9.5 |
| P12 | Tuning-dataset licensing verification before any copy enters the repo | doc 11 §9.2 |
| P12 | Tempo term in Phase B — add the parameter, let the tuner/SPRT decide | doc 11 §9.5 |
| P12 | Promote `threats_by_lesser` to a `Board` method if eval needs the same sets | doc 09 §9.6 |
| P13 | **decisions.md amendment required**: D14 category (d) / D15 shape for the TT huge-page path — must be recorded before that commit | doc 08 §9.1 |
| P13 | Net binary distribution (release assets vs git LFS); `include_bytes!` embedding is decided | doc 11 §9.3 |
| P13 | Complexity term once the net gains a PSQT output head; WDL fit + `UCI_ShowWDL` timing | docs 11 §9.1, 12 §9.5 |
| P14 | `tune` cfg mechanism for SPSA (consts → statics + UCI options) | doc 10 §9.1 |
| P14 | Bench-list curation revisit once eval/search diverge (list is frozen until a deliberate re-freeze) | doc 06 §9.1 |
| P14 | Root-parallel perft for the extended tier | doc 06 §9.2 |
| Opportunistic | `GenType` as enum const param if `adt_const_params` stabilizes; `ValueExt` trait for method syntax; SPRT-bound translation if ever forced onto cutechess-cli | docs 05 §9.3, 02 §9.2, 14 §9.3 |

---

## 7. Immediate next steps (P0, in commit order)

1. **Build profiles + crate hygiene** — add `[profile.release]`
   (`lto="fat"`, `codegen-units=1`, `panic="abort"`) and
   `[profile.profiling]` to `Cargo.toml`; `#![deny(unsafe_code)]` + crate doc
   comment in `lib.rs`; document the RUSTFLAGS fast config in CLAUDE.md
   (doc 01 plan 1–2).
2. **`types` growth** — `Value`/`Depth` aliases, all constant bands,
   predicates, `mate_in`/`mated_in`, `value_to_tt`/`value_from_tt`,
   `PIECE_VALUES` + `PieceType::value()`, `Piece::index`, with the doc 02 §5
   unit tests (constant table, band edges, TT roundtrips) (doc 02 steps 1–2).
3. **`Direction` + `Square` arithmetic + `Bitboard` helpers +
   `Move::raw16`** — checked/unchecked stepping, XOR flips, `shift`/
   `pawn_pushes`/`pawn_attacks`/`more_than_one`, raw16 round-trip; exhaustive
   64×8 and randomized-equivalence tests (doc 02 steps 3–5).
4. **`search/params.rs` seed + thin UCI skeleton** — provenance header +
   first constants; one-line `main.rs`; `uci`/`isready`/`quit`/garbage
   absorption smoke test; `tools/check.sh` v0 (doc 01 steps 4–5, doc 14
   step 2).
5. **Start `attacks`** — const foundations + leaper tables with their unit
   tests (doc 03 step 1) — the first P1 commit.

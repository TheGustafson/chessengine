# Design suite index

This directory is the engine's build plan. `decisions.md` is the
constitution: numbered, terse, binding decisions (D1–D44) that every doc and
every line of code must comply with — **decisions change there first, docs
follow** (a decision change is its own commit with rationale; the owning doc
is updated to match). The numbered docs `01`–`19` are prescriptive component
designs — full algorithms, constants, implementation plans, and acceptance
criteria (D26). They are distinct from `stockfish_reference/`, which holds
*descriptive* study notes (what current Stockfish does, with file/line
provenance at SF commit `eca43a97`); notes explain where ideas came from,
design docs say what we build, and we never copy SF code (D27). When unsure
during implementation: design doc → reference note → SF source, then extend
the note.

## Documents

| Doc | Title | Scope | Status |
|---|---|---|---|
| `README.md` | Suite index | This file: index, reading order, process rules | Living |
| `decisions.md` | Design decisions record | Numbered binding decisions D1–D44; every doc complies | Living |
| `00-master-plan.md` | Master implementation plan | Phases P0–P15, dependency graph, milestone ladder M1–M11, risks, consolidated open questions | Living |
| `01-architecture.md` | Architecture, module layout, conventions | Crate/module map, public seams, data flow, error/unsafe/dependency/build policies, `params.rs` convention | Reviewed v1 |
| `02-core-types.md` | Core types | `Value`/`Depth` bands + predicates, mate/TT score arithmetic, `Direction`/`Square`/`Bitboard` helpers, `Move::raw16`, `ExtMove` repr, SEE piece values | Reviewed v1 |
| `03-attack-tables.md` | Attack tables | Const leaper + ray-geometry tables, fancy magic bitboards behind `LazyLock`, the naive-ray correctness oracle | Reviewed v1 |
| `04-position-state.md` | Board state, Zobrist, make/unmake, SEE | `StateInfo` split + owned history stack, five-key Zobrist family with all SF tricks, make/unmake/null move, check info, draw rules, `see_ge`, hook seam | Reviewed v1 |
| `05-movegen.md` | Move generation | Monomorphized pseudo-legal generation per `GenType`, `legal`/`pseudo_legal` predicates, lazy legality | Reviewed v1 |
| `06-perft-validation.md` | Perft & validation | Perft/divide/checked-perft, test tiers, differential tooling, the `bench` subcommand + fingerprint suite | Reviewed v1 |
| `07-search-core.md` | Search core | Iterative deepening, aspiration windows, PVS node, qsearch, `Stack`/`PvLine`/`RootMove`, value semantics in search | Reviewed v1 |
| `08-transposition-table.md` | Transposition table | 10-byte relaxed-atomic entries in 32-byte clusters, `mul_hi64` indexing, replacement policy, prefetch, huge-page plan | Reviewed v1 |
| `09-move-ordering-history.md` | Move ordering & history | Staged `MovePicker` (no killers, D17), gravity `StatEntry`, all history tables incl. correction history, `ContHistId` | Reviewed v1 |
| `10-pruning-extensions.md` | Pruning, extensions, reductions | The full heuristic suite (RFP, NMP, LMR, LMP, futility, singular, multicut, ProbCut, ...) in SPRT-gated waves A–F | Reviewed v1 |
| `11-evaluation.md` | Evaluation | Phase A (material + tapered PSQT), Phase B (HCE + texel tuning), Phase C (NNUE with incremental accumulators) behind one `evaluate` seam (D22) | Reviewed v1 |
| `12-uci-time-management.md` | UCI & time management | Protocol loop, options, `Engine` glue, `Limits`, TM optimum/maximum formulas, soft stop, ponder, MultiPV, nodestime | Reviewed v1 |
| `13-threading-smp.md` | Threading (lazy SMP) | Persistent parked worker pool, job dispatch, `Arc<TT>` sharing, best-thread voting, stop discipline, race model (D23) | Reviewed v1 |
| `14-performance-testing.md` | Performance & testing | Measurement hierarchy (SPRT / tactics / bench / criterion), bench fingerprint policy, target ladder, profiling toolkit, SPRT/gauntlet protocol (D24/D25) | Reviewed v1 |
| `15-nnue-datagen.md` | NNUE training-data generation | The `bullet`/`bulletformat` on-disk format contract, curation/filtering spec, SF-teacher bootstrap pipeline, corpus volume/storage, D28 two-tier provenance progression | Draft v1 |
| `16-research-data-platform.md` | Research data platform | Four-layer observation model, the versioned self-describing Observation-Store schema (superset of doc 15's flat record), three-store data management, provenance, and the D29 staging/reconciliation discipline (build-now = Tier-A + schema; the rest deferred behind its consuming research) | Living |
| `17-nnue-research.md` | NNUE research predecessor | D34/D40 experiments and hypotheses that established learned move information can improve fixed-node search and exposed shared-training/data failures; partially superseded by D41/doc 19, retained as evidence and an idea catalogue | Historical/Living evidence |
| `18-experimental-ledger.md` | Experimental ledger | Content-addressed provenance DAG + append-only results ledger (nets/data/experiments JSONL + milestones) with a stdlib-only CLI; identity = sha256, every artifact carries a regen recipe; metadata in git, bytes in a content store (GCS + OpenBench later) (D38) | Living |
| `19-learned-search-allocation.md` | ARCHON learned counterfactual search allocation | Governing D41 research roadmap: external positions relabeled by ARCHON-native oracle search, protected value head, move/action consequence model, constrained planner, on-policy flywheel, staged authority, and fixed-node plus equal-time gates | Living |

The operational goal hierarchy, promotion thresholds, always-next rule, and
current research checkpoint live in
`docs/research/archon-execution-plan.md`; experimental continuation state lives
in `old/docs/research/policy-automl-state.md` (legacy filename retained).

(Some doc headers still read "Draft v1" from the drafting workflow; the
review pass is complete and this table is authoritative until the header
sweep in the docs-sync commit — see `00-master-plan.md` §6.2.)

## Recommended reading order (newcomer)

1. `decisions.md` — the constitution; everything cites it.
2. `00-master-plan.md` — where the project is and what comes next.
3. `01-architecture.md` — the module map, seams, and crate-wide policies.
4. `14-performance-testing.md` §3.1–§3.2 — the measurement discipline and
   the perft/bench/SPRT gates; internalize these before writing code,
   because they apply from the first commit.
5. Board layer, in dependency order: `02` → `03` → `04` → `05` → `06`.
6. Search layer: `07` → `08` → `09` → `10`.
7. `11` (evaluation), then `12` (UCI/TM), `13` (threading); `15` (NNUE
   training-data generation) accompanies `11` Phase C, and `16` (research
   data platform) is the forward-looking umbrella `15` is the first instance of.
8. Rest of `14` when you need targets, profiling recipes, or the strength
   harness.
9. `17` for the policy/value experimental history, `18` for provenance, then
   `19` for the governing ARCHON learned-allocation program.

## Process rules

- **Decisions lead docs.** A change to any pinned decision (D1–D44) lands in
  `decisions.md` first, as its own commit with rationale; the owning doc is
  updated to comply. A doc may not silently contradict `decisions.md` — a
  necessary deviation is called out in that doc's §9 Open Questions until
  the decision is amended (the D23 amendment of 2026-07-09 is the model).
- **Docs lead code.** These documents are contracts: code reviews check
  implementations against the owning doc, and a doc that disagrees with the
  code is a bug in the doc — when implementation forces a deviation, the doc
  is updated **in the same commit**.
- **Seams are synchronized.** The signatures in `01-architecture.md` §3.3
  are the cross-module contract; a change to a seam must edit doc 01 in the
  same commit. Inside a component's own scope, the component doc wins over
  doc 01, and doc 01 gets corrected.
- **Acceptance criteria stay current.** Every doc has a §8 acceptance
  section (D26); a phase is done when its criteria are *met* — perft suites,
  bench fingerprints, SPRT verdicts — not when its code exists. If a
  criterion changes, it changes in the doc before the code claims it.
- **Ownership is single.** Each topic has exactly one owning doc (see each
  doc's §1 "does NOT own" list); other docs reference, never redefine.
  Constants adopted from Stockfish trace to commit `eca43a97` via
  `stockfish_reference/` notes; tuned search constants live only in
  `search/params.rs` (D20) — a bare tunable literal in search-layer code is
  a review defect.
- **Status lifecycle.** `Reviewed v1` docs change only through the rules
  above (bumping to v2 after any substantial re-review); `README.md` and
  `00-master-plan.md` are Living and updated as phases complete.

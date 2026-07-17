# 16 — Research data platform (observation store + staging discipline)

Status: Living

This document is the umbrella design for datagen as a **research data
platform** (D29), built on one principle: **store expensive observations,
not just current labels.** A deep search is expensive to run and cheap to
re-label; if we preserve the raw circumstances and outcomes of the searches
we run — full position state, the search context, the search's observations,
and per-move observations — then every future training label (centipawn/WDL
value, policy, search-control targets, cascaded-evaluation routing,
mixture-of-experts routing, adversarial hard examples) becomes a cheap
*derivation* rather than a fresh multi-core-hour generation run. The platform
is **designed in full now and built incrementally**: the only build-now slice
is the Tier-A NNUE bootstrap (already specified and validated in doc 15) plus
the Observation-Store *schema* defined here; everything richer is designed but
deferred behind the research that consumes it. The engineering spine of this
doc is the **staging & reconciliation discipline** (D29) that keeps the grand
vision from turning into premature construction: what we build today, what we
merely reserve schema for, and why the rich store is inseparable from our own
instrumented engine (Tier-B), not the foreign Stockfish teacher (Tier-A).

---

## 1. Purpose and scope

**This document owns:**

- The **four-layer record model** (§3.2) — position state / search context /
  search observations / per-move observations — as the platform's canonical
  data shape, and the mapping of every generation mode and research target
  onto it.
- The **Observation-Store on-disk schema** (§3.5): a versioned,
  self-describing, forward-compatible binary format, its Rust logical types,
  and the wire rules that let an old reader tolerate a new field. This schema
  is a strict **superset** from which doc 15's flat 32-byte bullet record is a
  trivial projection (§3.6).
- The **three-store data-management model** (§3.3): Raw Position Store /
  Observation Store (permanent) / Derived Training Sets (disposable), and the
  export chain between them.
- The **staging & reconciliation discipline** (§3.7, D29): the build-now vs
  designed-deferred classification for every corpus and research target, each
  deferred item's **trigger**, the Tier-A/Tier-B seam (§3.8), and the
  storage-cost economics (§3.9) that justify the staging.
- The **mandatory provenance set** (§3.4) every record must carry for label
  reproducibility.

**This document does NOT own:**

- The **Tier-A bootstrap** itself — the SF teacher, its curation/filtering
  spec (§3.5 there), labeling settings, volume, and the flat
  `bulletformat::ChessBoard` contract — all owned by **doc 15**. This doc
  places doc 15 as *one instance* (the degenerate Self-Play mode) of the
  platform and consumes its flat record as the canonical Derived-layer target;
  it does not re-specify any of it.
- The **NNUE architecture, quantization, net format, and training recipe** —
  owned by doc 11 §3.4. The platform *feeds* the EVALUATION target into that
  pipeline; the net is doc 11's.
- The **search internals it instruments** — node structure, MovePicker
  staging, the pruning/reduction/extension heuristics whose decisions Layer 4
  records — owned by docs 07/09/10. Layer 4 *observes* those decisions; it
  does not define them.
- The **ML training/model code** for policy, search-control, cascade, or MoE
  research — that is downstream research tooling (discardable, non-engine, may
  use any deps), not part of the engine crate. This doc defines only the data
  contract it reads from.
- **Make/unmake, Zobrist, movegen** (docs 04/05) and the per-worker threading
  model (doc 13) — the emitter *hangs off* their seams; it does not redefine
  them.

## 2. References

- `design/decisions.md` — **D29** (this platform, the authoritative decision
  this doc elaborates), **D28** (the two-tier NNUE bootstrap: Tier-A
  discardable SF scaffold / Tier-B clean own-data), **D27** (never copy SF
  code, never ship SF nets), **D15** (zero runtime crate dependencies;
  external tooling is not a dependency), **D14** (unsafe budget), **D24** (the
  `Bench:` fingerprint, part of provenance), **D22/D21** (the `evaluate` seam
  and eval roadmap the value target trains), **D20** (SF constants as tunable
  starting points — the search-control targets aim to *learn* what D20 hard-
  codes), **D23** (per-worker single ownership — the shape of the emitter).
- `design/15-nnue-datagen.md` — the **Tier-A instance** and the owner of the
  flat bullet format (§3.2.1 there, the 32-byte `ChessBoard`), the curation
  spec (§3.5), the SF teacher (§3.3), regenerability metadata (§3.6), and the
  end-to-end validation (§10). Its record is this platform's Derived-layer
  target.
- `design/11-evaluation.md` §3.4 — the NNUE Phase C the EVALUATION target
  feeds; §3.4.7 the clean self-play generator (`bin/datagen.rs`) that, once
  instrumented, becomes the **Tier-B** producer that populates the Observation
  Store.
- `design/10-pruning-extensions.md` — the P9 heuristic waves (RFP/NMP/LMR/LMP/
  futility/singular/…) whose per-move decisions Layer 4 records and whose
  SEARCH-CONTROL targets will eventually supervise them.
- `design/09-move-ordering-history.md` — the MovePicker stages and history
  scores Layer 4's move-ordering fields observe.
- `design/13-threading-smp.md` — the per-worker `Worker` ownership (D23) the
  per-worker shard writer mirrors; the `MakeMoveHooks` seam (doc 04) the
  emitter shares with NNUE accumulator recording.
- `design/07-search-core.md` — node types (Root/PV/cut/all/qsearch), the
  iterative-deepening loop whose snapshots Layer 3 records, `RootMove`/`PvLine`.
- No new external references beyond doc 15's (`bullet`, `bulletformat`,
  python-chess). The wire format here is hand-rolled std-only (§3.5.2); it
  takes **no** serialization dependency.

## 3. Design

### 3.1 Objective and guiding principle

The objective is a **future-proof data-generation platform** that supports not
only conventional NNUE value training but also policy learning, search-control
learning, mixture-of-experts (MoE) routing, cascaded evaluation, adversarial
robustness, and research we have not yet imagined. The single guiding
principle, from which the whole design follows:

> **Store expensive observations, not just current labels.** Scores,
> policies, uncertainty, routing targets, pruning targets, and search-control
> targets can all be *derived later* if the raw search information is
> preserved. Regenerating deep search is expensive; deriving labels from
> stored observations is cheap.

Concretely: the corpus we would generate today to train a value net throws
away — the moment it writes a `(FEN, score, result)` line — the node counts,
the per-move cutoff behaviour, the reductions that were applied, and the
alternatives that were pruned. Those are exactly the signals a *later* project
on learned move ordering or learned pruning would need, and re-running the
searches to recover them costs the same core-hours all over again. The
platform's job is to make the write side capture the observation once, and the
read side (cheap, deterministic export) manufacture each experiment's labels
on demand.

### 3.2 The four-layer record model

Every observation is organized around four layers. They are the vision's
canonical data shape and the top-level structure of the on-disk record
(§3.5); the concrete field lists appear there.

1. **Position state** — everything needed to *fully reconstruct* the position
   and to identify it canonically: piece placement, side to move, castling
   rights, en passant, rule-50 clock, repetition/history context. Carries the
   **canonical position key** (placement + stm + castling + ep — doc 15
   §3.5.5's dedup identity) so positions deduplicate across the whole
   platform.

2. **Search context** — the *circumstances* under which the search ran: depth,
   node budget, the alpha-beta window, node type (root / PV / cut / all /
   qsearch), static eval at the node, TT state (hit, stored depth/bound/value/
   move), the active heuristic flags (improving, cutnode, in-check, singular,
   excluded-move, …), and a reference to the engine configuration. The context
   is what makes an outcome *interpretable* — a fail-high at a 3-ply
   null-window search means something different from one at a 30-ply PV search.

3. **Search observations** — the *outcomes*: best move, principal variation,
   final score, bound type (exact / upper / lower), nodes searched, search
   statistics (seldepth, …), and the per-depth iterative-deepening snapshots.

4. **Per-move observations** — for every move the node searched (or considered
   and pruned): move-ordering rank/stage/score, the depth it was actually
   given (post-reduction), reductions/extensions applied, the score it
   returned, nodes it consumed, whether it raised alpha or caused the cutoff,
   whether a re-search happened, and — for skipped moves — which pruner
   rejected it. **This layer is the search-control gold and is the crux of the
   Tier-A/Tier-B seam (§3.8): a foreign teacher physically cannot produce it,
   because these are *our* search's internal signals.**

#### 3.2.1 Evidence kind is part of the label

D43 requires two distinct record families:

- **Observational:** the current behavior policy encountered the node and chose
  its normal ordering/reduction/extension. These records are cheap and preserve
  the deployed distribution, but later moves are censored by earlier cutoffs and
  the chosen action is confounded with the heuristics that chose it.
- **Interventional:** the generator reconstructs the same pre-action state and
  forces one declared `CfAxis` while holding all other configuration and state
  regimes fixed. Only these records identify consequences of alternative
  reductions/extensions/policy tiers.

Derived exporters must never silently combine these evidence kinds. An
observational reduction outcome is not a counterfactual reduction label.

### 3.3 Data-management model — three stores

The platform's data lives in three stores with sharply different lifetimes:

1. **Raw Position Store** — canonical, deduplicated positions keyed by the
   Layer-1 canonical key. Permanent, and shared across every dataset so a
   position analysed by five different modes is one identity, not five.
2. **Observation Store** — the immutable, append-only record of search
   observations (Layers 1–4) and counterfactual experiments. **PERMANENT.**
   This is where the "store observations, not labels" investment lives.
3. **Derived Training Sets** — experiment-specific tensors/records generated
   *from* the Observation Store by cheap deterministic export. **DISPOSABLE**
   — regenerated whenever an experiment's labeling recipe changes. Doc 15's
   `train.bin`/`val.bin` flat bullet corpus is the first Derived set.

**Derived is disposable, raw is permanent.** The export direction is strictly
Observation Store → Derived; nothing flows back. A Derived set can always be
rebuilt from the store plus its recipe; the store can only be rebuilt by
re-running searches (expensive), which is the asymmetry the whole design
exploits.

```
   RAW POSITION STORE           OBSERVATION STORE                 DERIVED TRAINING SETS
   (canonical, deduped)   ◄──►  (Layers 1–4 + counterfactuals)    (per-experiment, disposable)
        permanent                     permanent            ──►    • bullet flat ChessBoard (doc 15)
        │  canonical key         append-only, immutable    export • policy tensors        (deferred)
        └──────────────────────────────┘   deterministic  ──►    • search-control tables  (deferred)
                                             cheap export         • columnar analytics mirror (deferred)
```

### 3.4 Provenance (mandatory, from day one)

Every record is reproducible only if we know exactly what produced it. The
following provenance set is **required** — the schema (§3.5) makes it a
non-optional block, and no observation is writable without it:

- **engine version** (git rev / semver) and its **`Bench:` signature** (D24) —
  identifies the exact search behaviour;
- **network version** — our net id (absent in Tier-A, which has no own net);
- **teacher identity** — the SF rev + binary `sha256` in Tier-A (absent in
  Tier-B self-play);
- **search config** — the full knob set (and a hash of it for quick joins);
- **generator version** — the datagen tool rev;
- **random seed** — per-worker PRNG seed (doc 15 §3.7 / doc 11 §3.4.7);
- **dataset source** — which generation mode (§3.6);
- **opening source** — book id or random-ply policy.

Provenance is interned once per shard/run (it is constant across a worker's
output), so per-record overhead is a small index, not a repeated blob. This is
the platform-wide generalization of doc 15 §3.6's `meta.json` regenerability
set.

### 3.5 The Observation-Store schema

#### 3.5.1 Format decision and its justification

The permanent store is written by **our own instrumented engine** (Tier-B,
§3.8) — an engine-side emitter that must respect **D15** (no runtime crate
dependency) and **D14** (no new `unsafe`). That rules out an off-the-shelf
serialization library (protobuf/prost, Arrow/Parquet, serde+bincode) *for the
write path*. It also wants **forward compatibility**: new fields and new
corpora must not break readers built against an older schema, because the store
is permanent and long-lived while the research that reads it evolves.

Decision: the store is a **versioned, self-describing, tag-length-value (TLV)
binary log**, hand-rolled in `std` (a ~40-line reader/writer), shaped after the
well-known protobuf wire model but taking **no** protobuf dependency (the wire
model is a public fact; we reimplement it, copying no code — D27/D15). Every
field carries a tag and a wire-type; a reader that meets an unknown tag skips
it by wire-type. That single property gives forward compatibility for free.
Row-oriented TLV is also the natural **streaming** write format for an emitter
that appends per node/per game, unlike a columnar format that must buffer and
transpose.

Columnar/analytics formats (Parquet) and embedded stores (sqlite/lmdb) are
*not banned* — they are **Derived-layer** artifacts, produced by discardable
export tooling (`tools/`, like doc 15's python `tools/nnue/`) that may use any
dependency it likes, exactly because the Derived layer is disposable and
non-engine. The permanent store stays zero-dep TLV; large-scale column access,
if it ever pays off, is a Derived mirror (§9 q1).

#### 3.5.2 Wire rules (forward-compatibility contract)

- A **shard file** = a header block, then a stream of episodes.
- Integers are LEB128 **varint**; signed integers are zigzag-encoded.
- A **field** is a varint key `(tag << 3) | wire_type` followed by its payload.
  Wire-types: `0` varint, `1` fixed64, `2` length-delimited (bytes / nested
  message / packed repeated), `5` fixed32.
- **Unknown-tag skip:** a reader that does not recognize a tag reads its
  wire-type and skips (varint: consume one varint; length-delimited: read the
  length, skip that many bytes; fixed: skip 4/8). ⇒ *an old reader tolerates a
  new field.*
- **Tag discipline:** tags are **append-only** — never renumbered, never
  reused, never repurposed. New fields get fresh tags; deprecated fields are
  tombstoned, not recycled. `EpisodeKind` and enum discriminants are likewise
  extensible; unknown kinds/values are skipped, not errors.
- `format_version` in the header is bumped only for a *breaking* wire change
  (which the append-only rule is designed to avoid); additive changes leave it
  untouched.

#### 3.5.3 Logical schema (Rust sketch)

The types below are the *logical* schema; on disk they are the TLV encoding of
§3.5.2. Engine-side they live behind a `datagen` build feature so the normal
search/bench binary compiles them out (§6). Tag numbers are illustrative and
grouped per message; the field lists — not the numbers — are the contract.

```rust
// src/obs/schema.rs  — engine-side, std-only, #[cfg(feature = "datagen")].
// On disk: the versioned self-describing TLV of §3.5.2. NO serde/prost/parquet
// dependency (D15); the reader/writer is hand-rolled std. Tags are append-only.

pub const OBS_MAGIC: [u8; 6] = *b"CEOBS\0";
pub const OBS_FORMAT_VERSION: u16 = 1;

pub struct ShardHeader {              // one file
    magic: [u8; 6],                   // (file framing preceding the header
                                      //  message, not a TLV field — impl note)
    format_version: u16,              // 2
    created_unix: u64,                // 3
    provenance: Vec<Provenance>,      // 4  interned; records reference by index
}

pub struct Provenance {               // §3.4 — mandatory, no record without it
    engine_version: String,           // 1
    engine_bench_sig: u64,            // 2  D24 Bench: nodes
    net_version: Option<String>,      // 3  our net; None in Tier-A
    teacher_identity: Option<String>, // 4  SF rev+sha256 in Tier-A; None in Tier-B
    search_config: SearchConfig,      // 5
    search_config_hash: u64,          // 6  fast join key
    generator_version: String,        // 7
    rng_seed: u64,                    // 8
    dataset_source: String,           // 9  which generation mode (§3.6)
    opening_source: String,           // 10 book id / random-ply policy
    hardware_identity: String,        // 11 appended 2026-07-17 (doc 19 §5.2):
                                      //    cpu model + feature flags + threads —
                                      //    cycle/time labels are machine-relative
}

#[repr(u8)]
pub enum EpisodeKind { SelfPlay, InternalSearch, RootMoveAnalysis,
                       Counterfactual, Adversarial } // extensible; unknown skipped

pub struct Episode {
    kind: EpisodeKind,                // 1
    episode_id: u64,                  // 2
    provenance_ref: u32,              // 3  index into ShardHeader.provenance
    game_result: Option<Wdl>,         // 4  white-relative; back-patched at game end
    cf_axis: Option<CfAxis>,          // 5  which knob differs across a counterfactual pair
    records: Vec<Observation>,        // 6  repeated
    replay_root: PositionState,       // 7  full episode-start state
    root_reversible_keys: Vec<u64>,   // 8  repetition history preceding replay_root
    event_hash_chain: Vec<u64>,       // 9  locate first deterministic divergence
}

pub struct Observation {              // one node — the four vision layers
    position: PositionState,          // 1  Layer 1
    context:  SearchContext,          // 2  Layer 2
    search:   SearchObservations,     // 3  Layer 3
    moves:    Vec<PerMoveObservation>,// 4  Layer 4 — repeated, may be sampled/empty
}

pub struct PositionState {            // Layer 1 — full reconstruction + identity
    occ: u64,                         // 1  D3 mapping — identical to bullet's occ
    pcs: Vec<u8>,                     // 2  nibble-per-occupied-square (doc 15 §3.2.1)
    stm: u8,                          // 3
    castling: u8,                     // 4  4-bit mask
    ep_square: Option<u8>,            // 5
    rule50: u8,                       // 6
    game_ply: u16,                    // 7
    plies_from_null: u16,             // 8
    repetition: i8,                   // 9
    canonical_key: u64,               // 10 placement+stm+castling+ep — dedup identity;
                                      //    the UNFOLDED main key (no rule50 fold —
                                      //    identity must be clock-independent, D33 r1)
    path_from_episode_root: Vec<u16>,  // 11 legal moves root→node
}

pub struct SearchContext {            // Layer 2 — circumstances of the search
    depth: i32,                       // 1
    node_budget: Option<u64>,         // 2
    alpha: i32, beta: i32,            // 3, 4
    node_type: NodeType,              // 5  Root/Pv/Cut/All/Qsearch
    static_eval: i32,                 // 6  raw eval at the node
    tt_hit: bool,                     // 7
    tt_depth: i32, tt_bound: Bound,   // 8, 9
    tt_value: i32, tt_move: u16,      // 10, 11
    flags: u32,                       // 12 bitset: improving/cutnode/in_check/singular/excluded/…
    event_ordinal: u64,               // 13 stable within deterministic episode
    event_fingerprint: u64,           // 14 hash(key,ply,depth,window,node-kind,move)
    ply: u16,                         // 15
    alpha_before_move: i32,           // 16 frozen pre-action window
    beta_before_move: i32,            // 17
    root_delta: i32,                  // 18 node-dither context
    tt_regime: TtRegime,              // 19 Empty/Restored/WarmObserved
    history_regime: HistoryRegime,    // 20 Fresh/Frozen/Restored/WarmObserved
    // 21–23 appended 2026-07-17 (doc 19 §5.2 completeness pass):
    legal_move_count: u8,             // 21 total legal moves at the node
    capture_move_count: u8,           // 22 tactical/quiet composition (quiets = 21 − 22)
    correction_value: i32,            // 23 unscaled corr-history sum at node entry (S1);
                                      //    /131072 is the eval delta the search applied
    // engine config lives in Provenance.search_config (interned)
}

pub struct SearchObservations {       // Layer 3 — outcomes
    best_move: u16,                   // 1
    pv: Vec<u16>,                     // 2
    final_score: i32,                 // 3  stm-relative Value
    bound: Bound,                     // 4  Exact/Upper/Lower
    nodes: u64,                       // 5  subtree
    seldepth: i32,                    // 6
    id_snapshots: Vec<IdSnapshot>,    // 7  per ID depth: {depth, score, best, nodes, time_ms, pv}
    completion: Completion,           // 8  appended 2026-07-17 (D33 r1): a timed-out
                                      //    REFERENCE search must never read as a
                                      //    complete Exact label (doc 19 §5.4/§11.1)
}

pub struct PerMoveObservation {       // Layer 4 — per searched/considered move
    mv: u16,                          // 1
    order_index: u16,                 // 2  rank in the move loop
    order_stage: u8,                  // 3  ttMove/goodCapture/quiet/badCapture (doc 09)
    order_score: i32,                 // 4  MovePicker/history score
    searched_depth: i32,              // 5  post-reduction depth given
    reduction: i16, extension: i16,   // 6, 7  LMR plies / extensions (doc 10)
    returned_score: i32,              // 8
    nodes: u64,                       // 9  subtree consumed by this move
    raised_alpha: bool,               // 10
    caused_cutoff: bool,              // 11
    researched: bool,                 // 12 full-window/full-depth re-search happened
    pruned_by: Prune,                 // 13 None if searched; else Lmp/Futility/See/History/…
    evidence: EvidenceKind,           // 14 Observational/Intervention/Reference
    baseline_reduction: i16,          // 15 classical finalized reduction
    requested_reduction: i16,         // 16 proposed/forced action
    actual_scout_depth: i32,          // 17 after reduction/extension
    policy_tier: PolicyTier,          // 18 None/Cheap/Full
    scout_score: i32,                 // 19 before any re-search
    scout_bound: Bound,               // 20 classification vs frozen alpha/beta
    scout_nodes: u64,                 // 21
    research_score: Option<i32>,      // 22 full-depth result when run
    research_bound: Option<Bound>,    // 23
    research_nodes: u64,              // 24
    research_changed_bound: bool,     // 25 not synonymous with trigger
    research_was_necessary: Option<bool>, // 26 requires accepted reference
    reference_bound: Option<Bound>,   // 27
    bound_error: Option<bool>,        // 28 action vs accepted full reference
    completion: Completion,           // 29 Complete/Stopped/TargetMiss/Invalid
}
```

Enums `Wdl`, `NodeType`, `Bound`, `Prune`, `CfAxis`, `EvidenceKind`,
`PolicyTier`, `Completion`, `TtRegime`, `HistoryRegime`, and `SearchConfig`/
`IdSnapshot` are small companion types; their discriminants follow the same
append-only, skip-unknown rule. Missing fields use explicit option/presence
encoding; readers never interpret an absent field as numeric zero.

**Presence semantics, per field class (D33 round 1, F2):** the presence
guarantee above is delivered by the `Option`-typed fields. The non-Option
numeric context fields (`static_eval`, the `tt_*` group, `correction_value`,
…) are always encoded, so "generator could not observe this" is expressed at
the RECORD-CLASS level, not per field: `EpisodeKind` + `evidence` +
`generator_version` scope which Layer-2 fields are meaningful. For
`RootMoveAnalysis` episodes produced by the external-engine root-v0
generator, only `depth`, `node_budget`, the window fields, and the
legal/capture move counts are meaningful; the engine-internal group is
Default-filled and MUST NOT be consumed. Corpora where those fields are
genuinely observed (`InternalSearch`) will populate them; if a future
consumer needs per-field presence inside one episode kind, the affected
fields get Option-ized with NEW tags (append-only).

**`search_config` v1 representation:** implemented as a canonical text
string plus `search_config_hash` (the doc's `SearchConfig` companion type
deferred until a consumer needs per-knob joins). The string INCLUDES the
root-analysis arm mode (`searchmoves` vs `child-negate`) — arms from
different modes are not comparable rows.

#### 3.5.4 Episode assembly note

`Episode.game_result` is known only at game end, so a self-play worker buffers
one episode (a game is ~100–400 records — trivially in memory), attaches the
white-relative WDL, and appends the whole episode. Counterfactual and
root-analysis episodes are naturally batch-shaped already.

#### 3.5.5 `InSearchCapture` episode kind — in-search observation capture v0 (D45a)

`InSearchCapture` (discriminant **5**, appended to `EpisodeKind` under the
append-only rule of §3.5.2) is the concrete engine-embedded realization of the
`InternalSearch` generation mode (§3.6.1 item 2): sampled *interior* main-search
nodes, captured live by the playing engine during ordinary SPRT/SPSA test games
so that the ~10⁴–10⁵ games/day the engine already plays feed the design/19
learned-search-allocation program instead of producing zero training data. It is
kept as a distinct discriminant from the reserved `InternalSearch` (=1) so that
game-embedded capture is provenance-separable from any future dedicated/offline
`InternalSearch` generator; both share this record framing.

**Record framing — one `Observation` per captured NODE.** A node is captured
only when it (a) reaches the interior move loop of the main negamax `search`
(not qsearch, and not a node that returns before the loop — RFP/NMP/ProbCut/
multicut cutoffs are not captured), (b) has remaining `depth ≥ ObsCaptureMinDepth`,
and (c) passes the deterministic position-keyed sample gate below, and (d)
actually searched at least one move (a terminal or fully-pruned node carries no
v0 Layer-4 data and is skipped). Each captured node emits exactly one
`Observation`:

- **Layer 1 `PositionState`** — packed with `convert::position_from_board(&board,
  0, vec![])`: an **EMPTY** `path_from_episode_root`. Game capture does not replay
  from a stored episode root, so the path is intentionally empty and node identity
  is `canonical_key = raw_key` (the UNFOLDED key, §3.5.3 / D33 r1). Consumers of
  `InSearchCapture` records dedup and join on `canonical_key`, never on the path.
  `repetition` is recorded as 0 (v0 does not thread per-node repetition state).
- **Layer 2 `SearchContext`** — the node's circumstances at loop entry: `depth`
  (post-hindsight, pre-singular-extension), `node_budget` (the `go nodes` limit if
  any, else absent), the entry `alpha`/`beta` window (also copied into
  `alpha_before_move`/`beta_before_move`), `node_type` (Root/Pv/Cut/All),
  `static_eval` (the corrected node eval), the `tt_*` group, `ply`,
  `correction_value`, and `legal_move_count`/`capture_move_count` (from a
  legal-move generation done only at the sampled node). The other engine-internal
  fields are Default-filled per the §3.5.3 F2 presence rule.
- **Layer 3 `SearchObservations`** — the node outcome: `best_move`,
  `final_score` (= `best_value`), `bound` (Lower/Exact/Upper, computed exactly as
  the TT-store bound), `nodes` (the node's own subtree cost), and
  `completion = Complete` (a stopped search returns before the node exit and is
  never captured). PV, seldepth, and id_snapshots are left empty in v0.
- **Layer 4 `PerMoveObservation`** — one row per move **actually searched** at the
  node (pruned-before-make moves are not rowed in v0). Each row carries: the move's
  `mv` (canonical `raw16`), `order_index` (1-based rank in the move loop, counting
  pruned siblings), `reduction` (**the finalized LMR reduction `r` in plies,
  `r / RED_UNIT` truncated toward zero** — RED_UNIT = 1024ths of a ply, doc 10 §3.3),
  `extension`, `searched_depth` (the nominal child depth after extension/
  deeper-shallower), `returned_score`, `nodes` (this move's subtree cost),
  `raised_alpha` (`value > alpha`), `caused_cutoff` (`value ≥ beta`) — together the
  fail-low/exact/cutoff value class vs the window — and `researched` (an LMR
  full-depth zero-window re-search fired). `pruned_by = None` and
  `evidence = Observational` on every v0 row.

**Sampling semantics.** Capture is deterministic and **position-keyed**, never
timing- or RNG-driven: a node is sampled iff `(raw_key & ((1 << ObsCaptureSampleBits)
− 1)) == 0`. Because the gate is a pure function of the position key, capture is
reproducible and never perturbs search (identical node counts / bestmove / score
with capture on or off — the D24 bench neutrality of the default build is
preserved because the whole path is behind `feature = "datagen"`). Transpositions
to the same key sample identically by construction.

**Episode framing & bounds.** One `Episode` per game: the per-`Worker` buffer
accumulates `Observation`s across the game's `go` commands and is flushed as a
single `InSearchCapture` episode on `ucinewgame` (game boundary) and on engine
drop/quit; an empty buffer flushes nothing (so *N* games → *N* episodes).
`game_result` is absent (the engine does not know the game outcome). The buffer is
a fixed-capacity `Vec` reserved once to `ObsCapturePerGameCap`; once full, further
sampled nodes are **counted, not silently lost** — the drop count is stored in the
new `Episode.dropped_records` field (tag **10**, append-only; encoded only when
non-zero so existing shards are byte-unchanged). Capture requires `Threads == 1`
(refused with an `info string` otherwise) so single-threaded determinism holds.

### 3.6 Reconciliation: generation modes, research targets, and the flat bullet projection

#### 3.6.1 The five generation modes (data products)

Each mode is an `EpisodeKind` that writes the same four-layer record; they
differ in *which positions* they search and *how*:

1. **Self-Play Corpus** (`SelfPlay`) — full games; value/WDL, generalization,
   deployment coverage. Doc 15's Tier-A is the degenerate instance of this
   mode (SF teacher, flat output).
2. **Internal Search Corpus** (`InternalSearch`) — sampled *internal* nodes
   (not just roots), for cutoff prediction, search-control, LMR/pruning
   research, and cascade gating. This is where Layer 4 is densest.
3. **Root Move Analysis Corpus** (`RootMoveAnalysis`) — every legal move at a
   position analysed independently (MultiPV-all / per-move fixed searches),
   for policy, move ranking, regret, and complexity.
4. **Counterfactual Corpus** (`Counterfactual`) — *paired* searches of the same
   position under different configs (reduced-vs-full depth, pruned-vs-unpruned,
   cheap-vs-expensive eval, different move-ordering, different experts), linked
   by shared `canonical_key` + a `cf_axis` tag. Direct supervision of
   search-control: the pair *is* the counterfactual label.
5. **Adversarial Corpus** (`Adversarial`) — positions selected to expose
   weaknesses (in eval, policy, search-control, MoE routing, or cascade
   gating), for robustness and hard-example mining.

#### 3.6.2 The five research-target categories

The platform must be able to *label* for all of the following; each is a
derivation from the stored layers (cheap) — except where a new generation mode
must first be run (noted in the staging table §3.7):

- **EVALUATION** — centipawn value, WDL, deep-search residuals, uncertainty.
  Derived from Layers 1+3 (+ episode WDL). *This is the build-now target*
  (doc 15/Phase C).
- **POLICY** — best-move prediction, move ranking, cutoff probability,
  search-cost-aware ordering. Derived from Layers 1+3+4 (Root Move Analysis /
  Internal Search corpora).
- **SEARCH CONTROL** — LMR risk, pruning safety, extension utility,
  search-depth requirements, re-search probability. Derived from Layer 4 and
  Counterfactual pairs.
- **CASCADED EVALUATION** — cheap-evaluator error, expensive-evaluator
  benefit, evaluator-invocation value, cost-aware routing. Derived from
  Counterfactual (cheap-vs-expensive-eval axis) + Layer 2 static evals.
- **MIXTURE OF EXPERTS** — oracle expert selection, router regret, expert
  specialization, expert-transition stability. Derived from Counterfactual
  (different-experts axis) + per-expert observations.

#### 3.6.3 The flat bullet record is a projection (superset proof)

Doc 15's flat 32-byte `bulletformat::ChessBoard` is exactly the Derived-layer
projection of an `Observation`:

```
bullet ChessBoard  ⟵  project(obs):
  occ, pcs                ⟵  obs.position.{occ, pcs}     (already the D3 mapping)
  ksq, opp_ksq            ⟵  obs.position.{occ, pcs}     (derive king squares)
  score (white-relative)  ⟵  obs.search.final_score, sign-mapped by obs.position.stm
  result                  ⟵  obs.episode.game_result
  (curation §3.5 doc 15)  ⟵  obs.context.flags(in_check), obs.search.final_score,
                              obs.search.best_move kind
```

The flat record is an `Observation` with Layers 2 and 4 dropped and Layers 1+3
projected — a strict subset. Therefore the Observation Store schema is a
**superset** and the bullet export is a *trivial, deterministic, cheap*
projection: read `position` + `final_score` + episode `game_result`, apply doc
15 §3.5's curation predicates, emit the white-relative text interchange line,
and let `bulletformat` pack. The user's three stores map exactly onto how
bullet consumes data: Raw Position Store = Layer-1 canonical positions;
Observation Store = the TLV shards; Derived Training Set = `train.bin`/
`val.bin`.

### 3.7 Staging & reconciliation discipline (D29)

The platform is **designed in full now, built incrementally.** The build-now
slice is essentially *only* the Tier-A bootstrap (which already exists and is
validated — doc 15) plus the Observation-Store schema *definition* (this doc).
Everything else is designed-but-deferred, each item gated on the specific
research that consumes it.

| Item | Class | Trigger (the research that consumes it) |
|---|---|---|
| Observation-Store schema (§3.5), staging, provenance set | **BUILD-NOW** | this doc; D29 pinned |
| Tier-A Self-Play (SF teacher, flat bullet output) | **BUILD-NOW (done)** | doc 15 — exists & validated (§10 there) |
| EVALUATION target (value/WDL) | **BUILD-NOW (done)** | NNUE Phase C (doc 11) — the flat export |
| Raw Position Store as a first-class deduped artifact | Deferred | Tier-B (cross-corpus joins); dedup key already defined (doc 15 §3.5.5) |
| Rich Observation Store emitter (engine-side TLV, Layers 1+3) | Deferred | **Tier-B** self-play lands (post-first-NNUE, D28b) |
| Layer 2 + Layer 4 capture (context + per-move) | Deferred | search-control / cutoff-prediction / LMR research |
| Self-Play Corpus (Tier-B, four-layer, PERMANENT) | Deferred | first NNUE net exists ⇒ own clean data |
| Internal Search Corpus | Deferred | search-control / LMR / pruning / cascade-gating research |
| Root Move Analysis Corpus | Deferred | policy learning |
| Counterfactual Corpus | Deferred | direct search-control / cascade / MoE supervision |
| Adversarial Corpus | Deferred | robustness / hard-example mining |
| POLICY target | Deferred | policy learning (needs Layers 3+4) |
| SEARCH-CONTROL target | Deferred | learned LMR/pruning (needs Layer 4 + Counterfactual) |
| CASCADED-EVALUATION target | Deferred | cascade research (needs cheap-vs-expensive Counterfactual) |
| MIXTURE-OF-EXPERTS target | Deferred | MoE research (needs experts to exist + Counterfactual) |
| Holdout stream (small, unbiased) | Deferred | first calibration/eval that needs an honest sample (doc 15 §5) |
| Derived exporters beyond bullet; columnar analytics mirror | Deferred | the consuming experiment; scale that justifies columns (§9 q1) |

**Sampling discipline (design intent, applied when each corpus is built).** The
training corpora are *intentionally* a diverse, curated distribution — across
search depths, game phases, node types, rare endgames, tactical vs strategic
positions, and hard/adversarial examples — because uniform self-play
under-samples the interesting tails (doc 15 §3.5.7). Cutting *against* that
curation, the platform always maintains a **small unbiased holdout stream** for
honest eval and calibration: sampled *without* the training curation bias,
reserved by canonical key so it never leaks into any training set even across
regenerations and net iterations (§9 q4).

### 3.8 The Tier-A/Tier-B seam — why the rich store is coupled to our own engine

The two provenance tiers (D28) are not just a data-quality progression; they
are the reason the build-now slice is so small.

**Tier-A stays simple — flat bullet records — for two reasons this doc must
state:**

1. **Rich capture on discardable scaffold is wasted investment.** Tier-A data
   is *build scaffold* (D28): the net-0 it trains is thrown away and the
   shipped net is retrained from clean data. Storing per-move observations for
   data we will delete buys nothing — we would never re-derive a label from it.
2. **Critically, a foreign teacher physically cannot emit our engine's
   signals.** Stockfish reports a score and a best move (Layers 1 + a scalar of
   3). It cannot report *our* search's internal-search / per-move / cutoff /
   reduction observations (Layers 2 and 4) — those do not exist in SF's run;
   they are the signals of the search we are trying to *learn to control*.

Therefore the rich Observation Store is **inherently coupled to Tier-B**: our
*own instrumented engine*, generating **PERMANENT** self-play data once a first
NNUE net exists (D28b). The Internal-Search, Root-Analysis, Counterfactual,
Adversarial, and MoE/search-control corpora each require instrumenting *our*
search and are built with Tier-B. Tier-A, lacking those signals, writes
**straight to the Derived layer** — the flat bullet records — bypassing the
Observation Store entirely. That bypass is not a shortcut to be fixed later; it
is correct, because SF cannot fill the store and the store's whole value is the
signals SF cannot produce.

```
 TIER A (D28a, doc 15, EXISTS)                    TIER B (D28b, DEFERRED)
   local Stockfish teacher                          OUR instrumented engine
   (foreign: emits Layer 1 + scalar score;          (emits all four layers;
    CANNOT emit Layers 2 & 4)                         self-play, PERMANENT)
        │                                                   │
        │ Layer-1 + scalar score only                       │ Layers 1–4 + counterfactuals
        ▼                                                    ▼
   ┌──────────────────────────┐                  ┌──────────────────────────────────┐
   │ DERIVED: flat bullet      │  ◄─ deterministic │ RAW POSITION STORE +             │
   │ ChessBoard 32-B (doc 15)  │     cheap export ──│ OBSERVATION STORE (Layers 1–4)   │
   │      DISPOSABLE           │                    │        PERMANENT                 │
   └──────────────────────────┘                    └──────────────────────────────────┘
```

### 3.9 Storage-cost economics (what to preserve, and where staging follows cost)

The "store everything because regeneration is expensive" principle is **not
uniform** — its strength tracks the *product* of two factors:

- **regeneration cost** — how expensive it is to re-run the search that would
  recover the observation, and
- **reuse value** — how likely the data is to be relabeled for a *different*
  future experiment (i.e. how permanent it is).

`preservation value ≈ regeneration cost × reuse value`.

- **Tier-A bootstrap (5000-node SF)** is *low reuse value* — it is discardable
  scaffold, deleted once Tier-B lands (D28). Even though a bulk Tier-A run is
  not cheap in wall-clock (doc 15 §10.3 measured ~1,700–2,100 core-hours for
  100 M at `p_cap 10`), we would *never re-derive a new label from it*, so
  storing rich observations for it is pure waste. Hence Tier-A keeps only the
  flat Derived record and, if ever needed again, regenerates from `meta.json`
  (doc 15 §3.6). **Storage staging follows: Tier-A → flat only.**
- **Tier-B deep-search analysis (high-node, our own, permanent)** is *high on
  both* — deep searches are the most expensive to regenerate, and the data is
  permanent and will be relabeled repeatedly for value, policy, search-control,
  cascade, and MoE experiments. This is where rich four-layer capture earns its
  keep. **Storage staging follows: Tier-B → full Observation Store.**

This is the quantitative justification for §3.7's staging and §3.8's seam: we
invest storage where preservation value is high (permanent deep analysis) and
stay flat where it is low (discardable bootstrap).

## 4. Implementation plan

Commit-sized steps; build-now steps are done or design-only, the rest are
deferred with their trigger. No engine code lands in this doc's scope beyond
the schema module when Tier-B is triggered.

1. **BUILD-NOW (this doc):** ratify the four-layer model, the Observation-Store
   schema (§3.5), the provenance set (§3.4), the staging table (§3.7), and the
   Tier-A/Tier-B seam (§3.8). *Done:* this doc reviewed; D29 pinned in
   decisions.md; README index updated.
2. **BUILD-NOW (done, doc 15):** Tier-A flat pipeline writes the Derived layer
   directly, bypassing the store (§3.8). *Done:* doc 15 §10 validation.
3. **Deferred (trigger: Tier-B):** `src/obs/` — the std-only TLV writer/reader
   behind `feature = "datagen"` (§3.5.2), with an append-only tag registry and
   an unknown-tag skip. *Done:* round-trip unit test; forward-compat test
   (§5).
4. **Deferred (trigger: Tier-B self-play):** wire the emitter into the clean
   generator `bin/datagen.rs` (doc 11 §3.4.7) via the `Worker` / `MakeMoveHooks`
   seam, Layers 1+3 first. *Done:* the flat bullet export of the store equals
   the direct flat path bit-for-bit on a sample (value-data parity).
5. **Deferred (trigger: search-control research):** add Layer 2 + Layer 4
   capture (the OUR-engine-only signals) as a sampled Internal Search Corpus.
   *Done:* Layer-4 fields reconcile against the search's own counters on a
   sample.
6. **Deferred (per consuming research):** the Root Move Analysis, Counterfactual,
   and Adversarial episode kinds and their generators.
7. **Deferred (per experiment):** Derived exporters beyond bullet (policy
   tensors, search-control tables) and an optional columnar analytics mirror —
   discardable tooling, may use deps (§3.5.1).

## 5. Testing and validation

- **Schema round-trip (build-now design test, when §4.3 lands).** Encode an
  `Observation` with all four layers, decode it, assert structural equality —
  the store's own read path, not a reimplementation.
- **Forward-compatibility proof.** A reader built against `format_version = N`
  reads a shard written with an *added* field tag and an *added* `EpisodeKind`
  and recovers every N-version field while skipping the new ones without error.
  This is the acceptance test for the §3.5.2 unknown-tag skip.
- **Bullet-projection round-trip (integrity).** Take a sample of `Observation`s,
  project to the flat bullet record (§3.6.3), and confirm the projected
  `(placement, white-relative score, result)` equals an independent recompute —
  extending doc 15 §5's perspective/dedup checks to run *through* the store.
  The store→bullet→trainer path ingests in `bullet` without error (doc 15 §4
  step 4, now sourced from the store).
- **Provenance completeness.** Assert every shard carries the full §3.4
  provenance block and every record references a valid provenance index; a
  record without provenance is unwritable.
- **Determinism / regenerability.** Same provenance (seed + config + engine
  bench signature) reproduces a byte-identical shard, generalizing doc 15
  §5's determinism check to the four-layer format.
- **Bench-neutrality (build-now guard).** With the `datagen` feature off, the
  D24 `bench` fingerprint is bit-identical to a build without this module —
  the emitter must never perturb normal search (§6).

## 6. Performance / storage notes

- **Bench-neutral when off.** The emitter hangs off the existing
  `MakeMoveHooks` / `Worker` seam (doc 04, shared with NNUE accumulator
  recording) behind a `datagen` build feature (or a runtime flag guarding a
  cold branch). The tournament/bench binary compiles it out, so it cannot move
  the D24 fingerprint — Layer-4 capture in particular allocates and is *only*
  ever on in the datagen build, never in the search binary that plays games.
- **Per-worker, lock-free write path.** Each `Worker` owns its own shard writer
  (D23), mirroring per-worker TT/PRNG ownership; no shared mutable writer, no
  lock on the search path. Records buffer per episode and flush at game end
  (§3.5.4); shards merge offline, exactly like doc 15's per-worker shards.
- **Volume asymmetry — why the store is sampled, not blanket.** A flat bullet
  record is 32 bytes. A full four-layer root `Observation` with Layer 4 over
  ~35 legal moves is ~1–2 KB — **30–60×** the flat record — and Layer 4 grows
  with branching × nodes if captured at every node. Blanket four-layer capture
  of every node of every Tier-B game is therefore infeasible; the Internal
  Search Corpus is *sampled* (which nodes get full Layer 4 vs Layer-1/3 only),
  and this sampling policy is the central open question §9 q3. The value-only
  export path (Layers 1+3) stays cheap and is what the EVALUATION target needs.
- **Text/analytics is Derived and offline.** Any columnar re-encoding, joins,
  and per-column scans happen in discardable tooling on the Derived layer, off
  the engine's hot path and outside D15.

## 7. Interactions

- **doc 15 (Tier-A instance).** Doc 15 owns the SF-teacher bootstrap, its
  curation/teacher/volume, and the flat `bulletformat::ChessBoard` contract.
  This doc places it as the degenerate Self-Play mode that writes *straight to
  the Derived layer* (§3.8) and consumes its flat record as the projection
  target (§3.6.3). This doc does not re-specify any of doc 15; a single
  cross-reference line is added there pointing back here.
- **doc 11 (NNUE / EVALUATION target).** Phase C NNUE consumes the EVALUATION
  target — value/WDL — a cheap derivation from Layers 1+3 + episode WDL. Doc
  11's own clean generator `bin/datagen.rs` (§3.4.7) *is* the Tier-B producer
  that populates the store once instrumented; doc 11 owns the net/arch/training,
  this doc owns the observation schema that generator emits.
- **doc 10 (pruning/reductions/extensions).** The SEARCH-CONTROL targets (LMR
  risk, pruning safety, extension utility, re-search probability) are labels
  derived from Layer-4 per-move observations, and they will eventually
  *supervise the very P9 heuristics doc 10 specifies* — closing the loop from
  D20's hand-tuned SF constants toward learned search control. Layer 4 records
  reductions/extensions/prune-reason/re-search precisely because those *are*
  the doc-10 decisions, captured as training signal.
- **doc 09 (move ordering).** Layer-4's `order_stage`/`order_index`/`order_score`
  observe the MovePicker stages and history scores doc 09 defines, feeding the
  POLICY (search-cost-aware ordering) target.
- **doc 13 (threading).** The emitter is per-worker (each `Worker` owns its
  shard writer, D23), no locks on the search path; hooks share the doc-04 seam
  with NNUE. Shards merge offline.
- **doc 07 (search core).** Node types, the iterative-deepening snapshots, and
  `RootMove`/`PvLine` are the sources of Layers 2 and 3.
- **D28 / D29.** This doc is the D29 spec and reconciles with D28's two tiers:
  Tier-A scaffold → flat/Derived; Tier-B clean → rich/permanent store (§3.8).

## 8. Acceptance criteria (D26)

The platform is **correctly designed and its build-now slice is done** when:

1. This doc captures the full vision — the four record layers (§3.2), five
   generation modes (§3.6.1), five research-target categories (§3.6.2), three
   stores (§3.3), the mandatory provenance set (§3.4), the sampling +
   holdout-stream intent (§3.7), and the long-term vision (§3.1) — and is
   reviewed.
2. The Observation-Store schema (§3.5) is concrete: a versioned, self-describing
   TLV format with the four-layer field map, the append-only/skip-unknown
   forward-compat rules, and a stated justification for the zero-dep
   hand-rolled choice under D15/D14.
3. The flat bullet record is shown to be a **trivial projection** of the schema
   (§3.6.3) — the superset property — so doc 15's Derived corpus is exactly
   `project(Observation)`.
4. The staging table (§3.7) classifies **every** corpus and research target as
   build-now or designed-deferred with a named trigger, and the build-now set
   is exactly `{ Tier-A bootstrap (doc 15, exists), Observation-Store schema
   (this doc), EVALUATION value/WDL export }`.
5. **Provenance completeness** is mandated: the full §3.4 set is a required,
   non-optional block; no observation is writable without it.
6. The Tier-A/Tier-B seam (§3.8) and the storage-cost economics (§3.9) are
   stated as the justification for the staging.

The Observation-Store build (deferred, gated on Tier-B) is done when:

7. **Round-trip integrity:** a sample written to the store, exported to bullet
   flat records, and ingested by the trainer round-trips — the projected
   `(placement, white-relative score, result)` equals an independent recompute
   (§5), and value-data parity with the direct flat path is bit-for-bit.
8. **Forward-compat proof:** a reader built at `format_version N` reads a shard
   written at `N`-plus-a-new-field-and-episode-kind without error, recovering
   all N-version fields and skipping the new ones (§5).
9. **Regenerability:** same provenance reproduces a byte-identical shard, and
   the `datagen`-off `bench` fingerprint is unchanged (§5, §6).

## 9. Open questions

1. **Storage backend / format final choice.** The permanent store is TLV row-log
   now (§3.5.1). Whether it later gains a **columnar/Parquet analytics mirror**
   (produced by discardable tooling) or a real embedded store (sqlite/lmdb —
   deps, so tooling-side only) depends on whether per-column scans at scale ever
   dominate. The row-log write path is fixed by D15; only the Derived-side read
   ergonomics are open.
2. **Dedup key across datasets.** The Layer-1 canonical key (placement + stm +
   castling + ep, doc 15 §3.5.5) dedups within a run. The **cross-corpus**
   identity for the Raw Position Store needs pinning before cross-dataset joins:
   does it fold ep only-when-legal (as the Zobrist key does, D9)? Does it bucket
   rule50? A mismatch here silently double-counts or wrongly merges positions.
3. **How much per-move (Layer-4) data is affordable at scale.** Full Layer 4 for
   every searched move at every node is 30–60× the flat record (§6) and grows
   with branching × nodes. The **sampling policy** — which nodes get full Layer
   4, which get Layers 1/3 only — is unresolved and is the main cost lever for
   the Internal Search Corpus (analogous to doc 15's `p_cap` lever, §10.4 there).
4. **Holdout-stream design.** How to carve and *freeze* the small unbiased
   holdout so it stays uncontaminated across corpus regenerations and net
   iterations — by canonical key reservation? by hashed game id? — and how to
   keep it sampled *without* the training curation bias so calibration stays
   honest.
5. **When Tier-B instrumentation lands.** The rich emitter is gated on the first
   NNUE net (D28b). The exact phase (post-P13) and whether a **minimal Layers
   1+3 emitter** ships earlier — purely to route value data through the store
   rather than the flat bypass — is open; the flat bypass is correct until then
   (§3.8).
6. **Counterfactual pairing representation.** Whether a counterfactual pair is
   two linked `Observation`s sharing a `canonical_key` + `cf_axis` (the §3.5
   sketch) or a single record carrying an array of per-config observations. The
   former composes better with the sampled Internal Search Corpus; the latter is
   denser. Decided when the Counterfactual Corpus is first built.

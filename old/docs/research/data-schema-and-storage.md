# ARCHON data schema & storage spec (capture-once superset + efficient store)

Status: Living. Owner doc for the research program's DATA CONTRACT and its
STORAGE architecture. Written 2026-07-18.

This doc formalizes the promise made in
`docs/research/data-quality-strategy.md` ("the obs schema ALREADY captures the
superset") into an enforceable contract, and specifies how that superset is
stored efficiently on a 14 GB-free box today and at D36/GCS scale later. It sits
on top of `design/16-research-data-platform.md` (the platform / obs-store
substrate) and `design/18` + `tools/ledger/ledger.py` (the D38 provenance
ledger). It is grounded in the ACTUAL record types in `src/obs/schema.rs`, the
generator `src/bin/root_analysis.rs`, and the export/view tool
`src/bin/obs_tool.rs`; every field name below is a real field.

**One-sentence mandate:** capture the expensive-to-regenerate superset ONCE, in
the append-only obs store, so every future net (value, D41 helm/allocation,
policy, time-management, learned pruning, and the self-improvement oracle) reads
a cheap DERIVATION instead of paying to re-run the search — and store it so the
never-pruned corpus fits a tiered local+archive budget.

Related live docs: `docs/research/data-quality-strategy.md` (the governing
two-dimension framework), `design/16` §3.2–§3.9 (four-layer model, three stores,
provenance, staging, storage economics), `design/18` (ledger), `design/19`
(learned search allocation — the D41 helm consumer), `docs/research/value-foundation-state.md`
(live leaderboard / in-flight jobs).

---

## 1. Purpose and principles

The obs store is already a strict SUPERSET of every training label we know how
to want (`design/16` §3.6.3 proves the flat 32-byte bullet record is a trivial
projection of one `Observation`). This spec exists to (a) pin that superset as
THE contract, (b) add the small, forward-looking extensions the data-quality
strategy needs (games, outcomes, multi-oracle labels), and (c) make the storage
survivable on one box. Seven principles govern everything below:

1. **Capture the superset once.** Store expensive OBSERVATIONS, not just current
   labels (`design/16` §3.1). A deep search is expensive to run and cheap to
   re-label; preserve the eval trajectory, per-move analysis, and full
   provenance and every future label becomes a derivation.
2. **Value-tiered capture.** Every deep search captures the FREE layers
   (trajectory + PV + metadata + provenance). HIGH-VALUE positions additionally
   get the rich per-move / effort layer (~35× cost). Bulk low-value positions
   get free layers only (§5).
3. **Never prune positions.** A low-relevance position with a GOOD label is
   net-positive. We improve labels IN PLACE (additively) and ADD positions; the
   corpus only grows (data-quality-strategy P1).
4. **Additive, versioned, multi-oracle labels.** A position accumulates
   `{oracle, budget, date}` labels; nothing is ever overwritten. Old labels stay
   as historical evidence; export MERGES them by precedence (§4.3).
5. **Views, not copies.** The obs store (and immutable external binpacks) are the
   SOURCE OF TRUTH; training corpora (`obs-tool export-train` NumPy SoA, bullet
   binpacks) are DERIVED and DISPOSABLE, regenerated from a recipe (`design/16`
   §3.3). Never treat a derived corpus as ground truth.
6. **Content-addressed + ledgered.** Every shard, corpus, and net is a D38 ledger
   row whose identity is its sha256 and which carries a regen recipe
   (`tools/ledger/ledger.py`). "Where the bytes physically live" is decoupled
   from "what the bytes are" — a locally-pruned shard is not lost, it is archived
   and provable by hash.
7. **Forward-compatible, zero-dep write path.** The store is versioned,
   self-describing TLV with APPEND-ONLY tags and skip-unknown decoding
   (`design/16` §3.5.2); a reader built today survives shards written years from
   now. New fields get fresh tag numbers; none are ever renumbered or reused.

---

## 2. Consumer map — who needs which record, and what it costs to regenerate

This is the justification for capturing the superset: the same deep search feeds
many nets, and re-running it per-consumer is the cost we refuse to pay. "Regen
cost" is the price of recovering the signal if we had NOT stored it.

| Consumer (net / research target) | Reads (record.field) | Implemented? | Regen cost |
|---|---|---|---|
| **Value net** (eval, build-now) | `PositionState` (L1) + `SearchObservations.final_score`/`bound` (L3) + `Episode.game_result` (WDL) | Yes — **`obs-tool export-value`** (`FEN\|cp\|wdl`, White-relative, Exact+Complete quarantine → bf-convert → bullet). NOT export-train (that needs Complete arms and skips reference-only shards) | Expensive — a fresh deep search per position (the whole point) |
| **D41 helm / learned allocation** (`design/19`) | `SearchObservations.id_snapshots` (eval-vs-depth trajectory: `{depth,score,nodes,time_ms,pv}`), `SearchContext.{depth,node_budget,static_eval,correction_value,tt_*}`, `PerMoveObservation.{nodes,searched_depth}` (effort) | Schema yes; trajectory populated by real searches, effort by `root-analysis` per-arm and `InSearchCapture` | Very expensive — instrumented search; trajectory is unrecoverable after the fact |
| **Policy / move-ordering** | `PerMoveObservation.{order_index,order_stage,order_score,returned_score,raised_alpha,caused_cutoff}`, `SearchObservations.{best_move,pv}`; `export-train` `value`/`margin`/`is_ref_best` | Yes (RootMoveAnalysis + export-train per-arm rows) | Very expensive — one search PER LEGAL MOVE (~35×) |
| **Time-management** | `IdSnapshot.{time_ms,nodes}` per depth (time-to-depth curve), `SearchObservations.completion` | Schema yes; needs trajectory-carrying searches | Expensive — re-run timed searches; machine-relative (`Provenance.hardware_identity`) |
| **Learned pruning / extensions ("beyond")** | `PerMoveObservation.{reduction,extension,pruned_by,researched,scout_*,research_*,reference_bound,bound_error}` + Counterfactual pairs (`Episode.cf_axis`, `EvidenceKind::Intervention`) | Schema yes (D43 fields present); Counterfactual generator deferred | Most expensive — instrumented + FORCED-interventional search |
| **Self-improvement oracle** (ARCHON-as-teacher flywheel) | full stack + multi-oracle labels (`Provenance.teacher_identity`) + disagreement views | Storage yes; ARCHON-teacher labeling is a future campaign | Whole-flywheel cost |

Read the table as: the value net is the only build-now consumer, but the SAME
`Observation` a value relabel produces already carries the trajectory, per-move,
and provenance that the helm/policy/time/pruning nets will need — so routing
relabels through the obs store (not the scalar shortcut) pays all of these for
free. That is the entire point of §7 step 1.

---

## 3. The canonical schema (the contract)

The record types in `src/obs/schema.rs` ARE the contract. On disk they are the
TLV encoding of `design/16` §3.5.2 (LEB128 varints, zigzag signed, `(tag<<3)|wire`
keys, unknown-tag skip). Tag numbers are the wire contract and are append-only.
Everything below is IMPLEMENTED unless marked **[proposed]** (§4).

### 3.1 Shard framing

A `.ceobs` shard = `OBS_MAGIC` (`b"CEOBS\0"`) + a `ShardHeader` + a stream of
`Episode`s. `ShardWriter`/`ShardReader` (`src/obs/shard.rs`) are the only
sanctioned read/write path; tooling decodes THROUGH them (never a second TLV
implementation — see `obs-tool export`).

| `ShardHeader` | tag | purpose |
|---|---|---|
| `format_version: u16` | 2 | bumped only on a breaking wire change (append-only rule avoids this) |
| `created_unix: u64` | 3 | shard write time — the **date** axis of a multi-oracle label (§4.3) |
| `provenance: Vec<Provenance>` | 4 | interned once per shard; episodes reference by index |

### 3.2 Provenance (mandatory — no record is writable without it)

`design/16` §3.4. Interned per shard/run; per-record cost is a `u32` index.

| `Provenance` | tag | purpose |
|---|---|---|
| `engine_version: String` | 1 | git rev / semver of the searching engine |
| `engine_bench_sig: u64` | 2 | D24 `Bench:` node count (0 = external-engine sentinel) |
| `net_version: Option<String>` | 3 | our net id (e.g. `EvalFile`); `None` for a foreign teacher |
| `teacher_identity: Option<String>` | 4 | producer engine rev + binary sha256 — **the ORACLE identity** for relabels |
| `search_config: String` | 5 | full canonical knob set (includes root-analysis arm mode) |
| `search_config_hash: u64` | 6 | fast join key over `search_config` |
| `generator_version: String` | 7 | the datagen tool rev |
| `rng_seed: u64` | 8 | per-worker PRNG seed |
| `dataset_source: String` | 9 | generation mode (`root_move_analysis`, `in_search_capture`, …) |
| `opening_source: String` | 10 | book id / random-ply policy |
| `hardware_identity: String` | 11 | cpu model + features + threads — time/cycle labels are machine-relative |

The `{teacher_identity, node_budget (§3.4), created_unix (§3.1)}` triple IS the
`{oracle, budget, date}` of the multi-oracle label model — already captured, no
new field (§4.3).

### 3.3 Layer 1 — `PositionState` (full reconstruction + identity)

| field | tag | purpose |
|---|---|---|
| `occ: u64` | 1 | occupancy (D3 mapping; identical to bullet's `occ`) |
| `pcs: Vec<u8>` | 2 | nibble-per-occupied-square placement |
| `stm: u8` | 3 | side to move — resolves per-ply producer in a game (§4.1) |
| `castling: u8` | 4 | 4-bit rights mask |
| `ep_square: Option<u8>` | 5 | en-passant target |
| `rule50: u8` | 6 | halfmove clock |
| `game_ply: u16` | 7 | ply linkage within a game |
| `plies_from_null: u16` | 8 | |
| `repetition: i8` | 9 | |
| `canonical_key: u64` | 10 | **THE dedup identity** — unfolded main key (placement+stm+castling+ep, NO rule50 fold; D33 r1) |
| `path_from_episode_root: Vec<u16>` | 11 | legal moves root→node (empty for `InSearchCapture`) |

`canonical_key` is the join/dedup key across the whole platform. A position
searched by five oracles is ONE identity, five labels.

### 3.4 Layer 2 — `SearchContext` (circumstances)

23 fields (tags 1–23). Grouped: window/shape (`depth`, `node_budget`, `alpha`,
`beta`, `alpha_before_move`, `beta_before_move`, `root_delta`, `node_type`,
`ply`); node signals (`static_eval`, `tt_hit`, `tt_depth`, `tt_bound`,
`tt_value`, `tt_move`, `flags`, `correction_value`, `legal_move_count`,
`capture_move_count`); determinism (`event_ordinal`, `event_fingerprint`);
regimes (`tt_regime`, `history_regime`). `node_budget: Option<u64>` (tag 2) is
the **budget** axis of a multi-oracle label. Per-episode-kind presence: the
engine-internal group is Default-filled for external-teacher RootMoveAnalysis
and MUST NOT be consumed there (`design/16` §3.5.3 F2 rule).

### 3.5 Layer 3 — `SearchObservations` (outcomes) + `IdSnapshot`

| `SearchObservations` | tag | purpose |
|---|---|---|
| `best_move: u16` | 1 | |
| `pv: Vec<u16>` | 2 | |
| `final_score: i32` | 3 | stm-relative value — **the scalar value label** |
| `bound: Bound` | 4 | Exact/Upper/Lower — the label-quarantine gate (only Exact is a clean value; §5) |
| `nodes: u64` | 5 | subtree cost |
| `seldepth: i32` | 6 | |
| `id_snapshots: Vec<IdSnapshot>` | 7 | **the eval TRAJECTORY** — the #1 capture-now / helm signal |
| `completion: Completion` | 8 | Complete/Stopped/TargetMiss/Invalid — a timed-out search is never a clean Exact label |

`IdSnapshot` (per iterative-deepening depth): `depth` (1), `score` (2),
`best` (3), `nodes` (4), `time_ms` (5), `pv` (6). This is the eval-and-effort
curve per depth — free the moment a real search runs, unrecoverable afterward.

### 3.6 Layer 4 — `PerMoveObservation` (per searched/considered move)

29 fields (tags 1–29). The search-control gold; a foreign teacher physically
cannot emit it (`design/16` §3.8). Grouped:

- **identity/order** — `mv`, `order_index`, `order_stage`, `order_score`
- **shaping** — `searched_depth`, `reduction`, `extension`, `baseline_reduction`,
  `requested_reduction`, `actual_scout_depth`, `policy_tier`
- **outcome** — `returned_score`, `nodes`, `raised_alpha`, `caused_cutoff`,
  `researched`, `pruned_by`
- **scout/research/reference** (D43) — `scout_score`, `scout_bound`,
  `scout_nodes`, `research_score`, `research_bound`, `research_nodes`,
  `research_changed_bound`, `research_was_necessary`, `reference_bound`,
  `bound_error`
- **meta** — `evidence` (Observational/Intervention/Reference), `completion`

### 3.7 Composition — `Observation` and `Episode`

`Observation` = the four layers: `position` (1), `context` (2), `search` (3),
`moves: Vec<PerMoveObservation>` (4).

| `Episode` | tag | purpose |
|---|---|---|
| `kind: EpisodeKind` | 1 | SelfPlay=0, InternalSearch=1, RootMoveAnalysis=2, Counterfactual=3, Adversarial=4, InSearchCapture=5 |
| `episode_id: u64` | 2 | |
| `provenance_ref: u32` | 3 | index into `ShardHeader.provenance` (the **white/self** producer) |
| `game_result: Option<Wdl>` | 4 | white-relative WDL, back-patched at game end — **the outcome value target** (§4.2) |
| `cf_axis: Option<CfAxis>` | 5 | which knob differs across a counterfactual pair |
| `records: Vec<Observation>` | 6 | |
| `replay_root: PositionState` | 7 | episode-start state |
| `root_reversible_keys: Vec<u64>` | 8 | repetition history preceding the root |
| `event_hash_chain: Vec<u64>` | 9 | locate first deterministic divergence |
| `dropped_records: u64` | 10 | sampled records the per-game cap refused (no silent loss) |
| `provenance_ref_black: Option<u32>` | **[proposed] 11** | the black-side producer for an `EngineGame` (§4.1) |

Enums (`Wdl`, `NodeType`, `Bound`, `Prune`, `CfAxis`, `EvidenceKind`,
`PolicyTier`, `Completion`, `TtRegime`, `HistoryRegime`, `EpisodeKind`) carry a
`Unknown(u8)` variant and follow the same append-only, skip-unknown rule.

---

## 4. Extensions needed for the forward-looking data (minimal, additive)

The data-quality strategy adds three things the current store does not yet
express first-class: strong-engine GAMES, game OUTCOMES on each position, and
MULTI-ORACLE labels. All three are additive; none breaks a reader. The design
decision throughout: **prefer a new tag or an export-time derivation over a wire
change; keep the write path zero-dep.**

### 4.1 GAME episode metadata — `EpisodeKind::EngineGame` [proposed]

Dimension-2 positions come from strong-engine games (SF18 v SF18, SF18 v
Reckless, SF18 v ARCHON, Reckless v ARCHON). A two-engine game differs from
`SelfPlay=0` (one engine) in one way: **each side has its own identity/config.**

Extension (both append-only, byte-unchanged for existing shards):

- **`EpisodeKind::EngineGame = 6`** — a two-engine game. Distinct discriminant so
  a two-engine game is provenance-separable from single-engine self-play.
- **`Episode.provenance_ref_black` (tag 11, `Option<u32>`)** — the black-side
  interned `Provenance`; `provenance_ref` (tag 3) stays the white producer.
  Encoded only when set (existing shards decode it as `None`).

Everything else the strategy asks for is ALREADY native and needs no field:

- **Both engines' identity/config** — two interned `Provenance` blocks (each
  carries `engine_version`, `net_version`, `teacher_identity`, `search_config`,
  `hardware_identity`).
- **Opening** — `Provenance.opening_source` + `Episode.replay_root`.
- **Result** — `Episode.game_result` (`Wdl`), back-patched at game end.
- **Per-move engine eval / depth / time for BOTH sides** — one `Observation` per
  ply; the moving side's search rides in `SearchObservations` (`final_score`,
  `bound`, `completion`) + `id_snapshots` (`depth`, `time_ms`, `nodes` per
  depth). The PRODUCER of a ply is resolved by `PositionState.stm` → white uses
  `provenance_ref`, black uses `provenance_ref_black`. No per-record producer
  field is needed.
- **Ply linkage** — `PositionState.game_ply` + record order within the episode.

A game producer buffers one episode (~100–400 records, trivially in memory,
`design/16` §3.5.4) and appends it whole with the WDL attached. In practice the
cheapest realization reuses the existing in-game `InSearchCapture` recorder
(`src/obs/recorder.rs`) plus a PGN/result join, upgraded to stamp
`EngineGame` + the second producer ref when two engines and a known result are
present.

### 4.2 Outcome propagation — a DERIVATION, not a field

AlphaZero-style value training wants the game WDL attached to each position
alongside the search label. `Episode.game_result` already holds the
white-relative `Wdl`; propagation is an EXPORT step, not storage: for each
record in a game episode, project `game_result` (sign-mapped by
`PositionState.stm`) into the training row's outcome column. This is exactly the
bullet projection `design/16` §3.6.3 already specifies
(`result ⟵ obs.episode.game_result`). Action: add a `wdl` column to
`export-train`'s value rows for game-derived episodes (RootMoveAnalysis /
InSearchCapture episodes have `game_result = None` and emit no WDL). No schema
change.

### 4.3 Additive multi-oracle labels — append-only storage + export MERGE

A position accumulates labels over time — SF18@25k, SF18@100k, Reckless@100k,
later ARCHON@N — and **none is ever overwritten**. This is already how the store
behaves: each relabel is a NEW episode in a NEW shard, sharing the position's
`canonical_key`, each carrying its own `Provenance`. The `{oracle, budget, date}`
tuple is fully captured with NO new field:

- **oracle** = `Provenance.teacher_identity` (interned; `obs-tool export-train`
  already interns it as `teacher_id`).
- **budget** = `SearchContext.node_budget` (already the `tier` column).
- **date** = `ShardHeader.created_unix`.

The extension is at the EXPORT/MERGE layer, where a policy decision currently
loses information. `obs-tool export-train`'s `CorpusBuilder` dedups by
`canonical_key` with FIRST-OCCURRENCE-WINS (`seen_keys.insert(key)` drops later
episodes). For multi-oracle that silently discards a better label. Replace it
with a **group-then-select merge**:

1. GROUP all episodes/records sharing `canonical_key` across every input shard.
2. SELECT the primary label by precedence:
   **deepest `node_budget` → newest `created_unix` → strongest oracle**, where
   oracle strength is an explicit configured ranking (today
   `SF18 > Reckless > ARCHON-current`; the flywheel promotes ARCHON as it
   surpasses an anchor in a regime).
3. KEEP ALL labels available — emit the chosen primary as `value`/`margin`, and
   thread the losing labels (or at minimum their `{teacher_id, tier, date}`)
   so disagreement views (§4.4) can be computed and so a later experiment can
   re-select a different primary without re-reading shards.

Threading `created_unix` into the exported row (from the shard header) is the
only new plumbing; it needs no schema field. This keeps the merge policy in
disposable tooling where it belongs, so precedence can change per experiment
without touching the permanent store.

### 4.4 Derived views, never stored: ARCHON-error and disagreement

`|SF18_deep − existing_label|` (the active-learning targeting signal) and
`|oracle_A − oracle_B|` (cross-oracle disagreement) and ARCHON-error
(`|ARCHON_in-game_score − oracle_relabel|`, joined by `canonical_key`) are
COMPUTED from two labels at analysis/export time — never stored fields.
`obs-tool disagree` already computes exactly this between two row-aligned shards
(best-move agreement, `|Δ final_score|` quantiles, Kendall-τ arm-ranking
correlation). Storing a derived score would violate principle 4 (it would go
stale the moment a new oracle label arrives). Keep them views.

### 4.5 Extension summary

| Extension | Kind | Wire impact |
|---|---|---|
| `EpisodeKind::EngineGame = 6` | new enum discriminant | append-only, skip-unknown safe |
| `Episode.provenance_ref_black` | new TLV tag 11 (`Option<u32>`) | encoded only when set; existing shards unchanged |
| Outcome (WDL) on each position | export-time derivation of `Episode.game_result` | none (schema already has the field) |
| Multi-oracle labels | append-only episodes (already) + export MERGE policy | none (thread `created_unix` through export) |
| ARCHON-error / disagreement | derived view (analysis-time) | none |

---

## 5. Value-tiered capture contract

Not every deep search should pay for the rich per-move layer. Three tiers, with
an explicit high-value gate. This is the agreed default (data-quality-strategy
"Value-tiered capture"), now pinned.

| Tier | When | Layers captured | ~cost |
|---|---|---|---|
| **Tier-F (Free)** | EVERY deep search / relabel | L1 `PositionState` + meaningful L2 `SearchContext` + L3 `SearchObservations` incl. `id_snapshots` trajectory + `pv` + `completion` + `Provenance` | 1 search (~0.4–1 KB/pos) |
| **Tier-R (Rich)** | HIGH-VALUE positions only | Tier-F **plus** full L4 `PerMoveObservation` per legal move (via `searchmoves`/child arms — the `root-analysis` shape) | ~35× search (~2–10 KB/pos) |
| **Tier-C (Counterfactual)** | THIN sample, far-future | paired interventional searches (`cf_axis`, `EvidenceKind::Intervention`) for true search-augmentation | ~2× per pair; keep hooks, do NOT scale on one box |

**Decision rule — a position qualifies for Tier-R if ANY of:**

1. **High disagreement** — `|strong-oracle − existing_label|` above a percentile
   threshold (active learning). `|existing_cp|` is a free coarse prior (the
   calibration shows label error concentrates in decisive positions: RMS 26cp
   balanced → 136cp in >400cp).
2. **Game-derived** — extracted from a strong-engine game (`EngineGame`),
   especially decisive lines our net under-samples.
3. **ARCHON blind-spot** — positions where ARCHON's in-game eval diverged most
   from the oracle in an ARCHON game (learn-from-losses, the highest-value
   subset).
4. **Holdout reservoir** — a small unbiased slice, reserved by `canonical_key`,
   always Tier-F and never curated (honest calibration).

Everything else (bulk low-disagreement) is **Tier-F only**. The rationale is the
preservation-value economics (`design/16` §3.9): rich capture earns its keep
only where regeneration cost × reuse value is high.

**Label-quarantine note (binding).** For `InSearchCapture` and any interior-node
capture, `Completion` is NOT an exactness signal (the recorder stamps every
capture `Complete`); the exactness gate is `SearchObservations.bound` — only
`Bound::Exact` (PV) nodes carry a clean scalar value; cut/all nodes are
aspiration bounds and are quarantined from L2 value losses (measured: in-search
capture is ~99.6% aspiration bounds, ~0.36% exact PV). Bounded labels are usable
as ORDERING/ranking signal, never as an L2 value target. This is the same
quarantine the 5k-node datagen labels sit under.

---

## 6. Storage architecture & efficiency

### 6.1 Two physical tiers, forced by the size ratio

There are two record shapes with a ~60–300× byte ratio, so they cannot share one
physical store.

| Store | Record | bytes/pos | 1 GB holds | 14 GB (local free) | 1 B positions |
|---|---|---|---|---|---|
| **BULK scalar** (bullet `ChessBoard`) | flat 32 B | 32 | ~31 M | ~440 M | ~32 GB |
| **BULK scalar** (official SF binpack, delta-coded) | single-digit B amortized | ~2–6 | ~200–500 M | ~3–7 B | ~2–6 GB |
| **RICH obs — Tier-F** (L1+L3+prov) | ~0.4–1 KB | ~500 | ~1–2.5 M | ~14–35 M | ~0.4–1 TB |
| **RICH obs — Tier-R** (L1+L3+L4, ~35 arms) | ~2–10 KB | ~2 K–10 K | ~100–500 K | ~1.4–7 M | ~2–10 TB |

Corroboration from disk: a `gold-v2` per-worker RootMoveAnalysis shard is
~39 MB; the whole `gold-v2` campaign (13 shards) ≈ 525 MB for a modest position
count — i.e. rich obs is hundreds of MB per campaign of ~10⁴–10⁵ positions,
while `data/corpora` holds 70 GB of BULK binpacks representing billions of
position-presentations. **The math is the whole argument:** a rich billion is
terabytes and cannot live locally, so the bulk scalar tier stays flat/binpack
(billions) and the rich obs tier is RESERVED for the high-value slice (millions).
This is why §5's tiering is not optional — it is forced by 14 GB of local disk
against a 300× record-size ratio.

Current disk reality (this box): 251 GB total, ~14 GB free (95% used);
`data/corpora` 70 GB (bulk binpacks + lichess evals), `data/nnue` 17 GB
(training artifacts), `data/obs` 2.3 GB (rich shards). Local CANNOT hold
everything → the plan is tiered + archive-aware from day one (§6.3).

### 6.2 Append-only, content-addressed, log-structured; dedup at scale

- **Append-only, immutable.** A new label is a NEW episode in a NEW shard; shards
  are never mutated (`design/16` §3.3). This is what makes multi-oracle
  accumulation (§4.3) and never-prune (principle 3) free, and what lets the D38
  ledger content-address a shard by a STABLE sha256.
- **Dedup identity = `canonical_key`** (unfolded raw Zobrist: placement+stm+
  castling+ep, no rule50 fold). Within an export, `CorpusBuilder` dedups on it;
  across the platform it is the join key.
- **Billion-scale dedup — NOT an in-RAM set.** Today `export-train` uses an
  in-RAM `HashSet<u64>` — correct at the current millions scale. It does NOT
  survive a billion keys: `1e9 × u64` is 8 GB of keys alone (2–3× with hashset
  overhead) on a 14 GB box. When the rich corpus crosses ~100 M keys, switch to
  a **disk-partitioned (grace-hash) dedup**: partition rows by the top bits of
  `canonical_key` into K buckets (e.g. 4096), each of which fits in RAM, and
  dedup each bucket independently; concatenate. Equivalent alternatives:
  external merge-sort the keys and drop adjacent duplicates, or a **bloom filter
  pre-filter + verify** against a persistent per-campaign SORTED key index (the
  bloom gives fast negatives at ~10 bits/key ≈ 1.25 GB for 1 B at ~1% FPR; the
  sorted index confirms candidates). The persistent per-campaign sorted
  `canonical_key` index doubles as the membership structure for the holdout
  reservation.

### 6.3 Compression, regenerable views, local vs archive

- **Compression (zstd at rest).** CEOBS TLV is varint-packed but still ~2–4×
  compressible (placement nibbles, small scores, repetitive per-move rows). Store
  FINISHED campaign shards zstd-compressed; decompress streaming on read. To keep
  the engine write path zero-dep (D15), the emitter writes RAW `.ceobs` and an
  OFFLINE tooling step (`zstd`) compresses the finished campaign — compression is
  disposable-tooling territory. The D38 ledger hashes the bytes AS STORED (the
  `.ceobs.zst` file), so identity tracks what is on disk.
- **Regenerable VIEWS.** Training corpora are DERIVED and DISPOSABLE:
  `obs-tool export-train` → NumPy SoA dir (`archon.obs-train.rootmove-value-v1`);
  bullet binpack → `train.bin`/`val.bin`. Their regen recipe is the export
  command over a NAMED set of obs campaigns. They are NOT the source of truth and
  are safe to delete under disk pressure. **Exception:** the official SF binpacks
  and lichess evals under `data/corpora/` are EXTERNAL immutable INPUTS (we cannot
  regenerate them) — they are the source of truth for the bulk scalar tier, kept
  or re-downloadable, registered with `add-data --source`.
- **Local working set vs D36/GCS archive.** Local disk holds only (a) the active
  experiment's derived corpus, (b) the current-cycle obs campaigns being
  labeled/exported, and (c) the ledger metadata (ALWAYS present). The permanent
  archive (D36 GCS mirror) holds all compressed obs campaigns, the immutable
  external binpacks, and historical nets. Local is a CACHE of the archive's
  working set; a shard pruned locally is provable-present in the archive by its
  ledger hash. Recommended ledger status for this: annotate `--status archived`
  (a sibling of the existing `lost`) so `verify` treats locally-absent-but-hashed
  bytes as expected, not drift.

### 6.4 The D38 ledger as the index / provenance / verify layer

`tools/ledger/ledger.py` (`design/18`) is the always-present, git-tracked index
over gitignored bytes. Every shard/campaign, corpus, and net is a row:

- `add-data` a finished obs campaign (sha256 via `sha256_tree` manifest hash over
  the shard dir — deterministic regardless of read order) with
  `--gen-recipe` (the `root-analysis`/recorder invocation) and `--source`.
- `add-data` a derived corpus with the `export-train` command as its recipe.
- `add-net` a trained net linking `--train-data` (a corpus id) and `--parent`.
- `add-exp` every match/SPRT/SPSA.
- `verify` recomputes hashes vs the registry, checks referential integrity
  (dangling `train_data`/`parent_net`/`A.net`/`B.net`), and honors `lost`/
  `archived` so acknowledged-absent bytes are not counted as drift.

The ledger is what makes principle 6 real: the metadata (hashes + recipes + DAG)
is small and diffable and git-tracked; the heavy bytes are tiered/archived; any
version is reproducible from its recipe and provable by its hash.

### 6.5 Concrete layout — directory / shard naming, growth over time

The never-pruned corpus is organized as an APPEND-ONLY SERIES OF DATED CAMPAIGNS.
A campaign is one labeling/generation run — a single `{oracle, budget, date}` —
immutable once finished and registered as one ledger data row. The path encodes
the three discriminators so they are legible without opening a shard (and they
are ALSO inside each shard's `Provenance`):

```
data/obs/
  <campaign>/                          # one campaign, immutable when done
    <producer>-w<worker>.ceobs[.zst]   # per-worker shard (D23 per-worker writer)
  games/<matchup>-<date>/              # EngineGame episodes (sf18-v-archon-2026-07-18/…)
  harvest/<program>/run.p<pid>.ceobs   # InSearchCapture runs (existing yield-probe shape)

data/corpora/
  <derived-name>/                      # DERIVED SoA / binpack — DISPOSABLE
  official-stockfish-master-binpacks/  # EXTERNAL immutable input (source of truth for bulk)
  lichess-chess-evaluations/           # EXTERNAL immutable input

ledger/{nets,data,experiments,annotations}.jsonl   # always-present, git-tracked index
```

Campaign naming carries oracle+budget+date, e.g. `sf18-100k-2026-07-18` (matching
today's `gold-v1`/`gold-v2` per-worker `sf18-w0..w5` / `reckless-w0..w5` shard
reality). Rules:

- A relabel NEVER mutates an existing campaign — it is a NEW campaign directory
  (new `canonical_key` labels accumulate across campaigns; §4.3 export merges
  them).
- One ledger `add-data` row per finished campaign (sha256 over the dir).
- Growth is monotone: campaigns are added, never edited or deleted from the
  ledger's history; under disk pressure the BYTES move to archive and the row is
  annotated `archived`, but the identity (hash) and recipe stay.

---

## 7. Migration / action plan (ordered)

1. **Route PRODUCTION relabel through `root-analysis`/obs.** ✅ DONE
   (2026-07-18): `root-analysis --reference-only` (Tier-F) captures L1+L3 (with
   the `id_snapshots` trajectory) + provenance WITHOUT the ~35× per-arm
   `searchmoves` loop, and **`obs-tool export-value`** materializes the
   scalar-cp corpus (`FEN | cp | wdl`, White-relative, Bound::Exact +
   Completion::Complete quarantine → bf-convert → bullet), verified end-to-end.
   Stop using the scalar shortcut (`tools/nnue/relabel_depth.py`) for PRODUCTION
   relabels — that path bypasses the store and throws away trajectory +
   provenance. NOTE the exporter split: `export-value` is the VALUE view
   (reference-only shards); `export-train` is the PER-MOVE POLICY view (needs
   Complete arms, skips reference-only shards) — do NOT use export-train for a
   value corpus. Reserve per-arm (Tier-R, no `--reference-only`) for the
   high-value gate (§5). **Keep the running scalar pilot AS-IS** — scalar cp is sufficient to
   answer the pilot's single question (do deep labels beat cheap labels), and it
   must not be disturbed mid-flight.
2. **Add the GAME episode kind + outcome propagation.** `EpisodeKind::EngineGame`
   (=6) + `Episode.provenance_ref_black` (tag 11); a game-capture harness (or the
   upgraded in-game recorder) that writes both-sides plies as one episode with
   `game_result` back-patched; `export-train` projects `game_result` onto each
   game-derived row as a `wdl` column (§4.1–4.2).
3. **Implement the additive-label MERGE in export.** Replace `CorpusBuilder`'s
   first-occurrence-wins dedup with group-by-`canonical_key` + precedence
   (deepest budget → newest date → strongest oracle), retaining all labels and
   threading `created_unix`; bump `TRAIN_WRITER`/`TRAIN_SCHEMA` (§4.3).
4. **Adopt the value-tiered defaults.** Tier-F on every deep search; Tier-R only
   on the §5 high-value gate; Tier-C hooks kept but sampled thin.
5. **Storage hygiene / scale-out.** zstd-at-rest for finished campaigns; one
   ledger `add-data` row per campaign; disk-partitioned dedup when the rich
   corpus crosses ~100 M keys; GCS archive (D36) for permanent shards with local
   as a working-set cache; add the `archived` ledger status.

Steps 1–4 are one-box-affordable and unblock the strategy; step 5 is the D36
scale-out. Nothing here changes the permanent wire format except two append-only
additions (a new enum value and a new optional tag), so every existing shard and
reader keeps working.

---

## 8. Open questions / far-future

- **Counterfactual traces for true search-augmentation (D19/D41).** The schema is
  ready (`Counterfactual` kind, `cf_axis`, `EvidenceKind::Intervention`, the
  `scout_*`/`research_*`/`reference_*` fields, `TtRegime`/`HistoryRegime`).
  Interventional records are the ONLY ones that identify the consequence of an
  alternative reduction/extension/policy-tier; observational and interventional
  evidence must never be silently merged (`design/16` §3.2.1). Keep the hooks,
  sample a thin slice, do NOT scale on one box.
- **ARCHON-as-oracle flywheel.** Once ARCHON surpasses an anchor in a regime, its
  own disagreement-with-labels guides relabeling and its ARCHON-loss games guide
  position generation — the AlphaZero-style loop. It slots into §4.3 as simply a
  new `teacher_identity`/`teacher_id` promoted up the oracle-strength ranking; no
  new mechanism.
- **Cross-corpus dedup identity (`design/16` §9 q2).** `canonical_key` is pinned
  as the unfolded main key; confirm the ep-square fold rule (ep-only-when-legal,
  as the Zobrist key does, D9) is consistent across every producer before
  cross-campaign joins, or positions silently double-count / wrongly merge.
- **D36 scale-out.** GCS mirror + OpenBench workers + distributed dedup; an
  OPTIONAL columnar analytics mirror (Parquet, tooling-side, disposable) if
  per-column scans at scale ever dominate (`design/16` §9 q1) — the permanent
  write path stays zero-dep TLV regardless.
- **Holdout freeze.** How to carve and freeze the small unbiased holdout by
  `canonical_key` reservation so it stays uncontaminated across regenerations and
  net iterations (`design/16` §9 q4).

---

## 9. The guarantee, in one line

Because the obs store is a superset (§3, proven in `design/16` §3.6.3), routing
production relabels and games through it (§7) captures — ONCE, at Tier-F cost —
the eval trajectory, per-move analysis, outcome, and full provenance that the
value, helm, policy, time-management, and pruning nets will each want later; the
two append-only extensions (§4) add games + multi-oracle labels without a wire
break; and the two-tier, content-addressed, archive-aware storage (§6) keeps the
never-pruned corpus inside a 14 GB-local / D36-archive budget. That is what
"capture the superset once, store it efficiently" means, made contractual.

# ARCHON cloud data factory — state assessment + scale-up update plan

Status: Assessment. Written 2026-07-18. Read-only audit of `tools/cloud/`
against today's repo and the new data strategy. Scope: what exists, what is
broken/stale, the delta vs the new plan, and an ordered update plan. **No cloud
runs, builds, or `gcloud`/`cargo` were executed for this doc; nothing in
`tools/cloud/` was modified.**

Aligns to: `docs/research/data-schema-and-storage.md` (the obs data contract +
storage spec — canonical substrate, `EpisodeKind::EngineGame=6` +
`Episode.provenance_ref_black` tag 11, multi-oracle additive labels,
value-tiered capture, GCS-as-archive, content-addressed + D38-ledgered) and
`docs/research/data-quality-strategy.md` (two dimensions: TARGETED
disagreement-first relabel; strong-engine games with ARCHON-loss focus; never
prune). Governance grounded in D36 (distributed compute) and design/18 (D38
ledger).

---

## 0. TL;DR verdict

**The cloud factory is architecturally sound but functionally stale, and it was
built for the OLD data model.** It is a well-governed GCP Batch pipeline
(dry-run default, Spot-only, cost-capped, keyless least-priv SA,
checksummed-resumable shards) — those rails are intact and worth keeping
verbatim. But it (a) **cannot build its worker image today** — all three engine
binaries drifted from their pinned SHAs (`image_manifest.json`), so
`build_worker_image.py` hard-fails; (b) **emits legacy scalar/JSONL formats, not
the obs `.ceobs` substrate** the new spec makes canonical; (c) **assumes
relabel-EVERYTHING over lossy LC0 feature arrays**, with no curated-subset path
for the TARGETED, disagreement-first relabel the new strategy is built around;
and (d) **has zero D38-ledger integration**. The two new obs entry points
(`root-analysis` emitting `.ceobs`, `obs-tool` / `ledger.py`) exist in the repo
but the cloud does not use them.

Read the pipeline as a **reference implementation of the governance rails to
port**, not a pipeline to patch in place. The value-generating cores
(dual-label, native-games) are superseded by routing through `root-analysis`.

**Top 5 concrete updates, in order:**
1. **Re-pin + repackage the worker image** — update the 3 drifted binary SHAs in
   `worker/image_manifest.json`; add `root-analysis`, `obs-tool`, and
   `tools/ledger/ledger.py` to the image; drop the LC0-feature scalar tools.
2. **Add a Tier-F "reference-only" mode to `root-analysis`** (LOCAL `src/bin`
   change) — a plain deep relabel emitting `L1+L3+Provenance` `.ceobs` without
   the ~35× per-arm loop. This is the obs entry point the cloud will call.
3. **New cloud worker `obs_relabel_shard.py`** that runs `root-analysis` over a
   curated FEN subset and publishes `.ceobs` shards (replaces
   `dual_label_shard.py` for production relabel).
4. **New curated-FEN manifest/partition tooling** — accept "here is a selected
   subset, relabel it," replacing the relabel-everything LC0-feature sharder.
5. **Reconcile → `ledger.py add-data` wiring** — one campaign row per finished
   run (sha256 over the shard dir, `{oracle,budget,date}` name, regen recipe);
   reconcile README + submit image defaults to ONE image identity.

Governance rails: **intact and good** (dry-run default, Spot-only, cost caps,
`--durable-data` gate, digest-pinned image, keyless least-priv SA, immutable
registry, checksummed-resumable + reconcile-replay). Keep them verbatim in the
new scripts.

---

## 1. Current state — file-by-file

`tools/cloud/` is 16 tracked files, ~3300 lines (all tracked in git; there is
also a `__pycache__/` that should be gitignored). Two independent data pipelines
plus provisioning/image/reconcile/inventory helpers.

| File | Purpose | Runs today? | Key staleness |
|---|---|---|---|
| `README.md` | Overview + cloud-eligibility gate | Doc only | Names image `archon-workers/research-worker@sha256:55bb5d…` (`README.md:52-56`) that matches NO submit default |
| `bootstrap_gcp.sh` | Keyless SA + immutable Artifact Registry | Yes | Fine; least-priv IAM intact |
| `build_worker_image.py` | Build SHA-pinned worker image from a manifest | **NO — hard-fails** | Enforces pinned hashes (`:52-56`); all 3 engine SHAs drifted (§Appendix A) |
| `worker/Dockerfile` | Python-3.12-slim image, `--require-hashes` deps | Yes (if inputs present) | Fine, but packages only Python scalar tools |
| `worker/image_manifest.json` | Pinned image inputs (bins, net, tools) | **Stale** | 3 drifted binary SHAs; packages LC0-scalar tools; **no obs bins, no ledger** (§Appendix A) |
| `worker/requirements.{in,lock}` | numpy / python-chess / gcs, hash-locked | Yes | Fine (no Rust/obs deps needed — obs is a compiled bin) |
| `worker/archon_worker.py` | Entry-point: verify hashes, then `run` | Yes | Fine; verifies against `provenance.json` |
| `shard_lc0_positions.py` | Shard LC0 feats.npy → immutable GCS position shards | Yes | **Wrong input shape** — LC0 32-int16 features, "piece_state_bootstrap_only" (`:143-152`); relabel-everything |
| `partition_position_manifest.py` | Split a position manifest into task manifests | Yes | Ordinal-modulo round-robin (`:34`); **no curated-subset path** |
| `dual_label_shard.py` | Worker: Reckless+SF18 paired scalar labels → NPZ | Yes | **Emits `archon.paired-value-labels.v1` scalar NPZ** (`:25`), the "scalar shortcut" the spec says to stop using; bootstrap-only board (`:83-101`) |
| `submit_dual_labels.py` | Plan/submit the paired-label Batch array | Yes* | Governance good; image default `data-worker@…9b0889` (`:16-19`) drifted |
| `reconcile_dual_labels.py` | Verify/replay paired-label outputs vs manifest | Yes | Fine for its format; no ledger write |
| `native_game_generator.py` | Worker: deterministic full-history games → gz JSONL | Yes | **Emits `archon.native-games.v1` JSONL** (`:28`), not obs episodes; blake2b state_key not `canonical_key` (`:180-183`) |
| `submit_native_generation.py` | Plan/submit the native-game Batch array | Yes* | Governance good; image default `data-worker@…187dac` (`:16-19`) drifted |
| `reconcile_native_games.py` | Verify/replay native-game outputs | Yes | Fine for its format; no ledger write |
| `native_dedup.py` | Resumable map/reduce/finalize global dedup | Yes | Dedups on blake2b state_key, not `canonical_key`; superseded by export dedup |
| `submit_native_dedup.py` | Plan/submit the dedup Batch arrays | Yes* | Governance good; image default drifted |
| `submit_http_to_gcs.py` | One checksum-gated HTTP→GCS transfer | Yes* | Governance good; image default drifted; generic, still useful |
| `archive_inventory.py` | Deterministic size+sha256 dir inventory | Yes | Useful precursor to a ledger `add-data`, but does not write one |

\* "Yes*" = the planner logic runs and dry-runs, but a real `--submit` needs a
buildable, digest-pinned image, which does not exist today (see
`build_worker_image.py` failure).

### 1.1 The image cannot be built today (hard blocker)

`build_worker_image.py:52-56` raises `SystemExit("pinned hash mismatch …")` when
any packaged artifact's sha256 ≠ the manifest pin. All three engine binaries
have drifted since the manifest was written (§Appendix A): `chessengine`,
`reckless`, and `stockfish-18` all differ; only `net-0.cenn` still matches. So
**no new worker image can be produced until the pins are refreshed**, and the
existing pushed images (`research-worker@…55bb5d`, `data-worker@…9b0889`,
`data-worker@…187dac`) are built from a stale ARCHON commit that predates the
current bench pin (701605), the D40 arch, and the PGO builds — not the engine we
now ship.

### 1.2 Image identity drift (three digests, two repo names)

- `README.md:52-56` — "current validated image" = `…/archon-workers/research-worker@sha256:55bb5d…`
- `submit_dual_labels.py:16-19` — default `…/archon-workers/data-worker@sha256:9b0889…`
- `submit_native_generation.py:16-19`, `submit_native_dedup.py:16-19`,
  `submit_http_to_gcs.py:14-17` — default `…/archon-workers/data-worker@sha256:187dac…`

Two repo names (`research-worker` vs `data-worker`) and three different digests.
None is reproducible today (binaries drifted). This must collapse to ONE
freshly-built, digest-pinned identity referenced everywhere.

### 1.3 The workers emit legacy formats, not obs `.ceobs`

- **`dual_label_shard.py`** (`SCHEMA = "archon.paired-value-labels.v1"`, `:25`)
  writes a compressed NPZ of flat per-teacher scalar arrays
  (`score_cp`/`mate_plies`/`best_move`/`depth`/`seldepth`/`nodes`/`bound`/`wdl`,
  `:215-225`). No eval trajectory (`id_snapshots`), no per-move layer, no obs
  `Provenance` record, no `canonical_key`. This is exactly the scalar-cp shortcut
  the storage spec §7 step 1 says to STOP using for PRODUCTION relabels because
  it "bypasses the store and throws away trajectory + per-move + provenance."
- **`native_game_generator.py`** (`SCHEMA = "archon.native-games.v1"`, `:28`)
  writes gz JSONL games with a per-move `SearchRecord`
  (`score_cp`/`mate_plies`/`best_move`/`chosen_move`/`chosen_rank`/`depth`/
  `seldepth`/`nodes`, `:34-46`, `:259-264`) plus full move history — richer than
  dual-label, but still NOT obs `Episode`s. It keys positions by a blake2b
  128-bit `state_key` over `fen|rep` (`:180-183`), NOT the platform
  `canonical_key` (unfolded Zobrist); it has no obs `Provenance`, no
  `EngineGame` kind, no `IdSnapshot` trajectory, no `PerMoveObservation`, and it
  attaches only a per-GAME `result_white` — not a per-position WDL derivation.

Neither format is a projection of the `Observation` superset; both would need a
converter to enter the obs store, and both re-run expensive search WITHOUT
capturing the Tier-F layers that ride along for free once you route through
`root-analysis`.

### 1.4 Input contract is LC0-feature-array + relabel-everything

`shard_lc0_positions.py` ingests LC0 `white/black/feats.npy` (2-D int16, 32
features/row, `:126-139`) and stamps `state_completeness:
"piece_state_bootstrap_only"` with `castling/en_passant/rule50` forced to
none/zero (`:143-152`). `dual_label_shard.py:83-101` reconstructs a board from
those features with `turn=WHITE, castling=EMPTY, ep=None, rule50=0`. This is a
**lossy, mover-normalized, piece-placement-only** representation: it cannot
represent castling/ep/rule50, so it is incompatible with `canonical_key` dedup
and with FEN-faithful obs episodes.

Sharding is **relabel-everything**: `shard_lc0_positions.py` cuts the whole
corpus into 250k-row shards; `partition_position_manifest.py:34` distributes
shards round-robin (`shards[partition::N]`). There is no "curated subset"
selector. The labels here are the historical `lc0-selected-101m` campaign
(`submit_dual_labels.py:146`) — the uniform relabel the new strategy explicitly
supersedes.

### 1.5 Schema extensions the new games plan needs are NOT implemented

`src/obs/schema.rs` `EpisodeKind` currently stops at `InSearchCapture = 5`
(`:61-71`); there is NO `EngineGame = 6` and NO `Episode.provenance_ref_black`
(tag 11). The storage spec marks both **[proposed]** (§4.1, §4.5). So even if the
cloud routed games through obs today, the two-engine-game + black-side-producer
representation does not exist yet — this is a LOCAL `src` prerequisite for the
Dimension-2 games work, not a cloud-only change.

### 1.6 `root-analysis` is Tier-R only (no Tier-F)

`src/bin/root_analysis.rs` always runs one unrestricted reference search PLUS one
`searchmoves`/`child` arm per legal move for every tier (`:554-580`) — i.e. the
~35× Tier-R shape for EVERY position. The storage spec (§5, §7 step 1) wants
Tier-F (reference-only: `L1+L3+Provenance`, one search/position) as the default
for bulk relabel, reserving Tier-R for the high-value gate. A Tier-F mode does
not exist yet; it is the LOCAL change that makes obs-routed bulk relabel
affordable. Note the RIGHT entry shape already exists: `root-analysis` takes
`--fens <file>` and `--engine <bin>` (`root_analysis.rs:10`), so a curated FEN
list + SF18/Reckless/ARCHON is directly drivable.

### 1.7 No ledger integration

No cloud script calls `tools/ledger/ledger.py` (verified: `ledger.py` exposes
`add-net`/`add-data`/`add-exp`/`annotate`/`verify`/`show`/`reproduce`). The
storage spec §6.4/§7 step 5 requires ONE `add-data` row per finished campaign
(sha256 over the shard dir, regen recipe, `--source`). `archive_inventory.py`
computes a deterministic dir sha256 inventory — a precursor to that hash — but
writes no ledger row; `reconcile_*` produce reconciliation JSON but no ledger
entry.

---

## 2. Gap vs the new strategy

### 2.a TARGETED relabel — selection is LOCAL; cloud relabels a curated subset

**Gap: the tooling assumes relabel-everything and has no curated-subset input.**
The new plan (data-quality-strategy Dimension 1) is: LOCALLY score positions by
`|SF18_moderate − existing_label|`, then relabel ONLY the top-disagreement subset
deep. The cloud's job is "here is a pre-selected FEN subset, deep-relabel it."
Today `shard_lc0_positions.py` + `partition_position_manifest.py` shard the WHOLE
corpus of LC0 feature arrays and round-robin them; there is no way to hand it a
curated list. The fix is a FEN-list manifest (not a feature-array manifest) that
partitions a curated subset into per-task FEN files feeding `root-analysis`.
(`root-analysis --fens` already exists — the entry point is there.)

### 2.b Obs-schema alignment — workers emit scalar/JSONL, not Tier-F `.ceobs`

**Gap: no worker emits obs shards conforming to `data-schema-and-storage.md`.**
Production relabel must emit `Observation`s so Tier-F (trajectory + `pv` +
`completion` + `Provenance`) rides along for free (spec §5, §7 step 1); today it
emits flat scalar NPZ (§1.3). For games, the workers must (once §1.5 lands) route
through obs as `EngineGame` episodes with both-side `Provenance` and
`game_result`, and export must derive the per-position WDL + the ARCHON-loss
signal (`|ARCHON_in-game − SF18_relabel|` joined by `canonical_key`,
data-quality-strategy step 5). Native-games JSONL captures neither obs provenance
nor `canonical_key`, so its ARCHON-loss extraction cannot be joined to relabels.

### 2.c Multi-oracle additive labels + content-addressed + D38 ledger

**Gap: no `canonical_key`, no additive campaign model, no ledger rows.** The spec
(§4.3, §6.5) models a position as accumulating `{oracle,budget,date}` labels
across immutable dated campaigns, joined by `canonical_key`, merged at export.
The cloud has no `canonical_key` (dual-label is bootstrap-only; native uses
blake2b), no per-campaign `add-data`, and no `{oracle,budget,date}` campaign
naming. The additive MERGE itself lives in LOCAL `obs-tool export-train` (spec §7
step 3) — not a cloud change — but the cloud must (i) route through obs so
`canonical_key`/`Provenance` exist, and (ii) name + ledger each run as one
campaign.

### 2.d Fresh worker image with current SHA-pinned SF18/ARCHON/Reckless + tooling

**Gap: the image is unbuildable and packages the wrong tools.** Re-pin the 3
drifted binary SHAs; add the obs Rust binaries (`root-analysis`, `obs-tool`) and
`tools/ledger/ledger.py`; drop `relabel_lc0_uci.py` /
`shard_lc0_positions.py`-scalar path from the packaged tools; collapse to one
digest-pinned identity referenced by README + all submits.

---

## 3. What still works / is worth keeping as-is (governance rails)

**All the D36 governance rails are present, correct, and worth preserving
verbatim in the new scripts.** Confirmed by reading each submit + worker:

| Rail | Where | Verdict |
|---|---|---|
| **Dry-run default** (must add `--submit`) | `submit_dual_labels.py:188-190`, `submit_native_generation.py:187-189`, `submit_native_dedup.py:207-209`, `submit_http_to_gcs.py:103-105` | Intact — good |
| **Spot-only** (`provisioningModel: "SPOT"`) | all four submit scripts (job `allocationPolicy`) | Intact — good |
| **Cost cap** (worst-case vs `--max-cost-usd`, `SystemExit` if over) | `submit_dual_labels.py:104-107`, `submit_native_generation.py:93-96`, `submit_native_dedup.py:151-154`, `submit_http_to_gcs.py:44-45` | Intact — good |
| **`--durable-data` gate** (cloud-eligibility) | `submit_dual_labels.py:82-83`, `submit_native_generation.py:69-70`, `submit_native_dedup.py:74-75`, `submit_http_to_gcs.py:38-39` | Intact — good |
| **Digest-pinned image required** | `submit_dual_labels.py:76-77`, `submit_native_generation.py:73-74`, `submit_native_dedup.py:78-79` | Intact — good |
| **Keyless least-priv SA** (report/log/read-image/read+create-object only; no delete/overwrite/admin; no SA key) | `bootstrap_gcp.sh:27-45` | Intact — good |
| **Immutable Artifact Registry** (`--immutable-tags`) | `bootstrap_gcp.sh:23` | Intact — good |
| **Checksummed, resumable, no-overwrite shards** (sha256 metadata, `if_generation_match=0`, validate-existing resume) | `dual_label_shard.py:416-459`, `native_game_generator.py:427-472`+`531-576`, `native_dedup.py:99-124` | Intact — good |
| **Reconcile + replay audit** (count/hash/replay a spread before accepting a wave) | `reconcile_dual_labels.py`, `reconcile_native_games.py` | Intact — good |
| **Bounded retries + max task seconds** | all four submit scripts (`maxRetryCount`, `maxRunDuration`) | Intact — good |
| **Determinism** (fixed seeds, `mtime=0` gzip, sorted-keys JSON) | `native_game_generator.py:417`+`:514`, throughout | Intact — good |
| **Image self-verify at container start** (hash-gate then run) | `worker/archon_worker.py:26-59` | Intact — good |

Also worth keeping structurally: the **planner→plan-artifact→submit** pattern
(persist a reviewable plan JSON before `--submit`), the **map/reduce/finalize
resumable stage** design in `native_dedup.py` (reusable if dedup ever needs the
cloud), and `submit_http_to_gcs.py` (a generic, well-gated durable ingest — keep,
just re-point the image).

---

## 4. The update plan (ordered)

### MUST-DO before any cloud run

**M1. Re-pin + repackage the worker image.** *(effort: S; ~1h + build)*
Edit `worker/image_manifest.json`: refresh the 3 drifted binary SHAs (build the
fast-config binaries first, then paste current sha256; §Appendix A); ADD entries
for `target/release/root-analysis`, `target/release/obs-tool`, and
`tools/ledger/ledger.py`; REMOVE the LC0-scalar tools no longer used in
production (`relabel_lc0_uci.py`, and the `shard_lc0_positions.py`/`dual_label`
path once M3 replaces it). `build_worker_image.py` needs no change (it already
copies manifest artifacts + validates hashes). Rebuild locally (no `--push`),
run `archon-worker verify`. This unblocks everything.

**M2. Add a Tier-F "reference-only" mode to `root-analysis`.** *(effort: M;
LOCAL `src/bin/root_analysis.rs`)* A `--reference-only` (or `--tier-f`) flag that
emits one Observation per tier with `L1 PositionState` + `L3 SearchObservations`
(incl. `id_snapshots` + `pv` + `completion`) + `Provenance`, SKIPPING the per-arm
`searchmoves`/`child` loop (`:554-580`). This is the affordable bulk-relabel
shape (spec §5 Tier-F, §7 step 1). Reserve the existing per-arm path for the
Tier-R high-value gate. Unit-test that a Tier-F shard round-trips through
`obs-tool export-train`.

**M3. New worker `obs_relabel_shard.py`.** *(effort: M)* Replaces
`dual_label_shard.py` for production relabel. Given a per-task FEN file + an
engine selector (SF18 / Reckless / ARCHON) + a node budget + a tier flag,
invokes the packaged `root-analysis` to produce a `.ceobs` shard, then publishes
it with the SAME governance the current worker has (sha256 metadata,
`if_generation_match=0`, validate-existing resume, atomic local write). Emit the
campaign discriminators (`{oracle,budget,date}`) into object metadata. Keep
`dual_label_shard.py` only as a historical control.

**M4. New curated-FEN manifest + partition tooling.** *(effort: S–M)* Replace
the LC0-feature sharder with a FEN-list ingester: take a curated
`FEN|cp|result` subset (the LOCAL disagreement-selected set), chunk it into
per-task FEN files, and emit a task manifest (`{fen_file_uri, rows, sha256,
ordinal}`) analogous to today's shard manifest. `partition_position_manifest.py`
can be adapted (it already does disjoint round-robin partitioning + hash
integrity) — the change is FEN-file records instead of feature-shard records.
This is the "here's a curated subset, relabel it" path (gap 2.a).

**M5. New submit script `submit_obs_relabel.py`.** *(effort: S — copy an
existing submit)* Fork `submit_dual_labels.py` verbatim for the governance rails
(dry-run default, `--durable-data`, digest-pinned image, Spot-only, cost cap,
bounded retries, plan artifact); swap the worker command to
`obs_relabel_shard.py`; swap the manifest reader to the FEN manifest (M4).

**M6. Reconcile → `ledger.py add-data` wiring.** *(effort: S–M)* After a wave
reconciles clean, run `ledger.py add-data` for the campaign: sha256 over the
shard dir (reuse `archive_inventory.py`'s deterministic tree hash / or ledger's
`sha256_tree`), `--source` = the GCS prefix, `--gen-recipe` = the exact
`root-analysis`/submit invocation, campaign id = `{oracle}-{budget}-{date}` (spec
§6.5). Add a Tier-F reconciler for `.ceobs` outputs (or extend
`obs-tool` to verify a campaign dir). Run `ledger.py verify` as the accept gate.

**M7. Collapse image identity + refresh README.** *(effort: S)* One
freshly-built digest referenced by README §"Worker image" AND every submit
`DEFAULT_IMAGE`. Pick one repo name (`archon-workers/…`); delete the stale
`research-worker`/`data-worker`/three-digest drift (§1.2). Add `__pycache__/` to
`.gitignore`.

### NICE-TO-HAVE / later (Dimension 2 + scale-out)

**N1. `EngineGame` schema extension** (LOCAL `src/obs/schema.rs`): add
`EpisodeKind::EngineGame = 6` + `Episode.provenance_ref_black` tag 11 (both
append-only, spec §4.1). Prerequisite for routing strong-engine games through
obs.

**N2. Game-capture worker → obs.** *(effort: L)* Replace/augment
`native_game_generator.py` so two-engine games (SF18 v SF18/Reckless/ARCHON) are
written as `EngineGame` obs episodes (both-side `Provenance`, `game_result`
back-patched), reusing the in-game recorder (`src/obs/recorder.rs`) per spec
§4.1. Enables the ARCHON-loss learn-from-losses signal (Dimension 2 step 5),
joinable to relabels by `canonical_key`.

**N3. Multi-oracle export MERGE** (LOCAL `obs-tool export-train`, spec §7 step 3;
NOT a cloud change) — group-by-`canonical_key` + precedence (deepest budget →
newest date → strongest oracle). The cloud only needs to name campaigns by
`{oracle,budget,date}` so the merge has clean inputs.

**N4. GCS archive + `archived` ledger status + zstd-at-rest** (spec §6.3, §6.5):
finished `.ceobs` campaigns zstd-compressed offline, mirrored to the GCS archive,
ledger-annotated `--status archived` so locally-pruned-but-hashed bytes are not
counted as drift.

**N5. Retire/park `native_dedup.py`** — dedup moves to export-time on
`canonical_key` (spec §6.2); the blake2b-state_key cloud dedup is superseded.
Keep the map/reduce/finalize scaffold as a template for the future
disk-partitioned (grace-hash) `canonical_key` dedup at >100M keys (spec §6.2).

### Minimal end-to-end validation plan (before scale)

1. **Local dry-run** each submit (already the default): verify plan JSON, cost
   cap math, `--durable-data`/digest-pinned refusals fire.
2. **Local image build** (`build_worker_image.py`, NO `--push`) → run
   `archon-worker verify` (hash gate) → 2-FEN local `root-analysis --reference-only`
   smoke that writes a `.ceobs`, then `obs-tool export-train` reads it back to a
   NumPy SoA (proves the obs round-trip end-to-end, offline).
3. **1–2 task cloud pilot**: a tiny curated FEN subset, Spot, `--durable-data`,
   digest-pinned image, tight `--max-cost-usd` (e.g. $1), `--max-retries 1`.
4. **Verify**: reconcile the wave; `--download-samples` replay-validate; run
   `obs-tool disagree` between the pilot `.ceobs` and a local reference shard to
   confirm the labels are sane; `ledger.py add-data` the pilot campaign +
   `ledger.py verify`.
5. **Scale** the FEN subset only after 1–4 are green, keeping the same governance
   flags and cost caps.

---

## Appendix A — exact drift (as of this audit)

Pinned SHAs live in `worker/image_manifest.json`; measured are current on-disk.

| Artifact | `image_manifest.json` pin | Current on disk | Match? |
|---|---|---|---|
| `target/release/chessengine` (`:7`) | `f55d0591…d4df698` | `c4686d29…441022f0` | **NO** |
| `reckless_reference/Reckless/target/release/reckless` (`:13`) | `8c4e36c0…1c2c66f9` | `2e9fc5dd…666f5cd30` | **NO** |
| `engines/stockfish-18-src/src/stockfish` (`:18`) | `1742703…905e86d3` | `3a429241…749b299d0` | **NO** |
| `data/nnue/nets/net-0.cenn` (`:21`) | `60138d33…178709111` | `60138d33…178709111` | yes |

Consequence: `build_worker_image.py` (`:52-56`) raises on the first mismatch;
the image is unbuildable until all three are re-pinned. The already-pushed images
predate the current engine (bench 701605, D40 arch, PGO) regardless.

## Appendix B — image identity drift

| Reference | Image |
|---|---|
| `README.md:52-56` | `…/archon-workers/research-worker@sha256:55bb5dc2…089a24` |
| `submit_dual_labels.py:16-19` | `…/archon-workers/data-worker@sha256:9b088970…d3573c3` |
| `submit_native_generation.py:16-19` | `…/archon-workers/data-worker@sha256:187dacdf…d43aa4b1` |
| `submit_native_dedup.py:16-19` | `…/archon-workers/data-worker@sha256:187dacdf…d43aa4b1` |
| `submit_http_to_gcs.py:14-17` | `…/archon-workers/data-worker@sha256:187dacdf…d43aa4b1` |

Two repo names, three digests, none reproducible today → collapse to one
freshly-built digest-pinned identity (M1 + M7).

## Appendix C — GCP resource dependencies (unchanged, still valid)

- Project `archon-502500`, region `us-east1` (defaults across all scripts).
- Bucket `gs://archon-main-bucket` (`bootstrap_gcp.sh:7`); worker SA can
  view+create objects, not overwrite/delete.
- Artifact Registry repo `archon-workers` (immutable tags).
- Worker SA `archon-batch-worker@archon-502500.iam.gserviceaccount.com`
  (keyless; roles: `batch.agentReporter`, `logging.logWriter`,
  `artifactregistry.reader`, `storage.objectViewer`, `storage.objectCreator`).
- Machine profile locked to `c2d-highcpu-16` Spot in the submit validators;
  measured throughput constants are baked into the cost estimators. These stay
  valid for the obs-relabel worker; re-measure `rows/sec` for `root-analysis`
  (deeper searches → lower throughput than the scalar labeler) before setting the
  cost estimator constant in `submit_obs_relabel.py`.

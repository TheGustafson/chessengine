# Archon cloud data factory

GCP is reserved for durable, reusable data generation. Training remains on the
local GPU.

## Cloud eligibility gate

A job may use GCP only when its outputs are intended to survive and serve
multiple experiments or model generations, such as corpus-scale external
teacher labels, counterfactual search records, or durable on-policy telemetry.

Keep these local:

- smoke tests and schema validation;
- one-off probes and tuning data;
- disposable bootstrap labels;
- all model training, until this decision is explicitly changed.

Every future submit command must default to dry-run, print a worst-case cost
estimate, use Spot provisioning only, cap task count and runtime, and require an
explicit durable-data declaration. Interruption-safe jobs publish immutable,
checksummed shards to GCS and resume by validating existing shards.

## Provisioning

`bootstrap_gcp.sh` creates a keyless Batch worker service account and immutable
Artifact Registry repository. The worker can report Batch status, write logs,
pull the repository image, and read/create bucket objects. It cannot overwrite
or delete corpus objects and has no training or broad project-admin role.

```bash
bash tools/cloud/bootstrap_gcp.sh
```

No service-account key is created. Batch uses the attached runtime identity.

## Worker image

The image includes SHA-pinned ARCHON, Reckless, Stockfish 18, net-0, the obs
binaries (`root-analysis`, `obs-tool`), the D38 ledger CLI, the obs-relabel
worker, the root-move oracle, and a hash-locked Python dependency graph. The
build uses a temporary allow-list context, so datasets and unrelated worktree
files cannot enter the image accidentally.

One image identity is used everywhere — repository `archon-workers`, name
`archon-worker`. There is no more `research-worker`/`data-worker` split or
multi-digest drift; the README block below and every `submit_*.py`
`DEFAULT_IMAGE` reference the same digest.

```bash
python3 tools/cloud/build_worker_image.py \
  --tag us-east1-docker.pkg.dev/archon-502500/archon-workers/archon-worker:NEW_IMMUTABLE_TAG
```

Before building, re-pin `worker/image_manifest.json`: every artifact marked
`_sha_refresh_needed` (the 3 drifted engines + the new `root-analysis`/`obs-tool`
bins) needs its `sha256` set to the freshly-built production binary. Add `--push`
only after the local verification and oracle smoke tests pass. Immutable tags
deliberately prevent replacing an experimental environment.

Current validated image (digest stamped by the M1 rebuild):

```text
us-east1-docker.pkg.dev/archon-502500/archon-workers/archon-worker@sha256:REPLACE_AT_BUILD
```

## Durable obs-relabel campaigns

The production, TARGETED relabel path emits obs `.ceobs` shards (not scalar
NPZ), so Tier-F layers (`PositionState` + `SearchObservations` trajectory + `pv`
+ `completion` + `Provenance`) ride along for free
(data-schema-and-storage.md §5, §7 step 1).

1. `partition_fen_manifest.py` chunks the LOCALLY disagreement-selected FEN
   subset (`FEN|cp|result` or plain FEN) into immutable per-task FEN shards and
   emits a `{fen_file_uri, rows, sha256, ordinal}` task manifest.
2. `submit_obs_relabel.py` plans/submits the Batch array (dry-run default,
   `--durable-data`, digest-pinned image, Spot-only, worst-case cost cap,
   bounded retries, persisted plan artifact). Each task drives the packaged
   `root-analysis` over one FEN shard at a pinned node budget, single-threaded,
   packed one-per-vCPU on the locked `c2d-highcpu-16`.
3. `obs_relabel_shard.py` (the worker) runs `root-analysis --reference-only`
   (Tier-F) and publishes the `.ceobs` with the same sha256-metadata,
   `if_generation_match=0`, validate-existing resume, and atomic-write
   governance as `dual_label_shard.py`. Campaign discriminators
   `{oracle, budget, date}` are stamped into the object metadata; the campaign
   id is `{oracle}-{budget}-{date}` (e.g. `sf18-100k-2026-07-18`).
4. `reconcile_obs_relabel.py` inventories the wave (count/hash/metadata,
   optional `obs-tool verify` replay on a downloaded spread) and, with
   `--ledger-add`, materializes + verifies the campaign locally then wires
   `tools/ledger/ledger.py add-data` + `ledger.py verify` as the accept gate.
5. **Build the training corpus from the reconciled shards (local):** a Tier-F
   (`--reference-only`) shard carries no per-move arms, so the value corpus
   comes out via `obs-tool export-value <out.txt> <shard>...` (`FEN | cp | wdl`,
   White-relative, Bound::Exact + Completion::Complete quarantine) → `bf-convert`
   → bullet `.bin`. Do NOT use `obs-tool export-train` on reference-only shards
   — that is the per-move POLICY exporter (it needs Complete arms and skips a
   reference-only shard). Rich (Tier-R) shards submitted without
   `--reference-only` feed `export-train`.

`dual_label_shard.py` / `shard_lc0_positions.py` / `partition_position_manifest.py`
stay as historical controls (the LC0-feature, relabel-everything, scalar path).

## Durable paired-label campaigns

`submit_dual_labels.py` reads and validates an immutable position manifest,
computes measured and retry-inclusive worst-case costs, and emits the exact
Batch request. It defaults to a dry run, requires `--durable-data`, requires a
digest-pinned image, and refuses to submit when the configured cost cap is
exceeded. Add `--submit` only after reviewing the persisted plan artifact.

## Native full-history generation

`native_game_generator.py` creates deterministic, task-local-deduplicated game
records for Reckless self-play, Stockfish 18 self-play, balanced cross-play,
and exploratory MultiPV play. Every sampled position includes exact FEN,
rule-50 state, repetition count, legal-move/check context, and a 128-bit state
key; the record also retains the complete standard-start move history and the
search provenance for every generated move. Outputs are deterministic gzip,
atomically published, hash-verified, and replay-validated on first creation and
retry. `reconcile_native_games.py` inventories an exact task interval, enforces
corpus/config/stratum identity, verifies counts and SHA-256 metadata, and can
download and fully replay a spread of task outputs before a wave is accepted.
`native_dedup.py` implements the second deduplication pass as resumable map,
reduce, and finalize stages. Mappers emit fixed-width state-key/source locators
into deterministic hash partitions; reducers retain the lexicographically first
source locator for each 128-bit state key; finalization requires every partition
and reconciles input, unique, and duplicate counts.
`submit_native_dedup.py` packages mapper and reducer arrays into digest-pinned,
Spot-only, retry-bounded Batch jobs with explicit worst-case cost caps.

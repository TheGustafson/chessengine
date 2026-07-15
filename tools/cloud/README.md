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

The image includes SHA-pinned ARCHON, Reckless, Stockfish 18, net-0, the value
relabeler, the root-move oracle, and a hash-locked Python dependency graph. The
build uses a temporary allow-list context, so datasets and unrelated worktree
files cannot enter the image accidentally.

```bash
python3 tools/cloud/build_worker_image.py \
  --tag us-east1-docker.pkg.dev/archon-502500/archon-workers/research-worker:NEW_IMMUTABLE_TAG
```

Add `--push` only after the local verification and oracle smoke tests pass.
Immutable tags deliberately prevent replacing an experimental environment.

Current validated image:

```text
us-east1-docker.pkg.dev/archon-502500/archon-workers/research-worker@sha256:55bb5dc258c98f3a928143d525ccef024fa600a115dd0eb8873bc45125089a24
```

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

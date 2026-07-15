# Archon durable data campaign

Status: existing-position campaign active. Benchmark, source audit, and cost
gates passed; native-play generation remains in local preflight.

## 1. Objective

Build two independently identifiable, paired-teacher corpora:

1. **LC0-state archive and dual labels:** preserve all local LC0 source data in
   GCS, extract the approximately 101M selected positions already used by the
   corrected 100M corpus and prospective 1M tail holdout, and label every row
   with pinned Reckless d445b8bf and Stockfish 18 at identical fixed-node
   budgets.
2. **Native-play 100M:** generate at least 100M unique full-state positions from
   Reckless and Stockfish play, then attach paired Reckless and Stockfish 18
   labels.

These are durable data assets. Model training remains local.

## 2. Critical distinction between the corpora

The existing extracted LC0 rows contain mover-normalized piece placement only.
They have no recoverable castling rights, en-passant square, rule-50 counter, or
repetition history. They must remain marked `piece_state_bootstrap_only`; the
labeler reconstructs white-to-move boards with no castling/ep and zero clocks.

The native-play corpus must preserve exact full FEN state; game start position
and complete move list once per game; sampled ply, game/opening ID, generator
and color; repetition/draw-claim state derived from history; and game result
plus termination reason. No loader may silently merge the two state classes.

## 3. Existing-data inventory and archive

Current local inventory on 2026-07-14:

- 86 raw LC0 tar files: approximately 66GB;
- corrected main corpus: exactly 100M rows, approximately 22GB;
- prospective tail holdout: 1M disjoint selected rows, approximately 221MB;
- small crossed/smoke datasets: retain as named lineage artifacts, but do not
  count their subset rows twice;
- local disk: 97% used, only approximately 8.3GB free.

Archive prefixes:

```text
gs://archon-main-bucket/corpora/lc0-test91/raw-tars/v1/
gs://archon-main-bucket/corpora/lc0-selected/v1/source-arrays/
gs://archon-main-bucket/corpora/lc0-selected/v1/position-shards/
gs://archon-main-bucket/corpora/lc0-selected/v1/dual-labels-r5k-sf18-5k/
```

The local-to-GCS transfer is not a Batch job: cloud VMs cannot read the local
disk. Upload original files directly with resumable transfers. Generate a
SHA-256 inventory first, rely on GCS CRC32C during transfer, compare object
sizes/counts, and independently download a stratified sample before deleting
anything local.

Preserve the full 87GB first. Then stream selected `feats.npy` arrays into
immutable 250K-row recovery shards without materializing another full local
copy. The manifest groups four adjacent recovery shards into a one-million-row
logical partition when larger sequential reads are useful.

Local deletion gate:

1. every expected object exists at generation 1;
2. local SHA-256 manifest and GCS inventory exist in GCS and the state log;
3. random first/middle/last objects download and hash correctly;
4. position shards reconstruct expected source indices and row hashes;
5. at least one cloud dual-label pilot consumes uploaded shards;
6. only then delete local raw tars and redundant policy arrays, retaining a
   small local audit cache.

## 4. Versioned row contract

Every row has a stable `position_id` from corpus ID, source shard, and source
row. A separate 128-bit chess-state key supports deduplication but is not the
primary identity.

Paired labels for each teacher contain signed side-to-move centipawns; mate
distance/status; UCI bound; lossless best move; depth, selective depth, and
actual nodes; optional WDL; and exact engine/options/request provenance. Paired
output publishes only after both teachers validate. Failures are explicit
rejects, never zero-filled labels.

## 5. Labeling semantics

The first durable value pass provisionally uses **5,000 nodes per teacher per
position**, matching the existing Reckless relabel. Engines are single-threaded.
Worker processes keep UCI engines alive but issue `ucinewgame` and clear search
state so TT/history leakage cannot make labels depend on shard order.

Both teachers receive the identical reconstructed or full-state board. Validate
score perspective, mate, best-move legality, requested/actual nodes, and source
identity before atomic publication.

## 6. Fail-fast cloud benchmark

Before the 101M job, build a GCS-aware paired labeler and run the same
representative 250K-position shard on `c2d-highcpu-16`, `c3d-highcpu-16`, and
`c4d-highcpu-16`; optionally include `e2-standard-16` as a control. Compare 8
and 16 single-thread worker processes to measure SMT contention.

Report positions/s, nodes/s, utilization, peak memory, phase timings,
preemption behavior, and dollars per million paired labels separately for each
teacher. Reject any machine/compiler combination with identity, score-probe, or
row-hash drift. Select throughput per dollar subject to quota, not raw speed.

Bulk launch gate: measured upper-bound compute, boot disk, and operations must
remain below **$50** for the existing approximately 101M paired-label job. A
higher estimate requires explicit review.

### Capacity request (2026-07-15)

Cloud Quotas preferences were submitted for `us-east1` so the current relabel
and subsequent native-position campaign are not unnecessarily serialized:

- `archon-c2d-cpus-us-east1-20k`: C2D vCPUs, 100 -> 20,000 requested;
- `archon-cpus-us-east1-20k`: regional vCPUs, 200 -> 20,000 requested;
- `archon-instances-us-east1-1250`: regional VM instances, 24 -> 1,250 requested;
- `archon-preemptible-cpus-us-east1-20k`: legacy preemptible-vCPU quota,
  0 -> 20,000 requested.

All four requests were accepted by the Cloud Quotas API and were reconciling
when submitted. Quota is only a capacity ceiling: every Batch submission must
still enforce Spot provisioning, useful task-count parallelism, and the
campaign's explicit dollar gate. The 404-shard LC0 relabel can use at most
6,464 vCPUs with the current 16-vCPU task shape; the planned approximately
1,100-task native generation campaign can use about 17,600 vCPUs.

The initial requests produced partial regional grants (300 regional CPUs and
300 C2D CPUs), but an initially overlooked `CPUS_ALL_REGIONS` quota remained
at 32 and constrained the campaign to two VMs. Global requests for 20,000 and
300 were denied. Incremental requests were accepted at 64 and then 96 vCPUs;
128 was denied. Stop escalation at the effective 96-vCPU/six-worker ceiling.

Because a single zone showed Spot exhaustion, the 404-shard manifest is split
by ordinal modulo into three disjoint task manifests pinned to
`us-east1-b/c/d` (135/135/134 tasks). Their retry-inclusive cost gates are
$16.47/$16.47/$16.35, exactly preserving the $49.29 whole-campaign ceiling.
The first published shard passed independent full-download reconciliation;
continue reconciling coverage while Batch runs.

## 7. Existing-position Batch job

Use one Batch job with one task group and approximately **404 independent
250K-row tasks**. This is one managed task array, not one monolithic worker.
Set task-group parallelism to the largest value allowed by the selected
machine-family quota and the 200-vCPU regional ceiling (for example, up to 12
concurrent 16-vCPU workers when family quota permits). Batch may reuse workers
across successive tasks, amortizing VM startup while retaining small recovery
units. Each task maps `BATCH_TASK_INDEX` to one input,
validates it, runs both teachers, checkpoints local subshards, validates paired
outputs, uploads with `if_generation_match=0`, and validates/skips completed
objects on retry.

Use Spot only, a digest-pinned image, no external state outside GCS, and one
task per VM. Initial cap is 30 minutes per task, revised from benchmark data.
Enable Spot retries only after the identical command passes local and one-task
cloud preflight.

Completion requires 100% source coverage, unique position IDs, paired labels,
teacher/config identity, and structural checks. Independently relabel a fixed
random 10K audit slice before declaring the corpus ready.

## 8. Native-play 100M design

Generate about 110M candidates in many deterministic seed-range tasks, globally
deduplicate by full state, and top up
to at least 100M accepted unique positions. Proposed named strata:

- 30% Reckless self-play;
- 30% Stockfish 18 self-play;
- 20% deterministic Reckless-versus-Stockfish, colors balanced;
- 20% reproducible exploratory/disagreement games using early MultiPV sampling.

Start with standard chess. Use versioned opening seeds and color reversal; skip
only trivial opening plies, not checks/tactics; cap sampled positions per game
(provisionally 40); record all moves and exact state; retain generator-budget
strata; and split by game/opening before sampling rows.

Select generator search budget and exploration temperature through a small
local quality/diversity study, then cloud throughput pilot. A provisional 10K
nodes/move is not locked without evidence.

Deduplicate in two passes: task-local deduplication, then deterministic hash
partitioning and global per-partition deduplication. Reserve at least 2M rows as
a prospective game-disjoint holdout. Apply the same paired 5K label contract
after deduplication.

Each native generation task targets a bounded number of games and at most
100K accepted candidates. A 10M pilot therefore has at least 100 independent
tasks; the final campaign has roughly 1,100 generation tasks plus hash-partition
deduplication and label tasks. Task indices fully determine seeds, so missing or
preempted tasks can be rerun without overlap.

The complete native-play campaign has an initial hard planning ceiling of
**$150**. Run one 10M-position pilot wave and reconcile quality, cost, and
duplicate rate; after that gate passes, submit the remaining sharded work at
maximum safe Spot parallelism rather than serial 10M waves.

## 9. Storage and lifecycle

Keep active manifests, positions, and labels in regional Standard Storage. Move
raw tars colder only after extractors stabilize and retrieval/minimum-duration
costs are understood. Keep object counts in the hundreds or low thousands.

GCS is the source of truth. Local training pulls only required generations,
splits, and columns into a bounded, hash-verified cache.

## 10. Ordered next actions

1. Freeze corpus IDs, manifests, row schema, and exact inventory.
2. Build the resumable archive/uploader and streaming position sharder; test a
   tiny named prefix.
3. Upload 86 raw tars and extracted arrays while local labeling continues only
   until the first cloud benchmark supersedes it.
4. Implement and locally test the GCS-aware paired-teacher labeler.
5. Build/push a new immutable data-worker image and run identical cloud probes.
6. Benchmark C2D/C3D/C4D worker counts and lock price/performance.
7. Produce a measured cost report and one-job plan for approximately 101M rows.
8. Launch only after coverage, identity, retry, and cost gates are green.
9. Verify the cloud corpus, then reclaim local storage in stages.
10. Implement the full-state game generator and run local quality/diversity
    probes before the first 10M native-play wave.

## 11. Live implementation checkpoint (2026-07-15)

The 101M paired-label array is active at the effective six-VM/96-vCPU ceiling.
At 01:35 EDT, zone D had 29 succeeded, six running, and 99 pending tasks (7.25M
durable rows); B/C were scheduled behind quota. There are no production task
errors. Brief reductions in the running-task count have been normal task
handoffs, not persistently idle VMs.

Native generation, exact reconciliation, hash-partitioned global dedup, and
cost-gated Batch submitters are implemented and pass local resume, replay,
duplicate-collapse, and mapper-overlap rejection smokes. The current immutable
image is:

```text
us-east1-docker.pkg.dev/archon-502500/archon-workers/data-worker@sha256:187dacdf4a95ed2333a1c2cefc97cc9eb11548399ca54ac0a93e8ceed43aa4b1
```

The corrected publication canary is `archon-native-canary-v2`. Do not launch
the 10M native wave until that and a small packed generation+dedup cloud pilot
pass GCS reconciliation.

The local raw-tar deletion gate passed again immediately before cleanup: all
86 cloud tar objects matched exact inventory names/sizes (69,951,662,080
bytes), the cloud inventory matched byte-for-byte, and spread downloads matched
SHA-256. The local tar copies were then deleted with user approval, recovering
about 65GB; extracted arrays and manifests remain local.

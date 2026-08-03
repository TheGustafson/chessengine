# `tools/ops/cloud/` — running the data campaign on GCP Batch

Five files. One of them can spend money, and it is named accordingly.

| script | purpose |
|---|---|
| `build_payload.sh` | Assemble the single artifact a worker needs: binaries, book, the pipeline scripts, vendored python-chess. sha256-pinned, with a per-file manifest. |
| `worker_shard.sh` | Produce ONE shard. The whole worker. Runs identically on Batch and on this box. |
| `plan_campaign.py` | Target positions → Batch job spec + worst-case cost + a budget verdict. Parallelism is **derived from a stated vCPU ceiling**. |
| `submit_campaign.sh` | **The only script here that writes to GCP.** Dry-run by default. |
| `fetch_shards.sh` | Progress, and bring shards home with sha256 verification on arrival. |

Merging is `tools/ops/data/merge_shards.py` — the same tool the local campaign uses.

## Runbook

```bash
# 1. build + upload the payload (~107 MB)
bash tools/ops/cloud/build_payload.sh
SHA=$(python3 -c "import json;print(json.load(open('data/cloud/payload/payload.meta.json'))['sha256'])")
gcloud storage cp data/cloud/payload/payload.tar.gz \
    gs://archon-main-bucket/payloads/payload-${SHA:0:12}.tar.gz

# 2. plan — no API calls, no money, prints the cost
python3 tools/ops/cloud/plan_campaign.py --target 100000000 --vcpu-limit 300 \
    --campaign oracle-100m --dest gs://archon-main-bucket/campaigns/oracle-100m \
    --payload-uri gs://archon-main-bucket/payloads/payload-${SHA:0:12}.tar.gz \
    --payload-sha "$SHA" --out plan.json

# 3. ONE task, real money, ~$0.01 — settles image/quota/Spot/GCS in one shot
bash tools/ops/cloud/submit_campaign.sh --plan plan.json --canary --i-mean-it

# 4. the campaign
bash tools/ops/cloud/submit_campaign.sh --plan plan.json --i-mean-it

# 5. collect and merge
bash tools/ops/cloud/fetch_shards.sh --dest gs://archon-main-bucket/campaigns/oracle-100m --status
bash tools/ops/cloud/fetch_shards.sh --dest gs://archon-main-bucket/campaigns/oracle-100m \
    --into data/nnue/selfplay/campaigns/oracle-100m-cloud
.venv/bin/python tools/ops/data/merge_shards.py \
    data/nnue/selfplay/campaigns/oracle-100m-cloud --out corpus.shuf.bin
```

## What was taken from the previous cloud generation, and what was not

The old system (`old/tools/cloud/`, `docs/research/cloud-data-campaign.md`) ran a
101M paired-label job to completion. Its **operating discipline was good and is
kept**:

* dry-run by default, worst-case cost printed, Spot only, task count and runtime
  capped, and an explicit dollar gate;
* a canary before any fan-out (`archon-native-canary-v2` ran before its 404-shard
  job);
* immutable checksummed shards, resume by validating what already exists;
* a provenance manifest written *before* the work starts, so a crashed run is
  still attributable.

Three things were **dropped**:

1. **The custom container image.** `build_worker_image.py`, an allow-list build
   context, Artifact Registry, a push. That is the largest piece of the old
   system and the fastest to rot — the manifest still in the tree pins
   `net-0.cenn` and three engine SHAs its own `_note` records as *already
   drifted at build time*. A sha256-pinned tarball on a public base image buys
   the same reproducibility with two hashes and no registry.

2. **The separate cloud generator.** `native_game_generator.py` is a
   multipv/temperature sampler that produces different positions with different
   WDL semantics from anything the local pipeline makes. Two generators means
   two corpora that cannot be pooled or compared, and no way to attribute a
   change in a trained net to the data rather than the generator. This ships
   `tools/ops/data/` **verbatim**; a cloud shard and a local shard are the same
   computation with a different book slice.

3. **Dual-teacher labelling and its reconcilers.** Oracle-vs-oracle play means
   the label is already in the PGN (see `label_inline.py`), so there is no
   second labelling pass to validate against a first.

## The quota correction

`docs/research/cloud-data-campaign.md` records four quota requests as "accepted
and reconciling" and plans around **17,600 vCPUs**. Verified 2026-08-03, the
live limits in `us-east1` are:

| quota | doc plans on | actual |
|---|---|---|
| `CPUS` | 20,000 | **300** |
| `C2D_CPUS` | 20,000 | **300** |
| `PREEMPTIBLE_CPUS` | 20,000 | **0** |
| `INSTANCES` | 1,250 | 1,250 ✓ |

Only the instance request landed. Every shape in that document is ~60× over
what the project can run, and nothing in the old tooling would have said so —
the job would simply have queued. **`plan_campaign.py` therefore derives
parallelism from a `--vcpu-limit` you must state**, and refuses a plan that
cannot fit it. Pass the real number.

`PREEMPTIBLE_CPUS = 0` is probably harmless: modern Batch `SPOT` provisioning
draws on the regular `CPUS` quota, not the legacy preemptible pool. That is an
inference, not a measurement, and it is the first thing the canary settles.

## Cost

At 300 vCPUs, worst case, `$0.012/vCPU-h` (**unverified list price — pass the
real one with `--vcpu-hour`**):

| target | wall clock | compute | egress | total |
|---|---|---|---|---|
| 100 M | 1.3 h | $4.09 | $0.61 | **$4.71** |
| 1.03 B | 13 h | $42 | $6 | **~$47** |

For comparison, the same 100 M locally is ~24 h on 12 cores.

## Design notes worth knowing before changing anything

**`DEST` may be a local directory instead of `gs://`.** That is not a
convenience — it is what let every path in `worker_shard.sh` be exercised on
this box before any cloud access existed. The old cloud workers reached for GCS
directly, so the first real test of that code was also the first time money was
spent. Six defects were found this way, in order: a preflight that used exit
codes to detect a missing `libstdc++` (and so failed on `nnue_convert`, which
exits non-zero on a usage error); a payload that invented its own directory
layout instead of mirroring repo paths; a hardcoded `.venv/bin/python` in
`run_pipeline.sh`; a missing `uciprobe.py`; **a silent shuffle skip in
`export_corpus.sh` that would have published unshuffled corpora**; and
`run_pipeline.sh` honouring `OUTDIR` for stage 1 only.

**Shards are published unshuffled, and named `corpus.bin` to say so.** The
per-shard shuffle is redundant — `merge_shards.py` does a global two-pass
external shuffle over every shard at merge time — and dropping it removes the
only cloud-side numpy dependency. numpy ships compiled extensions, so unlike
pure-Python python-chess it cannot be vendored by directory copy.

**The artifact uploads last.** `shard.json` and `games.pgn.zst` go first, so a
half-finished upload can never look complete: the presence of `corpus.bin` is
the commit point, and it is what the idempotency check reads. A preempted Spot
task re-runs from scratch and republishes; a completed one exits in seconds.

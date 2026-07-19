# GCE spot fleet — SF18 relabel at scale (plan, 2026-07-19)

User-greenlit the D36 fleet to do the SF18 relabel at scale — the transformational
eval-quality lever (four independent lines now say our eval is at the UHO+SF16-teacher
ceiling; only better *labels* break it). This is the plan + the exact access needed.

## The lever
Our nets are distilled from an **SF16** teacher on the UHO binpack labelled at
**5k nodes**. Relabelling those positions with **SF18** (stronger) at a **deeper**
budget produces labels beyond the SF16 ceiling → training on them should push the
net past SF16. The local pilot (~1.8M positions, SF18-deep, healthy distribution)
is the proof-of-concept; the fleet is the scale needed to train a real net on it.

## What's already built (task #32, ~90% done)
A complete, well-governed pipeline in `tools/cloud/`:
1. `partition_fen_manifest.py` — shard a FEN subset into immutable per-task inputs (2k FENs/task) + a task manifest.
2. `submit_obs_relabel.py` — plan/submit a GCE **Batch** array: **dry-run default**, `--durable-data` gate, **Spot-only**, digest-pinned image, **`--max-cost-usd` hard cap**, bounded retries. Each task runs `root-analysis --reference-only` driving **SF18** over one shard at a pinned node budget, one task per vCPU on `c2d-highcpu-16`.
3. `obs_relabel_shard.py` — the worker: publishes checksummed `.ceobs` shards to GCS, **resume by validating existing shards** (preemption-safe), atomic writes.
4. `reconcile_obs_relabel.py` — inventory/verify the wave + `--ledger-add` (D38) as the accept gate.
5. Local corpus build: `obs-tool export-value` (`FEN|cp|wdl`, bound/completion quarantine) → `bf-convert` → bullet `.bin` → train.
`bootstrap_gcp.sh` creates a keyless Spot Batch service account + Artifact Registry, minimal roles (batch.agentReporter, logging.logWriter, artifactregistry.reader, storage.objectViewer) — the worker **cannot delete corpus objects** and has no training/admin role.

## Cost model (measured)
- Spot `c2d-highcpu-16` = **$0.183/hr** → **$0.01144 / vCPU-hr**.
- SF18 throughput (measured locally, 1 core): **25k nodes 14.6 pos/s**, **50k 9.3**, **100k 5.3**.
- Cost ≈ positions ÷ (pos/s/core) ÷ 3600 × $0.01144. Table (+~30% for cloud core + retries):

| positions | 25k nodes | 50k nodes | 100k nodes |
|---|---|---|---|
| 100M | ~$28 | ~$44 | ~$78 |
| 200M | ~$57 | ~$89 | ~$156 |
| 500M | ~$142 | ~$222 | ~$390 |

Wall-clock (with 50–100 spot VMs = 800–1600 vCPU): **100M @ 50k ≈ 2–4 h**, 200M @ 50k ≈ 4–8 h. Spot scales VM count freely; the hard cost cap is the guardrail.

## Recommended first campaign
**200M UHO positions @ SF18 50k nodes ≈ $90, ~4–8 h.** 50k = 10× the SF16 5k
(meaningfully stronger labels), 200M = enough to train a full net. A cheaper
validation first cut: **100M @ 25k ≈ $30, ~2–4 h**. Cap `--max-cost-usd` at ~1.5×
the estimate. Campaign id `sf18-50k-<date>`.

## What needs YOUR access (the only blockers)
1. **A GCP project with billing enabled.** Scripts default to `archon-502500`,
   region `us-east1`. Confirm it exists + billing on (or give me a project id).
2. **Authenticate gcloud in this session:** `! gcloud auth login` (+ `gcloud config set project <id>`). I can't hold your credentials; once authed I drive the rest.
3. That unblocks, in order (I run these): `bootstrap_gcp.sh` → **build + push the worker image** (SF18 + ARCHON `root-analysis` + the relabel worker; the image isn't built yet — `worker/image_manifest.json` is absent, so this is the one real build step) → extract/partition FENs → **submit dry-run (you review the printed cost)** → submit → reconcile → build corpus → train → gate vs the incumbent.

## Readiness gaps I can close now (no access, in progress)
- **FEN extraction:** `tools/nnue/sfbinpack-audit/…/bulk_fens` dumps FENs from the UHO binpack → the partition input. I'll wire + test the extraction (sized to local disk — currently 8G free, so extract/upload in batches).
- **Worker image manifest:** assemble `worker/image_manifest.json` pinning SF18 + the ARCHON binaries so the build is one command once authed.
- **Dry-run rehearsal:** run `submit_obs_relabel.py` in dry-run against a tiny manifest to confirm the cost estimate + Batch request shape (may need auth for the image digest lookup — will confirm).

## Quota reality (user, 2026-07-19)
A GCE Spot vCPU quota limits how many instances run at once. **This changes
wall-clock, NOT cost** — cost is core-hours × price, independent of parallelism.
Fewer instances just runs longer at the same total spend. Implications:
- **Request a Spot vCPU (`c2d`/preemptible CPUs) quota increase** — that's the
  real wall-clock unblock; new projects often start at ~24–32 vCPU.
- Meanwhile, wall-clock ≈ core-hours ÷ quota-vCPUs. E.g. 100M @ 50k = ~3000
  core-hr → 32 vCPU ≈ **~4 days**; 128 vCPU ≈ **~24 h**; 512 vCPU ≈ **~6 h**.
- So the **first campaign should be sized to the granted quota + an acceptable
  wall-clock**: start with **50–100M @ 25k nodes** (~$15–30) and let it run; scale
  up once the quota is raised and the train+gate confirms the labels help.
- `submit_obs_relabel.py --parallel-tasks` caps concurrent tasks — set it at (or
  below) the quota vCPU count so Batch doesn't thrash against the limit.

## Prep status (2026-07-19, local — no access needed)
- ✅ **Pipeline** built + governed (`tools/cloud/`), verified end-to-end shape.
- ✅ **FEN extraction** confirmed: `sfbinpack-audit/…/bulk_fens <UHO> <count>` emits
  `FEN\twdl\tscore`; take col 1. (Slow under local I/O contention — runs on a quiet
  box or as a cloud prep step; batched extract→partition→upload to fit 8G disk.)
- ✅ **Worker image context present** (`tools/cloud/worker/`: Dockerfile,
  archon_worker.py entrypoint, image_manifest.json, requirements.lock).
- ⏳ **Image manifest SHA re-pin** — the pinned artifact SHAs drifted (binaries
  rebuilt since). Re-pin = build fresh production `root-analysis` / `obs-tool` /
  `chessengine` (+ the SF18 src build) and stamp their sha256 into the manifest.
  This is the last local step before the (access-gated) image build+push.
- ⛔ **Access-gated:** `bootstrap_gcp.sh`, image build+push to Artifact Registry,
  and submit — all need `gcloud auth` + the billed project.

## The gate discipline
The local pilot's train+gate (SF18-deep vs SF16 on 2M) is the cheap first read; but
2M undertrains a net, so the fleet's 100–200M is what actually tests whether
beyond-SF16 labels close the −41.5. Ship rule unchanged: equal-time vs the incumbent.

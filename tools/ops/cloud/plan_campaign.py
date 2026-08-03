#!/usr/bin/env python3
"""Turn a target position count into a Batch job spec, a cost, and a verdict.

THE vCPU CEILING IS MEASURED, NOT READ OFF A QUOTA ROW
    The compute quota reports CPUS 300 and C2D_CPUS 300 in us-east1. Planning on
    those is wrong: job #4 was submitted at parallelism 18 (288 vCPU) and settled
    at exactly 6 concurrent tasks / 96 vCPU, stable over minutes, with 6 live
    VMs and 67 tasks PENDING. Whatever binds -- a Spot-specific ceiling, a Batch
    scheduling limit, or regional Spot capacity -- it is not the row this tool
    was pointed at, and reading that row made every wall-clock estimate 3x
    optimistic.

    So --vcpu-limit means "what actually runs at once", and the number to pass
    today is 96. Confirm it against a live job before trusting a new one.

WHAT THIS EXISTS TO PREVENT
    The previous campaign's design doc plans around 17,600 vCPUs and records
    four quota requests as "accepted and reconciling". The live limits in
    us-east1 are CPUS 300, C2D_CPUS 300, PREEMPTIBLE_CPUS 0 — only the instance
    request landed. Every shape in that document is 60x over what the project
    can actually run, and nothing in the tooling would have said so; the job
    would simply have queued forever.

    So parallelism here is DERIVED FROM A STATED vCPU CEILING, never chosen.
    Pass the real limit; the planner sizes the fan-out to fit it and reports the
    wall clock that follows. A plan that would exceed the ceiling is an error,
    not a warning.

THE COST GATE
    Carried over from the old cloud README, which required every submit to
    "default to dry-run, print a worst-case cost estimate, use Spot
    provisioning only, cap task count and runtime". Those were good rules. The
    estimate here is deliberately WORST CASE: it charges full price for the
    maximum runtime of every task, ignores Spot restart credits, and adds
    egress for the full corpus. If the real bill is lower, good.

    PRICES ARE UNVERIFIED. This box has no billing API access, so the defaults
    below are published list figures and are marked as such in the output. They
    are a parameter for exactly that reason — check them before trusting a
    number, and pass --vcpu-hour with the real one.

USAGE
    python3 tools/ops/cloud/plan_campaign.py --target 1000000000 \\
        --vcpu-limit 300 --payload-sha <sha> --dest gs://bucket/campaigns/x \\
        --out plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# Measured on this box, 60-game oracle-vs-oracle smoke at 5k/5k:
#   12,000 games/h at conc 2  ->  2.90 core-h per 1M kept positions (gen only,
#   in-game labels). 57.5 kept positions per game after the D46 filter.
CORE_H_PER_M = 2.90
KEPT_PER_GAME = 57.5
BYTES_PER_POS = 32
PGN_ZST_GB_PER_M = 0.019


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, required=True, help="kept positions")
    p.add_argument("--shard-pos", type=int, default=2_000_000)
    p.add_argument("--vcpu-limit", type=int, required=True,
                   help="EFFECTIVE concurrent vCPU ceiling. MEASURED 2026-08-03 = 96 "
                        "(6 concurrent 16-vCPU tasks). Do NOT read this off the compute "
                        "CPUS/C2D_CPUS quota rows: those say 300 and something else "
                        "binds first. A job submitted with parallelism 18 sat at exactly "
                        "6 RUNNING / 67 PENDING and 6 live VMs, stable, not ramping.")
    p.add_argument("--machine", default="c2d-highcpu-16")
    p.add_argument("--vcpus-per-task", type=int, default=16)
    p.add_argument("--vcpu-hour", type=float, default=0.012,
                   help="Spot $/vCPU-hour. UNVERIFIED default; pass the real one.")
    p.add_argument("--egress-gb", type=float, default=0.12, help="$/GB out of GCS")
    p.add_argument("--budget", type=float, default=50.0, help="hard $ ceiling")
    p.add_argument("--region", default="us-east1")
    p.add_argument("--dest", required=True)
    p.add_argument("--payload-uri", required=True)
    p.add_argument("--payload-sha", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--image",
                   default="gcr.io/google.com/cloudsdktool/google-cloud-cli:stable",
                   help="PUBLIC base image. Probed locally with docker 2026-08-03 "
                        "rather than guessed: python:3.12 has python3 and libstdc++6 "
                        "but NO gcloud (the entrypoint needs it to fetch the payload), "
                        "and this image has gcloud and libstdc++6 but NO python3. "
                        "Neither is complete; this one is one apt line from complete "
                        "(python3 + zstd, measured 9s) whereas adding gcloud to a "
                        "python image means adding Google's apt repo.")
    p.add_argument("--nodes", type=int, default=5000)
    p.add_argument("--pergame", type=int, default=80)
    p.add_argument("--service-account",
                   default="archon-batch-worker@archon-502500.iam.gserviceaccount.com",
                   help="Least-privileged runtime identity. Verified 2026-08-03 to hold "
                        "exactly batch.agentReporter + logging.logWriter at project level "
                        "and objectCreator + objectViewer on the bucket — it can create "
                        "shard objects and cannot overwrite or delete a corpus. Omitting "
                        "this field does NOT mean 'no identity': Batch falls back to the "
                        "DEFAULT COMPUTE service account, which is broader.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    shards = math.ceil(a.target / a.shard_pos)
    core_h_total = a.target / 1e6 * CORE_H_PER_M
    core_h_shard = a.shard_pos / 1e6 * CORE_H_PER_M
    # A task owns a whole machine's vCPUs, so its wall clock is its core-hours
    # spread over them. Efficiency < 1 because fastchess concurrency does not
    # scale perfectly and a task also spends time on extract/export.
    eff = 0.85
    hours_per_task = core_h_shard / a.vcpus_per_task / eff

    max_tasks = a.vcpu_limit // a.vcpus_per_task
    if max_tasks < 1:
        print(f"FATAL: vcpu-limit {a.vcpu_limit} cannot fit one "
              f"{a.vcpus_per_task}-vCPU task", file=sys.stderr)
        return 2
    parallelism = min(shards, max_tasks)
    waves = math.ceil(shards / parallelism)
    wall_h = waves * hours_per_task

    # Worst case: every task runs the full cap, at full price.
    # 2x headroom on COMPUTE, plus a flat allowance for per-task setup that does
    # not scale with shard size: apt-get python3+zstd (~9 s measured), the
    # 107 MB payload download, and the untar. For a full 2M-position shard that
    # is a rounding error; for a canary it DOMINATES, and a cap derived from
    # compute alone (153 s for a 100k shard) would kill the task during setup.
    SETUP_S = 300
    max_run_s = max(900, int(hours_per_task * 3600 * 2.0) + SETUP_S)
    compute = shards * hours_per_task * a.vcpus_per_task * a.vcpu_hour
    corpus_gb = a.target * BYTES_PER_POS / 1e9
    pgn_gb = a.target / 1e6 * PGN_ZST_GB_PER_M
    egress = (corpus_gb + pgn_gb) * a.egress_gb
    total = compute + egress

    verdict = "OK" if total <= a.budget else "OVER BUDGET"

    print(f"campaign        {a.campaign}")
    print(f"target          {a.target:,} kept positions")
    print(f"shards          {shards} x {a.shard_pos:,}  (~{int(a.shard_pos/KEPT_PER_GAME):,} games each)")
    print(f"machine         {a.machine} ({a.vcpus_per_task} vCPU), SPOT")
    print()
    print(f"vCPU ceiling    {a.vcpu_limit}  -> at most {max_tasks} concurrent tasks")
    print(f"parallelism     {parallelism}  ({waves} wave(s))")
    print(f"per-task        {hours_per_task:.2f} h  (cap {max_run_s}s)")
    print(f"WALL CLOCK      {wall_h:.1f} h")
    print()
    print(f"compute         ${compute:,.2f}   ({core_h_total:,.0f} vCPU-h @ ${a.vcpu_hour}/vCPU-h, UNVERIFIED)")
    print(f"egress          ${egress:,.2f}   ({corpus_gb:.1f} GB corpus + {pgn_gb:.1f} GB pgn.zst)")
    print(f"TOTAL (worst)   ${total:,.2f}   budget ${a.budget:,.2f}   -> {verdict}")

    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{
                    "container": {
                        "imageUri": a.image,
                        "entrypoint": "/bin/bash",
                        # `-c`, NOT `-lc`. A LOGIN shell sources /etc/profile,
                        # which RESETS PATH and drops
                        # /usr/lib/google-cloud-sdk/bin — so `gcloud` vanishes
                        # and the entrypoint dies with exit 127 before it can
                        # fetch the payload. Measured twice: a local docker
                        # probe showed `bash -lc` reporting gcloud MISSING while
                        # `bash -c` found it, and the first canary then failed
                        # exactly this way. PATH is also set explicitly rather
                        # than trusted, so the image's own default cannot move
                        # this out from under us again.
                        "commands": ["-c", (
                            "set -euo pipefail; "
                            "export PATH=/usr/lib/google-cloud-sdk/bin:/usr/local/sbin:"
                            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
                            "command -v gcloud >/dev/null || "
                            "{ echo 'FATAL: gcloud not on PATH'; exit 2; }; "
                            "apt-get update -qq && apt-get install -y -qq python3 zstd >/dev/null; "
                            "mkdir -p /opt/archon && cd /opt/archon; "
                            f"gcloud storage cp {a.payload_uri} payload.tar.gz; "
                            f'echo "{a.payload_sha}  payload.tar.gz" | sha256sum -c -; '
                            "tar xzf payload.tar.gz; "
                            f"CAMPAIGN={a.campaign} DEST={a.dest} "
                            f"SHARD_POS={a.shard_pos} NODES={a.nodes} "
                            f"PERGAME={a.pergame} CONC={a.vcpus_per_task} "
                            "bash /opt/archon/worker_shard.sh"
                        )],
                    }
                }],
                "computeResource": {
                    "cpuMilli": a.vcpus_per_task * 1000,
                    "memoryMib": a.vcpus_per_task * 2048,
                },
                "maxRetryCount": 3,
                "maxRunDuration": f"{max_run_s}s",
            },
            "taskCount": shards,
            "parallelism": parallelism,
        }],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": a.machine,
                                      "provisioningModel": "SPOT"}}],
            "serviceAccount": {"email": a.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }

    plan = dict(
        schema="archon.cloud-campaign-plan.v1",
        campaign=a.campaign, region=a.region, dest=a.dest,
        payload=dict(uri=a.payload_uri, sha256=a.payload_sha),
        sizing=dict(target=a.target, shards=shards, shard_positions=a.shard_pos,
                    parallelism=parallelism, waves=waves,
                    vcpu_limit=a.vcpu_limit, vcpus_per_task=a.vcpus_per_task,
                    hours_per_task=round(hours_per_task, 3),
                    wall_clock_hours=round(wall_h, 2),
                    total_vcpu_hours=round(core_h_total, 1)),
        cost=dict(compute_usd=round(compute, 2), egress_usd=round(egress, 2),
                  total_usd=round(total, 2), budget_usd=a.budget,
                  verdict=verdict, prices_verified=False,
                  vcpu_hour_usd=a.vcpu_hour, egress_gb_usd=a.egress_gb),
        job=job,
    )

    if a.out:
        with open(a.out, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"\nplan -> {a.out}")
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())

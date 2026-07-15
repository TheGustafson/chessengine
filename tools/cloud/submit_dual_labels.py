#!/usr/bin/env python3
"""Plan or submit an immutable, budget-gated paired-label Batch task array."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any


DEFAULT_IMAGE = (
    "us-east1-docker.pkg.dev/archon-502500/archon-workers/data-worker@"
    "sha256:9b0889708691851f2b68a0033759622d890aeb911df72a18d38d27065d3573c3"
)
DEFAULT_SERVICE_ACCOUNT = "archon-batch-worker@archon-502500.iam.gserviceaccount.com"


def gcloud(*arguments: str, capture: bool = False) -> str:
    command = ["gcloud", *arguments]
    if capture:
        return subprocess.check_output(command, text=True)
    subprocess.run(command, check=True)
    return ""


def read_manifest(uri: str) -> dict[str, Any]:
    if uri.startswith("gs://"):
        return json.loads(gcloud("storage", "cat", uri, capture=True))
    return json.loads(Path(uri).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--region", default="us-east1")
    parser.add_argument("--zone")
    parser.add_argument("--machine", default="c2d-highcpu-16")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parallelism", type=int, default=6)
    limit = parser.add_mutually_exclusive_group(required=True)
    limit.add_argument("--nodes", type=int)
    limit.add_argument("--depth", type=int)
    parser.add_argument("--hash-mb", type=int, default=8)
    parser.add_argument("--max-task-seconds", type=int, default=1_200)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--spot-hourly-usd", type=float, default=0.183008)
    parser.add_argument("--measured-rows-per-second", type=float, default=278.8153295348857)
    parser.add_argument("--max-cost-usd", type=float, default=50.0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    parser.add_argument("--durable-data", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()

    positive = (
        args.workers,
        args.parallelism,
        args.nodes if args.nodes is not None else args.depth,
        args.hash_mb,
        args.max_task_seconds,
        args.spot_hourly_usd,
        args.measured_rows_per_second,
        args.max_cost_usd,
    )
    if any(value <= 0 for value in positive) or args.max_retries < 0:
        parser.error("numeric resource, throughput, and budget arguments must be positive")
    if not args.image.startswith("us-east1-docker.pkg.dev/") or "@sha256:" not in args.image:
        parser.error("--image must be a digest-pinned us-east1 Artifact Registry image")
    if args.machine != "c2d-highcpu-16":
        parser.error("this measured resource profile is locked to c2d-highcpu-16")
    if not args.output.startswith("gs://"):
        parser.error("--output must be a GCS prefix")
    if not args.durable_data:
        parser.error("refusing cloud compute without explicit --durable-data declaration")

    manifest = read_manifest(args.manifest)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise SystemExit("manifest has no shards")
    task_count = len(shards)
    total_rows = sum(int(shard["rows"]) for shard in shards)
    if manifest.get("total_shards") != task_count or manifest.get("total_rows") != total_rows:
        raise SystemExit("manifest summary does not match its shard records")
    ordinals = [int(shard["ordinal"]) for shard in shards]
    if len(set(ordinals)) != task_count:
        raise SystemExit("manifest contains duplicate source ordinals")
    if args.zone and not args.zone.startswith(args.region + "-"):
        parser.error("--zone must belong to --region")

    estimated_vm_hours = total_rows / args.measured_rows_per_second / 3600
    estimated_cost = estimated_vm_hours * args.spot_hourly_usd
    maximum_attempts = 1 + args.max_retries
    worst_case_vm_hours = task_count * args.max_task_seconds * maximum_attempts / 3600
    worst_case_cost = worst_case_vm_hours * args.spot_hourly_usd
    if worst_case_cost > args.max_cost_usd:
        raise SystemExit(
            f"worst-case compute ${worst_case_cost:.2f} exceeds cap ${args.max_cost_usd:.2f}"
        )

    command = "set -euo pipefail; " + shlex.join([
        "python", "/opt/archon/tools/dual_label_shard.py",
        "--manifest", args.manifest,
        "--output", args.output,
        "--reckless", "/opt/archon/bin/reckless",
        "--stockfish", "/opt/archon/bin/stockfish-18",
        "--nodes" if args.nodes is not None else "--depth",
        str(args.nodes if args.nodes is not None else args.depth),
        "--workers", str(args.workers),
        "--hash-mb", str(args.hash_mb),
    ])
    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": args.image,
                    "entrypoint": "/bin/bash",
                    "commands": ["-lc", command],
                }}],
                "computeResource": {"cpuMilli": 16_000, "memoryMib": 24_576},
                "maxRetryCount": args.max_retries,
                "maxRunDuration": f"{args.max_task_seconds}s",
            },
            "taskCount": task_count,
            "parallelism": min(args.parallelism, task_count),
        }],
        "allocationPolicy": {
            "instances": [{"policy": {
                "machineType": args.machine,
                "provisioningModel": "SPOT",
            }}],
            "serviceAccount": {"email": args.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {
            "purpose": "durable-paired-labels",
            "teachers": "reckless-sf18",
            "corpus": "lc0-selected-101m",
        },
    }
    if args.zone:
        job["allocationPolicy"]["location"] = {
            "allowedLocations": [f"zones/{args.zone}"]
        }
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "project": args.project,
        "region": args.region,
        "zone": args.zone,
        "manifest": args.manifest,
        "manifest_hash": manifest.get("manifest_hash"),
        "output": args.output,
        "image": args.image,
        "machine": args.machine,
        "workers_per_vm": args.workers,
        "parallelism": min(args.parallelism, task_count),
        "tasks": task_count,
        "rows": total_rows,
        "search_limit": {
            "kind": "nodes" if args.nodes is not None else "depth",
            "value": args.nodes if args.nodes is not None else args.depth,
        },
        "max_attempts": maximum_attempts,
        "max_task_seconds": args.max_task_seconds,
        "measured_rows_per_second": args.measured_rows_per_second,
        "estimated_vm_hours": estimated_vm_hours,
        "estimated_wall_hours": estimated_vm_hours / min(args.parallelism, task_count),
        "estimated_compute_usd": estimated_cost,
        "worst_case_compute_usd": worst_case_cost,
        "cost_cap_usd": args.max_cost_usd,
        "spot_only": True,
        "job": job,
    }
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not args.submit:
        print("DRY RUN: add --submit to create the Batch job")
        return
    with tempfile.TemporaryDirectory(prefix="archon-batch-submit-") as temporary:
        job_path = Path(temporary) / "job.json"
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        gcloud(
            "batch", "jobs", "submit", args.job_id,
            f"--project={args.project}", f"--location={args.region}",
            f"--config={job_path}", "--quiet",
        )


if __name__ == "__main__":
    main()

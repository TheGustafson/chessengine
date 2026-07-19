#!/usr/bin/env python3
"""Plan or submit cost-capped Batch arrays for native global deduplication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile


# ONE collapsed worker image identity (M1/M7). The digest is a placeholder,
# stamped by the M1 rebuild; README §"Worker image" + every submit_*.py share it.
DEFAULT_IMAGE = (
    "us-east1-docker.pkg.dev/archon-502500/archon-workers/archon-worker@"
    "sha256:REPLACE_AT_BUILD"
)
DEFAULT_SERVICE_ACCOUNT = "archon-batch-worker@archon-502500.iam.gserviceaccount.com"


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--parallel-tasks", type=int, default=96)
    parser.add_argument("--tasks-per-vm", type=int, default=16)
    parser.add_argument("--max-task-seconds", type=int, required=True)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--cpu-milli", type=int, default=1_000)
    parser.add_argument("--memory-mib", type=int, default=2_048)
    parser.add_argument("--spot-vm-hourly-usd", type=float, default=0.183008)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--region", default="us-east1")
    parser.add_argument("--zone")
    parser.add_argument("--machine", default="c2d-highcpu-16")
    parser.add_argument("--machine-vcpus", type=int, default=16)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    parser.add_argument("--durable-data", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--plan-output", type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    map_parser = sub.add_parser("map")
    common(map_parser)
    map_parser.add_argument("--source-task-index-start", type=int, required=True)
    map_parser.add_argument("--source-task-index-stop", type=int, required=True)
    map_parser.add_argument("--tasks-per-map", type=int, default=25)
    map_parser.add_argument("--map-index-offset", type=int, default=0)
    map_parser.add_argument("--map-task-count", type=int)
    reduce_parser = sub.add_parser("reduce")
    common(reduce_parser)
    reduce_parser.add_argument("--map-count", type=int, required=True)
    reduce_parser.add_argument("--partition-index-offset", type=int, default=0)
    reduce_parser.add_argument("--partition-task-count", type=int)
    args = parser.parse_args()

    positive = (
        args.partitions, args.parallel_tasks, args.tasks_per_vm,
        args.max_task_seconds, args.cpu_milli, args.memory_mib,
        args.spot_vm_hourly_usd, args.max_cost_usd, args.machine_vcpus,
    )
    if any(value <= 0 for value in positive) or args.max_retries < 0:
        parser.error("counts, resources, rates, and budgets must be positive")
    if args.partitions > 4096 or args.partitions & (args.partitions - 1):
        parser.error("--partitions must be a power of two no larger than 4096")
    if not args.durable_data:
        parser.error("refusing cloud compute without explicit --durable-data")
    if not args.input.startswith("gs://") or not args.output.startswith("gs://"):
        parser.error("--input and --output must be GCS prefixes")
    if "@sha256:" not in args.image:
        parser.error("--image must be digest-pinned")
    if args.machine != "c2d-highcpu-16" or args.machine_vcpus != 16:
        parser.error("resource profile is locked to c2d-highcpu-16")
    if args.tasks_per_vm > args.machine_vcpus:
        parser.error("tasks per VM cannot exceed machine vCPUs")
    if args.zone and not args.zone.startswith(args.region + "-"):
        parser.error("--zone must belong to --region")

    if args.stage == "map":
        if (
            args.source_task_index_start < 0
            or args.source_task_index_stop <= args.source_task_index_start
            or args.tasks_per_map <= 0
            or args.map_index_offset < 0
        ):
            parser.error("invalid source/map interval")
        total_maps = math.ceil(
            (args.source_task_index_stop - args.source_task_index_start) / args.tasks_per_map
        )
        task_count = args.map_task_count or (total_maps - args.map_index_offset)
        if task_count <= 0 or args.map_index_offset + task_count > total_maps:
            parser.error("requested mapper range is outside the full mapper plan")
        map_index = f"$((BATCH_TASK_INDEX+{args.map_index_offset}))"
        start_expr = (
            f"$(({args.source_task_index_start}+map_index*{args.tasks_per_map}))"
        )
        command = (
            "set -euo pipefail; "
            f"map_index={map_index}; task_start={start_expr}; "
            f"task_stop=$((task_start+{args.tasks_per_map})); "
            f"if (( task_stop > {args.source_task_index_stop} )); then "
            f"task_stop={args.source_task_index_stop}; fi; "
            + shlex.join([
                "python", "/opt/archon/tools/native_dedup.py", "map",
                "--input", args.input, "--output", args.output,
                "--partitions", str(args.partitions),
            ])
            + ' --map-index "$map_index" --task-index-start "$task_start"'
            + ' --task-index-stop "$task_stop"'
        )
        index_start = args.map_index_offset
        index_stop = args.map_index_offset + task_count
        total_work = total_maps
    else:
        if args.map_count <= 0 or args.partition_index_offset < 0:
            parser.error("invalid reducer range")
        task_count = args.partition_task_count or (
            args.partitions - args.partition_index_offset
        )
        if task_count <= 0 or args.partition_index_offset + task_count > args.partitions:
            parser.error("requested reducer range is outside the partition plan")
        partition = f"$((BATCH_TASK_INDEX+{args.partition_index_offset}))"
        command = (
            "set -euo pipefail; "
            f"partition={partition}; "
            + shlex.join([
                "python", "/opt/archon/tools/native_dedup.py", "reduce",
                "--input", args.input, "--output", args.output,
                "--partitions", str(args.partitions),
                "--map-count", str(args.map_count),
            ])
            + ' --partition-index "$partition"'
        )
        index_start = args.partition_index_offset
        index_stop = args.partition_index_offset + task_count
        total_work = args.partitions

    maximum_attempts = 1 + args.max_retries
    vcpu_hourly = args.spot_vm_hourly_usd / args.machine_vcpus
    task_vcpus = args.cpu_milli / 1000
    worst_core_hours = task_count * args.max_task_seconds * task_vcpus * maximum_attempts / 3600
    worst_cost = worst_core_hours * vcpu_hourly
    if worst_cost > args.max_cost_usd:
        raise SystemExit(
            f"worst-case compute ${worst_cost:.2f} exceeds cap ${args.max_cost_usd:.2f}"
        )

    parallelism = min(args.parallel_tasks, task_count)
    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": args.image, "entrypoint": "/bin/bash",
                    "commands": ["-lc", command],
                }}],
                "computeResource": {
                    "cpuMilli": args.cpu_milli, "memoryMib": args.memory_mib,
                },
                "maxRetryCount": args.max_retries,
                "maxRunDuration": f"{args.max_task_seconds}s",
            },
            "taskCount": task_count,
            "parallelism": parallelism,
            "taskCountPerNode": args.tasks_per_vm,
        }],
        "allocationPolicy": {
            "instances": [{"policy": {
                "machineType": args.machine, "provisioningModel": "SPOT",
            }}],
            "serviceAccount": {"email": args.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"purpose": f"native-dedup-{args.stage}"},
    }
    if args.zone:
        job["allocationPolicy"]["location"] = {
            "allowedLocations": [f"zones/{args.zone}"]
        }
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "job_id": args.job_id,
        "project": args.project, "region": args.region, "zone": args.zone,
        "input": args.input, "output": args.output, "image": args.image,
        "partitions": args.partitions, "full_stage_tasks": total_work,
        "index_start": index_start, "index_stop": index_stop,
        "task_count": task_count, "parallel_tasks": parallelism,
        "tasks_per_vm": args.tasks_per_vm,
        "maximum_parallel_vms": math.ceil(parallelism / args.tasks_per_vm),
        "maximum_attempts": maximum_attempts,
        "worst_case_compute_usd": worst_cost,
        "cost_cap_usd": args.max_cost_usd, "spot_only": True,
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
    with tempfile.TemporaryDirectory(prefix="archon-native-dedup-submit-") as temporary:
        job_path = Path(temporary) / "job.json"
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        subprocess.run([
            "gcloud", "batch", "jobs", "submit", args.job_id,
            f"--project={args.project}", f"--location={args.region}",
            f"--config={job_path}", "--quiet",
        ], check=True)


if __name__ == "__main__":
    main()

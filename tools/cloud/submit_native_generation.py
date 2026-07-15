#!/usr/bin/env python3
"""Plan or submit a packed, budget-gated native-game Batch task array."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile


DEFAULT_IMAGE = (
    "us-east1-docker.pkg.dev/archon-502500/archon-workers/data-worker@"
    "sha256:187dacdf4a95ed2333a1c2cefc97cc9eb11548399ca54ac0a93e8ceed43aa4b1"
)
DEFAULT_SERVICE_ACCOUNT = "archon-batch-worker@archon-502500.iam.gserviceaccount.com"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--stratum", choices=(
        "reckless_self", "sf18_self", "cross", "exploratory",
    ), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--task-index-offset", type=int, required=True)
    parser.add_argument("--target-positions-per-task", type=int, default=25_000)
    parser.add_argument("--positions-per-game", type=int, default=40)
    parser.add_argument("--move-nodes", type=int, default=5_000)
    parser.add_argument("--gate-nodes", type=int, default=500)
    parser.add_argument("--parallel-tasks", type=int, default=96)
    parser.add_argument("--tasks-per-vm", type=int, default=16)
    parser.add_argument("--max-task-seconds", type=int, default=2_400)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--measured-positions-per-core-second", type=float, default=15.0)
    parser.add_argument("--spot-vm-hourly-usd", type=float, default=0.183008)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--region", default="us-east1")
    parser.add_argument("--zone")
    parser.add_argument("--machine", default="c2d-highcpu-16")
    parser.add_argument("--machine-vcpus", type=int, default=16)
    parser.add_argument("--memory-mib-per-task", type=int, default=1_024)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    parser.add_argument("--durable-data", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()

    positive = (
        args.task_count, args.target_positions_per_task, args.positions_per_game,
        args.move_nodes, args.gate_nodes, args.parallel_tasks, args.tasks_per_vm,
        args.max_task_seconds, args.measured_positions_per_core_second,
        args.spot_vm_hourly_usd, args.max_cost_usd, args.machine_vcpus,
        args.memory_mib_per_task,
    )
    if any(value <= 0 for value in positive) or args.max_retries < 0:
        parser.error("counts, budgets, rates, and resources must be positive")
    if args.task_index_offset < 0:
        parser.error("--task-index-offset cannot be negative")
    if not args.durable_data:
        parser.error("refusing cloud compute without explicit --durable-data")
    if not args.output.startswith("gs://"):
        parser.error("--output must be a GCS prefix")
    if "@sha256:" not in args.image:
        parser.error("--image must be digest-pinned")
    if args.machine != "c2d-highcpu-16" or args.machine_vcpus != 16:
        parser.error("the measured resource profile is locked to c2d-highcpu-16")
    if args.tasks_per_vm > args.machine_vcpus:
        parser.error("tasks per VM cannot exceed the measured one-task-per-vCPU profile")
    if args.zone and not args.zone.startswith(args.region + "-"):
        parser.error("--zone must belong to --region")

    total_positions = args.task_count * args.target_positions_per_task
    estimated_core_hours = (
        total_positions / args.measured_positions_per_core_second / 3600
    )
    vcpu_hourly = args.spot_vm_hourly_usd / args.machine_vcpus
    estimated_cost = estimated_core_hours * vcpu_hourly
    maximum_attempts = 1 + args.max_retries
    worst_core_hours = (
        args.task_count * args.max_task_seconds * maximum_attempts / 3600
    )
    worst_cost = worst_core_hours * vcpu_hourly
    if worst_cost > args.max_cost_usd:
        raise SystemExit(
            f"worst-case compute ${worst_cost:.2f} exceeds cap ${args.max_cost_usd:.2f}"
        )

    generator_arguments = [
        "python", "/opt/archon/tools/native_game_generator.py",
        "--reckless", "/opt/archon/bin/reckless",
        "--stockfish", "/opt/archon/bin/stockfish-18",
        "--output", args.output,
        "--corpus-id", args.corpus_id,
        "--stratum", args.stratum,
        "--seed", str(args.seed),
        "--target-positions", str(args.target_positions_per_task),
        "--positions-per-game", str(args.positions_per_game),
        "--move-nodes", str(args.move_nodes),
        "--gate-nodes", str(args.gate_nodes),
    ]
    command = (
        "set -euo pipefail; "
        f"task_index=$((BATCH_TASK_INDEX+{args.task_index_offset})); "
        + shlex.join(generator_arguments)
        + ' --task-index "$task_index"'
    )
    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": args.image,
                    "entrypoint": "/bin/bash",
                    "commands": ["-lc", command],
                }}],
                "computeResource": {
                    "cpuMilli": 1_000,
                    "memoryMib": args.memory_mib_per_task,
                },
                "maxRetryCount": args.max_retries,
                "maxRunDuration": f"{args.max_task_seconds}s",
            },
            "taskCount": args.task_count,
            "parallelism": min(args.parallel_tasks, args.task_count),
            "taskCountPerNode": args.tasks_per_vm,
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
            "purpose": "durable-native-games",
            "stratum": args.stratum.replace("_", "-"),
            "corpus": "archon-native-v1",
        },
    }
    if args.zone:
        job["allocationPolicy"]["location"] = {
            "allowedLocations": [f"zones/{args.zone}"]
        }
    effective_parallel_tasks = min(args.parallel_tasks, args.task_count)
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "project": args.project,
        "region": args.region,
        "zone": args.zone,
        "corpus_id": args.corpus_id,
        "stratum": args.stratum,
        "output": args.output,
        "image": args.image,
        "task_index_start": args.task_index_offset,
        "task_index_stop": args.task_index_offset + args.task_count,
        "tasks": args.task_count,
        "positions_per_task": args.target_positions_per_task,
        "total_positions": total_positions,
        "parallel_tasks": effective_parallel_tasks,
        "tasks_per_vm": args.tasks_per_vm,
        "maximum_parallel_vms": math.ceil(effective_parallel_tasks / args.tasks_per_vm),
        "estimated_core_hours": estimated_core_hours,
        "estimated_wall_hours": estimated_core_hours / effective_parallel_tasks,
        "estimated_compute_usd": estimated_cost,
        "worst_case_compute_usd": worst_cost,
        "cost_cap_usd": args.max_cost_usd,
        "maximum_attempts": maximum_attempts,
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
    with tempfile.TemporaryDirectory(prefix="archon-native-submit-") as temporary:
        job_path = Path(temporary) / "job.json"
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        subprocess.run([
            "gcloud", "batch", "jobs", "submit", args.job_id,
            f"--project={args.project}", f"--location={args.region}",
            f"--config={job_path}", "--quiet",
        ], check=True)


if __name__ == "__main__":
    main()

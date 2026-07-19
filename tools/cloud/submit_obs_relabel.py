#!/usr/bin/env python3
"""Plan or submit an immutable, budget-gated obs-relabel Batch task array.

Forked from `submit_dual_labels.py` for the D36 governance rails — dry-run
default (requires `--submit`), `--durable-data` gate, digest-pinned image,
Spot-only, `--max-cost-usd` worst-case cost cap with `SystemExit`, bounded
retries, `maxRunDuration`, and the persisted plan artifact — with three swaps:

  * the worker command runs `obs_relabel_shard.py` (M3) over `root-analysis`
    instead of the scalar `dual_label_shard.py`;
  * the manifest is the curated FEN task manifest (M4), not the LC0-feature
    position manifest;
  * the cost estimator uses the one-task-per-vCPU packing model from
    `submit_native_generation.py` (each task is a single-threaded deep
    relabel; `taskCountPerNode` fills the c2d-highcpu-16 VM), with a
    conservative PLACEHOLDER throughput constant — see `--measured-rows-per-second`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any


# ONE collapsed worker image identity (M1/M7). The digest is a placeholder: the
# real one is stamped by the M1 image rebuild against the freshly-built
# production binaries. Every submit_*.py + README §"Worker image" reference this
# same identity — no more research-worker/data-worker three-digest drift.
DEFAULT_IMAGE = (
    "us-east1-docker.pkg.dev/archon-502500/archon-workers/archon-worker@"
    "sha256:REPLACE_AT_BUILD"
)
DEFAULT_SERVICE_ACCOUNT = "archon-batch-worker@archon-502500.iam.gserviceaccount.com"

# Oracle selector -> the packaged UCI engine binary root-analysis drives.
ORACLE_ENGINE_BIN = {
    "sf18": "/opt/archon/bin/stockfish-18",
    "reckless": "/opt/archon/bin/reckless",
    "archon": "/opt/archon/bin/archon",
}


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
    parser.add_argument("--manifest", required=True, help="FEN task manifest from partition_fen_manifest.py")
    parser.add_argument("--output", required=True, help="GCS campaign prefix for the .ceobs shards")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--oracle", choices=tuple(ORACLE_ENGINE_BIN), required=True)
    parser.add_argument("--nodes", type=int, required=True, help="node budget per position")
    parser.add_argument("--tier", choices=("reference-only", "rich"), default="reference-only")
    parser.add_argument("--arm-mode", choices=("searchmoves", "child"), default="searchmoves")
    parser.add_argument("--date", help="campaign date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--created-unix", type=int,
                        help="pin for byte-reproducible shards (default: --date midnight UTC)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hash-mb", type=int, default=16, help="driven-engine UCI Hash per task")
    parser.add_argument("--uci-option", action="append", default=[], metavar="NAME=VALUE",
                        help="extra passthrough (e.g. EvalFile=… for --oracle archon)")
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--region", default="us-east1")
    parser.add_argument("--zone")
    # Spot-availability adaptivity: `--machine` is a preferred->fallback list of
    # c2d-highcpu-{8,16,32}. Per-vCPU spot cost is FLAT across these sizes, so the
    # job is machine-size-independent: each task holds 1 vCPU (cpuMilli=1000) and
    # `taskCountPerNode` auto-derives from the chosen machine's vCPU count, so the
    # SAME shard array runs correctly on 8/16/32 and switching sizes is
    # cost-neutral. GCE Batch pins one machine type per job (instances[0]); if the
    # preferred size has no Spot capacity, resubmit with the next in the list (the
    # driving wrapper does this on a provisioning stall) — no cost or work change.
    parser.add_argument("--machine", default="c2d-highcpu-16,c2d-highcpu-32,c2d-highcpu-8",
                        help="preferred->fallback machine list; per-vCPU packing makes any size correct + cost-neutral")
    parser.add_argument("--machine-vcpus", type=int, default=16,
                        help="vCPU count the --spot-vm-hourly-usd price is quoted for (per-vCPU pricing is flat; default 16)")
    parser.add_argument("--tasks-per-vm", type=int, default=None,
                        help="tasks per VM (default: the chosen machine's vCPU count — one task per vCPU)")
    parser.add_argument("--parallel-tasks", type=int, default=96)
    parser.add_argument("--memory-mib-per-task", type=int, default=1_536)
    parser.add_argument("--max-task-seconds", type=int, default=1_800)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--spot-vm-hourly-usd", type=float, default=0.183008)
    # PLACEHOLDER (assessment Appendix C): single-threaded root-analysis rows/sec
    # at the node budget. Far LOWER than the 278.8 paired-rows/s scalar labeler —
    # that was two shallow searches; Tier-F is one DEEP search at a larger budget.
    # A low value only inflates the informational estimate (the hard cap is the
    # time-based worst case), so this is safe-by-construction. RE-MEASURE on a
    # pilot before scaling and set the true constant here.
    parser.add_argument("--measured-rows-per-second", type=float, default=5.0)
    parser.add_argument("--max-cost-usd", type=float, default=50.0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    parser.add_argument("--durable-data", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()

    positive = (
        args.nodes, args.machine_vcpus, args.parallel_tasks,
        args.memory_mib_per_task, args.max_task_seconds, args.hash_mb,
        args.spot_vm_hourly_usd, args.measured_rows_per_second, args.max_cost_usd,
    )
    if any(value <= 0 for value in positive) or args.max_retries < 0 or args.seed < 0:
        parser.error("numeric resource, throughput, and budget arguments must be positive")
    if not args.image.startswith("us-east1-docker.pkg.dev/") or "@sha256:" not in args.image:
        parser.error("--image must be a digest-pinned us-east1 Artifact Registry image")
    # Spot-adaptive machine list: preferred -> fallback across the measured
    # c2d-highcpu profile. Per-vCPU spot cost is flat across these sizes, so the
    # shard array is machine-size-independent (one task per vCPU); switching sizes
    # is cost-neutral. Batch pins instances[0]; the driving wrapper resubmits with
    # the next size on a Spot-provisioning stall.
    ALLOWED_MACHINES = {"c2d-highcpu-8": 8, "c2d-highcpu-16": 16, "c2d-highcpu-32": 32}
    machine_list = [m.strip() for m in args.machine.split(",") if m.strip()]
    if not machine_list or any(m not in ALLOWED_MACHINES for m in machine_list):
        parser.error(f"--machine must be a comma list drawn from {sorted(ALLOWED_MACHINES)}")
    chosen_machine = machine_list[0]
    chosen_vcpus = ALLOWED_MACHINES[chosen_machine]
    tasks_per_vm = args.tasks_per_vm if args.tasks_per_vm is not None else chosen_vcpus
    if tasks_per_vm <= 0 or tasks_per_vm > chosen_vcpus:
        parser.error("--tasks-per-vm must be in 1..chosen-machine-vCPUs (one task per vCPU)")
    if not args.output.startswith("gs://"):
        parser.error("--output must be a GCS prefix")
    if not args.durable_data:
        parser.error("refusing cloud compute without explicit --durable-data declaration")
    for option in args.uci_option:
        if "=" not in option:
            parser.error(f"--uci-option wants NAME=VALUE, got '{option}'")
    if args.zone and not args.zone.startswith(args.region + "-"):
        parser.error("--zone must belong to --region")

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        campaign_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        parser.error("--date must be YYYY-MM-DD")
    # Pin created_unix so every shard (and every retry) is byte-reproducible.
    created_unix = args.created_unix if args.created_unix is not None else int(campaign_date.timestamp())
    budget_label = (
        f"{args.nodes // 1_000_000}m" if args.nodes % 1_000_000 == 0
        else f"{args.nodes // 1_000}k" if args.nodes % 1_000 == 0
        else str(args.nodes)
    )
    campaign = f"{args.oracle}-{budget_label}-{date}"

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
        raise SystemExit("manifest contains duplicate task ordinals")

    # One-task-per-vCPU packing: each task is a single-threaded relabel, so a
    # VM-core does `measured_rows_per_second`; per-vCPU pricing prices it.
    vcpu_hourly = args.spot_vm_hourly_usd / args.machine_vcpus
    estimated_core_hours = total_rows / args.measured_rows_per_second / 3600
    estimated_cost = estimated_core_hours * vcpu_hourly
    maximum_attempts = 1 + args.max_retries
    # Worst case: every task runs to the wall on every attempt. Each holds one
    # vCPU, so core-hours = tasks x seconds x attempts (independent of packing;
    # the CAP is the hard guard).
    worst_core_hours = task_count * args.max_task_seconds * maximum_attempts / 3600
    worst_cost = worst_core_hours * vcpu_hourly
    if worst_cost > args.max_cost_usd:
        raise SystemExit(
            f"worst-case compute ${worst_cost:.2f} exceeds cap ${args.max_cost_usd:.2f}"
        )

    # Driven-engine UCI options: single-thread + Hash, user overrides win.
    options: dict[str, str] = {"Threads": "1", "Hash": str(args.hash_mb)}
    for option in args.uci_option:
        name, _, value = option.partition("=")
        options[name] = value
    option_args: list[str] = []
    for name in sorted(options):
        option_args += ["--uci-option", f"{name}={options[name]}"]

    command = "set -euo pipefail; " + shlex.join([
        "python", "/opt/archon/tools/obs_relabel_shard.py",
        "--manifest", args.manifest,
        "--output", args.output,
        "--root-analysis", "/opt/archon/bin/root-analysis",
        "--engine", ORACLE_ENGINE_BIN[args.oracle],
        "--oracle", args.oracle,
        "--nodes", str(args.nodes),
        "--date", date,
        "--created-unix", str(created_unix),
        "--tier", args.tier,
        "--arm-mode", args.arm_mode,
        "--seed", str(args.seed),
    ] + option_args)

    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": args.image,
                    "entrypoint": "/bin/bash",
                    "commands": ["-lc", command],
                }}],
                "computeResource": {"cpuMilli": 1_000, "memoryMib": args.memory_mib_per_task},
                "maxRetryCount": args.max_retries,
                "maxRunDuration": f"{args.max_task_seconds}s",
            },
            "taskCount": task_count,
            "parallelism": min(args.parallel_tasks, task_count),
            "taskCountPerNode": tasks_per_vm,
        }],
        "allocationPolicy": {
            "instances": [{"policy": {
                "machineType": chosen_machine,
                "provisioningModel": "SPOT",
            }}],
            "serviceAccount": {"email": args.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {
            "purpose": "durable-obs-relabel",
            "oracle": args.oracle,
            "tier": args.tier,
            "campaign": campaign,
        },
    }
    if args.zone:
        job["allocationPolicy"]["location"] = {
            "allowedLocations": [f"zones/{args.zone}"]
        }
    effective_parallel_tasks = min(args.parallel_tasks, task_count)
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "project": args.project,
        "region": args.region,
        "zone": args.zone,
        "manifest": args.manifest,
        "manifest_hash": manifest.get("manifest_hash"),
        "output": args.output,
        "campaign": campaign,
        "oracle": args.oracle,
        "tier": args.tier,
        "arm_mode": args.arm_mode,
        "date": date,
        "created_unix": created_unix,
        "seed": args.seed,
        "image": args.image,
        "machine": chosen_machine,
        "machine_fallback": machine_list[1:],
        "machine_vcpus": chosen_vcpus,
        "tasks_per_vm": tasks_per_vm,
        "parallel_tasks": effective_parallel_tasks,
        "maximum_parallel_vms": math.ceil(effective_parallel_tasks / tasks_per_vm),
        "tasks": task_count,
        "rows": total_rows,
        "uci_options": [f"{name}={options[name]}" for name in sorted(options)],
        "search_limit": {"kind": "nodes", "value": args.nodes},
        "max_attempts": maximum_attempts,
        "max_task_seconds": args.max_task_seconds,
        "measured_rows_per_second": args.measured_rows_per_second,
        "measured_rows_per_second_is_placeholder": True,
        "estimated_core_hours": estimated_core_hours,
        "estimated_wall_hours": estimated_core_hours / effective_parallel_tasks,
        "estimated_compute_usd": estimated_cost,
        "worst_case_compute_usd": worst_cost,
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
    with tempfile.TemporaryDirectory(prefix="archon-obs-relabel-submit-") as temporary:
        job_path = Path(temporary) / "job.json"
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        gcloud(
            "batch", "jobs", "submit", args.job_id,
            f"--project={args.project}", f"--location={args.region}",
            f"--config={job_path}", "--quiet",
        )


if __name__ == "__main__":
    main()

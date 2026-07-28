#!/usr/bin/env python3
"""Submit one checksum-gated durable HTTP-to-GCS transfer on Spot Batch."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path


# ONE collapsed worker image identity (M1/M7). The digest is a placeholder,
# stamped by the M1 rebuild; README §"Worker image" + every submit_*.py share it.
DEFAULT_IMAGE = (
    "us-east1-docker.pkg.dev/archon-502500/archon-workers/archon-worker@"
    "sha256:REPLACE_AT_BUILD"
)
DEFAULT_SERVICE_ACCOUNT = "archon-batch-worker@archon-502500.iam.gserviceaccount.com"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--project", default="archon-502500")
    parser.add_argument("--region", default="us-east1")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    parser.add_argument("--durable-data", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()
    if not args.durable_data:
        parser.error("refusing cloud transfer without --durable-data")
    if not args.output.startswith("gs://") or len(args.sha256) != 64 or args.bytes <= 0:
        parser.error("output, byte count, or SHA-256 is invalid")
    # e2-standard-2 Spot is comfortably below this conservative hourly cap.
    worst_cost = args.max_seconds / 3600 * 0.10
    if worst_cost > args.max_cost_usd:
        parser.error(f"worst-case ${worst_cost:.2f} exceeds cap ${args.max_cost_usd:.2f}")
    bucket, _, object_name = args.output[5:].partition("/")
    code = f"""
import hashlib, json, os, requests
from google.cloud import storage
url={args.url!r}; expected_bytes={args.bytes}; expected_sha={args.sha256!r}
path='/tmp/source.bin'; digest=hashlib.sha256(); size=0
with requests.get(url, stream=True, timeout=(30, 90)) as response:
    response.raise_for_status()
    with open(path, 'wb') as output:
        for chunk in response.iter_content(8 * 1024 * 1024):
            if not chunk: continue
            output.write(chunk); digest.update(chunk); size += len(chunk)
            if size % (256 * 1024 * 1024) < len(chunk): print(size, flush=True)
actual=digest.hexdigest()
if size != expected_bytes or actual != expected_sha:
    raise RuntimeError(f'identity mismatch bytes={{size}} sha={{actual}}')
client=storage.Client(project={args.project!r}); bucket=client.bucket({bucket!r})
blob=bucket.blob({object_name!r})
blob.upload_from_filename(path, if_generation_match=0, checksum='crc32c')
metadata={{'source':url,'bytes':size,'sha256':actual,'schema':'archon.http-source.v1'}}
bucket.blob({(object_name + '.json')!r}).upload_from_string(
    json.dumps(metadata, sort_keys=True)+'\\n', content_type='application/json',
    if_generation_match=0, checksum='crc32c')
print(json.dumps(metadata, sort_keys=True), flush=True)
"""
    encoded = base64.b64encode(code.encode()).decode()
    command = f"echo {encoded} | base64 -d > /tmp/fetch.py && python /tmp/fetch.py"
    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": args.image, "entrypoint": "/bin/bash",
                    "commands": ["-lc", "set -euo pipefail; " + command],
                }}],
                "computeResource": {"cpuMilli": 2_000, "memoryMib": 2_048},
                "maxRetryCount": 1, "maxRunDuration": f"{args.max_seconds}s",
            },
            "taskCount": 1, "parallelism": 1, "taskCountPerNode": 1,
        }],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": "e2-standard-2", "provisioningModel": "SPOT"}}],
            "serviceAccount": {"email": args.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"purpose": "durable-http-source"},
    }
    plan = {
        "job_id": args.job_id, "url": args.url, "output": args.output,
        "bytes": args.bytes, "sha256": args.sha256, "spot_only": True,
        "worst_case_compute_usd": worst_cost, "cost_cap_usd": args.max_cost_usd,
        "job": job,
    }
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(rendered)
    if not args.submit:
        print("DRY RUN: add --submit to create the Batch job")
        return
    with tempfile.TemporaryDirectory(prefix="archon-http-transfer-") as temporary:
        path = Path(temporary) / "job.json"
        path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        subprocess.run([
            "gcloud", "batch", "jobs", "submit", args.job_id,
            f"--project={args.project}", f"--location={args.region}",
            f"--config={path}", "--quiet",
        ], check=True)


if __name__ == "__main__":
    main()

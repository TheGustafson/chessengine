#!/usr/bin/env python3
"""Mirror official-stockfish/master-binpacks (HuggingFace) into our GCS bucket.

STREAMS each HF LFS file directly HF -> GCS on a Spot Batch fan-out (one task per
file), never staging to the VM's disk — so 47 GB files transfer on an
e2-standard-2 with a 30 GB boot disk. Integrity is verified against the HF LFS
sha256 (== the file's content hash; matches our ledger hashes) AFTER the stream;
a mismatch deletes the object and fails the task. Idempotent: a task whose object
already exists in GCS skips.

Why not Storage Transfer Service? Its URL-list (TsvHttpData) source requires a
base64 MD5 per file; HF exposes sha256 (LFS/xet), not MD5, and the resolve URL is
a short-lived signed redirect that expires mid-transfer. Computing MD5s ourselves
means downloading anyway. So we stream + sha256-verify here.

    # dry run (plan only)
    python tools/cloud/submit_hf_binpack_backup.py --manifest scratch/hf_binpacks.json
    # smoke one small file
    python tools/cloud/submit_hf_binpack_backup.py --manifest ... --only wrongNNUE_02_d9.binpack --submit
    # the rest (skips whatever is already in GCS)
    python tools/cloud/submit_hf_binpack_backup.py --manifest ... --submit
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

DEFAULT_IMAGE = (
    "us-east1-docker.pkg.dev/archon-502500/archon-workers/archon-worker@"
    "sha256:640b73509d6b8beba944c7c5e64c027cd3b8cf684e63bca2cf282fe5686d4b4e"
)
DEFAULT_SERVICE_ACCOUNT = "archon-batch-worker@archon-502500.iam.gserviceaccount.com"
DEFAULT_DEST = "gs://archon-main-bucket/corpora/official-stockfish/master-binpacks/main/"


def worker_code(files_json: str, dest_prefix: str) -> str:
    # Runs on the VM; picks its file by BATCH_TASK_INDEX and streams it to GCS.
    return f"""
import hashlib, json, os, sys, requests
from google.cloud import storage
FILES = json.loads({files_json!r})
DEST = {dest_prefix!r}
idx = int(os.environ.get('BATCH_TASK_INDEX', '0'))
f = FILES[idx]
bucket_name, _, prefix = DEST[5:].partition('/')
obj = prefix + f['path']
client = storage.Client(); bucket = client.bucket(bucket_name); blob = bucket.blob(obj)
if blob.exists():
    print('SKIP-EXISTS', obj, flush=True); sys.exit(0)
digest = hashlib.sha256(); size = 0; CH = 16 * 1024 * 1024
print('START', f['path'], f['bytes'], 'bytes', flush=True)
with requests.get(f['url'], stream=True, timeout=(30, 300)) as r:
    r.raise_for_status()
    with blob.open('wb', chunk_size=CH) as w:
        for chunk in r.iter_content(CH):
            if not chunk:
                continue
            w.write(chunk); digest.update(chunk); size += len(chunk)
            if size % (2 * 1024 * 1024 * 1024) < CH:
                print('progress', size, flush=True)
actual = digest.hexdigest()
if size != f['bytes'] or actual != f['sha256']:
    blob.delete()
    raise RuntimeError('IDENTITY MISMATCH %s bytes=%d/%d sha=%s/%s' % (obj, size, f['bytes'], actual, f['sha256']))
meta = {{'source': f['url'], 'bytes': size, 'sha256': actual, 'path': f['path'], 'schema': 'archon.hf-binpack.v1'}}
bucket.blob(obj + '.json').upload_from_string(
    json.dumps(meta, sort_keys=True) + '\\n', content_type='application/json')
print('OK', obj, size, actual, flush=True)
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="hf_binpacks.json: [{path,bytes,sha256,url}]")
    p.add_argument("--dest-prefix", default=DEFAULT_DEST)
    p.add_argument("--only", default="", help="comma list of paths to include (default: all)")
    p.add_argument("--skip", default="fishpack32.binpack", help="comma list of paths to exclude")
    p.add_argument("--job-id", default="hf-binpack-backup")
    p.add_argument("--project", default="archon-502500")
    p.add_argument("--region", default="us-east1")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    p.add_argument("--parallelism", type=int, default=8)
    p.add_argument("--max-run-seconds", type=int, default=7200)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--plan-output", type=Path)
    args = p.parse_args()

    if not args.dest_prefix.startswith("gs://") or not args.dest_prefix.endswith("/"):
        p.error("--dest-prefix must be a gs:// prefix ending in /")

    manifest = json.loads(Path(args.manifest).read_text())
    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    files = []
    for f in manifest:
        if only and f["path"] not in only:
            continue
        if f["path"] in skip:
            continue
        if not (f.get("sha256") and len(f["sha256"]) == 64 and f.get("bytes")):
            p.error(f"manifest entry {f['path']} missing sha256/bytes")
        files.append({"path": f["path"], "bytes": int(f["bytes"]), "sha256": f["sha256"], "url": f["url"]})
    if not files:
        p.error("no files selected")

    total = sum(f["bytes"] for f in files)
    files_json = json.dumps(files, sort_keys=True)
    code = worker_code(files_json, args.dest_prefix)
    encoded = base64.b64encode(code.encode()).decode()
    command = f"echo {encoded} | base64 -d > /tmp/fetch.py && python /tmp/fetch.py"

    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": args.image, "entrypoint": "/bin/bash",
                    "commands": ["-lc", "set -euo pipefail; " + command],
                }}],
                "computeResource": {"cpuMilli": 2000, "memoryMib": 4096},
                "maxRetryCount": 2, "maxRunDuration": f"{args.max_run_seconds}s",
            },
            "taskCount": len(files),
            "parallelism": min(args.parallelism, len(files)),
            "taskCountPerNode": 1,
        }],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": "e2-standard-2", "provisioningModel": "SPOT"}}],
            "serviceAccount": {"email": args.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"purpose": "hf-binpack-backup"},
    }
    plan = {
        "job_id": args.job_id, "dest_prefix": args.dest_prefix,
        "files": [{"path": f["path"], "gb": round(f["bytes"] / 1e9, 2)} for f in files],
        "total_gb": round(total / 1e9, 2), "task_count": len(files),
        "parallelism": job["taskGroups"][0]["parallelism"],
        "est_storage_usd_per_month": round(total / 1e9 * 0.020, 2),
    }
    print(json.dumps(plan, indent=2))
    if args.plan_output:
        args.plan_output.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.submit:
        print("DRY RUN: add --submit to create the Batch job")
        return
    with tempfile.TemporaryDirectory(prefix="hf-binpack-backup-") as tmp:
        path = Path(tmp) / "job.json"
        path.write_text(json.dumps(job, indent=2) + "\n")
        subprocess.run([
            "gcloud", "batch", "jobs", "submit", args.job_id,
            f"--project={args.project}", f"--location={args.region}",
            f"--config={path}", "--quiet",
        ], check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# tools/ops/cloud/submit_campaign.sh — the ONLY script here that writes to GCP.
#
# Everything else in tools/ops/cloud/ is read-only or local. That is deliberate:
# there is exactly one file to audit before believing nothing can be spent by
# accident, and it is this one.
#
# USAGE
#   # 1. see the plan and the cost, change nothing (DEFAULT):
#   bash tools/ops/cloud/submit_campaign.sh --plan plan.json
#
#   # 2. one task, real money, ~$0.01 — settles Spot/quota/GCS in one shot:
#   bash tools/ops/cloud/submit_campaign.sh --plan plan.json --canary --i-mean-it
#
#   # 3. the campaign:
#   bash tools/ops/cloud/submit_campaign.sh --plan plan.json --i-mean-it
#
# THE GATES, AND WHY EACH ONE IS HERE
#   --i-mean-it        Dry-run is the default. The old cloud README required
#                      this of every submit command and it was right.
#   --canary           Forces taskCount=parallelism=1. The old campaign ran
#                      `archon-native-canary-v2` before its fan-out and that is
#                      the practice worth keeping: one task proves the payload
#                      hash, the base image's libstdc++, Spot provisioning
#                      under a PREEMPTIBLE_CPUS=0 quota, and the GCS write path,
#                      for about a cent.
#   budget verdict     The plan carries its own OVER BUDGET verdict; this
#                      refuses to submit one. A cost gate that can be argued
#                      past in prose is not a gate.
#   payload existence  Refuses if the payload object is not already in GCS.
#                      A job whose first act is to fetch a missing tarball
#                      fails once per task, in parallel, and bills for each.
#   SPOT assertion     Refuses any plan not marked SPOT.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

PLAN=""; GO=0; CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --plan) PLAN=$2; shift 2;;
        --i-mean-it) GO=1; shift;;
        --canary) CANARY=1; shift;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done
[ -s "$PLAN" ] || { log "FATAL: --plan <plan.json> required"; exit 2; }

read -r CAMPAIGN REGION DEST PAYLOAD VERDICT TOTAL BUDGET PROV TASKS PAR <<EOF
$(python3 - "$PLAN" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
j = p["job"]
print(p["campaign"], p["region"], p["dest"], p["payload"]["uri"],
      p["cost"]["verdict"].replace(" ", "_"), p["cost"]["total_usd"],
      p["cost"]["budget_usd"],
      j["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"],
      j["taskGroups"][0]["taskCount"], j["taskGroups"][0]["parallelism"])
PYEOF
)
EOF

log "campaign   $CAMPAIGN  region=$REGION"
log "dest       $DEST"
log "payload    $PAYLOAD"
log "shape      taskCount=$TASKS parallelism=$PAR provisioning=$PROV"
log "cost       \$$TOTAL of \$$BUDGET  -> $VERDICT"

[ "$PROV" = "SPOT" ] || { log "FATAL: plan is not SPOT ($PROV). Refusing."; exit 3; }
[ "$VERDICT" = "OK" ] || { log "FATAL: plan is $VERDICT. Raise --budget deliberately or shrink the target."; exit 3; }

if [ "$GO" != "1" ]; then
    log "DRY RUN — nothing submitted. Add --i-mean-it to submit for real."
    log "  strongly recommended first: --canary --i-mean-it (one task, ~\$0.01)"
    exit 0
fi

gcloud storage ls "$PAYLOAD" >/dev/null 2>&1 || {
    log "FATAL: payload not in GCS: $PAYLOAD"
    log "  upload it first:  gcloud storage cp data/cloud/payload/payload.tar.gz $PAYLOAD"
    exit 4
}

JOBFILE=$(mktemp "${TMPDIR:-/tmp}/archon-job.XXXXXX.json")
python3 - "$PLAN" "$CANARY" > "$JOBFILE" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1])); job = p["job"]
if sys.argv[2] == "1":
    job["taskGroups"][0]["taskCount"] = 1
    job["taskGroups"][0]["parallelism"] = 1
json.dump(job, sys.stdout, indent=2)
PYEOF

SUFFIX=$([ "$CANARY" = "1" ] && echo "canary" || echo "run")
JOBNAME="${CAMPAIGN}-${SUFFIX}-$(date +%Y%m%d-%H%M%S)"
log "submitting $JOBNAME"
[ "$CANARY" = "1" ] && log "  CANARY: taskCount forced to 1"

gcloud batch jobs submit "$JOBNAME" \
    --location="$REGION" --config="$JOBFILE" 2>&1 | tail -20
rc=${PIPESTATUS[0]}
rm -f "$JOBFILE"
[ $rc -eq 0 ] || { log "FATAL: submit failed rc=$rc"; exit 5; }

log "submitted. watch:"
log "  gcloud batch jobs describe $JOBNAME --location=$REGION"
log "  bash tools/ops/cloud/fetch_shards.sh --dest $DEST --status"

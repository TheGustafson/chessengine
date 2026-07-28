#!/usr/bin/env bash
# Idempotently provision the keyless Archon Batch worker identity and registry.
set -euo pipefail

PROJECT="${ARCHON_GCP_PROJECT:-archon-502500}"
REGION="${ARCHON_GCP_REGION:-us-east1}"
BUCKET="${ARCHON_GCP_BUCKET:-archon-main-bucket}"
REPOSITORY="${ARCHON_GCP_REPOSITORY:-archon-workers}"
SERVICE_ACCOUNT_NAME="${ARCHON_GCP_WORKER_SA:-archon-batch-worker}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project "${PROJECT}" \
    --display-name "Archon Batch Worker" \
    --description "Keyless runtime identity for Archon durable-data Batch jobs" \
    --quiet
fi

if ! gcloud artifacts repositories describe "${REPOSITORY}" --project "${PROJECT}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --project "${PROJECT}" --location "${REGION}" \
    --repository-format docker --immutable-tags \
    --description "Pinned Archon durable-data worker images" --quiet
fi

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/batch.agentReporter --condition=None --quiet >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/logging.logWriter --condition=None --quiet >/dev/null
gcloud artifacts repositories add-iam-policy-binding "${REPOSITORY}" \
  --project "${PROJECT}" --location "${REGION}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/artifactregistry.reader --condition=None --quiet >/dev/null

# Viewer + creator permits resumable immutable output: workers may inspect and
# create objects, but cannot overwrite or delete corpus shards.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.objectViewer --quiet >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.objectCreator --quiet >/dev/null

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
case "${ACTIVE_ACCOUNT}" in
  *gserviceaccount.com) ACTIVE_MEMBER="serviceAccount:${ACTIVE_ACCOUNT}" ;;
  *) ACTIVE_MEMBER="user:${ACTIVE_ACCOUNT}" ;;
esac
gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT}" \
  --project "${PROJECT}" --member "${ACTIVE_MEMBER}" \
  --role roles/iam.serviceAccountUser --condition=None --quiet >/dev/null

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet >/dev/null
printf 'worker_sa=%s\nrepository=%s-docker.pkg.dev/%s/%s\n' \
  "${SERVICE_ACCOUNT}" "${REGION}" "${PROJECT}" "${REPOSITORY}"

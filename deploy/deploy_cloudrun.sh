#!/usr/bin/env bash
# Deploy the CosmUFR demo to Cloud Run.
#
# The image is built by Cloud Build, not locally, so nothing heavy runs on the
# development machine. Run from the repo root.
#
#   bash deploy/deploy_cloudrun.sh
#
# Set ALWAYS_ON=1 for a container that never scales to zero (instant response,
# billed continuously). Leave it unset for scale-to-zero (free when idle, with
# a cold start on the first request after a quiet period).

set -euo pipefail

PROJECT="${PROJECT:-gen-lang-client-0941231247}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-cosmufr-demo}"
REPO="${REPO:-cosmufr}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}"

# ALWAYS_ON=1 keeps one instance resident so there is never a cold start.
#
# Note it keeps CPU *throttling on*. Throttling only limits CPU between
# requests; the container is not torn down, so the Python process stays alive
# and the 136M-parameter model stays loaded in memory. A request arriving at a
# throttled instance unthrottles it and is served immediately: no image pull, no
# model load. --no-cpu-throttling would bill full vCPU rate around the clock for
# a service that does nothing between requests, which is roughly seven times the
# cost for no benefit here.
ALWAYS_ON="${ALWAYS_ON:-0}"
if [[ "${ALWAYS_ON}" == "1" ]]; then
  MIN_INSTANCES=1
else
  MIN_INSTANCES=0
fi
CPU_FLAG="--cpu-throttling"

echo "project      : ${PROJECT}"
echo "region       : ${REGION}"
echo "service      : ${SERVICE}"
echo "image        : ${IMAGE}"
echo "min instances: ${MIN_INSTANCES}  (${CPU_FLAG})"
echo

# 1. Artifact Registry repository (idempotent).
if ! gcloud artifacts repositories describe "${REPO}" \
      --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1; then
  echo "creating Artifact Registry repository ${REPO} ..."
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="CosmUFR container images" \
    --project="${PROJECT}"
fi

# 2. Build on Cloud Build. The checkpoint is pulled inside the build from the
#    public HuggingFace repo and its SHA256 verified, so the 545 MB file is
#    never uploaded from this machine.
echo "submitting build ..."
gcloud builds submit \
  --tag "${IMAGE}" \
  --project="${PROJECT}" \
  --timeout=30m \
  --machine-type=e2-highcpu-8 \
  .

# 3. Deploy.
#    --allow-unauthenticated: this is a public demo.
#    --concurrency=4: one model instance, and inference is CPU-bound, so a high
#    concurrency would only queue requests behind each other.
echo "deploying ..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --cpu=2 \
  --memory=4Gi \
  --concurrency=4 \
  --min-instances="${MIN_INSTANCES}" \
  --max-instances=3 \
  --timeout=120 \
  ${CPU_FLAG} \
  --startup-probe="httpGet.path=/health,initialDelaySeconds=10,periodSeconds=5,failureThreshold=30,timeoutSeconds=5"

URL=$(gcloud run services describe "${SERVICE}" \
        --project="${PROJECT}" --region="${REGION}" \
        --format='value(status.url)')

echo
echo "deployed: ${URL}"
echo
echo "verifying ..."
curl -sS -o /dev/null -w "  /health  HTTP %{http_code}  %{time_total}s\n" "${URL}/health"
curl -sS -o /dev/null -w "  /        HTTP %{http_code}  %{time_total}s\n" "${URL}/"
curl -sS -o /dev/null -w "  /audit   HTTP %{http_code}  %{time_total}s\n" "${URL}/audit"
curl -sS -o /dev/null -w "  /results HTTP %{http_code}  %{time_total}s\n" "${URL}/results"
curl -sS -o /dev/null -w "  POST /infer HTTP %{http_code}  %{time_total}s\n" \
     -X POST -d "example_id=0" "${URL}/infer"

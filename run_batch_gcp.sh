#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION=europe-west1
JOB=jdr-transcribe

IN_BUCKET="gs://${PROJECT_ID}-audio-in"
OUT_BUCKET="${PROJECT_ID}-audio-out"

SESSION="${1:-session-$(date +%Y%m%d-%H%M%S)}"
INPUT_DIR="${2:-input}"
CONTEXT_PATH="${3:-context/context.yaml}"
EMAIL_TO="${4:-rouquettej@gmail.com}"

PREFIX="sessions/${SESSION}"

echo ">> Upload ${INPUT_DIR} -> ${IN_BUCKET}/${PREFIX}"
gcloud storage rsync --recursive "${INPUT_DIR}" "${IN_BUCKET}/${PREFIX}"

CONTEXT_URI=""
if [[ -f "$CONTEXT_PATH" ]]; then
  echo ">> Upload contexte: ${CONTEXT_PATH}"
  gcloud storage cp "$CONTEXT_PATH" "${IN_BUCKET}/${PREFIX}/context/$(basename "$CONTEXT_PATH")"
  CONTEXT_URI="${IN_BUCKET}/${PREFIX}/context/$(basename "$CONTEXT_PATH")"
fi

echo ">> Execute Cloud Run Job (fire-and-forget)"
gcloud run jobs execute "$JOB" --region "$REGION" \
  --update-env-vars="INPUT_PREFIX=${IN_BUCKET}/${PREFIX},OUTPUT_BUCKET=${OUT_BUCKET},OUTPUT_PREFIX=${PREFIX},EMAIL_TO=${EMAIL_TO},CONTEXT_GCS_URI=${CONTEXT_URI}" \
  --format="value(latestCreatedExecution.name)"

echo "OK. Le job tourne côté GCP."
echo "Suivre: gcloud run jobs executions list --job $JOB --region $REGION"

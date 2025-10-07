#!/usr/bin/env bash

# run_batch_gcp.sh - Synchronise local audio files and context to Google Cloud Storage and triggers a Cloud Run batch job.
# Copyright (C) 2024  Jonathan Rouquette
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
# Usage:
#   ./run_batch_gcp.sh [session-name] [input-dir] [context-path] [email-to]
#   - session-name: Nom de la session (par défaut: session-YYYYMMDD-HHMMSS)
#   - input-dir: Répertoire local contenant les fichiers audio (par défaut: input)  
#   - context-path: Chemin local vers le fichier YAML de contexte (par défaut: context/context.yaml)
#   - email-to: Adresse email pour la notification (par défaut: email@example.com)
# Example:
#   ./run_batch_gcp.sh session-20240601-1230 input context/context.yaml email@example.com
# -----------------------------------------------------------------------------

set -euo pipefail

# Interactive GNU GPL notice
if [ -t 1 ]; then
  echo "run_batch_gcp.sh  Copyright (C) 2024  Jonathan Rouquette"
  echo "This program comes with ABSOLUTELY NO WARRANTY; for details type 'show w'."
  echo "This is free software, and you are welcome to redistribute it"
  echo "under certain conditions; type 'show c' for details."
fi

# Show warranty or copying info if requested
if [[ "${1:-}" == "show w" ]]; then
  echo "This program is distributed in the hope that it will be useful,"
  echo "but WITHOUT ANY WARRANTY; without even the implied warranty of"
  echo "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the"
  echo "GNU General Public License for more details."
  exit 0
fi

if [[ "${1:-}" == "show c" ]]; then
  echo "This program is free software: you can redistribute it and/or modify"
  echo "it under the terms of the GNU General Public License as published by"
  echo "the Free Software Foundation, either version 3 of the License, or"
  echo "(at your option) any later version."
  echo
  echo "You should have received a copy of the GNU General Public License"
  echo "along with this program.  If not, see <https://www.gnu.org/licenses/>."
  exit 0
fi

if ! command -v gcloud &> /dev/null; then
  echo "gcloud CLI n'est pas installé. Veuillez l'installer et réessayer."
  exit 1
fi

if [[ $# -eq 0 ]] || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  grep '^#' "$0" | sed 's/^#//'
  exit 0
fi

# -----------------------------------------------------------------------------

PROJECT_ID=$(gcloud config get-value project)
REGION=europe-west1
JOB=jdr-transcribe

IN_BUCKET="gs://${PROJECT_ID}-audio-in"
OUT_BUCKET="${PROJECT_ID}-audio-out"

SESSION="${1:-session-$(date +%Y%m%d-%H%M%S)}"
INPUT_DIR="${2:-input}"
CONTEXT_PATH="${3:-context/context.yaml}"
EMAIL_TO="${4:-email@example.com}"

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

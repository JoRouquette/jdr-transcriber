#!/usr/bin/env bash
# jdr-transcriber setup.sh - Installe les dépendances GCP et configure l'environnement batch Cloud Run.
# Copyright (C) 2025  <Votre Nom>
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. See <https://www.gnu.org/licenses/>.

set -euo pipefail

# -------------------- AIDE / LICENCE --------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage:
  ./setup.sh [--region <REGION>] [--email <EMAIL_ENVOI>] [--smtp-app-pass <APP_PASS>] [--no-venv]

Description:
  - Active les APIs GCP nécessaires
  - Crée 2 buckets privés: gs://<PROJECT_ID>-audio-in et gs://<PROJECT_ID>-audio-out
  - Crée un service account + IAM minimal
  - Crée le secret SMTP_PASSWORD (ou l’actualise)
  - Crée (ou met à jour) le Job Cloud Run “jdr-transcribe”
  - Construit et publie l’image Docker (Artifact Registry)
  - (Optionnel) installe un venv Python local

Options:
  --region <REGION>        Région GCP (défaut: europe-west1)
  --email <EMAIL_ENVOI>    Adresse SMTP d'envoi et From (ex: user@example.com)
  --smtp-app-pass <PASS>   Mot de passe d’application SMTP (sinon demandé en entrée masquée)
  --no-venv                N’installe pas d’environnement Python local

HELP
  exit 0
fi

echo "setup.sh  Copyright (C) 2025 Jonathan ROUQUETTE"
echo "This program comes with ABSOLUTELY NO WARRANTY. See 'show w'."
echo "This is free software under the GNU GPL v3+. See 'show c'."

if [[ "${1:-}" == "show w" ]]; then
  cat <<'WARRANTY'
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
WARRANTY
  exit 0
fi

if [[ "${1:-}" == "show c" ]]; then
  cat <<'COPY'
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
COPY
  exit 0
fi

# -------------------- PARAMÈTRES --------------------
REGION="europe-west1"
EMAIL_ENVOI=""
SMTP_APP_PASS=""
INSTALL_VENV="yes"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2;;
    --email) EMAIL_ENVOI="$2"; shift 2;;
    --smtp-app-pass) SMTP_APP_PASS="$2"; shift 2;;
    --no-venv) INSTALL_VENV="no"; shift;;
    *) echo "Option inconnue: $1. Utilise --help."; exit 1;;
  esac
done

# -------------------- PRÉREQUIS --------------------
command -v gcloud >/dev/null || { echo "gcloud n'est pas installé."; exit 1; }
command -v python3 >/dev/null || { echo "Python 3 n'est pas installé."; exit 1; }

# (Optionnel) Vérif Python 3.11, uniquement si tu tiens au venv 3.11 strict
if [[ "$INSTALL_VENV" == "yes" ]] && ! python3 -c 'import sys; exit(0 if sys.version_info[:2]>=(3,8) else 1)'; then
  echo "Python >= 3.8 requis pour le venv local. Continuer sans venv..."
  INSTALL_VENV="no"
fi

# Auth / Projet
echo ">> Vérification du compte et du projet"
gcloud auth list --filter=status:ACTIVE
PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Aucun projet GCP configuré. Exécute: gcloud config set project <PROJECT_ID>"
  exit 1
fi
echo "   Projet: ${PROJECT_ID}"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

# -------------------- ACTIVER LES APIS --------------------
echo ">> Activation des APIs requises"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com

# -------------------- BUCKETS PRIVÉS --------------------
IN_BUCKET="gs://${PROJECT_ID}-audio-in"
OUT_BUCKET="gs://${PROJECT_ID}-audio-out"

echo ">> Création des buckets (si absent)"
gcloud storage buckets create "$IN_BUCKET"  --location="$REGION" --uniform-bucket-level-access || true
gcloud storage buckets create "$OUT_BUCKET" --location="$REGION" --uniform-bucket-level-access || true

echo ">> Enforce Public Access Prevention (PAP) sur les deux buckets"
gcloud storage buckets update "$IN_BUCKET"  --public-access-prevention || true
gcloud storage buckets update "$OUT_BUCKET" --public-access-prevention || true

# -------------------- SERVICE ACCOUNT + IAM --------------------
SA_NAME="transcriber-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo ">> Service Account: ${SA_EMAIL}"
gcloud iam service-accounts create "$SA_NAME" --display-name="JDR Transcriber SA" || true

echo ">> IAM buckets (lecture IN / écriture OUT)"
gcloud storage buckets add-iam-policy-binding "$IN_BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectViewer" || true

gcloud storage buckets add-iam-policy-binding "$OUT_BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin" || true

echo ">> IAM Secret Manager (lecture)"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/secretmanager.secretAccessor" >/dev/null || true

# -------------------- SECRET SMTP --------------------
SECRET_NAME="SMTP_PASSWORD"
echo ">> Secret Manager: ${SECRET_NAME}"
gcloud secrets describe "$SECRET_NAME" >/dev/null 2>&1 || gcloud secrets create "$SECRET_NAME" --replication-policy=automatic

if [[ -z "$SMTP_APP_PASS" ]]; then
  echo -n "Entrez le mot de passe d'application SMTP (saisi masqué): "
  stty -echo; read -r SMTP_APP_PASS; stty echo; echo
fi

if [[ -n "$SMTP_APP_PASS" ]]; then
  printf '%s' "$SMTP_APP_PASS" | gcloud secrets versions add "$SECRET_NAME" --data-file=- >/dev/null
else
  echo "Aucun mot de passe d'application fourni. Le job ne pourra pas envoyer d'email tant que le secret n'est pas peuplé."
fi

# -------------------- ARTIFACT REGISTRY + BUILD --------------------
REPO="transcriber"
IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/jdr-transcriber:latest"

echo ">> Artifact Registry: ${REPO}"
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" || true

echo ">> Build & push: ${IMAGE_PATH}"
gcloud builds submit --tag "$IMAGE_PATH" .

# -------------------- JOB CLOUD RUN --------------------
echo ">> Création / Mise à jour du Job Cloud Run: jdr-transcribe"
JOB_EXISTS="$(gcloud run jobs describe jdr-transcribe --region "$REGION" --format='value(name)' 2>/dev/null || true)"

# Adresse d'envoi = SMTP_USER et From par défaut
if [[ -z "$EMAIL_ENVOI" ]]; then
  read -rp "Adresse d'envoi SMTP (ex: user@example.com): " EMAIL_ENVOI
fi

COMMON_FLAGS=(
  --region "$REGION"
  --image "$IMAGE_PATH"
  --service-account "$SA_EMAIL"
  --cpu=2
  --memory=8Gi
  --task-timeout=21600s
  --set-secrets "SMTP_PASSWORD=${SECRET_NAME}:latest"
  --set-env-vars "SMTP_HOST=smtp.gmail.com,SMTP_PORT=587,SMTP_USER=${EMAIL_ENVOI},EMAIL_FROM=${EMAIL_ENVOI},ATTACH_LIMIT_MB=20"
  --set-env-vars "WHISPER_MODEL_SIZE=large-v3,WHISPER_MODEL_DIR=/models/large-v3,HF_HUB_OFFLINE=1"
  --set-env-vars "MULTILANG=true,PRIMARY_LANG=fr,SECONDARY_LANG=en,MAX_CHUNK_MINUTES=15,CHUNK_OVERLAP_MS=2000,VAD_FRAME_MS=20"
)

if [[ -n "$JOB_EXISTS" ]]; then
  gcloud run jobs update jdr-transcribe "${COMMON_FLAGS[@]}"
else
  gcloud run jobs create jdr-transcribe "${COMMON_FLAGS[@]}"
fi

echo ">> Vérification du job"
gcloud run jobs describe jdr-transcribe --region "$REGION" \
  --format='value(template.template.template.containers[0].image)'

# -------------------- (OPTIONNEL) VENV LOCAL --------------------
if [[ "$INSTALL_VENV" == "yes" ]]; then
  echo ">> Installation venv local (optionnel)"
  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

cat <<'DONE'

OK. Environnement GCP prêt.

Prochaines étapes :
1) Placez vos audios dans ./input et (optionnel) le contexte dans ./context/context.yaml
2) Lancez un batch :
   ./run_batch_gcp.sh "session-01" "./input" "./context/context.yaml" "destinataire@example.com"

Le Job lira:
  INPUT_PREFIX = gs://<PROJECT_ID>-audio-in/sessions/session-01
et écrira:
  gs://<PROJECT_ID>-audio-out/sessions/session-01
et enverra un e-mail au destinataire.

DONE

# Guide de déploiement reproductible (anonymisé)

Ce document récapitule tout ce qu’il faut pour cloner/forker le repo et obtenir un traitement **batch “fire-and-forget”** sur Google Cloud : upload des audios + contexte, exécution dans un conteneur Docker (diarisation + transcription), archivage, envoi par email, puis arrêt automatique du job.
Aucune information personnelle n’apparaît ci-dessous ; remplace les valeurs entre chevrons `<>` par les tiennes.

---

## 1) Pré-requis

* Un projet Google Cloud actif : `<PROJECT_ID>`
* SDK Google Cloud installé et connecté au bon compte : `gcloud auth login && gcloud config set project <PROJECT_ID>`
* Facturation activée sur le projet
* Git + Docker (localement seulement si tu veux tester l’image ; sinon, Cloud Build suffit)
* Un compte email d’envoi (ex. Gmail) avec **mot de passe d’application** (2FA activée)

---

## 2) Arborescence du repo attendue

```
.
├─ app/
│  └─ process.py            # pipeline VAD/diarisation/transcription (faster-whisper)
├─ context/
│  ├─ context.yaml          # contexte optionnel (speakers, glossaire, voix, overrides)
│  └─ voices/               # échantillons de voix (optionnels)
├─ input/                   # audios à traiter (mp3, m4a, wav…)
├─ output/                  # vide (rempli par le job, localement seulement)
├─ entrypoint_gcp.py        # orchestrateur batch dans le conteneur (GCS in/out + email)
├─ requirements.txt         # dépendances Python (incluant GCP SDKs, webrtcvad, faster-whisper)
├─ Dockerfile               # image exécutable (CPU), modèle embarqué
└─ run_batch_gcp.sh         # script local “one-liner” pour lancer un batch sur Cloud Run Jobs
```

---

## 3) Fichiers clés du projet

### 3.1 `requirements.txt`

Inclure (en plus de tes libs existantes) :

```
google-cloud-storage>=2.17.0
google-cloud-secret-manager>=2.20.2
webrtcvad==2.0.10
faster-whisper>=1.0
huggingface_hub>=0.24.0
resemblyzer
tqdm
scikit-learn
PyYAML
```

### 3.2 Patch minimal dans `app/process.py`

Permet de charger un modèle depuis un chemin local si présent (le conteneur embarque le modèle) :

```python
from pathlib import Path
from faster_whisper import WhisperModel
# ...
model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
local_dir  = os.getenv("WHISPER_MODEL_DIR") or os.getenv("WHISPER_MODEL_PATH")
model_id   = local_dir if (local_dir and Path(local_dir).exists()) else model_size
model      = WhisperModel(model_id, device="cpu", compute_type="int8")
```

### 3.3 `entrypoint_gcp.py`

Fichier Python qui :

* télécharge `input/` (et le `context.yaml` si fourni) depuis GCS,
* lance `process.py`,
* uploade `output/` vers GCS,
* crée un ZIP et envoie un email avec la pièce jointe (ou un **lien signé** si trop volumineux).

Variables lues via env :

* Input/Output : `INPUT_PREFIX` (ex: `gs://<PROJECT_ID>-audio-in/sessions/<SESSION>`), `OUTPUT_BUCKET`, `OUTPUT_PREFIX`
* Contexte : `CONTEXT_GCS_URI` (optionnel)
* Email : `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` (secret), `EMAIL_FROM`, `EMAIL_TO`, `ATTACH_LIMIT_MB`
* Whisper : `WHISPER_MODEL_SIZE`, `WHISPER_MODEL_DIR`, etc.

### 3.4 `Dockerfile` (modèle **large-v3** embarqué, exécution offline)

```dockerfile
FROM python:3.11-slim

# Système : ffmpeg + toolchain pour compiler webrtcvad
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement du modèle faster-whisper (CTranslate2) AU BUILD
ENV WHISPER_MODEL_DIR=/models/large-v3
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Systran/faster-whisper-large-v3', local_dir='/models/large-v3', local_dir_use_symlinks=False)"

# Code
COPY app/ /app/app/
COPY entrypoint_gcp.py /app/entrypoint_gcp.py

# Runtime : offline côté HF, logs non-bufferisés
ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/entrypoint_gcp.py"]
```

---

## 4) Ressources Google Cloud

### 4.1 Activer les APIs

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com
```

### 4.2 Buckets GCS (privés, région identique partout, ex. `<REGION>=europe-west1`)

```bash
PROJECT_ID=<PROJECT_ID>
REGION=<REGION>

# buckets uniques globalement
IN="gs://${PROJECT_ID}-audio-in"
OUT="gs://${PROJECT_ID}-audio-out"

gcloud storage buckets create "$IN"  --location=$REGION --uniform-bucket-level-access
gcloud storage buckets create "$OUT" --location=$REGION --uniform-bucket-level-access

# Empêche tout accès public
gcloud storage buckets update "$IN"  --public-access-prevention
gcloud storage buckets update "$OUT" --public-access-prevention
```

### 4.3 Service Account + droits minimaux

```bash
gcloud iam service-accounts create transcriber-sa
SA="transcriber-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Lecture IN / écriture OUT
gcloud storage buckets add-iam-policy-binding "$IN"  \
  --member="serviceAccount:${SA}" --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding "$OUT" \
  --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin"

# Accès aux secrets
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"
```

### 4.4 Secret Manager (mot de passe d’application SMTP)

Créer un **mot de passe d’application** côté fournisseur mail (Gmail recommandé). Puis :

```bash
gcloud secrets create SMTP_PASSWORD --replication-policy=automatic
printf '%s' '<APP_PASSWORD_SANS_ESPACES>' | gcloud secrets versions add SMTP_PASSWORD --data-file=-
```

### 4.5 Artifact Registry + build d’image

```bash
gcloud artifacts repositories create transcriber \
  --repository-format=docker --location=$REGION

gcloud builds submit \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/transcriber/jdr-transcriber:latest .
```

### 4.6 Cloud Run Job (batch)

Création initiale (CPU/8Gi RAM pour `large-v3`, timeout 6h) :

```bash
gcloud run jobs create jdr-transcribe \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/transcriber/jdr-transcriber:latest \
  --region=$REGION \
  --service-account=$SA \
  --cpu=2 --memory=8Gi \
  --task-timeout=21600s \
  --set-secrets=SMTP_PASSWORD=SMTP_PASSWORD:latest \
  --set-env-vars=SMTP_HOST=smtp.gmail.com,SMTP_PORT=587,SMTP_USER=<EMAIL_ENVOI>,EMAIL_FROM="Ektaron Transcriber <EMAIL_ENVOI>" \
  --set-env-vars=ATTACH_LIMIT_MB=20,WHISPER_MODEL_SIZE=large-v3,WHISPER_MODEL_DIR=/models/large-v3,HF_HUB_OFFLINE=1,MULTILANG=true,PRIMARY_LANG=fr,SECONDARY_LANG=en,MAX_CHUNK_MINUTES=15,CHUNK_OVERLAP_MS=2000,VAD_FRAME_MS=20
```

Mise à jour ultérieure (ex. nouvelle image ou nouveau modèle) :

```bash
gcloud run jobs update jdr-transcribe \
  --region=$REGION \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/transcriber/jdr-transcriber:latest \
  --cpu=2 --memory=8Gi \
  --task-timeout=21600s \
  --set-env-vars=WHISPER_MODEL_SIZE=large-v3,WHISPER_MODEL_DIR=/models/large-v3,HF_HUB_OFFLINE=1
```

---

## 5) Lancer un batch depuis le repo

Le script `run_batch_gcp.sh` fait l’upload et lance le job avec des **overrides d’exécution**.

Contenu recommandé :

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION=${REGION:-europe-west1}
JOB=${JOB:-jdr-transcribe}

IN_BUCKET="gs://${PROJECT_ID}-audio-in"
OUT_BUCKET="${PROJECT_ID}-audio-out"

SESSION="${1:-session-$(date +%Y%m%d-%H%M%S)}"
INPUT_DIR="${2:-input}"
CONTEXT_PATH="${3:-context/context.yaml}"
EMAIL_TO="${4:-destinataire@example.com}"

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
  --wait

echo "OK. Le job est terminé côté GCP."
```

Utilisation :

```bash
chmod +x run_batch_gcp.sh
./run_batch_gcp.sh "session-01" "./input" "./context/context.yaml" "<EMAIL_DESTINATAIRE>"
```

Le job :

* lit `gs://<PROJECT_ID>-audio-in/sessions/session-01/...`
* écrit `gs://<PROJECT_ID>-audio-out/sessions/session-01/...`
* envoie un email (pièce jointe si ≤ `ATTACH_LIMIT_MB`, sinon URL signée valide 7 jours)

---

## 6) Logs, vérifications et nettoyage

Vérifier la dernière exécution :

```bash
gcloud run jobs executions list --job jdr-transcribe --region $REGION
gcloud run jobs executions describe <EXEC_NAME> --region $REGION
```

Lire des logs récents :

```bash
gcloud logging read \
  'resource.type=cloud_run_job AND resource.labels.job_name=jdr-transcribe' \
  --limit 100
```

Supprimer le Job (optionnel) :

```bash
gcloud run jobs delete jdr-transcribe --region $REGION
```

Supprimer l’image (optionnel) :

```bash
gcloud artifacts docker images delete \
  $REGION-docker.pkg.dev/$PROJECT_ID/transcriber/jdr-transcriber:latest \
  --delete-tags --quiet
```

Politique de rétention automatique des sorties (optionnel) :

```bash
cat > lifecycle.json << 'JSON'
{
  "rule": [
    { "action": { "type": "Delete" }, "condition": { "age": 30 } }
  ]
}
JSON

gcloud storage buckets update "gs://${PROJECT_ID}-audio-out" --lifecycle-file=lifecycle.json
```

---

## 7) Conseils coûts & perfs

* **Modèle** : `large-v3` donne le meilleur rendu, au prix d’une image plus lourde et plus de RAM. `medium` diminue le coût ; basculer se fait via `--set-env-vars=WHISPER_MODEL_SIZE=medium` et en reconstruisant une image si tu veux embarquer `medium` au lieu de `large-v3`.
* **Ressources** : `--cpu=2 --memory=8Gi` fonctionne bien pour `large-v3` en `int8`. Augmenter les vCPU accélère le run mais coûte un peu plus.
* **Stockage** : garde une seule image `:latest`. Active la lifecycle sur le bucket OUT pour supprimer les archives au bout de X jours.

---

## 8) FAQ rapide

**Q. Le job échoue avec “Config manquante: INPUT_PREFIX/OUTPUT_BUCKET/EMAIL_TO/SMTP_USER requis.”**
R. `SMTP_USER` est défini au niveau du Job (via `jobs create/update`). Les trois autres doivent être passés à l’exécution avec `--update-env-vars=INPUT_PREFIX=...,OUTPUT_BUCKET=...,OUTPUT_PREFIX=...,EMAIL_TO=...`.

**Q. Erreur `LocalEntryNotFoundError` / “outgoing traffic disabled” côté HuggingFace.**
R. Le modèle est **embarqué** dans l’image et chargé via `WHISPER_MODEL_DIR`. Vérifie que `process.py` utilise bien le chemin local et que l’image contient `/models/large-v3`.

**Q. Sous Windows Git Bash, `gsutil` casse.**
R. Utilise `gcloud storage` (déjà fait dans `run_batch_gcp.sh`).

---

## 9) Ancrage minimal du contexte (optionnel)

`context/context.yaml` peut contenir : speakers connus (id, nicknames, échantillons audio facultatifs), glossaire (personnages, lieux, termes), overrides pour renommer les `SPKx`, etc. Le pipeline les utilisera pour améliorer le rendu et mapper les voix aux noms.

Exemple ultra-court :

```yaml
speakers:
  - id: Alice
    nicknames: [Ali]
    voice_samples:
      - voices/alice_01.wav
      - voices/alice_02.wav
  - id: Bob

glossary:
  characters: [Ektaron, Silar]
  places: [Citadelle de Verre]
  terms: [Portail, Éclat]

overrides:
  speakers:
    SPK1: Alice
    SPK2: Bob
```

---

## 10) Résumé express

* Tu **buildes** l’image Docker avec le modèle **déjà dedans**.
* Tu crées **deux buckets** privés (`-audio-in` et `-audio-out`), un **service account** avec droits minimaux, un **secret SMTP**.
* Tu crées un **Cloud Run Job** avec quelques env vars “fixes” (SMTP, modèle, etc.).
* Tu lances un batch via `run_batch_gcp.sh` qui **sync** `input/` + `context.yaml` vers `-audio-in` et **execute** le job avec les env “dynamiques”.
* Tu reçois le résultat **par email** (pièce jointe ≤ 20 Mo, sinon **lien signé**), et les sorties restent aussi sur `-audio-out`.

Tout est **stateless** et **offline** côté modèle ; aucune donnée privée n’atterrit sur un repo public.

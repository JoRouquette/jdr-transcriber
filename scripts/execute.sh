#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# execute.sh — Relance un Cloud Run Job en réutilisant ce qui est DÉJÀ sur GCP
# Options:
#   --pick-session        : propose une liste des sessions présentes dans gs://<PROJECT>-audio-in/sessions/
#   --pick-context        : propose la liste des contextes (YAML/YML) dispo pour la session choisie
#   --list-sessions       : affiche les sessions détectées et quitte
#   --list-contexts <S>   : affiche les contextes pour la session S et quitte
#   --email <addr>        : destinataire (sinon demande interactivement si absent)
#   --session <name>      : nom de session à relancer (sinon auto ou --pick-session)
#   --context <gs://...>  : URI GCS du contexte (sinon auto/--pick-context)
#   --dry-run             : affiche ce qui serait exécuté, sans lancer
#
# Usage minimal (interaction pour choisir la session):
#   ./execute.sh --pick-session --pick-context
#
# Usage direct (sans interaction):
#   ./execute.sh --session session-20250101-120000 --email destinataire@example.com
#
# Requiert: gcloud (auth + projet sélectionné). fzf (optionnel).
# ---------------------------------------------------------------------------

set -euo pipefail

REGION=${REGION:-europe-west1}
JOB=${JOB:-jdr-transcribe}
PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
[[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]] && { echo "Aucun projet GCP (gcloud config set project …)"; exit 1; }

IN_BUCKET="gs://${PROJECT_ID}-audio-in"
OUT_BUCKET="${PROJECT_ID}-audio-out"

PICK_SESSION="no"
PICK_CONTEXT="no"
LIST_SESSIONS="no"
LIST_CONTEXTS_FOR=""
EMAIL_TO=""
SESSION=""
CONTEXT_URI=""
DRYRUN="no"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'

Options:
  --pick-session        : propose une liste des sessions présentes dans gs://<PROJECT>-audio-in/sessions/
  --pick-context        : propose la liste des contextes (YAML/YML) dispo pour la session choisie
  --list-sessions       : affiche les sessions détectées et quitte
  --list-contexts <S>   : affiche les contextes pour la session S et quitte
  --email <addr>        : destinataire (sinon demande interactivement si absent)
  --session <name>      : nom de session à relancer (sinon auto ou --pick-session)
  --context <gs://...>  : URI GCS du contexte (sinon auto/--pick-context)
  --dry-run             : affiche ce qui serait exécuté, sans lancer
Usage minimal (interaction pour choisir la session):
  ./execute.sh --pick-session --pick-context
Usage direct (sans interaction):
  ./execute.sh --session session-20250101-120000 --email destinataire@example.com
Requiert: gcloud (auth + projet sélectionné). fzf (optionnel).

Exemples:
  ./execute.sh --pick-session --pick-context
  ./execute.sh --session session-20250101-120000 --email destinataire@example.com

Licence: GNU GPL-3.0. Voir le fichier LICENSE.md.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
EOF
  exit 0
fi

# --------- parsing simple des args ---------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pick-session)   PICK_SESSION="yes"; shift;;
    --pick-context)   PICK_CONTEXT="yes"; shift;;
    --list-sessions)  LIST_SESSIONS="yes"; shift;;
    --list-contexts)  LIST_CONTEXTS_FOR="${2:-}"; shift 2;;
    --email)          EMAIL_TO="${2:-}"; shift 2;;
    --session)        SESSION="${2:-}"; shift 2;;
    --context)        CONTEXT_URI="${2:-}"; shift 2;;
    --dry-run)        DRYRUN="yes"; shift;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'; exit 0;;
    *)
      echo "Arg inconnu: $1 (use -h)"; exit 1;;
  esac
done

# --------- Logging ---------
if [ "$DRYRUN" == "yes" ]; then
  echo ">> Mode dry-run activé (aucune action ne sera effectuée)"
  echo ">> Projet GCP: $PROJECT_ID"
  echo ">> Bucket d'entrée: $IN_BUCKET"
  echo ">> Bucket de sortie: $OUT_BUCKET"
  echo ">> Email destinataire: $EMAIL_TO"
  echo ">> URI du contexte: $CONTEXT_URI"
  echo ">> Job Cloud Run: $JOB (région: $REGION)"
  echo 
  echo ">> Paramètres :"
  echo "   PICK_SESSION=$PICK_SESSION"
  echo "   PICK_CONTEXT=$PICK_CONTEXT"
  echo "   LIST_SESSIONS=$LIST_SESSIONS"
  echo "   LIST_CONTEXTS_FOR=$LIST_CONTEXTS_FOR"
  echo "   EMAIL_TO=$EMAIL_TO"
  echo "   SESSION=$SESSION"
  echo "   CONTEXT_URI=$CONTEXT_URI"
  echo "   DRYRUN=$DRYRUN"
  echo ">> ---------------------------------------------------"
fi

# --------- helpers ---------
have_fzf(){ command -v fzf >/dev/null 2>&1; }

list_sessions() {
  # extrait les noms de sessions (1er niveau sous sessions/)
  gcloud storage ls --recursive "${IN_BUCKET}/sessions/**" \
    | awk -F'sessions/' '/sessions\//{split($2,a,"/"); if(a[1]!="") print a[1]}' \
    | sort -u
}

list_contexts_for_session() {
  local s="$1"
  # liste YAML/YML dans context/ (si le dossier existe)
  gcloud storage ls "${IN_BUCKET}/sessions/${s}/context/**" 2>/dev/null \
    | awk '/\.ya?ml$/ {print $0}'
}

pick_from_list() {
  # $1: prompt, lit la liste sur stdin
  local prompt="$1"
  if have_fzf; then
    fzf --prompt="$prompt> " --height=20 --border
  else
    # menu numéroté
    mapfile -t LST
    [[ ${#LST[@]} -eq 0 ]] && return 1
    local i=1
    for it in "${LST[@]}"; do printf "%2d) %s\n" "$i" "$it"; ((i++)); done
    read -rp "$prompt (numéro)> " idx
    [[ "$idx" =~ ^[0-9]+$ ]] || return 1
    (( idx>=1 && idx<=${#LST[@]} )) || return 1
    echo "${LST[$((idx-1))]}"
  fi
}

# --------- modes “list-only” ---------
if [[ "$LIST_SESSIONS" == "yes" ]]; then
  echo "Sessions dans ${IN_BUCKET}/sessions/:"
  list_sessions || true
  exit 0
fi

if [[ -n "$LIST_CONTEXTS_FOR" ]]; then
  echo "Contextes pour session '${LIST_CONTEXTS_FOR}':"
  list_contexts_for_session "$LIST_CONTEXTS_FOR" || true
  exit 0
fi

# --------- choisir session si demandé ou absente ---------
if [[ -z "$SESSION" && "$PICK_SESSION" == "yes" ]]; then
  SESS_CHOICE="$(list_sessions | pick_from_list "Choisir la session")" || { echo "Aucune session sélectionnée."; exit 1; }
  SESSION="$SESS_CHOICE"
fi
# fallback si toujours vide: propose le plus “récent” (par dernier objet)
if [[ -z "$SESSION" ]]; then
  SESSION="$(list_sessions | tail -n1 || true)"
  [[ -z "$SESSION" ]] && { echo "Aucune session détectée dans ${IN_BUCKET}/sessions/."; exit 1; }
  echo "Session non fournie, utilisation de la plus récente détectée: ${SESSION}"
fi

INPUT_PREFIX="${IN_BUCKET}/sessions/${SESSION}"
OUT_PREFIX="sessions/${SESSION}"

# --------- choisir contexte si demandé ou auto ---------
if [[ -z "$CONTEXT_URI" && "$PICK_CONTEXT" == "yes" ]]; then
  CTX="$(list_contexts_for_session "$SESSION" | pick_from_list "Choisir le contexte (ou ESC pour aucun)")" || CTX=""
  CONTEXT_URI="$CTX"
fi
# auto-pick si un seul YAML existe
if [[ -z "$CONTEXT_URI" ]]; then
  ONE_CTX="$(list_contexts_for_session "$SESSION" | head -n1 || true)"
  MORE_CTX="$(list_contexts_for_session "$SESSION" | sed -n '2p' || true)"
  if [[ -n "$ONE_CTX" && -z "$MORE_CTX" ]]; then
    CONTEXT_URI="$ONE_CTX"
    echo "Contexte unique détecté: ${CONTEXT_URI}"
  fi
fi

# --------- email destinataire ---------
if [[ -z "$EMAIL_TO" ]]; then
  read -rp "EMAIL_TO (destinataire): " EMAIL_TO
  [[ -z "$EMAIL_TO" ]] && { echo "EMAIL_TO requis."; exit 1; }
fi

# --------- exécution ---------
CMD=( gcloud run jobs execute "$JOB" --region "$REGION"
      --update-env-vars "INPUT_PREFIX=${INPUT_PREFIX},OUTPUT_BUCKET=${OUT_BUCKET},OUTPUT_PREFIX=${OUT_PREFIX},EMAIL_TO=${EMAIL_TO},CONTEXT_GCS_URI=${CONTEXT_URI}" )

echo "Job   : $JOB  (region=$REGION, project=$PROJECT_ID)"
echo "Input : $INPUT_PREFIX"
echo "Output: gs://${OUT_BUCKET}/${OUT_PREFIX}"
[[ -n "$CONTEXT_URI" ]] && echo "Contexte: $CONTEXT_URI" || echo "Contexte: (aucun)"
echo "Email : $EMAIL_TO"

if [[ "$DRYRUN" == "yes" ]]; then
  echo "[dry-run] ${CMD[*]}"
  exit 0
fi

"${CMD[@]}" --wait
echo "OK. Exécution terminée."

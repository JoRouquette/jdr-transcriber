# jdr-transcriber

Transcripteur + diarisation “batch, fire-and-forget” pour vos sessions audio.
Pipeline CPU basé sur `faster-whisper` avec modèle embarqué, exécution sur **Cloud Run Jobs**, entrées/sorties sur **Cloud Storage**, envoi d’un **ZIP** par e-mail.

## Liens essentiels

* 👉 Guide de déploiement complet : [DEPLOYMENT-GUIDE.md](DEPLOYMENT-GUIDE.md)
* 👉 Création d’empreintes vocales (scripts à lire) : [context/VOICEPRINT.MD](context/VOICEPRINT.MD)

## Aperçu rapide

Vous uploadez vos fichiers audio (et le contexte optionnel) dans un bucket privé.
Un **Cloud Run Job** tire les fichiers, exécute `app/process.py`, produit SRT/TXT/MD/JSON, zippe le tout et envoie un e-mail (pièce jointe si ≤ 20 Mo, sinon lien signé temporaire).
Le job s’arrête automatiquement. Aucune VM à gérer.

## Arborescence

```
.
├─ app/
│  └─ process.py
├─ context/
│  ├─ context.yaml           # contexte optionnel (speakers, glossaire, overrides…)
│  ├─ VOICEPRINT.MD          # guide pour enregistrer des “voice prints”
│  └─ voices/                # placez ici les échantillons de voix (.wav/.mp3)
│     ├─ alice_01.wav
│     ├─ alice_02.wav
│     └─ bob_01.wav
├─ input/                    # déposer vos audios (.mp3, .m4a, .wav…)
├─ output/                   # (rempli en local uniquement)
├─ entrypoint_gcp.py         # orchestration batch (GCS in/out + e-mail)
├─ Dockerfile                # image CPU avec modèle embarqué
├─ requirements.txt
└─ run_batch_gcp.sh          # lance un batch (upload + exécution du Job)
```

Notes contexte/voix
Le champ `voice_samples` de `context.yaml` accepte des **chemins relatifs** depuis `context/`, par exemple :

```yaml
speakers:
  - id: Alice
    voice_samples:
      - voices/alice_01.wav
      - voices/alice_02.wav
  - id: Bob
    voice_samples:
      - voices/bob_01.wav
```

## Démarrage express

1. Prérequis: `gcloud` installé, projet sélectionné, facturation active.
2. Construire et publier l’image (ou suivez le guide détaillé) :

```bash
REGION=europe-west1
PROJECT_ID=<votre_project_id>
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/transcriber/jdr-transcriber:latest .
```

3. Créer/mettre à jour le **Cloud Run Job** (voir [DEPLOYMENT-GUIDE.md](./DEPLOYMENT-GUIDE.md) pour secrets SMTP, buckets et rôles).
4. Lancer un traitement depuis la racine du repo :

```bash
./run_batch_gcp.sh "session-01" "./input" "./context/context.yaml" "destinataire@example.com"
```

## Licence

GNU GPL-3.0. Voir le fichier [LICENSE](./LICENSE.md).

# entrypoint_gcp.py
import os, sys, shutil, smtplib, mimetypes, zipfile, time
from email.message import EmailMessage
from pathlib import Path
from google.cloud import storage, secretmanager

# ---------- Inputs attendus (env overrides à l'exécution du Job) ----------
# gs://bucket/prefix/  (le dossier de tes audios)
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "").rstrip("/") + "/"
# bucket GCS pour les résultats
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "")
# sous-dossier de sortie (ex: session/05-2025-10-07)
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "batch")
# contexte optionnel (fichier YAML/JSON)
CONTEXT_GCS_URI = os.getenv("CONTEXT_GCS_URI", "")

# Email
EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", os.getenv("SMTP_USER", ""))
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv(
    "SMTP_PASSWORD", ""
)  # injecté via Secret Manager dans la conf du Job
ATTACH_LIMIT = int(os.getenv("ATTACH_LIMIT_MB", "20")) * 1024 * 1024  # 20 Mo par défaut

# Whisper/env déjà gérés par ton app/process.py via variables d'env
# --------------------------------------------------------------------------

ALLOWED = {".mp3", ".m4a", ".wav", ".mp4", ".mpeg", ".mpga", ".ogg", ".webm"}
TMP = Path("/tmp")
IN_DIR = TMP / "input"
OUT_DIR = TMP / "output"
CTX_LOCAL = TMP / "context"
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CTX_LOCAL.mkdir(exist_ok=True)


def log(msg):
    print(msg, flush=True)


def gcs_client():
    return storage.Client()


def list_blobs_with_prefix(client, prefix):
    if not prefix.startswith("gs://"):
        raise ValueError("INPUT_PREFIX doit être un gs://bucket/prefix")
    _, _, rest = prefix.partition("gs://")
    bucket_name, _, folder = rest.partition("/")
    bucket = client.bucket(bucket_name)
    return bucket, list(client.list_blobs(bucket, prefix=folder))


def download_to(client, blob, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))


def upload_dir_recursive(client, bucket_name, src, dest_prefix):
    bucket = client.bucket(bucket_name)
    for p in Path(src).rglob("*"):
        if p.is_file():
            rel = p.relative_to(src).as_posix()
            blob = bucket.blob(f"{dest_prefix.rstrip('/')}/{rel}")
            blob.upload_from_filename(str(p))


def zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(src_dir).rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(src_dir).as_posix())


def send_email(subject, body, attachments=None):
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    for path in attachments or []:
        ctype, _ = mimetypes.guess_type(path)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(
                f.read(), maintype=maintype, subtype=subtype, filename=Path(path).name
            )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


def signed_url_for(bucket, object_name, seconds=7 * 24 * 3600):
    # URL signée V4
    client = gcs_client()
    blob = client.bucket(bucket).blob(object_name)
    return blob.generate_signed_url(version="v4", expiration=seconds, method="GET")


def main():
    if not INPUT_PREFIX or not OUTPUT_BUCKET or not EMAIL_TO or not SMTP_USER:
        log("Config manquante: INPUT_PREFIX/OUTPUT_BUCKET/EMAIL_TO/SMTP_USER requis.")
        sys.exit(2)

    client = gcs_client()

    # 1) Télécharger les audios
    bucket_in, blobs = list_blobs_with_prefix(client, INPUT_PREFIX)
    audio_blobs = [b for b in blobs if Path(b.name).suffix.lower() in ALLOWED]
    if not audio_blobs:
        log("Aucun audio trouvé sous INPUT_PREFIX")
        sys.exit(1)

    for b in audio_blobs:
        dest = IN_DIR / Path(b.name).name
        log(f"DL {b.name} -> {dest}")
        download_to(client, b, dest)

    # 2) Télécharger le contexte éventuel
    ctx_local_path = None
    if CONTEXT_GCS_URI:
        _, _, rest = CONTEXT_GCS_URI.partition("gs://")
        buck, _, key = rest.partition("/")
        b = client.bucket(buck).blob(key)
        ctx_local_path = CTX_LOCAL / Path(key).name
        log(f"DL contexte {key} -> {ctx_local_path}")
        download_to(client, b, ctx_local_path)

    # 3) Lancer le traitement
    cmd = [
        "python",
        "/app/app/process.py",
        "--input",
        str(IN_DIR),
        "--output",
        str(OUT_DIR),
    ]
    if ctx_local_path:
        cmd += ["--context", str(ctx_local_path)]
    log(f"RUN: {' '.join(cmd)}")
    rc = os.spawnvp(os.P_WAIT, cmd[0], cmd)
    if rc != 0:
        log(f"process.py a échoué (rc={rc})")
        sys.exit(rc)

    # 4) Upload des sorties + ZIP
    out_prefix = f"{OUTPUT_PREFIX.rstrip('/')}/{Path(INPUT_PREFIX).name}"
    upload_dir_recursive(client, OUTPUT_BUCKET, OUT_DIR, out_prefix)

    zip_path = TMP / "transcripts.zip"
    zip_dir(OUT_DIR, zip_path)
    zip_blob = f"{out_prefix}/transcripts.zip"
    client.bucket(OUTPUT_BUCKET).blob(zip_blob).upload_from_filename(str(zip_path))

    # 5) Envoi email (pièce jointe si zip <= ATTACH_LIMIT, sinon URL signée)
    size = zip_path.stat().st_size
    if size <= ATTACH_LIMIT:
        send_email(
            subject=f"[Transcripts] {Path(INPUT_PREFIX).name}",
            body="Résultats en pièce jointe.",
            attachments=[str(zip_path)],
        )
    else:
        url = signed_url_for(OUTPUT_BUCKET, zip_blob, seconds=7 * 24 * 3600)
        send_email(
            subject=f"[Transcripts] {Path(INPUT_PREFIX).name}",
            body=f"Zip trop volumineux pour une pièce jointe.\nTéléchargement (7 jours):\n{url}\n",
        )

    log("Terminé.")
    # Cloud Run Job s'arrête ici; pas de “conteneur à supprimer”.
    # (Optionnel: on peut aussi supprimer /tmp/* pour libérer l’espace éphémère)
    try:
        shutil.rmtree(IN_DIR, ignore_errors=True)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        shutil.rmtree(CTX_LOCAL, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink()
    except:
        pass


if __name__ == "__main__":
    main()

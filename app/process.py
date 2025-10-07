import os, re, json, wave, contextlib, argparse, subprocess
from pathlib import Path
from datetime import timedelta
import yaml
import warnings
import numpy as np
import webrtcvad
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel

try:
    _ = np.bool
except AttributeError:
    np.bool = np.bool_

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*np\.bool.*")

# ---------- utilitaires temps / ffmpeg ----------
def hhmmss(ms):
    td = timedelta(milliseconds=int(ms)); s = int(td.total_seconds())
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d},{int(ms - s*1000):03d}"

def run(cmd):
    subprocess.run(cmd, check=True)

def convert_to_wav16k_mono(src, dst):
    run(["ffmpeg","-y","-i",src,"-ac","1","-ar","16000","-sample_fmt","s16",dst,"-loglevel","error"])

def get_wav_duration_ms(wav_path):
    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        frames = wf.getnframes(); rate = wf.getframerate()
    return int(frames * 1000 / rate)

def file_size_mb(path):
    return os.path.getsize(path) / 1e6

def split_wav_ffmpeg(in_wav, chunk_ms, overlap_ms):
    """Retourne une liste [(chunk_path, offset_ms), ...]"""
    out = []
    dur = get_wav_duration_ms(in_wav)
    if dur <= chunk_ms:  # pas besoin
        return [(in_wav, 0)]
    # planning des débuts avec recouvrement
    step = max(1000, chunk_ms - overlap_ms)
    starts = list(range(0, dur, step))
    base = Path(in_wav).with_suffix("").name
    tmpdir = Path(in_wav).parent
    for i, start in enumerate(starts):
        end = min(start + chunk_ms, dur)
        out_wav = tmpdir / f"{base}__chunk{i:03d}.wav"
        # découpe exacte sans réencodage additionnel
        run([
            "ffmpeg","-y","-i",in_wav,
            "-ss", f"{start/1000:.3f}",
            "-t",  f"{(end-start)/1000:.3f}",
            "-ac","1","-ar","16000","-sample_fmt","s16",
            str(out_wav), "-loglevel","error"
        ])
        out.append((str(out_wav), start))
        if end == dur: break
    return out

def load_context(path: str | None):
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    if p.suffix.lower() in [".yaml", ".yml"]:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return {}

def build_initial_prompt(ctx: dict, primary_lang: str, secondary_lang: str, repeat: int = 1):
    """
    Construit une phrase amorce pour Whisper avec noms propres et termes clés.
    Répéter les noms aide à la stabilisation orthographique.
    """
    if not ctx: return None
    names = []
    for s in ctx.get("speakers", []):
        names.append(s.get("id",""))
        names += s.get("nicknames", []) or []
    gl = ctx.get("glossary", {})
    terms = (gl.get("characters", []) or []) + (gl.get("places", []) or []) + (gl.get("terms", []) or [])
    npcs = ctx.get("npcs_controlled_by_mj", []) or []
    bag = [n for n in names if n] + terms + npcs
    if not bag: return None

    bag = list(dict.fromkeys([x.strip() for x in bag if x and isinstance(x, str)]))  # déduplique
    bag = sum([[x]*int(max(1, repeat)) for x in bag], [])  # répète

    # Une phrase courte suffit, bilingue si besoin
    return (
        f"Langue principale: {primary_lang}. Langue secondaire: {secondary_lang}. "
        f"Noms propres et termes importants: " + ", ".join(bag) + "."
    )

def load_voiceprints(ctx: dict):
    """
    Charge des empreintes vocales optionnelles par intervenant à partir de fichiers courts (10–30 s).
    Retourne { 'Nom': embedding[np.array] }.
    """
    m = {}
    samples_root = Path("context")
    enc = VoiceEncoder()
    for spk in ctx.get("speakers", []) or []:
        name = spk.get("id")
        paths = spk.get("voice_samples") or []
        embs = []
        for rel in paths:
            f = (samples_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            if not f.exists(): continue
            wav = preprocess_wav(f)
            emb = enc.embed_utterance(wav)
            embs.append(emb)
        if embs:
            m[name] = np.mean(np.vstack(embs), axis=0)
    return m

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def compute_cluster_centroids(wav_path, diar_segments):
    """
    Calcule un embedding centroidal par label SPKx à partir des segments diarizés.
    Retourne { 'SPK1': emb, ... }.
    """
    if not diar_segments: return {}
    wav = preprocess_wav(wav_path)
    sr = 16000
    enc = VoiceEncoder()
    by = {}
    for s,e,lab in diar_segments:
        s_idx = int(s/1000*sr); e_idx = int(e/1000*sr)
        s_idx = max(0, min(s_idx, len(wav))); e_idx = max(0, min(e_idx, len(wav)))
        if e_idx - s_idx < int(0.5*sr):  # ignore trop court
            continue
        seg = wav[s_idx:e_idx]
        by.setdefault(lab, []).append(enc.embed_utterance(seg))
    out = {}
    for lab, lst in by.items():
        if lst:
            out[lab] = np.mean(np.vstack(lst), axis=0)
    return out

def rename_speakers_by_similarity(segs, spk_centroids, voiceprints, threshold=0.76, overrides=None):
    """
    Mappe SPKx -> Nom si similarité cosinus dépasse 'threshold'.
    Respecte les overrides éventuels (SPK1: 'Jonathan', ...).
    """
    if overrides:
        for s in segs:
            if s["speaker"] in overrides:
                s["speaker"] = overrides[s["speaker"]]

    if not spk_centroids or not voiceprints:
        return segs

    used = set()
    # Tri: on nomme d'abord les SPK qui parlent le plus
    talk = {}
    for s in segs:
        talk[s["speaker"]] = talk.get(s["speaker"], 0) + (s["end_ms"] - s["start_ms"])
    labels = sorted(spk_centroids.keys(), key=lambda k: -talk.get(k, 0))

    for lab in labels:
        emb = spk_centroids[lab]
        best_name, best_sim = None, 0.0
        for name, vemb in voiceprints.items():
            if name in used: continue
            sim = cosine_sim(emb, vemb)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_name and best_sim >= threshold:
            for s in segs:
                if s["speaker"] == lab:
                    s["speaker"] = best_name
            used.add(best_name)
    return segs

# ---------- VAD / diarisation / transcription ----------
def read_pcm_frames(wav_path, frame_ms=30):
    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000
        rate = wf.getframerate()
        frame_bytes = int(rate * (frame_ms/1000.0) * 2)  # 2 bytes (s16)
        pcm = wf.readframes(wf.getnframes())
    # On ne garde que des frames de taille exacte
    usable = len(pcm) - (len(pcm) % frame_bytes)
    frames = [pcm[i:i+frame_bytes] for i in range(0, usable, frame_bytes)]
    return frames, rate, frame_ms

def vad_segments(wav_path, aggressiveness=2, frame_ms=30, pad_ms=150):
    vad = webrtcvad.Vad(aggressiveness)
    frames, rate, frame_ms = read_pcm_frames(wav_path, frame_ms)
    frame_bytes = int(rate * (frame_ms/1000.0) * 2)
    speech = []
    cur_start = None
    ms = 0
    pad_frames = int(pad_ms/frame_ms)
    consec_sil = 0

    for fr in frames:
        if len(fr) != frame_bytes:
            ms += frame_ms
            continue
        try:
            is_speech = vad.is_speech(fr, rate)
        except webrtcvad.Error:
            # On ignore la frame et on avance proprement
            ms += frame_ms
            continue

        if is_speech and cur_start is None:
            cur_start = ms
            consec_sil = 0
        elif not is_speech and cur_start is not None:
            consec_sil += 1
            if consec_sil >= pad_frames:
                end = ms + frame_ms * consec_sil
                speech.append((cur_start, end))
                cur_start = None
                consec_sil = 0
        else:
            consec_sil = 0 if is_speech else consec_sil
        ms += frame_ms

    if cur_start is not None:
        speech.append((cur_start, ms))

    # fusion de segments proches
    merged = []
    for s,e in speech:
        if not merged or s - merged[-1][1] > 200:
            merged.append([s,e])
        else:
            merged[-1][1] = e
    return [(s,e) for s,e in merged]

def sliding_windows(segment, win_ms=1600, hop_ms=800):
    s,e = segment; cur=s
    while cur+win_ms <= e:
        yield (cur, cur+win_ms); cur += hop_ms

def embed_segments(wav_path, segments_ms):
    wav = preprocess_wav(wav_path); sr=16000
    enc = VoiceEncoder(); embs=[]; spans=[]
    for (s,e) in segments_ms:
        s_idx=int(s/1000*sr); e_idx=int(e/1000*sr); e_idx=min(e_idx,len(wav))
        if e_idx - s_idx < int(0.5*sr): continue
        seg = wav[s_idx:e_idx]; emb = enc.embed_utterance(seg)
        embs.append(emb); spans.append((s,e))
    return (np.vstack(embs) if embs else np.empty((0,256))), spans

def choose_k(embeddings, min_k=1, max_k=8):
    n=len(embeddings); 
    if n<=1: return 1
    best_k, best = 2, -1
    for k in range(2, min(max_k,n)+1):
        cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = cl.fit_predict(embeddings)
        try: sc = silhouette_score(embeddings, labels, metric='cosine')
        except: sc = -1
        if sc>best: best, best_k = sc, k
    return 1 if best < 0.12 else best_k

def diarize(wav_path, speech_spans, min_k=1, max_k=8):
    wins=[]; [wins.extend(list(sliding_windows(x))) for x in speech_spans]
    if not wins: return []
    embs, spans = embed_segments(wav_path, wins)
    if len(embs)==0: return [(s,e,"SPK1") for s,e in speech_spans]
    k = choose_k(embs, min_k, max_k)
    cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    labels = cl.fit_predict(embs)
    by_spk={}
    for (s,e),lab in zip(spans, labels):
        spk=f"SPK{lab+1}"; by_spk.setdefault(spk, []).append((s,e))
    out=[]
    for spk,lst in by_spk.items():
        lst.sort(); merged=[]
        for s,e in lst:
            if not merged or s-merged[-1][1] > 400: merged.append([s,e])
            else: merged[-1][1]=max(merged[-1][1], e)
        out += [(s,e,spk) for s,e in merged]
    out.sort()
    return out

def intersect(a0,a1,b0,b1): return max(0, min(a1,b1) - max(a0,b0))

def transcribe_words(model, wav_path, language=None, initial_prompt=None, beam_size=5, best_of=5):
    """
    language: None => auto (multilingue). "fr" => force français.
    initial_prompt: amorce de contexte (noms propres, lieux, jargon).
    """
    segments, _ = model.transcribe(
        wav_path,
        word_timestamps=True,
        vad_filter=False,
        language=language,                 # None = auto
        initial_prompt=initial_prompt,
        beam_size=int(beam_size) if int(beam_size) > 0 else None,
        best_of=int(best_of) if int(best_of) > 0 else None,
        condition_on_previous_text=True,
        temperature=0.0                    # plus strict, moins d'inventions
    )
    words=[]
    for seg in segments:
        if not seg.words: continue
        for w in seg.words:
            if w.start is None or w.end is None: continue
            words.append({"word": w.word.strip(), "start_ms": int(w.start*1000), "end_ms": int(w.end*1000)})
    return words

def assign_words_to_speakers(words, spk_segments):
    spk_segments=sorted(spk_segments, key=lambda x:x[0])
    assigned=[]; last=None
    for w in words:
        w_s, w_e = w["start_ms"], w["end_ms"]; best=0; who=None
        for s,e,spk in spk_segments:
            if e < w_s: continue
            if s > w_e: break
            ov = intersect(w_s,w_e,s,e)
            if ov>best: best, who = ov, spk
        if who is None: who = last or "SPK1"
        last = who; w["speaker"]=who; assigned.append(w)
    return assigned

def words_to_segments(assigned, gap_ms=900):
    segs=[]; 
    if not assigned: return segs
    cur_spk=assigned[0]["speaker"]; cur_start=assigned[0]["start_ms"]
    tokens=[assigned[0]["word"]]; last_end=assigned[0]["end_ms"]
    for w in assigned[1:]:
        if w["speaker"]!=cur_spk or w["start_ms"]-last_end>gap_ms:
            txt=" ".join(tokens).strip()
            if txt: segs.append({"start_ms":cur_start,"end_ms":last_end,"speaker":cur_spk,"text":txt})
            cur_spk=w["speaker"]; cur_start=w["start_ms"]; tokens=[w["word"]]
        else: tokens.append(w["word"])
        last_end=w["end_ms"]
    txt=" ".join(tokens).strip()
    if txt: segs.append({"start_ms":cur_start,"end_ms":last_end,"speaker":cur_spk,"text":txt})
    return segs

def merge_adjacent_segments(segs, gap_ms=900):
    """Fusionne si même speaker et silence court entre deux segments consécutifs (après concat des chunks)."""
    if not segs: return segs
    segs = sorted(segs, key=lambda s: s["start_ms"])
    out=[segs[0]]
    for s in segs[1:]:
        last = out[-1]
        if s["speaker"]==last["speaker"] and s["start_ms"] - last["end_ms"] <= gap_ms:
            last["end_ms"] = max(last["end_ms"], s["end_ms"])
            last["text"] = (last["text"] + " " + s["text"]).strip()
        else:
            out.append(s)
    return out

def reconcile_speakers(prev_segs, new_segs, seam_start_ms, window_ms=5000):
    """
    Aligne les labels de locuteurs d’un chunk avec les précédents,
    en comparant les segments autour de la jointure (recouvrement).
    """
    if not prev_segs or not new_segs: return new_segs
    prev_window = []
    new_window  = []
    for s in prev_segs:
        if s["end_ms"] > seam_start_ms - window_ms and s["end_ms"] <= seam_start_ms:
            prev_window.append(s)
    for s in new_segs:
        if s["start_ms"] >= seam_start_ms and s["start_ms"] < seam_start_ms + window_ms:
            new_window.append(s)
    if not prev_window or not new_window:
        return new_segs

    # calcule les recouvrements cumulés par paire (new_spk -> prev_spk)
    overlap = {}
    for a in new_window:
        for b in prev_window:
            ov = intersect(a["start_ms"], a["end_ms"], b["start_ms"], b["end_ms"])
            if ov <= 0: continue
            overlap.setdefault(a["speaker"], {}).setdefault(b["speaker"], 0)
            overlap[a["speaker"]][b["speaker"]] += ov

    # mapping greedy: pour chaque speaker du chunk, on mappe vers le prev_spk le plus chevauchant
    mapping = {}
    used_prev = set()
    for new_spk, d in overlap.items():
        prev_spk = max(d.items(), key=lambda x: x[1])[0]
        if prev_spk not in used_prev:
            mapping[new_spk] = prev_spk
            used_prev.add(prev_spk)

    # applique le mapping
    for s in new_segs:
        if s["speaker"] in mapping:
            s["speaker"] = mapping[s["speaker"]]
    return new_segs

# ---------- sorties ----------
def save_srt(segs, path):
    with open(path,"w",encoding="utf-8") as f:
        for i,s in enumerate(segs, start=1):
            f.write(f"{i}\n{hhmmss(s['start_ms'])} --> {hhmmss(s['end_ms'])}\n[{s['speaker']}] {s['text']}\n\n")

def save_txt(segs, path):
    with open(path,"w",encoding="utf-8") as f:
        for s in segs: f.write(f"{s['speaker']}: {s['text']}\n")

def save_md(segs, path, title):
    with open(path,"w",encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for s in segs:
            t0=hhmmss(s['start_ms'])
            f.write(f"- **{s['speaker']}** [{t0}] — {s['text']}\n")

# ---------- pipeline par WAV ----------
def process_one_wav(model, wav_path, vad_aggr, min_k, max_k, ctx=None, lang_cfg=None, beam=5, best_of=5, prompt_repeat=1):
    frame_ms = int(os.getenv("VAD_FRAME_MS", "30"))
    speech = vad_segments(wav_path, aggressiveness=vad_aggr, frame_ms=frame_ms)
    diar   = diarize(wav_path, speech, min_k=min_k, max_k=max_k)

    # langue
    language = None if (lang_cfg and lang_cfg.get("multilang")) else (lang_cfg.get("primary") if lang_cfg else None)

    # prompt de contexte
    init_prompt = build_initial_prompt(
        ctx, 
        (lang_cfg or {}).get("primary","fr"),
        (lang_cfg or {}).get("secondary","en"),
        repeat=int(prompt_repeat)
    )

    words  = transcribe_words(
        model, wav_path,
        language=language,
        initial_prompt=init_prompt,
        beam_size=beam,
        best_of=best_of
    )
    assigned = assign_words_to_speakers(words, diar)
    segs = words_to_segments(assigned)
    return segs, diar

def sort_key(p: Path):
    stem = p.stem 

    m = re.search(r"^(.*)\.Session(\d+)\.(\d+)$", stem, re.IGNORECASE)
    if m:
        prefix = m.group(1).lower()
        session = int(m.group(2))
        part    = int(m.group(3))
        # Catégorie 0 = motif reconnu
        return (0, prefix, session, part)

    tokens = re.findall(r'\d+|\D+', stem)
    norm = []
    for t in tokens:
        if t.isdigit():
            norm.append((0, int(t)))        # 0 = numérique
        else:
            norm.append((1, t.lower()))     # 1 = texte

    return (1, tuple(norm))

# ---------- main ----------
def main(indir, outdir):
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Collecte des fichiers audio et tri "naturel" + pattern SessionXX.YY
    files = [Path(indir)/p for p in os.listdir(indir)]
    files = [p for p in files if p.is_file() and p.suffix.lower() in [".mp3",".m4a",".wav",".mp4",".mpeg",".mpga",".ogg",".webm"]]
    files.sort(key=sort_key)
    if not files:
        print("Aucun fichier audio dans input/.")
        return

    # Modèle Whisper (CPU) et paramètres globaux
    model_size   = os.getenv("WHISPER_MODEL_SIZE","small")
    model        = WhisperModel(model_size, device="cpu", compute_type="int8")
    vad_aggr     = int(os.getenv("VAD_AGGRESSIVENESS","2"))
    min_k        = int(os.getenv("MIN_SPEAKERS","1"))
    max_k        = int(os.getenv("MAX_SPEAKERS","8"))

    max_chunk_min = int(os.getenv("MAX_CHUNK_MINUTES","15"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP_MS","3000"))
    max_wav_mb    = float(os.getenv("MAX_WAV_MB","150"))

    # Contexte (facultatif)
    ctx_path_env = os.getenv("CTX_ARG_PATH")  # alimenté par __main__ si --context fourni
    ctx_global   = load_context(ctx_path_env) if ctx_path_env else {}
    voiceprints  = load_voiceprints(ctx_global) if ctx_global else {}

    # Langues & décodage
    lang_cfg = {
        "multilang": os.getenv("MULTILANG","true").lower() == "true",
        "primary":   os.getenv("PRIMARY_LANG","fr"),
        "secondary": os.getenv("SECONDARY_LANG","en"),
    }
    beam           = int(os.getenv("WHISPER_BEAM_SIZE","5"))
    best_of        = int(os.getenv("WHISPER_BEST_OF","5"))
    prompt_repeat  = int(os.getenv("INITIAL_PROMPT_REPEAT","1"))
    sim_threshold  = float(os.getenv("SPEAKER_SIM_THRESHOLD","0.76"))

    for src in tqdm(files, desc="Traitement"):
        base = src.stem
        tmp  = Path(outdir)/f"{base}__16k.wav"
        convert_to_wav16k_mono(str(src), str(tmp))

        # Découpage si nécessaire (par durée OU taille)
        dur_ms     = get_wav_duration_ms(str(tmp))
        need_split = (dur_ms > max_chunk_min*60*1000) or (file_size_mb(str(tmp)) > max_wav_mb)
        chunks     = split_wav_ffmpeg(str(tmp), max_chunk_min*60*1000, chunk_overlap) if need_split else [(str(tmp), 0)]

        all_segs = []
        diar_all = []

        for i, (chunk_wav, offset) in enumerate(chunks):
            segs, diar = process_one_wav(
                model, chunk_wav, vad_aggr, min_k, max_k,
                ctx=ctx_global, lang_cfg=lang_cfg,
                beam=beam, best_of=best_of, prompt_repeat=prompt_repeat
            )

            # Applique l’offset temporel global du chunk
            for s in segs:
                s["start_ms"] += offset
                s["end_ms"]   += offset

            # Conserve la diarisation décalée pour calculer les centroides finaux
            diar_shifted = [(s+offset, e+offset, lab) for (s,e,lab) in diar]
            diar_all.extend(diar_shifted)

            # Réconciliation des labels aux jointures à partir du 2e chunk
            if i > 0:
                seam = offset
                segs = reconcile_speakers(all_segs, segs, seam_start_ms=seam, window_ms=min(5000, chunk_overlap+2000))

            all_segs.extend(segs)

        # Mapping SPKx -> noms (si empreintes vocales fournies)
        overrides = (ctx_global.get("overrides", {}) or {}).get("speakers", {}) if ctx_global else {}
        if voiceprints and diar_all:
            centroids = compute_cluster_centroids(str(tmp), diar_all)
            all_segs  = rename_speakers_by_similarity(all_segs, centroids, voiceprints, threshold=sim_threshold, overrides=overrides)
        elif overrides:
            # Applique juste les overrides explicites si pas de voiceprints
            for s in all_segs:
                if s["speaker"] in overrides:
                    s["speaker"] = overrides[s["speaker"]]

        # Fusion finale pour lisibilité
        all_segs = merge_adjacent_segments(all_segs, gap_ms=900)

        # Sauvegardes
        with open(Path(outdir)/f"{base}.json","w",encoding="utf-8") as f:
            json.dump(all_segs, f, ensure_ascii=False, indent=2)
        save_srt(all_segs, Path(outdir)/f"{base}.srt")
        save_txt(all_segs, Path(outdir)/f"{base}.txt")
        save_md(all_segs, Path(outdir)/f"{base}.md", f"Transcription diarisée — {base}")

        # Nettoyage
        try:
            if need_split:
                for (chunk_wav, _) in chunks:
                    p = Path(chunk_wav)
                    if p.exists() and p != Path(tmp):
                        p.unlink()
            if Path(tmp).exists():
                Path(tmp).unlink()
        except:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--context", required=False, help="Chemin vers context.yaml|json (optionnel)")
    args = ap.parse_args()
    if args.context:
        os.environ["CTX_ARG_PATH"] = args.context
    main(args.input, args.output)

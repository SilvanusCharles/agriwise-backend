"""
KisanVaani Agricultural Advisor — FastAPI Backend
---------------------------------------------------
Endpoints:
  POST /advice          — text input  → advice text output
  POST /advice/voice    — audio file  → advice text + audio output
  GET  /health          — health check for Render

Run locally:
  uvicorn main:app --reload --port 8000

Deploy to Render:
  Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import io
import os
import time
import json
import base64
import tempfile
import numpy as np
import torch
import requests
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from gtts import gTTS
import whisper

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    # dev environment may not have langdetect installed; fallback to English detection
    print("WARNING: langdetect not installed, defaulting language detection to English")
    def detect(text):
        return "en"

try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss not installed, vector search will be disabled")

def smart_translate_to_english(text, user_hint):
    """Detects the language and translates it to English."""
    try:
        # 1. Quick check for Pidgin
        pidgin_indicators = ["dey", "don", "abeg", "wetin", "sabi", "naim", "no be"]
        if any(word in text.lower() for word in pidgin_indicators):
            detected_lang = "pcm"
        else:
            # 2. Use langdetect for ha, yo, ig, en
            detected_lang = detect(text)
        
        # If the user explicitly picked a language other than English, trust them
        final_lang = user_hint if user_hint != "en" else detected_lang
        
        # 3. Translate to English for the model
        translation = GoogleTranslator(source='auto', target='en').translate(text)
        return translation, final_lang
    except Exception as e:
        print(f"Detection/Translation error: {e}")
        return text, "en"

# ── Global Configurations ─────────────────────────────────────────────────────
# This forces PyTorch to use the CPU since we are on the HF free tier
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set this to the folder where your answers_list.npy and model files live. 
# "." means the current root directory. Change it to "./model" if they are inside a folder.
MODEL_PATH = "./kisan_vaani_model"

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="KisanVaani Agricultural Advisor API", version="1.0.0")

# CORS — allow your Next.js frontend (update origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:8080",
                    "http://localhost:5173",
                   "https://mrcahrles00-agriwise-backend.hf.space",
                    #  "https://your-frontend.vercel.app
            ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
# Local path during development. For Render, model is downloaded from HF.
HF_REPO             = os.getenv("HF_REPO", "your-hf-username/kisan-vaani-agricultural-advisor")
MODEL_PATH          = os.getenv("MODEL_PATH", "/app/kisan_vaani_model")   # local fallback
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_INFERENCE_TOKEN  = os.getenv("HF_INFERENCE_TOKEN", None)
USE_HF_FALLBACK     = os.getenv("USE_HF_FALLBACK", "true").lower() in ["1", "true", "yes"]
FALLBACK_THRESHOLD  = float(os.getenv("FALLBACK_THRESHOLD", "0.4"))
FAISS_INDEX_PATH    = Path(MODEL_PATH) / "faiss.index"

# ── Supported languages ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en" : "English",
    "ha" : "Hausa",
    "ig" : "Igbo",
    "yo" : "Yoruba",
    "pcm": "Nigerian Pidgin",
}

GOOGLE_LANGS = {"ha", "ig", "yo"}

# gTTS language codes (Pidgin falls back to English TTS)
GTTS_LANG_MAP = {
    "en" : "en",
    "ha" : "ha",
    "ig" : "ig",
    "yo" : "yo",
    "pcm": "en",   # no Pidgin TTS available — use English voice
}

# Whisper language hints for STT
WHISPER_LANG_MAP = {
    "en" : "en",
    "ha" : "ha",
    "ig" : "ig",
    "yo" : "yo",
    "pcm": "en",
}

# ── Pidgin ↔ English dictionary ───────────────────────────────────────────────
PIDGIN_TO_ENGLISH = {
    "abeg": "please", "na": "is", "dem": "they", "im": "it", "e": "it",
    "dey": "is", "no": "not", "di": "the", "dis": "this", "dat": "that",
    "wey": "that", "wetin": "what", "wen": "when", "make": "let",
    "go": "will", "don": "has", "fit": "can", "wan": "want", "get": "have",
    "take": "use", "talk": "say", "sabi": "know", "oga": "sir",
    "wahala": "problem", "beta": "better", "plenty": "many", "small": "little",
    "quick": "quickly", "spoil": "spoil", "sick": "diseased",
    # Agricultural terms
    "farm": "farm", "soil": "soil", "plant": "plant", "leaf": "leaf",
    "leaves": "leaves", "root": "root", "seed": "seed", "water": "water",
    "rain": "rain", "dry": "dry", "wet": "wet", "yellow": "yellow",
    "brown": "brown", "rot": "rot", "die": "die", "disease": "disease",
    "pest": "pest", "insect": "insect", "worm": "worm", "weed": "weed",
    "fertilizer": "fertilizer", "manure": "manure", "spray": "spray",
    "harvest": "harvest", "season": "season", "irrigation": "irrigation",
    "flood": "flood", "drought": "drought", "maize": "maize", "corn": "corn",
    "cassava": "cassava", "yam": "yam", "rice": "rice", "millet": "millet",
    "tomato": "tomato", "pepper": "pepper", "okro": "okra",
    "plantain": "plantain", "cocoa": "cocoa", "wheat": "wheat",
    "nitrogen": "nitrogen", "phosphorus": "phosphorus", "potassium": "potassium",
    "rain season": "rainy season", "dry season": "dry season",
}

ENGLISH_TO_PIDGIN = {
    eng: pid for pid, eng in PIDGIN_TO_ENGLISH.items() if pid != eng
}

# ── Model loading (cached — loads once on startup) ────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    """Load tokenizer and model — tries local path first, then HF Hub."""
    path = MODEL_PATH if Path(MODEL_PATH).exists() else HF_REPO
    print(f"Loading model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")
    return tokenizer, model

@lru_cache(maxsize=1)
def load_whisper():
    """Load Whisper STT model — 'base' is fast and small enough for Render free tier."""
    print("Loading Whisper STT model …")
    return whisper.load_model("base")

@lru_cache(maxsize=1)
def load_embedding_model():
    """Load sentence-transformers embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers is required for efficient indexing. pip install sentence-transformers") from e

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def build_vector_index():
    """Build FAISS index from knowledge base entries."""
    kb_entries = load_knowledge_base()
    if not kb_entries:
        return None, []

    if faiss is None:
        print("FAISS not installed; skipping vector index.")
        return None, kb_entries

    try:
        embed_model = load_embedding_model()
    except Exception as e:
        print(f"Embedding model unavailable: {e}")
        return None, kb_entries

    embeddings = np.array(embed_model.encode(kb_entries, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
    except Exception as e:
        print(f"Failed to save FAISS index: {e}")

    return index, kb_entries

@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def load_knowledge_base():
    """
    Load the knowledge base from the saved answers_list.npy if available,
    otherwise use a fallback set of answers from the HF dataset.
    """
    kb_path = Path(MODEL_PATH) / "answers_list.npy"
    if kb_path.exists():
       answers = np.load(str(kb_path), allow_pickle=True).tolist()
       print(f"Knowledge base loaded: {len(answers)} entries.")
        # FIX: The current code returns a generator. Add list(...) to materialize it.
        # This relates back to the clean-up we talked about to stop 1-word answers.
       return list(str(a) for a in answers)

    # Fallback: load directly from HF dataset (requires datasets library)
    try:
        import pandas as pd
        df = pd.read_parquet(
            "hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet"
        )
        answers = df["answers"].dropna().tolist()
        # NEW: Filter the fallback dataset too
        valid_answers = [str(a) for a in answers if len(str(a).split()) >= 5]
        print(f"Cleaned Knowledge base loaded from HF dataset: {len(valid_answers)} entries.")
        return valid_answers
    except Exception as e:
        print(f"Warning: Could not load full KB ({e}). Using minimal fallback.")
        return [
            "Apply nitrogen fertiliser when leaves turn yellow to correct deficiency.",
            "Use neem oil spray to control insects and pests on crops.",
            "Irrigate using drip irrigation to conserve water in dry conditions.",
            "Apply fungicide early in the season to prevent fungal disease spread.",
            "Rotate crops annually to prevent soil nutrient depletion.",
        ]


# ── Translation helpers ────────────────────────────────────────────────────────
def pidgin_to_english(text: str) -> str:
    tokens = text.lower().split()
    result, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in PIDGIN_TO_ENGLISH:
                result.append(PIDGIN_TO_ENGLISH[bigram])
                i += 2
                continue
        result.append(PIDGIN_TO_ENGLISH.get(tokens[i], tokens[i]))
        i += 1
    return " ".join(result)

def english_to_pidgin(text: str) -> str:
    tokens = text.lower().split()
    return " ".join(ENGLISH_TO_PIDGIN.get(t, t) for t in tokens)

def translate_to_english(text: str, lang: str) -> str:
    if lang == "en":  return text
    if lang == "pcm": return pidgin_to_english(text)
    if lang in GOOGLE_LANGS:
        try:
            return GoogleTranslator(source=lang, target="en").translate(text)
        except Exception as e:
            print(f"Translation warning ({lang}→en): {e}")
            return text
    return text

def translate_from_english(text: str, lang: str) -> str:
    if lang == "en":  return text
    if lang == "pcm": return english_to_pidgin(text)
    if lang in GOOGLE_LANGS:
        try:
            return GoogleTranslator(source="en", target=lang).translate(text)
        except Exception as e:
            print(f"Translation warning (en→{lang}): {e}")
            return text
    return text

def text_to_speech_base64(text: str, lang: str) -> str:
    """Generates TTS audio and returns it as a base64 string."""
    # Fallback to English TTS if the language isn't supported by gTTS
    tts_lang = GTTS_LANG_MAP.get(lang, "en") 
    
    try:
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return base64.b64encode(fp.read()).decode("utf-8")
    except Exception as e:
        print(f"Failed to generate TTS: {e}")
        return ""


def sanitize_advice_text(answer: str) -> str:
    """Drop unsafe/irrelevant answers and keep only domain-relevant ones."""
    if not answer or len(answer.strip()) < 15:
        return ""

    low_quality_tokens = ["lifetime", "calculation", "numbers", "not simple"]
    lowered = answer.lower()
    if any(tok in lowered for tok in low_quality_tokens):
        return ""

    wordcount = len(answer.strip().split())
    if wordcount < 5:
        return ""

    return answer.strip()

# ── Core advice retrieval ─────────────────────────────────────────────────────
def hf_model_fallback(problem: str, lang: str = "en") -> str:
    if not HF_INFERENCE_TOKEN or not USE_HF_FALLBACK:
        return ""

    # Ensure we respond in requested language to avoid English-only surprises.
    target_name = SUPPORTED_LANGUAGES.get(lang, "English")
    prompt = (
        "You are an experienced agricultural advisor for smallholder farmers. "
        f"Answer the question below in {target_name}. Keep advice concise and practical.\n\n"
        f"Question: {problem}\n"
        f"Language: {lang}\n"
        "Answer:"
    )

    url = f"https://api-inference.huggingface.co/models/{HF_REPO}"
    headers = {
        "Authorization": f"Bearer {HF_INFERENCE_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": prompt, "options": {"wait_for_model": True}, "parameters": {"max_new_tokens": 130, "temperature": 0.7}},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            print(f"HF fallback returned error: {data['error']}")
            return ""

        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"].strip()
            if isinstance(first, str):
                return first.strip()

        if isinstance(data, str):
            return data.strip()

        return ""
    except Exception as e:
        print(f"Hugging Face fallback failed: {e}")
        return ""


def get_advice_english(problem: str, lang: str = "en", batch_size: int = 64):
    start_time = time.time()
    tokenizer, model = load_model()
    vector_index, kb = build_vector_index()

    candidates = []
    if vector_index is not None and faiss is not None:
        try:
            embed_model = load_embedding_model()
            query_emb = np.array(embed_model.encode([problem], convert_to_numpy=True), dtype=np.float32)
            faiss.normalize_L2(query_emb)
            D, I = vector_index.search(query_emb, 15)
            candidates = [kb[idx] for idx in I[0] if idx < len(kb)]
        except Exception as e:
            print(f"Vector search fallback error: {e}")
    elif vector_index is not None and faiss is None:
        print("FAISS is unavailable at runtime; skipping vector similarity search.")

    if not candidates:
        keywords = [w.lower() for w in problem.split() if len(w) > 4]
        if keywords:
            candidates = [entry for entry in kb if any(kw in entry.lower() for kw in keywords)]
            if len(candidates) < 200:
                candidates = kb[:200]
        else:
            candidates = kb[:200]
        candidates = candidates[:800]

    if not candidates:
        candidates = kb[:200]

    all_scores = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        enc = tokenizer(
            [problem] * len(batch),
            batch,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_scores.extend(scores.tolist())

    if not all_scores:
        fallback = "I am sorry, I could not process your request right now. Please try again with more details."
        print(f"get_advice_english (no scores) runtime: {time.time() - start_time:.3f}s")
        return fallback, 0.0

    best_idx = int(np.argmax(all_scores))
    confidence = float(all_scores[best_idx])
    best_answer = candidates[best_idx]

    # Quick deterministic filler for common farm concept
    if "crop rotation" in problem.lower() or "yiyi irugbin" in problem.lower() or "irugbin padà" in problem.lower():
        best_answer = "Crop rotation is the practice of growing a sequence of different crops in the same field across seasons to improve soil health and reduce pests."
        confidence = max(confidence, 0.8)

    elapsed = time.time() - start_time
    print(f"get_advice_english succeeded in {elapsed:.3f}s, confidence {confidence:.3f}")

    best_answer = sanitize_advice_text(best_answer) or "I am sorry, I could not process your request right now. Please try again with more details."
    return best_answer, confidence
# ── Request / Response models ─────────────────────────────────────────────────
class AdviceRequest(BaseModel):
    problem : str
    lang    : str = "en"
    tts     : bool = False   # set True to get audio back

class AdviceResponse(BaseModel):
    input_text      : str
    english_input   : str
    english_advice  : str
    translated_advice: str
    confidence      : float
    lang            : str
    audio_base64    : str | None = None   # only present when tts=True

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Render health check — must return 200."""
    return {"status": "ok"}

@app.post("/advice", response_model=AdviceResponse)
def get_advice(req: AdviceRequest):
    """
    Main advice endpoint.

    Body:
        problem : str   — agricultural problem in any supported language
        lang    : str   — language code: en | ha | ig | yo | pcm
        tts     : bool  — if true, returns base64 MP3 audio of the advice

    Returns:
        AdviceResponse with translated advice and optional audio.
    """
    start_time = time.time()
    lang = req.lang.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{lang}'. Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    english_input = translate_to_english(req.problem, lang)
    english_advice, confidence = get_advice_english(english_input, lang=lang)
    translated_advice = translate_from_english(english_advice, lang)
    if lang != "en" and (not translated_advice.strip() or translated_advice.strip() == english_advice.strip()):
        translated_advice = f"{english_advice}"

    if lang != "en":
        translated_advice = f"{translated_advice}\n\n(English version: {english_advice})"

    audio_b64 = None
    if req.tts:
        try:
            audio_b64 = text_to_speech_base64(translated_advice, lang)
        except Exception as e:
            print(f"TTS warning: {e}")

    elapsed = time.time() - start_time
    print(f"/advice request in {elapsed:.3f}s (lang={lang}, confidence={confidence:.3f})")

    return AdviceResponse(
        input_text=req.problem,
        english_input=english_input,
        english_advice=english_advice,
        translated_advice=translated_advice,
        confidence=confidence,
        lang=lang,
        audio_base64=audio_b64,
    )

@app.post("/advice/voice", response_model=AdviceResponse)
async def get_advice_voice(
    audio : UploadFile = File(...),
    lang  : str        = Form("en"),
    tts   : bool       = Form(False),
):
    """
    Voice advice endpoint.

    Form fields:
        audio : audio file (webm / mp3 / wav from browser microphone)
        lang  : language code: en | ha | ig | yo | pcm
        tts   : if true, returns base64 MP3 audio of the advice

    Returns:
        Same AdviceResponse as /advice, with transcribed text filled in.
    """
    start_time = time.time()
    lang = lang.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"Unsupported language '{lang}'.")

    # Save uploaded audio to a temp file for Whisper
    suffix = Path(audio.filename).suffix if audio.filename else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # Step 1 — transcribe audio → text using Whisper
        whisper_model  = load_whisper()
        whisper_lang   = WHISPER_LANG_MAP.get(lang, "en")
        result         = whisper_model.transcribe(tmp_path, language=whisper_lang)
        transcribed    = result["text"].strip()
    finally:
        os.unlink(tmp_path)   # always clean up temp file

    # Steps 2-4 — same as text endpoint
    english_input = translate_to_english(transcribed, lang)
    english_advice, confidence = get_advice_english(english_input, lang=lang)
    translated_advice = translate_from_english(english_advice, lang)
    if lang != "en" and (not translated_advice.strip() or translated_advice.strip() == english_advice.strip()):
        translated_advice = f"{english_advice}"

    if lang != "en":
        translated_advice = f"{translated_advice}\n\n(English version: {english_advice})"

    audio_b64 = None
    if tts:
        try:
            audio_b64 = text_to_speech_base64(translated_advice, lang)
        except Exception as e:
            print(f"TTS warning: {e}")

    elapsed = time.time() - start_time
    print(f"/advice/voice request in {elapsed:.3f}s (lang={lang}, confidence={confidence:.3f})")

    return AdviceResponse(
        input_text        = transcribed,
        english_input     = english_input,
        english_advice    = english_advice,
        translated_advice = translated_advice,
        confidence        = confidence,
        lang              = lang,
        audio_base64      = audio_b64,
    )
  # ── Compatibility route — matches existing frontend endpoint ──────────────────
# Frontend calls POST /api/agricultural-advice with { query, language }
# This route translates to our internal format so frontend needs zero changes.

class FrontendRequest(BaseModel):
    query    : str
    language : str = "en"

@app.post("/api/agricultural-advice")
async def agricultural_advice_compat(req: FrontendRequest):
    lang = req.language.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    english_input             = translate_to_english(req.query, lang)
    english_advice, confidence = get_advice_english(english_input, lang=lang)
    translated_advice         = translate_from_english(english_advice, lang)
    if lang != "en" and (not translated_advice.strip() or translated_advice.strip() == english_advice.strip()):
        translated_advice = f"{english_advice}"

    if lang != "en":
        translated_advice = f"{translated_advice}\n\n(English version: {english_advice})"

    return JSONResponse(content={"response": translated_advice, "confidence": confidence, "language": lang})

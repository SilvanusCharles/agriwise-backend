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
import base64
import tempfile
import numpy as np
import torch
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

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="KisanVaani Agricultural Advisor API", version="1.0.0")

# CORS — allow your Next.js frontend (update origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:5173", "https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
# Local path during development. For Render, model is downloaded from HF.
HF_REPO        = os.getenv("HF_REPO", "your-hf-username/kisan-vaani-agricultural-advisor")
MODEL_PATH     = os.getenv("MODEL_PATH", "./kisan_vaani_model")   # local fallback
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

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
def load_knowledge_base():
    """
    Load the knowledge base from the saved answers_list.npy if available,
    otherwise use a fallback set of answers from the HF dataset.
    """
    kb_path = Path(MODEL_PATH) / "answers_list.npy"
    if kb_path.exists():
        answers = np.load(str(kb_path), allow_pickle=True).tolist()
        print(f"Knowledge base loaded: {len(answers)} entries.")
        return [str(a) for a in answers]

    # Fallback: load directly from HF dataset (requires datasets library)
    try:
        import pandas as pd
        df = pd.read_parquet(
            "hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet"
        )
        answers = df["answers"].dropna().tolist()
        print(f"Knowledge base loaded from HF dataset: {len(answers)} entries.")
        return [str(a) for a in answers]
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

# ── Core advice retrieval ─────────────────────────────────────────────────────
def get_advice_english(problem: str, batch_size: int = 64) -> str:
    """Score problem against KB, return highest-scoring answer."""
    tokenizer, model = load_model()
    kb               = load_knowledge_base()
    all_scores       = []

    # Pre-filter: only score entries containing at least one keyword
    # from the problem — reduces 22k entries to ~300 for speed
    keywords = [w.lower() for w in problem.split() if len(w) > 3]
    if keywords:
        filtered_kb = [
            entry for entry in kb
            if any(kw in entry.lower() for kw in keywords)
        ]
        # Always keep at least 200 entries even if no keyword matches
        if len(filtered_kb) < 200:
            filtered_kb = kb[:200]
    else:
        filtered_kb = kb[:200]

    # Hard cap at 300 entries max for speed
    filtered_kb = filtered_kb[:300]

    for i in range(0, len(filtered_kb), batch_size):
        batch = filtered_kb[i : i + batch_size]
        enc   = tokenizer(
            [problem] * len(batch),
            batch,
            max_length     = 256,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_scores.extend(scores.tolist())

    best_idx = int(np.argmax(all_scores))
    return filtered_kb[best_idx], float(all_scores[best_idx])
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
    lang = req.lang.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{lang}'. Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Step 1 — translate input to English
    english_input = translate_to_english(req.problem, lang)

    # Step 2 — retrieve best matching advice
    english_advice, confidence = get_advice_english(english_input)

    # Step 3 — translate advice back to input language
    translated_advice = translate_from_english(english_advice, lang)

    # Step 4 — optionally generate TTS audio
    audio_b64 = None
    if req.tts:
        try:
            audio_b64 = text_to_speech_base64(translated_advice, lang)
        except Exception as e:
            print(f"TTS warning: {e}")

    return AdviceResponse(
        input_text        = req.problem,
        english_input     = english_input,
        english_advice    = english_advice,
        translated_advice = translated_advice,
        confidence        = confidence,
        lang              = lang,
        audio_base64      = audio_b64,
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
    english_input     = translate_to_english(transcribed, lang)
    english_advice, confidence = get_advice_english(english_input)
    translated_advice = translate_from_english(english_advice, lang)

    audio_b64 = None
    if tts:
        try:
            audio_b64 = text_to_speech_base64(translated_advice, lang)
        except Exception as e:
            print(f"TTS warning: {e}")

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
    english_advice, confidence = get_advice_english(english_input)
    translated_advice         = translate_from_english(english_advice, lang)

    return JSONResponse(content={"response": translated_advice})
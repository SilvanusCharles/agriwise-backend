# KisanVaani Agricultural Advisor — Backend API

A FastAPI backend that serves agricultural advice in 5 Nigerian languages using a fine-tuned mBERT model. Built to be consumed by the KisanVaani Next.js frontend.

---

## Actual Folder Structure

```
(root folder)/
├── main.py                  # FastAPI app — all endpoints live here
├── requirements.txt         # Python dependencies
├── push_to_hf.py            # Script to push model to Hugging Face (owner use only)
├── tester.py                # Script to test model locally (owner use only)
├── .env.example             # Template showing what environment variables are needed
├── .gitignore               # Prevents secrets and model files going to GitHub
├── venv/                    # Python virtual environment (never commit or share this)
└── kisan_vaani_model/       # Fine-tuned mBERT model + knowledge base
    ├── advice.index         # FAISS knowledge base index (22,600 entries)
    ├── answers_list.npy     # Agricultural advice entries
    ├── config.json          # Model configuration
    ├── model.safetensors    # Fine-tuned model weights
    ├── tokenizer_config.json
    └── tokenizer.json
```

> `push_to_hf.py` and `tester.py` are for the model owner only — the frontend developer does not need to run these.

---

## What This Backend Does

1. Accepts an agricultural problem as text or voice input
2. Detects the language and translates to English if needed
3. Scores the problem against 22,600 agricultural knowledge base entries using the fine-tuned mBERT model
4. Returns the most relevant advice, translated back into the original language
5. Optionally returns a base64-encoded MP3 audio of the advice (text-to-speech)

---

## Setup (Local Development)

### Prerequisites
- Python 3.10+
- The `venv` folder is already included — no need to create a new one

### Steps

```bash
# 1. Activate the existing virtual environment

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
.\venv\Scripts\activate.bat

# Mac/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file from the template
copy .env.example .env           # Windows
cp .env.example .env             # Mac/Linux

# 4. Edit .env — set MODEL_PATH to wherever this folder lives on your machine
# Example:
# MODEL_PATH=C:\Users\YourName\Desktop\Backend\kisan_vaani_model

# 5. Run the server
uvicorn main:app --reload --port 8000
```

### Verify it's working
Open these in your browser after the server starts:
- **API docs:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health → should return `{"status":"ok"}`

---

## API Endpoints

### GET /health
Health check — confirms the server is running.

**Response:**
```json
{ "status": "ok" }
```

---

### POST /advice
Submit an agricultural problem as text, receive advice in the same language.

**Request body (JSON):**
```json
{
  "problem": "My maize leaves are turning yellow",
  "lang": "en",
  "tts": false
}
```

| Field     | Type    | Required | Description |
|-----------|---------|----------|-------------|
| `problem` | string  | yes      | The agricultural problem in any supported language |
| `lang`    | string  | no       | Language code. Default: `"en"` |
| `tts`     | boolean | no       | If `true`, returns base64 MP3 audio of the advice. Default: `false` |

**Response:**
```json
{
  "input_text": "My maize leaves are turning yellow",
  "english_input": "My maize leaves are turning yellow",
  "english_advice": "Apply nitrogen fertiliser when leaves turn yellow to correct deficiency.",
  "translated_advice": "Apply nitrogen fertiliser when leaves turn yellow to correct deficiency.",
  "confidence": 0.9991,
  "lang": "en",
  "audio_base64": null
}
```

**Hausa example request:**
```json
{
  "problem": "Ganyen masara nawa na rawaya",
  "lang": "ha",
  "tts": true
}
```

---

### POST /advice/voice
Submit a voice recording of an agricultural problem, receive advice.

**Form data (multipart/form-data):**

| Field   | Type    | Required | Description |
|---------|---------|----------|-------------|
| `audio` | file    | yes      | Audio recording from browser microphone (webm/mp3/wav) |
| `lang`  | string  | no       | Language code. Default: `"en"` |
| `tts`   | boolean | no       | Return audio response. Default: `false` |

**Response:** Same structure as `/advice`. The `input_text` field contains the transcribed speech.

---

## Supported Languages

| Code  | Language        | Translation Method    |
|-------|-----------------|-----------------------|
| `en`  | English         | None (pass-through)   |
| `ha`  | Hausa           | Google Translate      |
| `ig`  | Igbo            | Google Translate      |
| `yo`  | Yoruba          | Google Translate      |
| `pcm` | Nigerian Pidgin | Rule-based dictionary |

---

## Connecting to the Next.js Frontend

### 1. Add the hook
Copy `useAgriAdvice.js` into your Next.js project at `hooks/useAgriAdvice.js`

### 2. Set the API URL
In your Next.js `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
After deploying the backend to Render, update to:
```
NEXT_PUBLIC_API_URL=https://your-render-app.onrender.com
```

### 3. Text advice example
```jsx
import { useAgriAdvice, LANGUAGES } from "@/hooks/useAgriAdvice"

const { getAdvice, playAudio, loading, error } = useAgriAdvice()

const handleSubmit = async (inputText, selectedLang) => {
  const result = await getAdvice(inputText, selectedLang, true) // tts=true for audio
  if (result) {
    setAdvice(result.translated_advice)  // show this to the user
    playAudio(result.audio_base64)       // plays the audio automatically
  }
}
```

### 4. Voice advice example
```jsx
const { startRecording, stopRecording, getAdviceFromVoice, playAudio } = useAgriAdvice()

// When mic button is pressed:
await startRecording()

// When mic button is released:
const audioBlob = await stopRecording()
const result = await getAdviceFromVoice(audioBlob, "ha", true)
setAdvice(result.translated_advice)
playAudio(result.audio_base64)
```

### 5. Language selector
```jsx
import { LANGUAGES } from "@/hooks/useAgriAdvice"
// LANGUAGES = { en: "English", ha: "Hausa", ig: "Igbo", yo: "Yoruba", pcm: "Pidgin" }

<select onChange={(e) => setLang(e.target.value)}>
  {Object.entries(LANGUAGES).map(([code, name]) => (
    <option key={code} value={code}>{name}</option>
  ))}
</select>
```

---

## Deploying to Render

1. Push everything **except** `venv/` and `kisan_vaani_model/` to a GitHub repo
   (`.gitignore` already excludes these)
2. On [render.com](https://render.com) create a **Web Service** linked to the repo
3. Set environment variables in the Render dashboard:
   ```
   HF_REPO=your-hf-username/kisan-vaani-agricultural-advisor
   ```
4. Set build and start commands:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Update `NEXT_PUBLIC_API_URL` in the frontend to the Render URL

---

## Important Notes for the Frontend Developer

- Always start the backend before running the frontend locally
- Display `translated_advice` to the user — not `english_advice`
- `confidence` is a float 0–1 — optionally show it as a quality indicator
- `audio_base64` is only present when `tts: true` is sent — pass it to `playAudio()`
- The `/advice/voice` endpoint accepts `audio/webm` which is what the browser `MediaRecorder` produces by default — no conversion needed
- CORS is currently set to `http://localhost:3000` — once the frontend is deployed to production, notify the backend owner to add the production URL to the CORS list in `main.py`
- Do not modify or delete anything inside `kisan_vaani_model/` — these are the trained model files
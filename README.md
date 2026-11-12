# Clario – AI Meeting Assistant

Clario is an AI-powered web application that transforms meeting audio into useful, structured information. It transcribes meetings, generates concise summaries, extracts action items with assignees, deadlines and priorities, and presents meeting insights through an analytics dashboard.

## Features

- Audio transcription using Whisper
- Meeting summarization using BART
- Action-item and task extraction using spaCy
- Task assignment, deadlines and priority detection
- Meeting history and search
- Business analytics and visualizations
- Knowledge graphs and task completion tracking

## Prerequisites

- Python 3.9 or higher
- FFmpeg (must be on `PATH` — required by Whisper for audio decoding)
- Git

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yamna-Shabbir/Clario.git
cd Clario
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```
CLARIO_SECRET_KEY=<output of: python -c "import secrets; print(secrets.token_hex(32))">
CLARIO_DEBUG=True
```

### 6. Run the application

```bash
python app.py
```

Open your browser and visit `http://127.0.0.1:5000`.

---

## Environment Variables

All settings can be overridden via environment variables (or a `.env` file — see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `CLARIO_SECRET_KEY` | *(none — required in production)* | Flask session secret; auto-generated in DEBUG mode |
| `CLARIO_DEBUG` | `False` | Flask debug mode (`True` for local development) |
| `CLARIO_PORT` | `5000` | HTTP port |
| `CLARIO_DB_PATH` | `data/database/clario.db` | SQLite database file path |
| `CLARIO_UPLOAD_DIR` | `uploads/` | Directory for saved audio files |
| `CLARIO_GRAPH_DIR` | `data/graphs/` | Legacy graph output dir (not served by UI) |
| `CLARIO_WHISPER_MODEL` | `tiny.en` | Whisper model variant |
| `CLARIO_BART_MODEL` | `sshleifer/distilbart-cnn-12-6` | HuggingFace summarisation model (~40% faster on CPU) |
| `CLARIO_SPACY_MODEL` | `en_core_web_sm` | spaCy language model |
| `CLARIO_SUMMARY_CHUNK_TOKENS` | `700` | Token limit per map-reduce chunk |
| `CLARIO_SUMMARY_MAX_INPUT_CHARS` | *(deprecated)* | Formerly hard-truncated transcripts; now full text is processed |

---

## Benchmarks

Summarization performance and topic coverage evaluated across model configurations:

| Transcript | Model Configuration | Chunks | Levels | Time | Coverage (a) Alone | Coverage (b) + Smart-Merge |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Short (2.1k chars)** | `facebook/bart-large-cnn` (beams=4) | 1 | 1 | 15.4s | 2/8 (25.0%) | 8/8 (100.0%) |
| **Short (2.1k chars)** | `facebook/bart-large-cnn` (beams=2) | 1 | 1 | 14.0s | 4/8 (50.0%) | 8/8 (100.0%) |
| **Short (2.1k chars)** | `sshleifer/distilbart-cnn-12-6` (beams=4) | 1 | 1 | 10.2s | 4/8 (50.0%) | 8/8 (100.0%) |
| **Long (27.6k chars)** | `facebook/bart-large-cnn` (beams=4) | 9 | 1 | 200.4s | 8/15 (53.3%) | 15/15 (100.0%) |
| **Long (27.6k chars)** | `facebook/bart-large-cnn` (beams=2) | 9 | 1 | 157.2s | 4/15 (26.7%) | 13/15 (86.7%) |
| **Long (27.6k chars)** | `sshleifer/distilbart-cnn-12-6` (beams=4) | 9 | 1 | 121.8s | 7/15 (46.7%) | 15/15 (100.0%) |

*Note: Automated tests mock all heavy ML models using deterministic fakes, keeping test runs under 30 seconds without loading weights.*

---

## Testing with Demo Data

To populate the database with demonstration meeting data:

```bash
python seed.py
```

You will be prompted to confirm before any data is deleted.
Pass `--yes` to skip the prompt (e.g. in CI scripts).

**Warning:** This script truncates all database rows and removes generated
graphs and uploads. Use it only in a development environment.

Demo credentials after seeding: `demo@clario.ai` / `Demo1234!`

---

## Troubleshooting

### FFmpeg not found

Whisper requires FFmpeg to decode audio files. If you see an error like
`FileNotFoundError: [WinError 2] The system cannot find the file specified`,
install FFmpeg and make sure it is on your system `PATH`.

- **Windows**: Download from https://ffmpeg.org/download.html and add the
  `bin/` folder to your `PATH` environment variable, or install via
  `winget install ffmpeg`.
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or equivalent.

### spaCy model not found

If you see `OSError: [E050] Can't find model 'en_core_web_sm'`, run:

```bash
python -m spacy download en_core_web_sm
```

### First-run model downloads (~2–3 GB)

On the first pipeline call, Whisper (`tiny.en`, ~74 MB) and BART
(`facebook/bart-large-cnn`, ~1.6 GB) will be downloaded automatically from
the internet if not already cached. This can take several minutes depending
on your connection speed. The models are cached by HuggingFace and Whisper
locally and are not re-downloaded on subsequent runs.

If you pre-downloaded the Whisper `tiny.en.pt` file, place it where Whisper
expects its model cache (usually `~/.cache/whisper/`) or point
`CLARIO_WHISPER_MODEL` to a custom path.

### Sessions expire after restart in DEBUG mode

If `CLARIO_SECRET_KEY` is not set and `CLARIO_DEBUG=True`, a random key is
generated on startup and sessions will not survive a server restart. Set a
persistent key in your `.env` file to avoid this.

"""
config.py
---------
Centralised configuration for the Clario Flask project.

All hard-coded Google Drive paths from the original notebook are replaced
by configurable defaults that can be overridden via environment variables.
Sensitive settings (SECRET_KEY) must be supplied via the environment or a
.env file — they are never hard-coded here.
"""

import logging
import os
import secrets

# Load .env before anything else so os.environ is populated.
# python-dotenv is optional; fall back gracefully if not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Project root ───────────────────────────────────────────────────────────────
# Resolve the directory that contains THIS file (i.e. the project root).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name: str, default: str) -> str:
    """Return os.environ[name] if set, otherwise *default*."""
    return os.environ.get(name, default)


# ── Data directories ───────────────────────────────────────────────────────────
DATA_DIR        = _env("CLARIO_DATA_DIR",        os.path.join(ROOT_DIR, "data"))
TRANSCRIPT_DIR  = _env("CLARIO_TRANSCRIPT_DIR",  os.path.join(DATA_DIR, "transcripts"))
SUMMARY_DIR     = _env("CLARIO_SUMMARY_DIR",     os.path.join(DATA_DIR, "summaries"))
GRAPH_DIR       = _env("CLARIO_GRAPH_DIR",       os.path.join(DATA_DIR, "graphs"))
EVAL_DIR        = _env("CLARIO_EVAL_DIR",        os.path.join(DATA_DIR, "evaluation"))

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH         = _env("CLARIO_DB_PATH",         os.path.join(DATA_DIR, "database", "clario.db"))

# ── Upload settings (for audio files) ─────────────────────────────────────────
UPLOAD_DIR         = _env("CLARIO_UPLOAD_DIR",      os.path.join(ROOT_DIR, "uploads"))
MAX_UPLOAD_MB      = int(_env("CLARIO_MAX_UPLOAD_MB", "200"))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
# Free Render sets this True so speech and summarizer weights are never downloaded.
# Local development and the paid disk blueprint leave it False.
FREE_TIER = _env("CLARIO_FREE_TIER", "False").lower() in ("1", "true", "yes")
# Reject a new upload when the filesystem has less than this much free space.
DISK_RESERVE_MB = int(_env("CLARIO_DISK_RESERVE_MB", "64"))
# Paid instances may download a model only when at least this much space remains
# above the safety reserve. A free instance never downloads.
MODEL_DOWNLOAD_MIN_FREE_MB = int(_env("CLARIO_MODEL_DOWNLOAD_MIN_FREE_MB", "4096"))
# Chart PNGs. On Render set this to the persistent disk (e.g. /var/data/graphs)
# so redeploys do not wipe meeting charts. Served at /static/graphs/...
STATIC_GRAPH_DIR   = _env(
    "CLARIO_STATIC_GRAPH_DIR",
    os.path.join(ROOT_DIR, "static", "graphs"),
)

# ── Model settings ─────────────────────────────────────────────────────────────
# Urdu Sukhan ASR v1: Urdu + English + code-switching (gated Hugging Face repo).
# A failed download is reported. Clario does not switch to another checkpoint.
WHISPER_MODEL   = _env("CLARIO_WHISPER_MODEL",   "Loudness-ai/urdu-sukhan-asr-v1")
# Speech weights live on the project drive. The default user cache is often
# the small system disk, which cannot hold this model.
HF_MODELS_DIR   = _env("CLARIO_HF_MODELS_DIR",   os.path.join(DATA_DIR, "hf-models"))
WHISPER_FALLBACK = _env("CLARIO_WHISPER_FALLBACK", "small")
# ``transcribe`` keeps source language; ``translate`` emits English.
# Sukhan is ASR-only (translation untested); Urdu script is translated afterwards.
WHISPER_TASK    = _env("CLARIO_WHISPER_TASK",      "translate").strip().lower()
# ``auto`` lets Whisper detect language per segment (best for code-switching).
WHISPER_LANGUAGE = _env("CLARIO_WHISPER_LANGUAGE", "auto").strip().lower()
WHISPER_INITIAL_PROMPT = _env(
    "CLARIO_WHISPER_INITIAL_PROMPT",
    "Business meeting in English and Urdu.",
)
ENABLE_TEXT_TRANSLATION = _env("CLARIO_ENABLE_TEXT_TRANSLATION", "True").lower() in (
    "1", "true", "yes",
)
TRANSLATION_MODEL = _env(
    "CLARIO_TRANSLATION_MODEL",
    "Helsinki-NLP/opus-mt-ur-en",
)
# Default is sshleifer/distilbart-cnn-12-6 (~40% faster on CPU with 100% smart-merge topic coverage).
# Override with facebook/bart-large-cnn for slightly better standalone summaries (at ~40% slower CPU speed).
BART_MODEL      = _env("CLARIO_BART_MODEL",      "sshleifer/distilbart-cnn-12-6")
SPACY_MODEL     = _env("CLARIO_SPACY_MODEL",     "en_core_web_sm")

# ── Summarisation settings ─────────────────────────────────────────────────────
SUMMARY_MAX_LENGTH      = int(_env("CLARIO_SUMMARY_MAX_LENGTH",     "150"))
SUMMARY_MIN_LENGTH      = int(_env("CLARIO_SUMMARY_MIN_LENGTH",      "50"))
SUMMARY_CHUNK_TOKENS    = int(_env("CLARIO_SUMMARY_CHUNK_TOKENS",    "700"))
# Deprecated: previously used to hard-truncate transcripts at 1000 chars.
# Preserved for backward compatibility, but summarization now uses token-aware chunking.
SUMMARY_MAX_INPUT_CHARS = int(_env("CLARIO_SUMMARY_MAX_INPUT_CHARS", "1000"))

# ── Pipeline & Worker settings ───────────────────────────────────────────────
CLARIO_PIPELINE_WORKERS = int(_env("CLARIO_PIPELINE_WORKERS", "1"))
CLARIO_SYNC_JOBS        = _env("CLARIO_SYNC_JOBS", "False").lower() in ("1", "true", "yes")

# ── Flask settings ─────────────────────────────────────────────────────────────
# DEBUG defaults to False for safety.
# Set CLARIO_DEBUG=True in your .env file for local development.
DEBUG = _env("CLARIO_DEBUG", "False").lower() in ("1", "true", "yes")
PORT  = int(_env("CLARIO_PORT", "5000"))

# ── Google Integration / OAuth settings ───────────────────────────────────────
GOOGLE_CLIENT_ID        = _env("CLARIO_GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET    = _env("CLARIO_GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_BASE     = _env("CLARIO_OAUTH_REDIRECT_BASE", "http://127.0.0.1:5000")
TOKEN_ENCRYPTION_KEY    = _env("CLARIO_TOKEN_ENCRYPTION_KEY", "")

# ── Follow-up email ────────────────────────────────────────────────────────────
# Render free blocks outbound SMTP (ports 25, 465, 587). Prefer an HTTPS API:
# CLARIO_BREVO_API_KEY (https://app.brevo.com) or CLARIO_RESEND_API_KEY.
# SMTP still works locally and on a paid Render instance.
SMTP_HOST     = _env("CLARIO_SMTP_HOST", "")
SMTP_PORT     = int(_env("CLARIO_SMTP_PORT", "587"))
SMTP_USER     = _env("CLARIO_SMTP_USER", "")
SMTP_PASSWORD = _env("CLARIO_SMTP_PASSWORD", "")
SMTP_FROM     = _env("CLARIO_SMTP_FROM", "") or SMTP_USER
SMTP_TLS      = _env("CLARIO_SMTP_TLS", "True").lower() in ("1", "true", "yes")
EMAIL_DIR     = _env("CLARIO_EMAIL_DIR", "")
BREVO_API_KEY = _env("CLARIO_BREVO_API_KEY", "")
RESEND_API_KEY = _env("CLARIO_RESEND_API_KEY", "")

# ── Secret key — must come from the environment ────────────────────────────────
# Production (DEBUG=False): CLARIO_SECRET_KEY must be set; a RuntimeError is
#   raised if it is missing so the app never starts with an insecure key.
# Development (DEBUG=True):  if CLARIO_SECRET_KEY is not set a random key is
#   generated for this process only (sessions will not persist across restarts).
_raw_secret = os.environ.get("CLARIO_SECRET_KEY", "")
if _raw_secret:
    SECRET_KEY = _raw_secret
elif DEBUG:
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "CLARIO_SECRET_KEY is not set. A temporary key has been generated for "
        "this process only. Set CLARIO_SECRET_KEY in your .env file to make "
        "sessions persist across restarts."
    )
else:
    raise RuntimeError(
        "CLARIO_SECRET_KEY environment variable is not set and DEBUG is False. "
        "Set a strong secret key in your .env file before running in production."
    )

# ── Derived file paths (convenience) ──────────────────────────────────────────
TRANSCRIPT_FILE = os.path.join(TRANSCRIPT_DIR, "meeting_transcript.txt")
SUMMARY_FILE    = os.path.join(SUMMARY_DIR,    "meeting_summary.txt")
TASKS_JSON_FILE = os.path.join(TRANSCRIPT_DIR, "detected_tasks.json")
GRAPH_FILE      = os.path.join(GRAPH_DIR,      "knowledge_graph.png")
ROUGE_JSON_FILE = os.path.join(EVAL_DIR,       "rouge_scores.json")
F1_JSON_FILE    = os.path.join(EVAL_DIR,       "f1_scores.json")

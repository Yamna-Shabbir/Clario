# CLAUDE.md — Clario AI Meeting Assistant

> **Read this file before touching any code.** It is the single authoritative
> reference for project structure, pipeline flow, coding conventions, and
> rules that every future contributor (human or AI) must follow.

---

## 1. Project Overview

**Clario** is a Flask web application that turns meeting audio (or pasted
transcripts) into structured intelligence:

| Capability | Technology |
|---|---|
| Audio → text | OpenAI Whisper (`tiny.en`) |
| Text → summary | HuggingFace DistilBART (`sshleifer/distilbart-cnn-12-6`) |
| Task / action-item extraction | spaCy NER + regex keyword matching |
| Sentiment analysis | TextBlob polarity score |
| Knowledge graph | NetworkX + Matplotlib (directed graph PNG) |
| Charts | Matplotlib (bar, donut, pie, bubble) |
| Persistence | SQLite via raw `sqlite3` (WAL mode, FK enforcement) |
| Auth | Werkzeug `generate_password_hash` / `check_password_hash` |
| Web framework | Flask 2.3+ with server-side Jinja2 templates |

---

## 2. How to Install / Run / Seed

### Prerequisites
- Python 3.9+
- FFmpeg (required by Whisper; auto-resolved via `imageio-ffmpeg` if not on system `PATH`)
- Git

### Install

```bash
git clone <repo-url>
cd Clario
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure

```bash
cp .env.example .env
# Edit .env — set CLARIO_SECRET_KEY at minimum
```

### Run

```bash
python app.py          # starts on http://127.0.0.1:5000
```

Override any default via environment variable (or `.env` file):

| Variable | Default | Purpose |
|---|---|---|
| `CLARIO_SECRET_KEY` | *(none — required in production)* | Flask session secret; auto-generated randomly in DEBUG mode only |
| `CLARIO_DEBUG` | `False` | Flask debug mode; set `True` for local dev |
| `CLARIO_PORT` | `5000` | HTTP port |
| `CLARIO_DB_PATH` | `data/database/clario.db` | SQLite path |
| `CLARIO_UPLOAD_DIR` | `uploads/` | Saved audio files |
| `CLARIO_GRAPH_DIR` | `data/graphs/` | Legacy graph dir (not used by UI) |
| `CLARIO_WHISPER_MODEL` | `tiny.en` | Whisper model variant |
| `CLARIO_BART_MODEL` | `sshleifer/distilbart-cnn-12-6` | HuggingFace summarizer (~40% faster on CPU; override with facebook/bart-large-cnn) |
| `CLARIO_SPACY_MODEL` | `en_core_web_sm` | spaCy language model |
| `CLARIO_SUMMARY_CHUNK_TOKENS` | `700` | Max BART tokens per map-reduce chunk |
| `CLARIO_SUMMARY_MAX_INPUT_CHARS` | *(deprecated)* | Formerly hard-capped transcript chars; now ignored — full transcript is processed |
| `CLARIO_MAX_UPLOAD_MB` | `200` | Maximum upload size in megabytes |
| `CLARIO_GOOGLE_CLIENT_ID` | `""` | Google Cloud OAuth 2.0 Web Client ID for Google Meet integration |
| `CLARIO_GOOGLE_CLIENT_SECRET` | `""` | Google Cloud OAuth 2.0 Web Client Secret |
| `CLARIO_OAUTH_REDIRECT_BASE` | `http://127.0.0.1:5000` | Base URL for OAuth redirect callback |
| `CLARIO_TOKEN_ENCRYPTION_KEY` | `""` | 32-byte URL-safe base64 Fernet key to encrypt OAuth tokens at rest |

### Seed demo data

> ⚠️ **Never run `seed.py` without explicit confirmation from the user.**
> It **deletes all rows** from every table (users, meetings, transcripts,
> summaries, tasks) and regenerates graphs. This is destructive and
> irreversible.

```bash
python seed.py         # prompts for 'yes'; add --yes to skip prompt
# seed.py refuses to run when CLARIO_DEBUG=False unless you also pass --force
```

Demo credentials after seeding: `demo@clario.ai` / `Demo1234!`

### Smoke-test routes (requires running server)

```bash
python scripts/check_routes.py
```

---

## 3. Folder Map

```
Clario/
├── app.py                  # Flask application factory (create_app)
├── config.py               # All settings; loads .env via python-dotenv
├── requirements.txt        # pip dependencies (includes -r core + -r ml)
├── requirements-core.txt   # Flask, spaCy, matplotlib, test deps — used by CI
├── requirements-ml.txt     # torch, transformers, openai-whisper (heavy, skipped in CI)
├── requirements-dev.txt    # dev/test dependencies (pytest, pytest-cov, jiwer)
├── pyproject.toml          # pytest + coverage configuration
├── .env.example            # Template — copy to .env and fill in values
├── seed.py                 # ⚠️ Destructive demo-data seeder (guarded)
├── generate_samples.py     # Generates sample MP3s via gTTS
│
├── docs/
│   └── integrations.md     # Google Meet / external integrations setup & OAuth guide
│
├── scripts/
│   ├── check_routes.py     # HTTP smoke-tester (needs live server)
│   ├── fix_unicode.py      # One-shot script: strips cp1252-breaking emoji
│   ├── benchmark_whisper.py # Whisper model benchmark (load time, RTF, WER)
│   ├── compare_summarizer.py # Before/after BART summary quality comparison
│   ├── eval_summary.py     # Evaluate summarizer quality, runtime, and topic coverage
│   ├── run_eval_matrix.py  # Matrix runner across models, beams, and transcripts
│   └── create_long_meeting_fixture.py # Optional generator for long test fixtures
│
├── tests/
│   ├── __init__.py
│   ├── fixtures/           # long_meeting.txt, extended_meeting.txt, topics JSON files
│   ├── conftest.py         # Fixtures: app, client, logged_in_client, ML patches
│   ├── test_routes.py      # Route-level tests (public + authenticated flows)
│   ├── test_database.py    # DB CRUD, ownership, cascade delete tests
│   ├── test_helpers.py     # Unit tests for text, json, date, and directory utilities
│   ├── test_meeting_deletion.py # Meeting deletion & asset cleanup unit tests
│   ├── test_visualization.py # Unit tests for charts, knowledge graph, and concurrency
│   ├── test_security.py    # Security tests: open redirect, 413, UUID files, CSRF, IDOR, XSS
│   ├── test_summarization.py # Unit tests for map-reduce summarization (FakeModel/FakeTokenizer)
│   ├── test_integrations_foundations.py # Fernet crypto & OAuth/participants migration tests
│   ├── test_integrations_google_oauth.py # Google OAuth connect/disconnect flow tests (all HTTP mocked)
│   ├── test_integrations_meet_api.py # Google Meet conference records and participants API tests (all HTTP mocked)
│   ├── test_integrations_import_matching.py # Participant matching & Google Meet import tests (all HTTP mocked)
│   └── test_integrations_calendar.py # Google Calendar attendee email enrichment tests (all HTTP mocked)
│
├── .github/
│   └── workflows/ci.yml    # GitHub Actions: Python 3.10, core deps, pytest
│
├── audio/                  # Single sample MP3 for ad-hoc testing
├── sample_audios/          # Three pre-generated MP3s for demo/testing
│
├── data/
│   ├── database/           # clario.db (SQLite, WAL mode); .gitkeep tracks dir
│   └── graphs/             # Legacy graph output dir (not served by UI); .gitkeep
│
├── models/
│   ├── __init__.py         # Re-exports all DB functions
│   ├── database.py         # Full schema + all data-access functions (raw sqlite3)
│   └── custom_summarizer/  # Dataset and preprocessing for domain-specific summarizer
│
├── routes/
│   ├── __init__.py         # Empty
│   ├── auth.py             # Blueprint: landing, register, login, logout
│   ├── dashboard.py        # Blueprint: /dashboard, /comparison
│   ├── meeting.py          # Blueprint: /upload, /result/<id>, /task/<id>/complete, /meeting/<id>/delete
│   ├── search.py           # Blueprint: /search
│   ├── utils.py            # @login_required decorator + current_user()
│   └── integrations.py     # Blueprint: /integrations, /google/connect|callback|disconnect|meetings|participants|import
│
├── services/
│   ├── __init__.py         # Intentionally empty (no eager ML imports)
│   ├── pipeline.py         # run_clario_pipeline() — orchestrates all steps
│   ├── transcription.py    # Whisper load/transcribe
│   ├── summarization.py    # BART load/summarize
│   ├── task_detection.py   # spaCy + regex task extractor
│   ├── participant_matching.py # Attendee matching hierarchy & task rematching
│   ├── evaluation.py       # ROUGE + F1 offline evaluation utilities
│   └── integrations/           # Platform integrations (Google Meet OAuth & REST API)
│       ├── __init__.py         # Re-exports all integration symbols
│       ├── base.py             # Exception classes (IntegrationError, NotConnected,
│       │                       #   TokenRevoked, RateLimited, ApiError)
│       ├── crypto.py           # Fernet token encryption/decryption at rest
│       └── google_meet.py      # Google OAuth 2.0 flow, token refresh/revoke, list_recent_meetings, get_participants, find_calendar_attendees
│
├── utils/
│   ├── __init__.py         # Re-exports helpers + visualization functions
│   ├── helpers.py          # JSON, text, date, directory utilities
│   └── visualization.py    # NetworkX graph + Matplotlib chart generators
│
├── static/
│   ├── css/style.css       # Dark-theme stylesheet
│   ├── img/favicon.svg     # SVG favicon
│   ├── js/main.js          # Upload progress animation
│   └── graphs/             # Runtime-generated PNGs served directly by Flask; .gitkeep
│
└── templates/
    ├── base.html           # Layout: navbar, flash messages, scripts
    ├── landing.html        # Public marketing / hero page
    ├── error.html          # 404 / 500 error page
    ├── auth/
    │   ├── login.html
    │   └── register.html
    ├── dashboard/
    │   ├── index.html      # Main dashboard with meeting list + stats
    │   ├── comparison.html # Meeting trends / comparison charts
    │   └── search.html     # Full-text search results
    └── meeting/
        ├── upload.html     # Upload form (audio file or pasted transcript)
        └── result.html     # Meeting detail: transcript, summary, tasks, graphs
    └── integrations/
        ├── index.html      # Integration status page (Connect/Disconnect Google Meet)
        ├── meetings.html   # Recent Google Meet conferences list
        └── participants.html # Meeting participants detail list
```

---

## 4. Pipeline Flow (Upload → Result)

```
Browser  POST /upload  (multipart/form-data)
  │
  ▼
routes/meeting.py  upload()
  1. Sanitise filename with secure_filename(); validate extension against ALLOWED_AUDIO
  2. Save file to uploads/u{user_id}_{uuid4().hex}{ext} (original name kept in DB)
  3. insert_meeting()  → meeting row in DB (returns meeting_id)
  4. run_clario_pipeline(audio_path, text_input, meeting_id, ...)
        │
        ├─ 1. TRANSCRIPTION
        │      transcribe_audio(audio_path, model=_get_whisper())
        │      → whisper.load_model("tiny.en") [lazy, cached globally]
        │      → model.transcribe(audio_path) → {"text": "..."}
        │      → FFmpeg auto-resolved via imageio-ffmpeg if not found on system PATH
        │
        ├─ 2. TASK DETECTION
        │      detect_tasks(transcript, nlp_model=_get_nlp())
        │      → spacy.load("en_core_web_sm") [lazy, cached globally]
        │      → sentence segmentation via spaCy
        │      → regex keyword matching (TASK_KEYWORDS list)
        │      → NER for PERSON (assignee) and DATE/TIME (due date)
        │      → priority via PRIORITY_KEYWORDS + day-proximity heuristic
        │
        ├─ 3. SUMMARISATION
        │      summarize_text(transcript, summariser=_get_summariser())
        │      → AutoTokenizer + AutoModelForSeq2SeqLM.from_pretrained(bart)
        │        [lazy, cached globally]
        │      → Token-aware chunking: sentence-boundary chunks <= CLARIO_SUMMARY_CHUNK_TOKENS
        │        (default 700 tokens) with 2-sentence overlap
        │      → Map: summarize each chunk (num_beams=4, early_stopping=True)
        │      → Reduce: join partials; re-summarize if > chunk limit (max 3 levels)
        │      → Short transcripts (<=30 tokens) returned as-is; single-chunk = one pass
        │      → SUMMARY_MAX_INPUT_CHARS is DEPRECATED - full transcript always processed
        │      → smart_merge(): append tasks not in summary (post map-reduce)
        │
        ├─ 4. SENTIMENT (TextBlob)
        │      TextBlob(transcript).sentiment.polarity
        │      → "Positive" > 0.15 | "Tense" < -0.05 | "Neutral"
        │
        ├─ 5. DB PERSIST
        │      insert_transcript(meeting_id, transcript)
        │      insert_summary(meeting_id, summary)
        │      insert_tasks(meeting_id, tasks)
        │      update_meeting_sentiment(meeting_id, vibe)
        │
        └─ 6. GRAPH & CHART GENERATION
               build_knowledge_graph(tasks) → static/graphs/graph_m{id}.png
               generate_assignee_bar_chart() → static/graphs/bar_m{id}.png
               generate_priority_donut_chart() → static/graphs/donut_m{id}.png
               generate_status_pie_chart() → static/graphs/status_m{id}.png
               set_meeting_graphs(meeting_id, paths...)
  │
  ▼
redirect  GET /result/<meeting_id>
  │
  ▼
routes/meeting.py  result()
  get_meeting_detail(meeting_id, user_id)   ← ownership enforced (AND m.user_id=?)
  get_tasks_by_meeting(meeting_id)
  render_template("meeting/result.html", ...)
```

> **Blocking pipeline**: The entire pipeline runs synchronously inside the HTTP
> request. For large audio files this means multi-minute request times. The
> upload page shows a JS fake-progress spinner to mask the latency.

---

## 5. Database Schema

Access method: **raw `sqlite3`** — no ORM. Every function opens a fresh
connection via `get_connection()` (which sets `row_factory = sqlite3.Row`,
`PRAGMA foreign_keys = ON`, `PRAGMA journal_mode = WAL`) and closes it via
the context manager on exit.

```
users
  id            INTEGER PK AUTOINCREMENT
  username      TEXT NOT NULL UNIQUE
  email         TEXT NOT NULL UNIQUE
  password_hash TEXT NOT NULL
  created_at    TEXT DEFAULT datetime('now')

meetings
  id                  INTEGER PK AUTOINCREMENT
  user_id             INTEGER NOT NULL → users.id ON DELETE CASCADE
  title               TEXT DEFAULT 'Untitled Meeting'
  sentiment           TEXT DEFAULT 'Neutral'
  meeting_date        TEXT DEFAULT date('now')
  audio_filename      TEXT
  graph_path          TEXT   (relative: "graphs/graph_m{id}.png")
  bar_chart_path      TEXT
  donut_chart_path    TEXT
  status_chart_path   TEXT
  platform            TEXT   (nullable, e.g. 'google')
  external_meeting_id TEXT   (nullable)
  created_at          TEXT DEFAULT datetime('now')

transcripts                           (1-to-1 with meetings)
  id         INTEGER PK AUTOINCREMENT
  meeting_id INTEGER NOT NULL UNIQUE → meetings.id ON DELETE CASCADE
  content    TEXT

summaries                             (1-to-1 with meetings)
  id         INTEGER PK AUTOINCREMENT
  meeting_id INTEGER NOT NULL UNIQUE → meetings.id ON DELETE CASCADE
  content    TEXT

tasks                                 (many-to-1 with meetings)
  id             INTEGER PK AUTOINCREMENT
  meeting_id     INTEGER NOT NULL → meetings.id ON DELETE CASCADE
  description    TEXT
  assigned_to    TEXT DEFAULT 'Team'
  due_date       TEXT DEFAULT 'Not specified'
  status         TEXT DEFAULT 'pending'
  keyword        TEXT
  priority       TEXT DEFAULT 'Low'
  assigned_email TEXT (nullable)

oauth_connections                     (many-to-1 with users)
  id                INTEGER PK AUTOINCREMENT
  user_id           INTEGER NOT NULL → users.id ON DELETE CASCADE
  provider          TEXT NOT NULL
  access_token_enc  TEXT NOT NULL
  refresh_token_enc TEXT (nullable)
  expires_at        TEXT (nullable)
  scope             TEXT (nullable)
  created_at        TEXT DEFAULT datetime('now')
  updated_at        TEXT DEFAULT datetime('now')
  UNIQUE(user_id, provider)

meeting_participants                  (many-to-1 with meetings)
  id          INTEGER PK AUTOINCREMENT
  meeting_id  INTEGER NOT NULL → meetings.id ON DELETE CASCADE
  name        TEXT NOT NULL
  email       TEXT (nullable)
  source      TEXT DEFAULT 'manual'
  external_id TEXT (nullable)
  joined_at   TEXT (nullable)
  left_at     TEXT (nullable)
```

---

## 6. Coding Conventions Observed

- **Application factory pattern** — `create_app()` in `app.py`; blueprints
  registered inside the factory.
- **Lazy ML model loading** — models are module-level `None` globals in
  `services/pipeline.py`; loaded on first pipeline call, cached for the
  process lifetime. `services/__init__.py` is intentionally empty to prevent
  eager imports.
- **Raw sqlite3 everywhere** — all queries use `?` placeholders (safe from
  SQL injection). No ORM.
- **`with get_connection() as conn`** pattern for auto-commit/rollback; a few
  sites call `conn.close()` manually — these are consistent but diverge from
  the context-manager style.
- **Ownership enforcement** — every query that fetches meeting-scoped data
  includes `AND m.user_id = ?` with the session user ID.
- **Blueprint-per-concern** — `auth`, `dashboard`, `meeting`, `search`, `integrations` each
  live in their own file and register on the Flask app in `create_app()`.
  When adding a new blueprint with a manual test app (as in `test_meeting_deletion.py`),
  register `integrations_bp` in that app too so `base.html`'s `url_for('integrations.index')` resolves.
- **Config via `config.py`** — all paths, model names, and tunable constants
  live in one file; each can be overridden via an environment variable.
  `config.py` calls `load_dotenv()` at import time so `.env` is always loaded.
- **Static graphs served directly** — generated PNGs are written to
  `static/graphs/` and served as ordinary static files by Flask.
- **Type hints** on all public function signatures.
- **Docstrings** on all public functions; keep them when editing.

---

## 7. Rules for Future Work

1. **Never run `seed.py` without explicit user confirmation.** It
   unconditionally deletes every row in every table. The script now requires
   typing 'yes' at a prompt (or passing `--yes`), and refuses to run when
   `CLARIO_DEBUG=False` unless `--force` is also supplied.

2. **Do not load ML models at import time.** Keep the lazy-loading pattern in
   `services/pipeline.py`. Never add `import whisper` or
   `from transformers import …` at module top-level in service files.

3. **Mock heavy models in tests.** Always mock Whisper and BART:
   use `unittest.mock.patch` on `services.pipeline._get_whisper` and
   `services.pipeline._get_summariser`.  spaCy (`en_core_web_sm`) may be
   loaded for real — it is small enough to include in CI.  The `real_nlp`
   session-scoped fixture in `tests/conftest.py` loads it once per test
   session and wires it into `services.pipeline._get_nlp`.  Never load
   Whisper or BART weights in any automated test.

4. **Keep function signatures backward compatible unless told otherwise.**
   Add new parameters as keyword-only with defaults. Never remove or reorder
   existing positional parameters.

5. **All DB queries must include user ownership checks** (`AND m.user_id = ?`)
   when accessing meeting-scoped data. Never fetch a meeting by ID alone.

6. **Sanitise filenames before saving uploads.** `routes/meeting.py` calls
   `werkzeug.utils.secure_filename()` on every upload, then saves to disk as
   `u{user_id}_{uuid4().hex}{ext}` so filenames never collide and path-traversal
   attacks are impossible. The user's original filename is kept in
   `meetings.audio_filename` for display only. The extension is validated against
   `ALLOWED_AUDIO` *after* sanitisation.

7. **Upload size is enforced via `MAX_CONTENT_LENGTH`.** The limit (default
   200 MB) is read from `CLARIO_MAX_UPLOAD_MB` env var in `config.py` and set
   on `app.config["MAX_CONTENT_LENGTH"]` in `create_app()`. Flask returns HTTP
   413 automatically; a custom `request_entity_too_large` handler in `app.py`
   renders `error.html` with a clear message. Never add manual size-checking
   inside route handlers.

8. **CSRF protection is active via Flask-WTF CSRFProtect.** Every POST form
   includes `{{ csrf_token() }}`. JavaScript fetch/XHR calls are wrapped by
   `setupCSRF()` in `static/js/main.js`, which injects the `X-CSRFToken`
   header automatically. Tests disable CSRF with
   `flask_app.config["WTF_CSRF_ENABLED"] = False`; use a separate `csrf_app`
   fixture when testing CSRF enforcement specifically.

9. **The pipeline is synchronous / blocking.** Do not introduce `time.sleep()`
   or busy-waits as a workaround. Any long-running work should be moved to a
   task queue (Celery, RQ) with a proper async result endpoint.

10. **`SECRET_KEY` must come from the environment in production.** `config.py`
    raises a `RuntimeError` on startup if `CLARIO_SECRET_KEY` is unset and
    `CLARIO_DEBUG` is `False`. Never hard-code a secret key in source.

11. **`next` redirect on login is validated.** `routes/auth.py login()` only
    follows `next` when it starts with a single `/` (not `//`) and has no
    scheme or netloc. Any other value falls back to `/dashboard`. Never relax
    this check.

12. **Charts are regenerated per-upload and on task completion.** The status
    pie chart is also regenerated in `complete_task()`. Keep this pattern — do
    not cache chart data in the session or in a separate background thread.

13. **`utils/__init__.py` eagerly imports from `visualization.py`**, which
    triggers `import os` and `import json` but NOT matplotlib (matplotlib is
    imported inside each function). This is safe; keep it that way.

14. **Summarization is token-aware map-reduce; do not reintroduce hard truncation.**
    `summarize_text()` chunks the full transcript by sentence boundary (<= 700
    tokens per chunk, env `CLARIO_SUMMARY_CHUNK_TOKENS`), summarizes each chunk,
    and reduces hierarchically (up to 3 levels). deduplicate_sentences() removes repeated or near-duplicate sentences (difflib ratio > 0.9) after the map step and on final output. SUMMARY_MAX_INPUT_CHARS is
    deprecated -- it remains in the function signature for backward compatibility
    but is never applied as `text[:max_input_chars]`. Do not reintroduce that cap.
    `smart_merge()` in `services/summarization.py` appends action items not already
    in the summary and is called after map-reduce completes. Tests use
    `FakeModel`/`FakeTokenizer` (1 word = 1 token) -- never load real BART weights
    in automated tests.

15. **Participant matching and meeting platform imports.**
    - `match(assignee_name, participants)` in `services/participant_matching.py`
      implements a three-tier matching hierarchy: normalised exact match, unique
      first-name match, and difflib fuzzy match (ratio >= 0.8 with unique best
      candidate). Pronouns ("he", "she", "they", etc.) and generic group terms
      ("Team", "all") must NEVER be matched to a participant.
    - `rematch_meeting_tasks(meeting_id, user_id)` reapplies matching to existing
      meeting tasks, updating only tasks with matching participants and leaving
      others untouched.
    - Importing participants into an existing meeting (`POST /integrations/google/import`)
      enforces ownership (`m.user_id = ?`, returns 404 if unowned), is idempotent
      (`replace_meeting_participants`), and updates `meetings.platform` and
      `meetings.external_meeting_id`.

---

## 8. Sprint Working Rules

These rules apply to every task in every future sprint:

1. **Keep public function signatures backward compatible.** New parameters must
   be keyword-only with sensible defaults. Never remove or reorder existing
   positional parameters unless explicitly instructed.

2. **Run `pytest` before declaring any task done and show the result.**
   Install dev dependencies first: `pip install -r requirements-dev.txt`.
   A passing test suite is the minimum bar for "done".

3. **Update the affected CLAUDE.md sections after every change.** If you add
   a new folder, route, config variable, or rule, update the relevant section
   of this file in the same commit/diff.

4. **Never run `seed.py` without asking the user first.** The interactive
   confirmation in the script is a last-resort guard, not a substitute for
   human approval.

5. **Prefer small, reviewable diffs; no unrelated refactors.** Each task should
   touch only the files required to accomplish that task. If you notice a
   separate issue, file it as a note — do not fix it inline.

6. **Charts use matplotlib: always the Agg backend, guarded by a threading
   lock.** Every chart function in `utils/visualization.py` already calls
   `matplotlib.use("Agg")`. When adding new chart functions, also acquire a
   module-level `threading.Lock` around the matplotlib draw/save calls to
   prevent race conditions when multiple pipeline calls run concurrently in
   the future.

---

## 9. Security Architecture

- **CSRF Protection:** Flask-WTF `CSRFProtect` initialized in `create_app()`. Meta tag `<meta name="csrf-token">` in `base.html`, hidden `csrf_token` input in all POST forms. `static/js/main.js` automatically intercepts `fetch` and `XMLHttpRequest` to set `X-CSRFToken`. Disabled in test fixtures with `WTF_CSRF_ENABLED = False`, tested explicitly with `csrf_app` fixture.
- **Upload Hardening:** User-uploaded files are sanitised with `werkzeug.utils.secure_filename()`. Filenames saved on disk use `u{user_id}_{uuid4().hex}{ext}` to eliminate collisions and path traversal. Original filename is stored in SQLite for UI presentation. File extensions are validated against `ALLOWED_AUDIO` post-sanitisation.
- **Upload Size Limit:** `MAX_CONTENT_LENGTH` is derived from `CLARIO_MAX_UPLOAD_MB` (default: 200 MB). Over-limit uploads are caught by a 413 error handler rendering `error.html`.
- **Session Security:** Session cookies configured with `HTTPONLY=True`, `SAMESITE="Lax"`, and `SECURE=not DEBUG`.
- **Open Redirect Protection:** Login redirect (`?next=`) is strictly validated to ensure it starts with a single `/` and does not contain `//`, scheme, or netloc.
- **IDOR / Object-Level Access Control:** `/result/<int:meeting_id>` returns HTTP 404 (via `abort(404)`) instead of redirects or data leakage when accessed by an unauthenticated or non-owner user.
- **XSS Prevention:** Contextual escaping across templates. Jinja2's `|tojson` filter encodes server-side data passed to inline JavaScript, and helper `escapeHtml()` sanitises dynamic DOM insertions.

---

## 10. Model Benchmarks

Empirical performance evaluation across summarization configurations on CPU:

| Transcript | Model Configuration | Chunks | Levels | Summarize Time | Coverage (a) Alone | Coverage (b) + Smart-Merge |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Short (2.1k chars)** | `facebook/bart-large-cnn` (beams=4) | 1 | 1 | 15.43 s | 2 / 8 (25.0%) | 8 / 8 (100.0%) |
| **Short (2.1k chars)** | `facebook/bart-large-cnn` (beams=2) | 1 | 1 | 13.97 s | 4 / 8 (50.0%) | 8 / 8 (100.0%) |
| **Short (2.1k chars)** | `sshleifer/distilbart-cnn-12-6` (beams=4) | 1 | 1 | 10.21 s | 4 / 8 (50.0%) | 8 / 8 (100.0%) |
| **Long (27.6k chars)** | `facebook/bart-large-cnn` (beams=4) | 9 | 1 | 200.43 s | 8 / 15 (53.3%) | 15 / 15 (100.0%) |
| **Long (27.6k chars)** | `facebook/bart-large-cnn` (beams=2) | 9 | 1 | 157.22 s | 4 / 15 (26.7%) | 13 / 15 (86.7%) |
| **Long (27.6k chars)** | `sshleifer/distilbart-cnn-12-6` (beams=4) | 9 | 1 | 121.84 s | 7 / 15 (46.7%) | 15 / 15 (100.0%) |

**Findings & Decision:**
- `sshleifer/distilbart-cnn-12-6` is the production default: runs ~40% faster on CPU, halves load time (8.5s vs 18.4s), and achieves 100% topic coverage (15/15) after task smart-merge.
- Automated tests mock all heavy ML models using deterministic fakes, keeping CI test runs under 30 seconds without loading weights.

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
| Task / action-item extraction | spaCy NER + regex keyword matching + dependency parse + dateparser |
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
| `CLARIO_WHISPER_MODEL` | `Loudness-ai/urdu-sukhan-asr-v1` | Urdu + English + code-switching ASR (gated Hugging Face repo) |
| `CLARIO_HF_MODELS_DIR` | `data/hf-models/` | Where speech-model weights are stored. Kept off the small system cache |
| `CLARIO_STATIC_GRAPH_DIR` | `static/graphs/` | Meeting chart PNGs; on Render use the persistent disk path |
| `CLARIO_WHISPER_FALLBACK` | `small` | Public Whisper size used when the primary HF model cannot load |
| `CLARIO_WHISPER_TASK` | `translate` | Whisper fallback task. Sukhan always transcribes, then Urdu script is translated |
| `CLARIO_WHISPER_LANGUAGE` | `auto` | Force `en`/`ur` or `auto` for code-switching |
| `CLARIO_WHISPER_INITIAL_PROMPT` | *(bilingual hint)* | Optional context for mixed meetings |
| `CLARIO_ENABLE_TEXT_TRANSLATION` | `True` | Neural Marian Urdu→English only. The meeting lexicon still rewrites Urdu when this is False |
| `CLARIO_TRANSLATION_MODEL` | `Helsinki-NLP/opus-mt-ur-en` | Fallback text translator |
| `CLARIO_BART_MODEL` | `sshleifer/distilbart-cnn-12-6` | HuggingFace summarizer (~40% faster on CPU; override with facebook/bart-large-cnn) |
| `CLARIO_SPACY_MODEL` | `en_core_web_sm` | spaCy language model |
| `CLARIO_SUMMARY_CHUNK_TOKENS` | `700` | Max BART tokens per map-reduce chunk |
| `CLARIO_SUMMARY_MAX_INPUT_CHARS` | *(deprecated)* | Formerly hard-capped transcript chars; now ignored — full transcript is processed |
| `CLARIO_MAX_UPLOAD_MB` | `200` | Maximum upload size in megabytes |
| `CLARIO_FREE_TIER` | `False` | When `True` (Render free), refuse speech, BART, and Marian downloads and ask the user to paste a transcript. Local development stays `False` so Whisper and other models still load. Audio upload is allowed on the free plan only when the local transcription URL and token below are both set |
| `CLARIO_LOCAL_TRANSCRIPTION_URL` | `""` | Optional HTTPS tunnel to the laptop ASR service. Render forwards audio there and does not download Whisper |
| `CLARIO_LOCAL_TRANSCRIPTION_TOKEN` | `""` | Shared bearer token for that service. Never logged or sent to the browser. The laptop process reads the same variable |
| `CLARIO_DISK_RESERVE_MB` | `64` | Reject a new upload when free space would fall below this reserve. Render free sets 64; the paid blueprint sets 512 |
| `CLARIO_MODEL_DOWNLOAD_MIN_FREE_MB` | `4096` | Paid instances download a missing model only when this much space remains above the reserve |
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
├── requirements-core.txt   # Flask, spaCy, matplotlib, gunicorn, test deps — used by CI
├── requirements-ml.txt     # torch, transformers, openai-whisper (heavy, skipped in CI)
├── requirements-dev.txt    # dev/test dependencies (pytest, pytest-cov, jiwer)
├── pyproject.toml          # pytest + coverage configuration
├── .env.example            # Template — copy to .env and fill in values
├── render.yaml             # Render Blueprint — FREE plan only (default)
├── render.standard.yaml    # Paid Standard + disk (do not apply without billing)
├── build.sh                # Render build: core deps on free, full ML when not free
├── Procfile                # Gunicorn start command
├── runtime.txt             # Python 3.12.8 for Render
├── docs/deploy-render.md   # Free vs Standard deploy guide and hosting limits
├── seed.py                 # ⚠️ Destructive demo-data seeder (guarded)
├── generate_samples.py     # Generates sample MP3s via gTTS
│
├── docs/
│   └── integrations.md     # Google Meet / external integrations setup & OAuth guide
│
├── scripts/
│   ├── check_routes.py     # HTTP smoke-tester (needs live server)
│   ├── local_transcription_server.py # Laptop Whisper API (localhost only)
│   ├── fix_unicode.py      # One-shot script: strips cp1252-breaking emoji
│   ├── benchmark_whisper.py # Whisper model benchmark (load time, RTF, WER)
│   ├── generate_multilingual_samples.py # gTTS Urdu/English/mixed fixtures for eval
│   ├── eval_multilingual_transcription.py # WER matrix → docs/multilingual-transcription-eval.md
│   ├── compare_summarizer.py # Before/after BART summary quality comparison
│   ├── eval_summary.py     # Evaluate summarizer quality, runtime, and topic coverage
│   ├── eval_tasks.py       # Sentence-level task precision/recall → docs/task-detection-eval.md
│   ├── eval_meeting_tasks.py # Meeting-level task precision/recall → docs/meeting-task-eval.md
│   ├── run_eval_matrix.py  # Matrix runner across models, beams, and transcripts
│   └── create_long_meeting_fixture.py # Optional generator for long test fixtures
│
├── tests/
│   ├── __init__.py
│   ├── fixtures/           # long_meeting.txt, extended_meeting.txt, topics JSON files
│   ├── conftest.py         # Fixtures: app, client, logged_in_client, ML patches
│   ├── test_routes.py      # Route-level tests (public + authenticated flows)
│   ├── test_remote_transcription.py # Laptop ASR client and local service tests
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
│   ├── live.py             # Blueprint: /live/start, /live/<token>, recording, invite, end
│   ├── tasks.py            # Blueprint: /tasks status board and shared-name confirmation
│   ├── search.py           # Blueprint: /search
│   ├── utils.py            # @login_required decorator + current_user(); drops a session whose user row was wiped (Render free disk)
│   └── integrations.py     # Blueprint: /integrations, /google/connect|callback|disconnect|meetings|participants|import
│
├── services/
│   ├── __init__.py         # Intentionally empty (no eager ML imports)
│   ├── jobs.py             # ThreadPoolExecutor background pipeline job runner & hooks
│   ├── pipeline.py         # run_clario_pipeline() — five report phases
│   ├── evaluation.py       # Phase 5: ROUGE + F1 offline evaluation
│   ├── notifications.py    # Post-meeting assignee email + summary PDF, join-link invites
│   ├── live.py             # Live room: per-account microphone tracks → named transcript
│   ├── preprocessing/      # Phase 1: input and preprocessing
│   │   ├── transcription.py    # Whisper load/transcribe
│   │   ├── remote_transcription.py  # Forward audio to the laptop service
│   │   ├── local_asr.py        # Authenticated localhost transcription API
│   │   ├── cleaning.py         # Fillers, noise markers, timestamps
│   │   ├── roman_urdu.py       # Roman Urdu meeting lexicon → English
│   │   └── urdu_norm.py        # Urdu phrase table before Marian fallback
│   ├── nlp/                # Phase 2: NLP and entity extraction
│   │   ├── utterances.py       # Decision / Task Assignment / General Discussion
│   │   ├── task_detection.py   # spaCy NER + regex + dep-parse + dateparser
│   │   ├── task_verification.py # Speaker context, evidence, duplicates, review list
│   │   ├── summarization.py    # BART load/summarize
│   │   └── participant_matching.py # Attendee matching hierarchy
│   ├── knowledge_graph/    # Phase 3: Decision knowledge graph (NetworkX)
│   │   └── builder.py          # Person, Task, Decision, Deadline, Meeting
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
    ├── meeting/
    │   ├── upload.html     # Upload form (audio file or pasted transcript)
    │   └── result.html     # Meeting detail: transcript, summary, tasks, graphs
    ├── live/
    │   ├── start.html      # Create a join link and email invites
    │   └── room.html       # Join, copy link, record microphone, end meeting
    ├── tasks/
    │   └── index.html      # Shared task board and name-collision confirmation
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
  3. insert_meeting()  → meeting row in DB with status='queued' (returns meeting_id)
  4. submit_pipeline_job(...)  → non-blocking submission to ThreadPoolExecutor (services/jobs.py)
  5. redirect immediately to GET /result/<meeting_id>
        │
        ▼ (in background thread pool)
services/jobs.py  _run_pipeline_job()
  1. update_meeting_status(meeting_id, status='processing', progress_stage=...)
  2. run_clario_pipeline(audio_path, text_input, meeting_id, ..., progress_callback=...)
        │
        ├─ 1. PREPROCESSING (stage: "transcribing")
        │      transcribe_audio(audio_path, model=_get_whisper(), task=translate)
        │      → When CLARIO_LOCAL_TRANSCRIPTION_URL and
        │        CLARIO_LOCAL_TRANSCRIPTION_TOKEN are both set, the audio file is
        │        POSTed to that tunnel with a bearer token. Render does not load
        │        Whisper. Connect timeout is 15s and the read timeout is 16 minutes.
        │        An offline service, a bad token, or a timeout fails that meeting
        │        unless the same upload also included pasted text
        │      → Primary ASR: Urdu Sukhan v1 (Urdu, English, code-switching)
        │      → Weights are stored in data/hf-models (CLARIO_HF_MODELS_DIR)
        │      → CLARIO_FREE_TIER=True returns before snapshot_download and asks
        │        the user to paste a transcript, unless the laptop service above is
        │        configured. Pasted text still runs. A paid
        │        instance downloads only when free space covers
        │        CLARIO_MODEL_DOWNLOAD_MIN_FREE_MB plus the disk reserve
        │      → If the primary HF model cannot load, CLARIO_WHISPER_FALLBACK
        │        is loaded through Transformers (openai/whisper-*) and the
        │        server log names it. openai-whisper is not used for that
        │        fallback because numba is blocked here
        │      → With no fallback configured, the primary load error is raised
        │      → Sukhan writes Urdu in Urdu script; prepare_meeting_turns()
        │        rewrites names, weekdays, and action verbs with the lexicon
        │        even when CLARIO_ENABLE_TEXT_TRANSLATION is false. Marian
        │        runs only for a line the lexicon cannot render, and only
        │        when that flag is on
        │      → FFmpeg auto-resolved via imageio-ffmpeg if not on PATH
        │      clean_transcript(): drop fillers (uh, um), noise markers, timestamps
        │
        ├─ 2. NLP (stage: "detecting_tasks", then "summarizing")
        │      verify_meeting_tasks() runs detect_tasks() on the full English transcript
        │      Speaker labels attach "I will" / "I promised" to that person
        │      A reply such as "I can do that" keeps the request sentence as evidence
        │      Repeated commitments collapse; a later deadline replaces the earlier one
        │      Unnamed commitments go to task_reviews instead of a guessed assignee
        │      "Name (email):" is read from the original paste before rendering
        │      drops the address, so follow-up mail can match that person
        │      verify_meeting_tasks receives the meeting's calendar day, and
        │      a resolved due_date_iso is what insert_tasks stores
        │      If spaCy cannot load (blocked native libs, or msgpack rejecting int keys), keyword detection still runs and the meeting is not marked failed
        │      On this PC, Windows Application Control blocks spacy/strings.pyd;
        │      that is an OS allow-list, and the regex detector is the path that runs
        │      → spacy.load("en_core_web_sm") [lazy, cached globally, thread-safe]
        │      → regex keyword matching with verb POS validation
        │      → NER for PERSON (assignee) and DATE/TIME (due date)
        │      → priority via PRIORITY_KEYWORDS + day-proximity heuristic
        │      summarize_text(transcript, summariser=_get_summariser())
        │      → AutoTokenizer + AutoModelForSeq2SeqLM.from_pretrained(bart)
        │        [lazy, cached globally, thread-safe]
        │      → Token-aware chunking: sentence-boundary chunks <= CLARIO_SUMMARY_CHUNK_TOKENS
        │        (default 700 tokens) with 2-sentence overlap
        │      → Map: summarize each chunk (num_beams=4, early_stopping=True)
        │      → Reduce: join partials; re-summarize if > chunk limit (max 3 levels)
        │      → Extractive summarization fallback when model is unavailable or OOM
        │      → smart_merge(): append tasks not in summary (post map-reduce)
        │
        ├─ 3. SAVE (stage: "saving")  — security/storage layer is SQLite + session auth
        │      insert_transcript(meeting_id, transcript)
        │      insert_summary(meeting_id, summary)
        │      insert_tasks(meeting_id, tasks)
        │      insert_decisions(meeting_id, decisions)
        │      update_meeting_sentiment(meeting_id, vibe)
        │
        └─ 4. KNOWLEDGE GRAPH + CHARTS (stage: "generating_charts")
               build_decision_graph() → Person, Task, Decision, Deadline, Meeting
               relations: ASSIGNED_TO, DECIDED_BY, DEPENDS_ON, RELATED_TO, DEADLINE, BELONGS_TO
               build_knowledge_graph(..., decision_graph=...) → static/graphs/graph_m{id}.png
               generate_assignee_bar_chart() → static/graphs/bar_m{id}.png
               generate_priority_donut_chart() → static/graphs/donut_m{id}.png
               generate_status_pie_chart() → static/graphs/status_m{id}.png
               set_meeting_graphs(meeting_id, paths...)
  3. update_meeting_status(meeting_id, status='completed', progress_stage='completed')
  4. Run post-completion hooks. send_meeting_followups emails each
     assignee their tasks and attaches a PDF of the summary, decisions,
     and the full task list. The meeting owner receives the PDF too.
     Mail prefers CLARIO_BREVO_API_KEY or CLARIO_RESEND_API_KEY (HTTPS).
     Brevo accepts the meeting host as From only when that address is an
     active sender (`GET /v3/senders`, `active: true`). The host's username
     is the display name and the same address is Reply-To. Summary mail and
     per-assignee task mail both use that host. An unverified host is
     refused in the meeting email log. Clario does not substitute
     `CLARIO_SMTP_FROM` or the recipient. Live invitations use the same
     host (`send_invite_emails(..., host_email=session email)`).
     SMTP (CLARIO_SMTP_*) is the local fallback. Render free blocks ports
     587 and 465, so Gmail SMTP cannot send there. Without a transport,
     copies go to CLARIO_EMAIL_DIR or data/outbox in debug.
  5. On error: update_meeting_status(meeting_id, status='failed', error_message=...)
  │
  ▼
Browser  GET /result/<meeting_id>
  │
  ├─ If status == 'queued' | 'processing':
  │    Render loading UI; JS polls GET /meeting/<id>/status every 2 seconds
  │    When completed: window.location.reload() renders final results
  │    When failed: display error message with link to upload again
  │
  ├─ If status == 'failed':
  │    Render failure UI with error message and 'Upload Again' button
  │
  └─ If status == 'completed':
       get_meeting_detail(meeting_id, user_id)   ← ownership enforced (AND m.user_id=?)
       get_tasks_by_meeting(meeting_id)
       get_decisions_by_meeting(meeting_id)
       render_template("meeting/result.html", ...)
```

> **Asynchronous pipeline**: File uploads insert the meeting row with `status='queued'`,
> dispatch the heavy AI processing to an in-process `ThreadPoolExecutor` worker pool
> (`services/jobs.py`), and immediately redirect the user to `/result/<meeting_id>`. The
> browser polls `/meeting/<id>/status` every 2s for live stage-driven progress updates.

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

meeting_emails                        (many-to-1 with meetings)
  id         INTEGER PK AUTOINCREMENT
  meeting_id INTEGER NOT NULL → meetings.id ON DELETE CASCADE
  recipient  TEXT NOT NULL
  subject    TEXT
  status     TEXT NOT NULL          (sent, saved, failed, skipped, waiting)
  detail     TEXT
  created_at TEXT DEFAULT datetime('now')

decisions                             (many-to-1 with meetings)
  id          INTEGER PK AUTOINCREMENT
  meeting_id  INTEGER NOT NULL → meetings.id ON DELETE CASCADE
  content     TEXT NOT NULL
  confidence  REAL DEFAULT 0.8
  source_text TEXT

tasks                                 (many-to-1 with meetings)
  id             INTEGER PK AUTOINCREMENT
  meeting_id     INTEGER NOT NULL → meetings.id ON DELETE CASCADE
  description    TEXT
  assigned_to    TEXT DEFAULT 'Team'
  due_date       TEXT DEFAULT 'Not specified'
  status            TEXT DEFAULT 'pending'  (pending, started, doing, completed)
  keyword           TEXT
  priority          TEXT DEFAULT 'Low'
  assigned_email    TEXT (nullable)
  assigned_by_email TEXT (nullable; account that spoke the assignment)
  source_text       TEXT (spoken sentence; evidence, not a paraphrase)
  evidence_time     TEXT (timestamp from the transcript, when one was present)
  source_context    TEXT (earlier line a short reply was answering)

task_reviews                          (many-to-1 with meetings)
  Commitments that are not saved as tasks because the assignee, deadline,
  or meaning is unclear, plus duplicates and deadlines that were restated.
  kind, description, assigned_to, due_date, source_text, source_context,
  evidence_time, reason. Meeting delete cascades.

live_rooms / live_members / live_recordings / live_messages / live_signals
  A host creates a join token. Logged-in participants are stored with the
  account name and email. Each person's newest microphone file is one row
  in live_recordings. Chat lines are stored in live_messages. Ending the room
  creates a normal meetings row.

meeting_speaker_turns
  One row per transcribed speaker track (name, email, spoken text).

task_confirmations
  Pending when an assignee name matches two or more participant emails.
  Mail waits until the assigner or the meeting host chooses one.

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
   session and wires it into `services.pipeline._get_nlp`.  If Windows
   Application Control blocks `spacy/strings.pyd`, that fixture returns
   None and `tests/test_task_detection.py` skips.  Never load
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

9. **Asynchronous background-thread design.** AI pipeline execution is non-blocking,
   managed by an in-process `ThreadPoolExecutor` worker pool (`services/jobs.py`,
   `CLARIO_PIPELINE_WORKERS=1` by default). State transitions follow
   `queued` -> `processing` -> `completed` / `failed` in the database.
   For testing, setting `CLARIO_SYNC_JOBS=True` runs jobs synchronously inline.
   Do not introduce `time.sleep()` or busy-waits. For production scaling across
   multiple server instances, this thread pool design is architected to transition
   seamlessly to distributed task queues (Celery or RQ with Redis).

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
    `smart_merge()` in `services/nlp/summarization.py` appends action items not already
    in the summary and is called after map-reduce completes. Tests use
    `FakeModel`/`FakeTokenizer` (1 word = 1 token) -- never load real BART weights
    in automated tests.

15. **Participant matching and meeting platform imports.**
    - `match(assignee_name, participants)` in `services/nlp/participant_matching.py`
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

16. **Task detection algorithm (`services/nlp/task_detection.py`).**
    The detector runs five stages on each sentence in order:
    1. **Negation guard** — skip sentences containing `will not / won't / nobody /
       no one / doesn't / didn't / never / cannot`.
    2. **Hypothetical guard** — skip sentences starting with `if / whether` or
       containing `would / could / might / may` without an obligation cue.
       Exception: polite-indirect requests (`"It would be great if [Name] could…"`)
       are explicitly exempted by `_POLITE_REQUEST_RE`.
    3. **Question guard** — skip bare questions (ends `?` or auxiliary-led) unless
       a strong obligation cue (`can you`, `make sure`, etc.) is also present.
    4. **Keyword matching** — `STRONG_CUES` (will, needs to, must, has to, please,
       deadline, responsible for, can you, let's, make sure, follow up, ensure,
       promised to…) or `STRONG_ACTION_VERBS` (submit, fix, deliver, deploy, etc.).
       Weak verbs (review, check, discuss) only count when a strong cue is also present.
       Without spaCy, a listed verb used as a noun ("test cases", "the release",
       "an update") is not treated as a task.
    5. **Dep-parse fallback** — uses `token.dep_` to detect imperatives (ROOT VERB
       with no subject and not past tense) or future/modal patterns (ROOT VERB +
       modal aux `will/must/shall` + named nsubj). Past-tense ROOT verbs (tag
       `VBD/VBN`) are excluded.

    Relative date phrases (`tomorrow`, `next Friday`, day names) and absolute
    dates (`November 20`, `2026-11-20`) are captured in `due_date`. Relative
    phrases are resolved to ISO `YYYY-MM-DD` via `dateparser` when a
    `meeting_date` is passed. The ISO form goes into `due_date_iso`.

    **Verification** (`services/nlp/task_verification.py`) runs after detection
    on the whole meeting. It keeps the spoken sentence and any timestamp,
    attributes first-person commitments to the current speaker, collapses
    repeats, and stores uncertain lines in `task_reviews`. It does not invent
    a person's name when the transcript does not say who will do the work.
    `verify_meeting_tasks(transcript, tasks=None, *, nlp=None, participants=None,
    meeting_date=None, segments=None)` is keyword-only after the transcript.
    `prepare_meeting_turns()` keeps each line's original wording as `source_text`,
    moves timestamps onto tasks, and rewrites Arabic-script lines with `urdu_norm`
    or the Roman Urdu lexicon, keeping the gloss that preserves more of the
    commitment. It always runs, including when `CLARIO_ENABLE_TEXT_TRANSLATION`
    is false; that flag only skips the Marian model. A transliterated name is
    aligned to a Latin speaker label already on the transcript. A named "please"
    is saved for that person even when nobody answers; a question still waits
    for the reply. Agreement words (`بالکل`, `ٹھیک ہے`, okay, absolutely) are
    never assignees or speaker names. A later please-line from the same speaker
    keeps the person just named when the line itself has no new name. A line
    that is still in Arabic script goes to review instead of becoming a task.
    Code-switched imperatives such as `add کریں` and `fix کریں` become
    please-add and please-fix. Marian runs only when Arabic script remains
    and text translation is enabled.

    **Public signatures are locked** — `detect_tasks(text, nlp=None, nlp_model=None,
    *, participants=None, meeting_date=None)` and `get_task_summary(tasks)`. Add
    new behaviour only via keyword-only parameters with defaults. Never copy fixture
    sentences into rules; tune only on the dev split of `tests/fixtures/task_labels.json`.

    **Eval scripts:** `python scripts/eval_tasks.py` prints precision/recall/F1 on
    dev (20) and held-out (10) sentence splits and appends a dated section to
    `docs/task-detection-eval.md`. Always record baseline before changing the detector.
    `python scripts/eval_meeting_tasks.py` scores annotated English, mixed, and
    Urdu meetings in `tests/fixtures/meeting_actions.json` and appends missed and
    incorrect tasks to `docs/meeting-task-eval.md`. Urdu lines use the phrase
    table only; a line it cannot render is a miss, not a neural-model guess.

17. **Live meetings and the shared task board.**
    - `POST /live/start` creates `live_rooms` and can email the join link.
      The invitation is sent from the logged-in host. From and Reply-To
      are that account's email, and the display name is the account name.
      Brevo must already list that address as an active sender. In-room
      invites use the same host. API keys stay on the server.
      `GET /live/<token>` renders the Meet-style room (`templates/live/room.html`,
      `static/js/live-room.js`): stage (local camera preview), sidebar (People |
      Chat), bottom toolbar (mic, camera, record, save, host end). Recording does
      **not** auto-start; the user enables mic/camera then taps Record. Uploads go
      to `POST /live/<token>/recording` (audio `.webm` / `.ogg`). In-room chat
      uses `live_messages` via `GET /live/<token>/messages` and
      `POST /live/<token>/chat` (form `csrf_token` + `message`). Participant
      video and audio use a mesh of peer connections. Admitted people appear
      together in one gallery. Audio and video are negotiated as soon as
      someone is in the meeting, so a mic or camera turned on afterwards still
      reaches everyone. Remote video stays muted so the browser can show it
      immediately; their voice plays on a separate audio element. A polite
      peer asks the offerer to renegotiate (`refresh`) instead of sending a
      new offer every few seconds. The browser asks for the mic and camera on entry.
      Signaling is stored in `live_signals` via `GET|POST /live/<token>/signal`.
      `GET` with `bootstrap=1` returns a cursor and does not replay old
      signals; the participant with the lower user id keeps sending a fresh
      offer until the other side answers. Roster polling uses `GET /live/<token>/roster`.
      **Waiting room:** guests join as `pending` on `live_members.admission_status`;
      host approval sets `host_admitted_at` (legacy ``admitted`` rows without that
      timestamp count as waiting). Guests POST `/live/<token>/request` to register;
      host admits via `POST /live/<token>/admit` or denies via `POST /live/<token>/deny`;
      guests poll `GET /live/<token>/admission` until `admitted`.
      **Record** saves **microphone audio only** (for Whisper); live camera is not
      uploaded unless present in the audio file format.
      The name saved on the room page is the speaker label. Everyone who joined
      is stored as a participant, even if they did not speak. The host ends the
      room; `services/live.py` transcribes each track, prefixes `Name (email):`,
      and submits that transcript to the existing pipeline.
    - The upload form collects `attendee_name` / `attendee_email` rows.
      `roster_from_pairs()` stores them as meeting participants before the
      pipeline runs, and those addresses are what task matching uses.
    - `ambiguous_matches()` reports a tie. A spoken name that matches nobody
      on that roster is held too. `send_meeting_followups` writes
      `task_confirmations` and status `waiting` instead of sending. The
      uploader (meeting host) or the assigner confirms on the result page or
      `/tasks`, and mail is sent only after no choices remain.
    - `/tasks` lists every task in meetings the user owns, joined, or was assigned
      in. Only the assignee email can set status to `pending`, `started`,
      `doing`, or `completed`. Other people see that status and cannot change it.

18. **Hybrid transcription (Render free + laptop Whisper).**
    `scripts/local_transcription_server.py` binds to `127.0.0.1` only and
    serves `POST /transcribe`. It loads the first complete checkpoint already
    under `CLARIO_HF_MODELS_DIR` (configured model, then
    `CLARIO_WHISPER_FALLBACK`) and sets `HF_HUB_OFFLINE` so it cannot download
    weights. It does not import spaCy or the summarizer. Publish it with a
    tunnel you manage. Clario calls it only when both
    `CLARIO_LOCAL_TRANSCRIPTION_URL` and `CLARIO_LOCAL_TRANSCRIPTION_TOKEN`
    are set. The token travels in the `Authorization` header, is compared
    with a hash, and is never written to logs, meeting errors, or templates.
    `CLARIO_FREE_TIER=True` still blocks model downloads on Render. With the
    pair unset, free-tier audio uploads stay refused and pasted transcripts
    still run. In-process Whisper is unchanged when the pair is unset.
    Live recordings use the same choice.

---

## 8. Design system (Sage + Warm Ivory)

Server-rendered UI uses `static/css/style.css` and `templates/macros/icons.html`
(inline SVG, no icon font). Fonts: **Sora** (headings) and **DM Sans** (body) via
`base.html`.

| Token | Value | Use |
|---|---|---|
| `--color-forest` | `#263A35` | Primary text, primary buttons, video stage |
| `--color-sage` | `#829B8A` | Accents, links, chart series, success |
| `--color-mint-soft` | `#DCE8DF` | Table headers, hovers, subtle fills |
| `--color-ivory` | `#F7F5EF` | Page background |
| `--color-white` | `#FFFFFF` | Cards, nav pill, inputs |
| `--bg` / `--surface` | ivory / white | Layout surfaces |
| `--text-muted` | `#5F726B` | Secondary copy |
| `--accent-coral` | `#8B6348` | Warm accent (priority, tense vibe) |

**Components:** floating pill navbar (`navbar-shell`), buttons (`btn-primary`,
`btn-outline`, `btn-ghost`), `card` + `card-pad` / `card-flush`, `form-control`,
`alert-*`, badge variants (`badge-mint`, `badge-lilac`, `badge-queued`, …),
`empty-state`, `bento-stats` / `bento-main`, `listening-orb` + CSS waveform on
landing and live room.

**Motion:** CSS only (`page-rise`, hover lift, orb/waveform). Respect
`prefers-reduced-motion`. No Framer Motion.

**Layout helpers:** `page-title`, `page-lead`, `back-link`, `container-narrow`,
`container-medium`, `container-wide`. Avoid inline styles in templates — extend
`style.css` instead.

**Mobile:** `@media (max-width: 768px/480px)` stacks forms and dashboard grids,
makes the live-room toolbar a safe-area scroll strip, and shrinks the video
stage for phones. Cache-bust `style.css` / `live-room.js` query `v=` when those
files change.

---

## 9. Sprint Working Rules

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

## 10. Security Architecture

- **CSRF Protection:** Flask-WTF `CSRFProtect` initialized in `create_app()`. Meta tag `<meta name="csrf-token">` in `base.html`, hidden `csrf_token` input in all POST forms. `static/js/main.js` automatically intercepts `fetch` and `XMLHttpRequest` to set `X-CSRFToken`. Disabled in test fixtures with `WTF_CSRF_ENABLED = False`, tested explicitly with `csrf_app` fixture.
- **Upload Hardening:** User-uploaded files are sanitised with `werkzeug.utils.secure_filename()`. Filenames saved on disk use `u{user_id}_{uuid4().hex}{ext}` to eliminate collisions and path traversal. Original filename is stored in SQLite for UI presentation. File extensions are validated against `ALLOWED_AUDIO` post-sanitisation.
- **Upload Size Limit:** `MAX_CONTENT_LENGTH` is derived from `CLARIO_MAX_UPLOAD_MB` (default: 200 MB). Over-limit uploads are caught by a 413 error handler rendering `error.html`.
- **Session Security:** Session cookies configured with `HTTPONLY=True`, `SAMESITE="Lax"`, and `SECURE=not DEBUG`.
- **Open Redirect Protection:** Login redirect (`?next=`) is strictly validated to ensure it starts with a single `/` and does not contain `//`, scheme, or netloc.
- **IDOR / Object-Level Access Control:** `/result/<int:meeting_id>` returns HTTP 404 (via `abort(404)`) instead of redirects or data leakage when accessed by an unauthenticated or non-owner user.
- **XSS Prevention:** Contextual escaping across templates. Jinja2's `|tojson` filter encodes server-side data passed to inline JavaScript, and helper `escapeHtml()` sanitises dynamic DOM insertions.

---

## 11. Model Benchmarks

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

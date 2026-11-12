# Deploy Clario on Render

This guide prepares a production deploy without changing Clario’s SQLite
schema or AI pipeline. Models stay lazy-loaded. Local `data/` and `uploads/`
are not uploaded to Git; a Render disk holds them in production.

## What this does *not* do

- It does **not** migrate SQLite to MongoDB or Postgres.
- It does **not** claim the app is live until you deploy and smoke-test it.
- It does **not** commit `.env` or any secrets.

## Hosting limits

The checked-in `render.yaml` targets the **Free** web plan only (no paid
disk, no Standard/Pro). That avoids charges.

| Resource | Free web (this Blueprint) | Paid (optional later) |
|---|---|---|
| RAM | 512 MB | ≥ 2 GB for Torch + Whisper + DistilBART |
| Disk | Ephemeral (data lost on spin-down) | Persistent disk at `/var/data` |
| Idle | Spins down after ~15 min | Always-on |
| ML | **Not loaded** (`CLARIO_FREE_TIER=True` → core deps only) | Full `requirements.txt` |
| What works | Register/login, paste transcript, extractive summary, spaCy tasks, UI | Audio ASR + neural summary |

Sukhan / Whisper / DistilBART need a paid instance and a disk. Do not add those
to `render.yaml` without explicitly choosing a paid plan.

There is **no Zoom API** in this repo. Integrations are Google Meet OAuth and
Clario’s own live rooms.

## One-time local checks

1. Confirm `.env` is gitignored and never committed.
2. Keep a backup of `data/database/clario.db` before any production cutover.
3. Push the deployment files (`render.yaml`, `build.sh`, `Procfile`, …) to GitHub.

## Deploy with the Blueprint

1. Open [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect the GitHub repo that contains this project.
3. Review `render.yaml`. Confirm:
   - Plan is **Standard** (or higher), not Free.
   - Disk mount is `/var/data` (20 GB).
4. In the service **Environment** tab, set every `sync: false` variable (see below).
5. Under **Native Environment** / apt packages, add **`ffmpeg`**.
6. Deploy. Watch the build log for `spacy download` success.
7. After the service shows Live, open `https://<service>.onrender.com/health`.
   Expect `{"status":"ok"}`.
8. Register a user, paste a short English transcript (no audio first), and
   confirm a meeting result page. Then try a short audio upload.

## Environment variables

### Required

| Variable | Purpose |
|---|---|
| `CLARIO_SECRET_KEY` | Flask sessions. Blueprint can auto-generate; keep it stable. |
| `CLARIO_DEBUG` | Must be `False` in production. |
| `CLARIO_DATA_DIR` | `/var/data` |
| `CLARIO_DB_PATH` | `/var/data/database/clario.db` |
| `CLARIO_UPLOAD_DIR` | `/var/data/uploads` |
| `CLARIO_HF_MODELS_DIR` | `/var/data/hf-models` |
| `CLARIO_STATIC_GRAPH_DIR` | `/var/data/graphs` (meeting chart PNGs) |

### Strongly recommended

| Variable | Purpose |
|---|---|
| `CLARIO_WHISPER_FALLBACK` | `small` (public Whisper when Sukhan is blocked) |
| `CLARIO_BART_MODEL` | `sshleifer/distilbart-cnn-12-6` |
| `CLARIO_PIPELINE_WORKERS` | `1` |
| `CLARIO_MAX_UPLOAD_MB` | `100` |
| `HF_HUB_DISABLE_XET` | `1` |
| `HF_TOKEN` | Hugging Face token (needed for gated Sukhan) |

### Google Meet (optional)

| Variable | Purpose |
|---|---|
| `CLARIO_GOOGLE_CLIENT_ID` | OAuth web client ID |
| `CLARIO_GOOGLE_CLIENT_SECRET` | OAuth web client secret |
| `CLARIO_OAUTH_REDIRECT_BASE` | `https://<service>.onrender.com` (must match Google Console) |
| `CLARIO_TOKEN_ENCRYPTION_KEY` | Fernet key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

### Email follow-ups (optional)

| Variable | Purpose |
|---|---|
| `CLARIO_SMTP_HOST` | SMTP host |
| `CLARIO_SMTP_PORT` | Usually `587` |
| `CLARIO_SMTP_USER` / `CLARIO_SMTP_PASSWORD` | Credentials |
| `CLARIO_SMTP_FROM` | From address |
| `CLARIO_EMAIL_DIR` | `/var/data/outbox` when SMTP is unset |

## Local data vs Render

- Git does **not** ship `clario.db`, uploads, or model weights.
- A fresh Render deploy starts with an **empty** database unless you copy a
  backup onto the disk (via SSH/one-off job) yourself.
- Do **not** run `seed.py` on production unless you intentionally wipe data.

## Smoke test checklist

- [ ] `/health` returns 200
- [ ] Register / login works over HTTPS
- [ ] Paste-transcript meeting completes
- [ ] Audio upload either transcribes or fails with a clear model error (not a 500 blank page)
- [ ] Charts appear under the meeting result
- [ ] Google OAuth redirect (if configured) uses the `https://…onrender.com` base
- [ ] After a restart, meetings and uploads on `/var/data` are still present

## Gunicorn notes

Start command:

```bash
gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
```

One worker keeps a single copy of lazy ML models and matches the in-process
`ThreadPoolExecutor` pipeline. Raising `--workers` multiplies RAM use.

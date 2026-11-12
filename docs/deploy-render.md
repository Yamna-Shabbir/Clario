# Deploy Clario on Render

Two Blueprint files are kept in the repo:

| File | Plan | Disk | Billing |
|---|---|---|---|
| **`render.yaml`** (default) | **Free** | None (ephemeral) | No card required for Free compute |
| **`render.standard.yaml`** | Standard + 20 GB disk | `/var/data` | **Paid — do not apply unless you accept charges** |

This guide focuses on the **free** path. It does **not** claim the app is live
until you have a working public URL. It does not commit `.env` or touch your
local SQLite database.

## Why Render may ask for a card

Render’s Free web plan itself does not require a payment method. A card prompt
usually means one of these happened in the UI:

1. You selected a **paid** compute plan (Starter / Standard / Pro), or
2. The Blueprint included a **persistent disk** (paid), or
3. You opened **Billing** / tried to upgrade the workspace.

**Fix:** deploy with `render.yaml` only (`plan: free`, no `disk:` block), or
create a Web Service manually and choose **Free**. Skip any “Add payment
method” step. If the dashboard still forces a card before you can pick Free,
that is an account/UI gate on Render’s side — stop there; do not enter a card
for this project unless you choose to.

Without a card, exceeding free bandwidth/build minutes suspends free services
for the month instead of charging you.

## What runs on free hosting

`build.sh` with `CLARIO_FREE_TIER=True` installs **`requirements-core.txt` only**
(no Torch / Whisper / transformers). Features stay in the code; they are not
deleted.

| Feature | Free online |
|---|---|
| Landing, register, login, dashboard, search UI | Yes |
| Paste / type transcript → meeting pipeline | Yes (lightweight path) |
| Extractive summary (BART unavailable → fallback) | Yes |
| spaCy task extraction + verification (English) | Yes (if `en_core_web_sm` build succeeds) |
| Charts / knowledge graph PNGs | Yes until instance spin-down |
| `/health` | Yes |
| Audio upload → Whisper / Sukhan ASR | **No** (needs Torch + RAM + often ffmpeg) |
| DistilBART / BART neural summarization | **No** |
| Marian Urdu→English translation | **No** |
| Durable meetings/uploads across restarts | **No** (ephemeral disk; wiped on spin-down ~15 min idle) |
| Google Meet OAuth | Optional; needs Google client secrets |
| Invite / follow-up email | HTTPS only. Set `CLARIO_BREVO_API_KEY` and `CLARIO_SMTP_FROM` to a sender verified in Brevo. Gmail SMTP (587/465) is blocked on free |

Full ML + durable storage: use `render.standard.yaml` later on a paid plan.

## Deploy free configuration (exact steps)

### Option A — Blueprint (uses `render.yaml`)

1. Push the latest `dev` branch to GitHub (already contains `render.yaml`).
2. Open [dashboard.render.com](https://dashboard.render.com) and sign in.
3. Stay on a **Hobby** workspace. Do **not** add a payment method.
4. **New → Blueprint**.
5. Connect `Yamna-Shabbir/Clario`, branch **`dev`**.
6. Confirm the Blueprint preview shows:
   - Service name `clario`
   - Plan **`free`**
   - **No** disk / storage add-on
7. Apply the Blueprint. Do not switch the plan to Standard in the review UI.
8. Wait for the build (`pip install -r requirements-core.txt`, spaCy model).
9. Open `https://<service-name>.onrender.com/health` — expect `{"status":"ok"}`.
10. Open the site root, register a user, paste a short English transcript, and
    confirm a meeting result. Do not expect audio ASR to work on free.

### Option B — Manual Web Service (if Blueprint still nudges billing)

1. **New → Web Service** → select the same GitHub repo / `dev` branch.
2. Runtime: **Python**.
3. Build command: `bash build.sh`
4. Start command:
   ```text
   gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
   ```
5. Instance type: **Free**.
6. Add env vars from `render.yaml` (`CLARIO_DEBUG=False`, `CLARIO_FREE_TIER=True`,
   path vars, etc.). Let Render generate `CLARIO_SECRET_KEY`, or set one yourself.
7. Deploy. Test `/health` as above.

### Do not do (avoids charges)

- Do not apply `render.standard.yaml`.
- Do not attach a persistent disk.
- Do not change the instance type away from Free.
- Do not enter card details for this free demo.

## Environment variables (free)

| Variable | Value |
|---|---|
| `CLARIO_DEBUG` | `False` |
| `CLARIO_SECRET_KEY` | auto-generate or strong random string |
| `CLARIO_FREE_TIER` | `True` |
| `CLARIO_DATA_DIR` | `data` |
| `CLARIO_DB_PATH` | `data/database/clario.db` |
| `CLARIO_UPLOAD_DIR` | `uploads` |
| `CLARIO_HF_MODELS_DIR` | `data/hf-models` |
| `CLARIO_STATIC_GRAPH_DIR` | `static/graphs` |
| `CLARIO_PIPELINE_WORKERS` | `1` |
| `CLARIO_MAX_UPLOAD_MB` | `25` |
| `CLARIO_ENABLE_TEXT_TRANSLATION` | `False` |

## Smoke test checklist (after you have a URL)

- [ ] `/health` returns 200 and `{"status":"ok"}`
- [ ] Landing page loads over HTTPS
- [ ] Register / login works
- [ ] Paste-transcript meeting completes (extractive summary / tasks)
- [ ] Audio upload fails clearly or is skipped (expected on free)
- [ ] After ~15 minutes idle, cold start takes ~1 minute (expected)

## Paid Standard config (reference only)

See `render.standard.yaml`. Apply it only when you intentionally accept billing
for ≥2 GB RAM, a disk at `/var/data`, full `requirements.txt`, and ffmpeg.

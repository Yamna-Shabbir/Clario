# External Integrations Guide

This document outlines the setup, architecture, and security configuration for integrating Clario with external meeting platforms.

---

## Google Meet Integration Setup

Clario connects to Google Meet via Google Cloud OAuth 2.0 to import meeting spaces, metadata, and participant details.

### 1. Google Cloud Project & API Enablement

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., `Clario Meeting Assistant`) or select your existing project.
3. Navigate to **APIs & Services > Library**.
4. Search for and enable:
   - **Google Meet REST API** (`meetings.googleapis.com`)
   - *(Optional for calendar event lookups)* **Google Calendar API** (`calendar-json.googleapis.com`)

### 2. Configure OAuth Consent Screen

1. In Google Cloud Console, navigate to **APIs & Services > OAuth consent screen**.
2. Select **External** user type and click **Create**.
3. Fill in the required application branding details:
   - **App name**: `Clario`
   - **User support email**: Your developer email
   - **Developer contact information**: Your developer email
4. **Publishing Status**: Keep in **Testing** status during development.
5. **Test Users**:
   - Under the **Test users** tab, click **Add Users**.
   - Add every Google account (email address) that will authorize or test Google Meet connections with Clario.
   > ⚠️ **Important:** In **Testing** status, only explicitly added test users can authorize through the OAuth consent screen.
6. **Scopes**:
   - Click **Add or Remove Scopes**.
   - Select the approved scopes:
     - `https://www.googleapis.com/auth/meetings.space.readonly`
     - `https://www.googleapis.com/auth/calendar.readonly` *(if calendar meeting sync is enabled)*
   > ⚠️ **Scope Restriction:** Do not request any scopes beyond this approved list without prior consent screen updates.

### 3. Create OAuth 2.0 Client Credentials

1. Navigate to **APIs & Services > Credentials**.
2. Click **Create Credentials > OAuth client ID**.
3. Select **Web application** as the application type.
4. Set **Name** to `Clario Web Client`.
5. Under **Authorized redirect URIs**, add:
   ```
   http://127.0.0.1:5000/integrations/google/callback
   ```
   *(Ensure the host, port, and path match your local development environment exact URL).*
6. Click **Create** and securely copy the generated **Client ID** and **Client Secret**.

---

## 4. Token Expiration Warning (Testing Status)

> [!WARNING]
> When a Google OAuth App is in **Testing** publishing status, all generated **refresh tokens expire after approximately 7 days**.
>
> Users will need to re-authenticate / reconnect their Google account periodically in local development. When moving to production with non-expiring refresh tokens, the OAuth consent screen must undergo Google verification or be published to **In Production** with an Internal Google Workspace organization audience.

---

## 5. Environment Configuration

Add the following variables to your local `.env` file (see `.env.example`):

```bash
# Google OAuth 2.0 Web Client credentials
CLARIO_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
CLARIO_GOOGLE_CLIENT_SECRET=your-client-secret

# Base URL for OAuth redirect callback
CLARIO_OAUTH_REDIRECT_BASE=http://127.0.0.1:5000

# 32-byte URL-safe base64 Fernet key for encrypting OAuth tokens in SQLite
CLARIO_TOKEN_ENCRYPTION_KEY=
```

### Generating the Token Encryption Key

Generate a valid 32-byte URL-safe base64 Fernet key using Python:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the resulting output string into `CLARIO_TOKEN_ENCRYPTION_KEY` in `.env`.

---

## 6. Security Architecture & Data Storage

- **Database Persistence**:
  - `oauth_connections`: Stores per-user OAuth tokens.
  - `meeting_participants`: Stores meeting attendees linked to meetings.
- **At-Rest Token Encryption**:
  - OAuth `access_token` and `refresh_token` are symmetrically encrypted with Fernet (`AES-128-CBC` with `HMAC-SHA256`) using `services/integrations/crypto.py` before insertion into SQLite (`access_token_enc`, `refresh_token_enc`).
  - Raw tokens, keys, and ciphertexts are never logged.
  - If `CLARIO_TOKEN_ENCRYPTION_KEY` is missing or invalid, cryptographic operations fail safely with generic, secret-free error messages.

---

## 7. OAuth Flow Routes

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| `GET`  | `/integrations` | Required | Status page — shows connected/disconnected badge, scope list, Connect/Disconnect buttons |
| `GET`  | `/integrations/google/connect` | Required | Generates a random `state` token (stored in session), redirects to Google's authorization server |
| `GET`  | `/integrations/google/callback` | Required | Validates `state` (constant-time), handles `error=access_denied` and other Google errors, exchanges `code` for tokens, encrypts and stores them |
| `POST` | `/integrations/google/disconnect` | Required + CSRF | Best-effort token revocation at Google, then deletes the `oauth_connections` row |

### Anti-forgery State

`state` is a `secrets.token_urlsafe(32)` value stored in `session["google_oauth_state"]`. The callback compares received vs. stored with `secrets.compare_digest` and rejects any mismatch or missing state to prevent CSRF in the OAuth flow.

### Error handling in callback

| Condition | Action |
|-----------|--------|
| `error=access_denied` | Flash "cancelled or denied", redirect to `/integrations` |
| Any other `error` | Flash message includes the error code |
| State missing/mismatch | Flash "Invalid or expired OAuth state" |
| No `code` param | Flash "Authorization code missing" |
| Google returns 400 on token exchange | Flash "Could not complete Google authorization" |
| `invalid_grant` on refresh | Delete connection row, raise `TokenRevoked` |

---

## 8. `get_valid_access_token(user_id, db_path)` Helper

Located in `services/integrations/google_meet.py`. Used by any code that needs to call a Google API on behalf of a user.

- Returns a **valid, decrypted** access token (automatically refreshes if expired within 5 minutes).
- Raises `NotConnected` if the user has no stored connection.
- Raises `TokenRevoked` on `invalid_grant` (also deletes the stale row — user must reconnect).
- Raises `RateLimited` or `ApiError` for HTTP 429 or other network/API failures.

---

## 9. Manual Test Steps (Real Google Account)

1. Set in `.env`:
   ```
   CLARIO_GOOGLE_CLIENT_ID=<your-client-id>
   CLARIO_GOOGLE_CLIENT_SECRET=<your-client-secret>
   CLARIO_OAUTH_REDIRECT_BASE=http://127.0.0.1:5000
   CLARIO_TOKEN_ENCRYPTION_KEY=<generated Fernet key>
   ```
2. Add your Google account email as a **Test User** in GCP console → OAuth consent screen → Test users.
3. Run: `python app.py`
4. Register/log in to Clario, navigate to **Integrations** (navbar).
5. Click **Connect Google Meet →** — you are redirected to Google's consent screen.
6. Grant permissions — you are redirected back to `/integrations` with a flash "Google Meet connected successfully! 🎉"
7. Verify the page shows a **Connected** badge, your connection timestamp, and the granted scope list.
8. Click **Disconnect** and confirm — flash "Google Meet disconnected successfully." and status returns to Not Connected.

> [!NOTE]
> Refresh tokens expire after ~7 days in Testing publishing status. If `get_valid_access_token` raises `TokenRevoked`, the user will need to reconnect.


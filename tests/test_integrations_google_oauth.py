"""
tests/test_integrations_google_oauth.py
----------------------------------------
Tests for the Google OAuth 2.0 connect/disconnect flow.

Coverage:
 - Login-required enforcement on all integration routes
 - Connect: redirect to Google, state stored in session
 - Callback: state mismatch, missing state, access_denied, server error,
             no code, Google error response, successful exchange (tokens encrypted)
 - Disconnect: not connected, row deleted after disconnect
 - get_valid_access_token: returns valid token, auto-refresh, invalid_grant,
                           not connected
 - User isolation: one user cannot see another's connection

ALL external HTTP calls are mocked — no real network traffic.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

# Fixtures: app, client, logged_in_client come from conftest.py
# Autouse patches for whisper / summariser / nlp are also in conftest.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_FERNET_KEY = Fernet.generate_key()

FAKE_SCOPE = "https://www.googleapis.com/auth/meetings.space.readonly"


def _set_encryption_key(monkeypatch):
    """Inject a consistent Fernet key so encrypt/decrypt work in every test."""
    monkeypatch.setenv("CLARIO_TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())
    import config
    monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())


def _set_google_creds(monkeypatch):
    """Inject dummy Google credentials so connect routes don't bail out."""
    import config
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID",     "test-client-id")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(config, "OAUTH_REDIRECT_BASE",  "http://127.0.0.1:5000")
    monkeypatch.setenv("CLARIO_GOOGLE_CLIENT_ID",     "test-client-id")
    monkeypatch.setenv("CLARIO_GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("CLARIO_OAUTH_REDIRECT_BASE",  "http://127.0.0.1:5000")


def _login(client, app):
    """
    Register + log in a fresh user, return (user_id, username, email).
    """
    creds = {
        "username": "oauthuser",
        "email":    "oauthuser@example.com",
        "password": "StrongPass1!",
    }
    client.post(
        "/register",
        data={
            "username": creds["username"],
            "email":    creds["email"],
            "password": creds["password"],
            "confirm":  creds["password"],
        },
        follow_redirects=True,
    )
    # Retrieve the user_id from the DB
    import routes.integrations as _r_integ
    from models.database import get_user_by_email
    user = get_user_by_email(creds["email"], db_path=_r_integ.DB_PATH)
    return user["id"], creds["username"], creds["email"]


def _seed_connection(app, user_id, monkeypatch, expires_at=None):
    """
    Persist an oauth_connections row using encrypted dummy tokens.
    Returns the plaintext access and refresh tokens.
    """
    _set_encryption_key(monkeypatch)
    from services.integrations.crypto import encrypt_token
    from models.database import upsert_oauth_connection
    import routes.integrations as _r_integ

    at_plain = "ya29.access_token_seed"
    rt_plain = "1//refresh_token_seed"
    at_enc = encrypt_token(at_plain)
    rt_enc = encrypt_token(rt_plain)

    if expires_at is None:
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    upsert_oauth_connection(
        user_id=user_id,
        provider="google",
        access_token_enc=at_enc,
        refresh_token_enc=rt_enc,
        expires_at=expires_at,
        scope=FAKE_SCOPE,
        db_path=_r_integ.DB_PATH,
    )
    return at_plain, rt_plain


# =============================================================================
# 1. Login-required enforcement
# =============================================================================

class TestLoginRequired:

    def test_integrations_index_requires_login(self, client):
        resp = client.get("/integrations")
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]

    def test_connect_requires_login(self, client):
        resp = client.get("/integrations/google/connect")
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]

    def test_callback_requires_login(self, client):
        resp = client.get("/integrations/google/callback?code=abc&state=xyz")
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]

    def test_disconnect_requires_login(self, client):
        resp = client.post("/integrations/google/disconnect")
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]


# =============================================================================
# 2. GET /integrations — status page
# =============================================================================

class TestIntegrationsIndex:

    def test_index_200_when_logged_in(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        resp = client.get("/integrations")
        assert resp.status_code == 200
        assert b"Google" in resp.data

    def test_index_shows_not_connected_by_default(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        resp = client.get("/integrations")
        assert resp.status_code == 200
        assert b"Not Connected" in resp.data or b"not connected" in resp.data.lower()


# =============================================================================
# 3. GET /integrations/google/connect
# =============================================================================

class TestGoogleConnect:

    def test_connect_redirects_to_google(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        _login(client, app)

        resp = client.get("/integrations/google/connect")
        assert resp.status_code in (301, 302)
        location = resp.headers["Location"]
        assert "accounts.google.com" in location
        assert "response_type=code" in location
        assert "access_type=offline" in location
        assert "prompt=consent" in location
        assert "meetings.space.readonly" in location

    def test_connect_stores_state_in_session(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        _login(client, app)

        with client.session_transaction() as sess:
            assert "google_oauth_state" not in sess

        client.get("/integrations/google/connect")

        with client.session_transaction() as sess:
            assert "google_oauth_state" in sess
            assert len(sess["google_oauth_state"]) >= 32

    def test_connect_redirects_to_integrations_when_not_configured(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        import config
        monkeypatch.setattr(config, "GOOGLE_CLIENT_ID",     "")
        monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "")
        _login(client, app)

        resp = client.get("/integrations/google/connect", follow_redirects=False)
        assert resp.status_code in (301, 302)
        assert "/integrations" in resp.headers["Location"]


# =============================================================================
# 4. GET /integrations/google/callback
# =============================================================================

class TestGoogleCallback:

    def _set_session_state(self, client, state):
        with client.session_transaction() as sess:
            sess["google_oauth_state"] = state

    # -- error param from Google -----------------------------------------------

    def test_callback_access_denied(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        self._set_session_state(client, "mystate")

        resp = client.get(
            "/integrations/google/callback?error=access_denied&state=mystate",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"cancelled" in resp.data.lower() or b"denied" in resp.data.lower()

    def test_callback_other_error(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        self._set_session_state(client, "mystate")

        resp = client.get(
            "/integrations/google/callback?error=server_error&state=mystate",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"server_error" in resp.data

    # -- state validation -------------------------------------------------------

    def test_callback_missing_state_in_session(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        # do NOT set session state

        resp = client.get(
            "/integrations/google/callback?code=abc&state=something",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Invalid" in resp.data or b"expired" in resp.data.lower()

    def test_callback_state_mismatch(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        self._set_session_state(client, "correct_state")

        resp = client.get(
            "/integrations/google/callback?code=abc&state=wrong_state",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Invalid" in resp.data or b"expired" in resp.data.lower()

    # -- missing code -----------------------------------------------------------

    def test_callback_no_code(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)
        self._set_session_state(client, "mystate")

        resp = client.get(
            "/integrations/google/callback?state=mystate",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        # Flash message about missing code
        assert b"code" in resp.data.lower() or b"missing" in resp.data.lower()

    # -- Google error on token exchange -----------------------------------------

    def test_callback_google_error_response(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        _login(client, app)
        self._set_session_state(client, "mystate")

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "invalid_client", "error_description": "Bad client"}

        with patch("requests.post", return_value=mock_resp):
            resp = client.get(
                "/integrations/google/callback?code=bad_code&state=mystate",
                follow_redirects=True,
            )

        assert resp.status_code == 200
        assert b"Could not complete" in resp.data or b"failed" in resp.data.lower()

    # -- successful exchange stores encrypted tokens ----------------------------

    def test_callback_successful_exchange_stores_encrypted_tokens(
        self, client, app, monkeypatch
    ):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        user_id, _, _ = _login(client, app)
        self._set_session_state(client, "goodstate")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token":  "ya29.fresh_access_token",
            "refresh_token": "1//fresh_refresh_token",
            "expires_in":    3600,
            "scope":         FAKE_SCOPE,
            "token_type":    "Bearer",
        }

        with patch("requests.post", return_value=mock_resp):
            resp = client.get(
                "/integrations/google/callback?code=goodcode&state=goodstate",
                follow_redirects=False,
            )

        # Should redirect to /integrations after success
        assert resp.status_code in (301, 302)
        assert "/integrations" in resp.headers["Location"]

        # Verify the row was stored with encrypted tokens (not plaintext)
        import routes.integrations as _r_integ
        from models.database import get_oauth_connection
        from services.integrations.crypto import decrypt_token

        conn = get_oauth_connection(user_id, "google", db_path=_r_integ.DB_PATH)
        assert conn is not None, "No oauth_connection row stored"

        # Must NOT store plaintext
        assert conn["access_token_enc"] != "ya29.fresh_access_token"
        assert conn["refresh_token_enc"] != "1//fresh_refresh_token"

        # Must decrypt back to plaintext
        assert decrypt_token(conn["access_token_enc"]) == "ya29.fresh_access_token"
        assert decrypt_token(conn["refresh_token_enc"]) == "1//fresh_refresh_token"
        assert conn["scope"] == FAKE_SCOPE


# =============================================================================
# 5. POST /integrations/google/disconnect
# =============================================================================

class TestGoogleDisconnect:

    def test_disconnect_when_not_connected(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        resp = client.post("/integrations/google/disconnect", follow_redirects=True)
        assert resp.status_code == 200
        assert b"not currently connected" in resp.data.lower() or b"not" in resp.data.lower()

    def test_disconnect_deletes_row(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        user_id, _, _ = _login(client, app)
        _seed_connection(app, user_id, monkeypatch)

        # Verify seeded
        import routes.integrations as _r_integ
        from models.database import get_oauth_connection
        assert get_oauth_connection(user_id, "google", db_path=_r_integ.DB_PATH) is not None

        # Mock the revoke HTTP call (best-effort — doesn't matter what it returns)
        mock_revoke = MagicMock()
        mock_revoke.status_code = 200

        with patch("requests.post", return_value=mock_revoke):
            resp = client.post("/integrations/google/disconnect", follow_redirects=True)

        assert resp.status_code == 200
        assert b"disconnected" in resp.data.lower()
        assert get_oauth_connection(user_id, "google", db_path=_r_integ.DB_PATH) is None

    def test_disconnect_still_deletes_row_even_if_revoke_fails(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        user_id, _, _ = _login(client, app)
        _seed_connection(app, user_id, monkeypatch)

        import routes.integrations as _r_integ
        from models.database import get_oauth_connection

        # Revoke raises a network error — disconnect should still succeed
        with patch("requests.post", side_effect=Exception("network down")):
            resp = client.post("/integrations/google/disconnect", follow_redirects=True)

        assert resp.status_code == 200
        assert get_oauth_connection(user_id, "google", db_path=_r_integ.DB_PATH) is None


# =============================================================================
# 6. get_valid_access_token
# =============================================================================

class TestGetValidAccessToken:

    def _get_db_path(self, app):
        import routes.integrations as _r_integ
        return _r_integ.DB_PATH

    def test_raises_not_connected_when_no_row(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        from services.integrations.google_meet import get_valid_access_token
        from services.integrations.base import NotConnected

        db_path = self._get_db_path(app)
        # Insert a user just to have a valid user_id; no oauth row
        from models.database import insert_user
        import werkzeug.security as ws
        uid = insert_user(
            "tokenuser", "tokenuser@example.com",
            ws.generate_password_hash("x"), db_path=db_path
        )
        with pytest.raises(NotConnected):
            get_valid_access_token(uid, db_path=db_path)

    def test_returns_access_token_when_valid(self, app, monkeypatch, client):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        user_id, _, _ = _login(client, app)
        at_plain, _ = _seed_connection(app, user_id, monkeypatch)

        from services.integrations.google_meet import get_valid_access_token

        db_path = self._get_db_path(app)
        # No HTTP call should be made (token is fresh)
        with patch("requests.post") as mock_post:
            result = get_valid_access_token(user_id, db_path=db_path)
            mock_post.assert_not_called()

        assert result == at_plain

    def test_auto_refreshes_expired_token(self, app, monkeypatch, client):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        user_id, _, _ = _login(client, app)

        # Seed a connection with an already-expired expires_at
        expired_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        _seed_connection(app, user_id, monkeypatch, expires_at=expired_at)

        from services.integrations.google_meet import get_valid_access_token
        from models.database import get_oauth_connection
        from services.integrations.crypto import decrypt_token

        db_path = self._get_db_path(app)

        new_access_token = "ya29.refreshed_token"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": new_access_token,
            "expires_in": 3600,
            "scope": FAKE_SCOPE,
            "token_type": "Bearer",
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = get_valid_access_token(user_id, db_path=db_path)
            mock_post.assert_called_once()

        assert result == new_access_token

        # DB should have the new encrypted access token
        updated = get_oauth_connection(user_id, "google", db_path=db_path)
        assert decrypt_token(updated["access_token_enc"]) == new_access_token

    def test_invalid_grant_deletes_connection_and_raises_token_revoked(
        self, app, monkeypatch, client
    ):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        user_id, _, _ = _login(client, app)

        expired_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        _seed_connection(app, user_id, monkeypatch, expires_at=expired_at)

        from services.integrations.google_meet import get_valid_access_token
        from services.integrations.base import TokenRevoked
        from models.database import get_oauth_connection

        db_path = self._get_db_path(app)

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Token has been expired or revoked.",
        }

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(TokenRevoked):
                get_valid_access_token(user_id, db_path=db_path)

        # Row must be deleted after invalid_grant
        assert get_oauth_connection(user_id, "google", db_path=db_path) is None


# =============================================================================
# 7. User isolation
# =============================================================================

class TestUserIsolation:

    def test_user_b_cannot_see_user_a_connection(self, client, app, monkeypatch):
        """User B's /integrations page must show 'Not Connected', not user A's data."""
        _set_encryption_key(monkeypatch)

        # Register user A
        client.post(
            "/register",
            data={
                "username": "usera",
                "email":    "usera@example.com",
                "password": "StrongPass1!",
                "confirm":  "StrongPass1!",
            },
            follow_redirects=True,
        )

        import routes.integrations as _r_integ
        from models.database import get_user_by_email
        user_a = get_user_by_email("usera@example.com", db_path=_r_integ.DB_PATH)
        assert user_a is not None
        _seed_connection(app, user_a["id"], monkeypatch)

        # Log out user A
        client.get("/logout", follow_redirects=True)

        # Register and log in as user B
        client.post(
            "/register",
            data={
                "username": "userb",
                "email":    "userb@example.com",
                "password": "StrongPass1!",
                "confirm":  "StrongPass1!",
            },
            follow_redirects=True,
        )

        resp = client.get("/integrations")
        assert resp.status_code == 200
        # User B must NOT see "Connected" for user A's integration
        assert b"Not Connected" in resp.data or b"not connected" in resp.data.lower()
        # Also must not leak user A's scope
        assert b"usera" not in resp.data

    def test_cannot_disconnect_another_users_connection(self, client, app, monkeypatch):
        """POSTing disconnect as user B must not delete user A's row."""
        _set_encryption_key(monkeypatch)

        # Register and log in as user A, seed a connection
        client.post(
            "/register",
            data={
                "username": "alice",
                "email":    "alice@example.com",
                "password": "StrongPass1!",
                "confirm":  "StrongPass1!",
            },
            follow_redirects=True,
        )
        import routes.integrations as _r_integ
        from models.database import get_user_by_email, get_oauth_connection
        alice = get_user_by_email("alice@example.com", db_path=_r_integ.DB_PATH)
        _seed_connection(app, alice["id"], monkeypatch)

        client.get("/logout", follow_redirects=True)

        # Register and log in as user B
        client.post(
            "/register",
            data={
                "username": "bob",
                "email":    "bob@example.com",
                "password": "StrongPass1!",
                "confirm":  "StrongPass1!",
            },
            follow_redirects=True,
        )

        # Bob tries to disconnect — his own session has no connection
        resp = client.post("/integrations/google/disconnect", follow_redirects=True)
        assert resp.status_code == 200

        # Alice's row must still exist
        assert (
            get_oauth_connection(alice["id"], "google", db_path=_r_integ.DB_PATH)
            is not None
        )


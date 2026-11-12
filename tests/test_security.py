"""
tests/test_security.py
----------------------
Security-focused tests for Clario.

Covers:
  - Open redirect rejected on login ?next= parameter
  - Oversize upload returns HTTP 413
  - Duplicate filenames do not collide on disk (UUID naming)
  - POST without CSRF token is rejected (WTF_CSRF_ENABLED=True fixture)
  - User B gets 404 on user A's /result/<id>  (IDOR protection)
  - XSS payload in a task description is HTML-escaped in the response
"""

import io
import os
import re

import pytest

from app import create_app as _create_app

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_CREDS_A = dict(
    username="secUserA",
    email="sec_a@example.com",
    password="SecurePass1!",
    confirm="SecurePass1!",
)
_CREDS_B = dict(
    username="secUserB",
    email="sec_b@example.com",
    password="SecurePass1!",
    confirm="SecurePass1!",
)


def _register(client, creds: dict):
    """Register a user and follow the redirect (auto-login)."""
    return client.post("/register", data=creds, follow_redirects=True)


def _login(client, email: str, password: str, next_url: str = ""):
    url = f"/login?next={next_url}" if next_url else "/login"
    return client.post(
        url,
        data={"email": email, "password": password},
        follow_redirects=False,
    )


def _logout(client):
    client.get("/logout", follow_redirects=True)


def _upload_text(client, text: str = "Alice will finish the report by Friday."):
    return client.post(
        "/upload",
        data={"text_input": text, "meeting_title": "Security Test Meeting"},
        follow_redirects=False,
    )


# ---------------------------------------------------------------------------
# Fixture: CSRF-enabled app (separate from the global `app` fixture which
# disables CSRF).  Uses the same tmp_dirs infrastructure.
# ---------------------------------------------------------------------------

@pytest.fixture()
def csrf_app(tmp_dirs):
    """App with CSRF protection turned ON — used only by CSRF tests."""
    flask_app = _create_app(
        test_config={
            "db_path":    tmp_dirs["db_path"],
            "upload_dir": tmp_dirs["upload_dir"],
            "graph_dir":  tmp_dirs["graph_dir"],
            "TESTING":    True,
            "WTF_CSRF_ENABLED": True,
        }
    )
    # Do NOT set WTF_CSRF_ENABLED = False here — that is the whole point
    yield flask_app


@pytest.fixture()
def csrf_client(csrf_app):
    """Test client for the CSRF-enabled app."""
    return csrf_app.test_client()


# ---------------------------------------------------------------------------
# Fixture: tiny upload limit (10 bytes) for 413 testing
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_limit_client(tmp_dirs):
    """App wired with a 10-byte MAX_CONTENT_LENGTH to trigger 413 easily."""
    flask_app = _create_app(
        test_config={
            "db_path":    tmp_dirs["db_path"],
            "upload_dir": tmp_dirs["upload_dir"],
            "graph_dir":  tmp_dirs["graph_dir"],
            "TESTING":    True,
            "MAX_CONTENT_LENGTH": 10,  # 10 bytes
        }
    )
    flask_app.config["WTF_CSRF_ENABLED"] = False
    from models.database import init_db, insert_user
    from werkzeug.security import generate_password_hash
    init_db(tmp_dirs["db_path"])
    user_id = insert_user(
        "secUserA", "sec_a@example.com", generate_password_hash("SecurePass1!"),
        db_path=tmp_dirs["db_path"],
    )
    c = flask_app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = "secUserA"
    yield c


# ===========================================================================
# 1. Open redirect rejected
# ===========================================================================

class TestOpenRedirect:
    """Login ?next= must only follow single-/ relative paths."""

    def _register_and_logout(self, client):
        """Ensure user exists, then log out so we can test login redirect."""
        _register(client, _CREDS_A)
        _logout(client)

    def test_absolute_url_rejected(self, client):
        """?next=https://attacker.com must NOT redirect to attacker."""
        self._register_and_logout(client)
        resp = _login(
            client,
            email=_CREDS_A["email"],
            password=_CREDS_A["password"],
            next_url="https://attacker.com",
        )
        # Must redirect to dashboard, not to attacker
        assert resp.status_code == 302
        location = resp.headers.get("Location", "")
        assert "attacker.com" not in location
        assert location.endswith("/dashboard") or "/dashboard" in location

    def test_protocol_relative_url_rejected(self, client):
        """?next=//attacker.com must NOT redirect to attacker."""
        self._register_and_logout(client)
        resp = _login(
            client,
            email=_CREDS_A["email"],
            password=_CREDS_A["password"],
            next_url="//attacker.com",
        )
        assert resp.status_code == 302
        location = resp.headers.get("Location", "")
        assert "attacker.com" not in location
        assert "/dashboard" in location

    def test_valid_relative_next_followed(self, client):
        """?next=/dashboard should be followed (a legitimate redirect)."""
        self._register_and_logout(client)
        resp = _login(
            client,
            email=_CREDS_A["email"],
            password=_CREDS_A["password"],
            next_url="/dashboard",
        )
        assert resp.status_code == 302
        assert "/dashboard" in resp.headers.get("Location", "")

    def test_scheme_in_next_rejected(self, client):
        """?next=javascript:alert(1) must NOT be followed."""
        self._register_and_logout(client)
        resp = _login(
            client,
            email=_CREDS_A["email"],
            password=_CREDS_A["password"],
            next_url="javascript:alert(1)",
        )
        assert resp.status_code == 302
        location = resp.headers.get("Location", "")
        assert "javascript" not in location
        assert "/dashboard" in location


# ===========================================================================
# 2. Oversize upload returns 413
# ===========================================================================

class TestOversizeUpload:
    """Uploading a file larger than MAX_CONTENT_LENGTH must return 413."""

    def test_oversize_gives_413(self, tiny_limit_client):
        c = tiny_limit_client
        # Send 50 bytes — well above the 10-byte limit
        payload = b"X" * 50
        resp = c.post(
            "/upload",
            data={
                "audio_file": (io.BytesIO(payload), "recording.mp3"),
                "meeting_title": "Big File",
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        assert resp.status_code == 413, (
            f"Expected 413 for oversize upload, got {resp.status_code}"
        )

    def test_413_body_mentions_size(self, tiny_limit_client):
        """The error page should mention the size limit."""
        c = tiny_limit_client
        payload = b"X" * 50
        resp = c.post(
            "/upload",
            data={
                "audio_file": (io.BytesIO(payload), "recording.mp3"),
                "meeting_title": "Big File",
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 413
        body = resp.data.lower()
        assert b"large" in body or b"413" in body or b"exceeded" in body or b"size" in body


# ===========================================================================
# 3. Duplicate filenames do not collide on disk
# ===========================================================================

class TestDuplicateFilenames:
    """Two uploads with the same filename must produce different disk paths."""

    def test_no_collision(self, logged_in_client, tmp_dirs):
        c, _ = logged_in_client
        upload_dir = tmp_dirs["upload_dir"]

        fake_audio = b"RIFF" + b"\x00" * 40  # plausible but fake WAV header

        def _upload_named(name):
            return c.post(
                "/upload",
                data={
                    "audio_file": (io.BytesIO(fake_audio), name),
                    "meeting_title": "Duplicate Test",
                },
                content_type="multipart/form-data",
                follow_redirects=False,
            )

        resp1 = _upload_named("recording.mp3")
        resp2 = _upload_named("recording.mp3")

        # Both must succeed (redirect to /result/)
        assert resp1.status_code == 302, f"Upload 1 failed: {resp1.status_code}"
        assert resp2.status_code == 302, f"Upload 2 failed: {resp2.status_code}"

        # Must redirect to different meeting pages
        loc1 = resp1.headers["Location"]
        loc2 = resp2.headers["Location"]
        assert loc1 != loc2, "Both uploads redirected to the same meeting!"

        # Disk must contain two *different* files in the upload_dir
        files_on_disk = os.listdir(upload_dir)
        mp3_files = [f for f in files_on_disk if f.endswith(".mp3")]
        assert len(mp3_files) >= 2, (
            f"Expected at least 2 .mp3 files on disk; found: {mp3_files}"
        )
        assert len(set(mp3_files)) == len(mp3_files), "Filenames on disk are not unique!"

    def test_original_name_preserved_in_response(self, logged_in_client, tmp_dirs):
        """The user's original filename must be preserved in meetings.audio_filename."""
        c, _ = logged_in_client
        fake_audio = b"RIFF" + b"\x00" * 40

        resp = c.post(
            "/upload",
            data={
                "audio_file": (io.BytesIO(fake_audio), "my_meeting.mp3"),
                "meeting_title": "Name Preservation Test",
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        assert resp.status_code == 302
        m = re.search(r"/result/(\d+)", resp.headers["Location"])
        assert m, "Did not redirect to /result/<id>"
        meeting_id = int(m.group(1))

        from models.database import get_meeting_detail
        meeting = get_meeting_detail(meeting_id, 1, tmp_dirs["db_path"])
        assert meeting is not None
        assert meeting["audio_filename"] == "my_meeting.mp3"


# ===========================================================================
# 4. POST without CSRF token is rejected (WTF_CSRF_ENABLED = True)
# ===========================================================================

class TestCSRFEnforcement:
    """With CSRF on, POST requests missing the token must be rejected."""

    def test_register_without_csrf_rejected(self, csrf_client):
        resp = csrf_client.post(
            "/register",
            data=dict(
                username="noCsrfUser",
                email="nocsrf@example.com",
                password="Password1!",
                confirm="Password1!",
            ),
            follow_redirects=False,
        )
        # Flask-WTF returns 400 when CSRF token is missing/invalid
        assert resp.status_code == 400, (
            f"Expected 400 for missing CSRF token on /register, got {resp.status_code}"
        )

    def test_login_without_csrf_rejected(self, csrf_client):
        resp = csrf_client.post(
            "/login",
            data=dict(email="any@example.com", password="Password1!"),
            follow_redirects=False,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for missing CSRF token on /login, got {resp.status_code}"
        )


# ===========================================================================
# 5. IDOR — user B gets 404 on user A's /result/<id>
# ===========================================================================

class TestIDOR:
    """Accessing another user's meeting must return 404 (not 200 or 302)."""

    def test_unowned_meeting_returns_404(self, app):
        # -- User A: register and upload
        c_a = app.test_client()
        _register(c_a, _CREDS_A)
        resp = _upload_text(c_a)
        assert resp.status_code == 302
        location = resp.headers["Location"]
        m = re.search(r"/result/(\d+)", location)
        assert m, f"Expected /result/<id> redirect, got: {location}"
        meeting_id = m.group(1)

        # -- User B: register (auto-logs in as B)
        c_b = app.test_client()
        _register(c_b, _CREDS_B)

        # B tries to access A's meeting directly
        resp_b = c_b.get(f"/result/{meeting_id}", follow_redirects=True)
        assert resp_b.status_code == 404, (
            f"Expected 404 for unowned meeting, got {resp_b.status_code}"
        )
        assert b"not found" in resp_b.data.lower()

    def test_unowned_meeting_no_data_leak(self, app):
        """The 404 response body must not contain the meeting transcript."""
        c_a = app.test_client()
        _register(c_a, dict(
            username="leakUserA", email="leak_a@example.com",
            password="SecurePass1!", confirm="SecurePass1!",
        ))
        resp = _upload_text(c_a, text="Top secret project Nightingale briefing.")
        assert resp.status_code == 302
        meeting_id = re.search(r"/result/(\d+)", resp.headers["Location"]).group(1)

        c_b = app.test_client()
        _register(c_b, dict(
            username="leakUserB", email="leak_b@example.com",
            password="SecurePass1!", confirm="SecurePass1!",
        ))
        resp_b = c_b.get(f"/result/{meeting_id}", follow_redirects=True)
        assert b"nightingale" not in resp_b.data.lower()


# ===========================================================================
# 6. XSS payload in task description is HTML-escaped
# ===========================================================================

class TestXSSEscaping:
    """Script tags in task content must not appear raw in HTML output."""

    XSS_PAYLOAD = "<script>alert('xss')</script>"

    def test_xss_in_transcript_escaped_on_result_page(self, logged_in_client):
        c, _ = logged_in_client
        # Inject payload as part of the transcript
        transcript = f"{self.XSS_PAYLOAD} Alice will finish the report by Friday."
        resp = c.post(
            "/upload",
            data={"text_input": transcript, "meeting_title": "XSS Test"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        result_resp = c.get(resp.headers["Location"])
        assert result_resp.status_code == 200

        body = result_resp.data
        # Raw script tag must NOT appear verbatim
        assert b"<script>alert('xss')</script>" not in body, (
            "XSS payload appeared unescaped in the result page HTML"
        )

    def test_xss_in_meeting_title_escaped(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.post(
            "/upload",
            data={
                "text_input": "Alice will finish the report by Friday.",
                "meeting_title": f"Meeting {self.XSS_PAYLOAD}",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302
        result_resp = c.get(resp.headers["Location"])
        body = result_resp.data
        assert b"<script>alert('xss')</script>" not in body, (
            "XSS payload in meeting title appeared unescaped in the result page"
        )

    def test_json_in_page_is_safe(self, logged_in_client):
        """
        The tasksData JSON block embedded in result.html must encode the payload
        as a JSON string (not raw HTML), making it safe in a JS context.
        """
        c, _ = logged_in_client
        transcript = f"Alice must do {self.XSS_PAYLOAD} by Friday."
        resp = c.post(
            "/upload",
            data={"text_input": transcript, "meeting_title": "JSON Safety Test"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        result_resp = c.get(resp.headers["Location"])
        body_text = result_resp.data.decode("utf-8", errors="replace")

        # If the payload appears in the JS block it must be JSON-encoded
        # (Jinja2 tojson escapes < > & by default → \\u003cscript\\u003e)
        raw_in_script = "<script>alert('xss')</script>" in body_text
        # Jinja tojson converts < → \u003c, so the raw tag won't appear
        assert not raw_in_script, (
            "Raw XSS payload found inside the page's JavaScript block"
        )


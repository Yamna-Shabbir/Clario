"""
tests/test_routes.py
--------------------
Route-level tests for the Clario Flask application.

Covers:
  - Public pages (unauthenticated)
  - Auth flows: register, login, logout, redirect behaviour
  - Authenticated end-to-end: upload (text) → result → complete task
    → search → dashboard → comparison
"""

import re

import pytest


# ---------------------------------------------------------------------------
# 1. Public / unauthenticated routes  (mirrors scripts/check_routes.py)
# ---------------------------------------------------------------------------

class TestPublicRoutes:
    def test_health_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_landing_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_register_200(self, client):
        resp = client.get("/register")
        assert resp.status_code == 200

    def test_login_200(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_dashboard_redirects_when_anonymous(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_upload_redirects_when_anonymous(self, client):
        resp = client.get("/upload")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_result_redirects_when_anonymous(self, client):
        resp = client.get("/result/1")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_404_on_unknown_path(self, client):
        resp = client.get("/notfound")
        assert resp.status_code == 404

    def test_search_redirects_when_anonymous(self, client):
        resp = client.get("/search")
        assert resp.status_code == 302


# ---------------------------------------------------------------------------
# 2. Auth flows
# ---------------------------------------------------------------------------

class TestAuth:
    def test_register_success(self, client):
        resp = client.post(
            "/register",
            data={
                "username": "alice",
                "email":    "alice@example.com",
                "password": "Password1!",
                "confirm":  "Password1!",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"alice" in resp.data or b"dashboard" in resp.data.lower()

    def test_register_duplicate_email(self, client):
        data = dict(
            username="bob",
            email="dup@example.com",
            password="Password1!",
            confirm="Password1!",
        )
        client.post("/register", data=data)
        # Registering automatically logs in; log out first so the second register is evaluated
        client.get("/logout")
        resp = client.post(
            "/register",
            data={**data, "username": "bob2"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"already exists" in resp.data

    def test_register_password_too_short(self, client):
        resp = client.post(
            "/register",
            data=dict(username="short", email="s@e.com", password="abc", confirm="abc"),
            follow_redirects=True,
        )
        assert b"8 characters" in resp.data

    def test_register_password_mismatch(self, client):
        resp = client.post(
            "/register",
            data=dict(username="mm", email="mm@e.com", password="Password1!", confirm="Wrong1!"),
            follow_redirects=True,
        )
        assert b"do not match" in resp.data

    def test_login_bad_password(self, client):
        # Register first
        client.post(
            "/register",
            data=dict(username="carol", email="carol@e.com", password="Password1!", confirm="Password1!"),
        )
        client.get("/logout")
        resp = client.post(
            "/login",
            data=dict(email="carol@e.com", password="WrongPass1!"),
            follow_redirects=True,
        )
        assert b"Invalid" in resp.data

    def test_login_unknown_email(self, client):
        resp = client.post(
            "/login",
            data=dict(email="nobody@example.com", password="Password1!"),
            follow_redirects=True,
        )
        assert b"Invalid" in resp.data

    def test_logout_clears_session(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/logout", follow_redirects=True)
        assert resp.status_code == 200
        # After logout dashboard should redirect
        resp2 = c.get("/dashboard")
        assert resp2.status_code == 302

    def test_landing_redirects_logged_in_user(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/")
        # Should redirect to dashboard, not show landing page
        assert resp.status_code == 302
        assert "dashboard" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# 3. Authenticated flows
# ---------------------------------------------------------------------------

class TestAuthenticatedFlows:

    def _upload_text(self, client, text="Alice will finish the report by Friday.", title="Test Meeting"):
        """Helper: POST a text transcript to /upload and return the response."""
        return client.post(
            "/upload",
            data={"text_input": text, "meeting_title": title},
            follow_redirects=False,
        )

    def test_upload_get_returns_200(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/upload")
        assert resp.status_code == 200

    def test_upload_no_input_flashes_error(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.post("/upload", data={}, follow_redirects=True)
        assert resp.status_code == 200
        assert b"audio file or paste" in resp.data

    def test_upload_bad_extension_flashes_error(self, logged_in_client):
        c, _ = logged_in_client
        from io import BytesIO
        resp = c.post(
            "/upload",
            data={"audio_file": (BytesIO(b"fake"), "recording.exe")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Unsupported format" in resp.data

    def test_full_pipeline_text_input(self, logged_in_client):
        """
        End-to-end: upload text → redirect to /result/<id> → 200 with content.
        """
        c, _ = logged_in_client
        resp = self._upload_text(c)
        # Should redirect to /result/<meeting_id>
        assert resp.status_code == 302
        location = resp.headers["Location"]
        assert "/result/" in location

        # Follow redirect
        resp2 = c.get(location, follow_redirects=False)
        assert resp2.status_code == 200
        assert b"Test Meeting" in resp2.data or b"result" in resp2.data.lower()

    def test_result_contains_summary(self, logged_in_client):
        c, _ = logged_in_client
        resp = self._upload_text(c)
        location = resp.headers["Location"]
        resp2 = c.get(location)
        # The fake summariser returns FAKE_SUMMARY
        assert b"milestones" in resp2.data or b"summary" in resp2.data.lower()

    def test_result_ownership_enforced(self, app):
        """User B must not be able to read user A's meeting."""
        # Create user A and upload a meeting
        c_a = app.test_client()
        c_a.post(
            "/register",
            data=dict(username="userA", email="a@a.com", password="Password1!", confirm="Password1!"),
        )
        resp = c_a.post(
            "/upload",
            data={"text_input": "Alice will finish the report.", "meeting_title": "A meeting"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        meeting_url = resp.headers["Location"]
        meeting_id = re.search(r"/result/(\d+)", meeting_url).group(1)

        # Create user B
        c_b = app.test_client()
        c_b.post(
            "/register",
            data=dict(username="userB", email="b@b.com", password="Password1!", confirm="Password1!"),
        )
        c_b.post("/login", data=dict(email="b@b.com", password="Password1!"))

        # User B tries to access user A's meeting
        resp_b = c_b.get(f"/result/{meeting_id}", follow_redirects=True)
        assert resp_b.status_code == 404
        assert b"not found" in resp_b.data.lower() or b"access denied" in resp_b.data.lower()

    def test_complete_task(self, logged_in_client):
        c, _ = logged_in_client
        # Upload a meeting with a task-triggering sentence
        resp = self._upload_text(c, text="Bob must review the design documents immediately.")
        location = resp.headers["Location"]
        resp2 = c.get(location)
        assert resp2.status_code == 200

        from models.database import get_connection, init_db
        from config import DB_PATH

        loc = location if isinstance(location, str) else location.decode()
        m = re.search(r"/result/(\d+)", loc)
        meeting_id = int(m.group(1)) if m else None
        if meeting_id:
            init_db(DB_PATH)
            with get_connection(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE meeting_id = ? LIMIT 1",
                    (meeting_id,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE tasks SET assigned_email = ? WHERE id = ?",
                        ("test@example.com", row["id"]),
                    )
                    conn.commit()
                    task_id = row["id"]
                    resp3 = c.post(
                        f"/tasks/{task_id}/status",
                        data={"status": "completed"},
                        follow_redirects=True,
                    )
                    assert resp3.status_code == 200

    def test_complete_task_wrong_user(self, app):
        """User B must not complete user A's task."""
        c_a = app.test_client()
        c_a.post(
            "/register",
            data=dict(username="u1", email="u1@a.com", password="Password1!", confirm="Password1!"),
        )
        resp = c_a.post(
            "/upload",
            data={"text_input": "Dave must fix the bug today.", "meeting_title": "u1 meeting"},
            follow_redirects=False,
        )
        location = resp.headers["Location"]
        result_resp = c_a.get(location)
        loc = location if isinstance(location, str) else location.decode()
        meeting_match = re.search(r"/result/(\d+)", loc)
        task_id = None
        if meeting_match:
            from models.database import get_connection, init_db
            from config import DB_PATH
            init_db(DB_PATH)
            mid = int(meeting_match.group(1))
            with get_connection(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE meeting_id = ? LIMIT 1",
                    (mid,),
                ).fetchone()
                if row:
                    task_id = row["id"]
                    conn.execute(
                        "UPDATE tasks SET assigned_email = ? WHERE id = ?",
                        ("u2@b.com", task_id),
                    )
                    conn.commit()

        c_b = app.test_client()
        c_b.post(
            "/register",
            data=dict(username="u2", email="u2@b.com", password="Password1!", confirm="Password1!"),
        )
        c_b.post("/login", data=dict(email="u2@b.com", password="Password1!"))

        if task_id:
            resp_b = c_b.post(
                f"/task/{task_id}/complete",
                follow_redirects=True,
            )
            assert resp_b.status_code == 200
            assert b"only the person assigned" in resp_b.data.lower()

    def test_search_returns_results(self, logged_in_client):
        c, _ = logged_in_client
        self._upload_text(c, title="Searchable Meeting Alpha")
        resp = c.get("/search?q=Alpha")
        assert resp.status_code == 200
        assert b"Alpha" in resp.data

    def test_search_empty_query(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/search?q=")
        assert resp.status_code == 200

    def test_dashboard_200(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_shows_uploaded_meeting(self, logged_in_client):
        c, _ = logged_in_client
        self._upload_text(c, text="Unique meeting summary snippet", title="Dashboard Visible Meeting")
        resp = c.get("/dashboard")
        assert resp.status_code == 200
        assert b"Unique meeting summary snippet" in resp.data or b"/result/" in resp.data

    def test_comparison_200(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/comparison")
        assert resp.status_code == 200

    def test_upload_page_asks_for_attendee_emails(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.get("/upload")
        assert b"People in this meeting" in resp.data
        assert b"attendee_email" in resp.data

    def test_upload_rejects_a_name_without_an_email(self, logged_in_client):
        c, _ = logged_in_client
        resp = c.post(
            "/upload",
            data={
                "text_input": "John will deploy the schema tomorrow.",
                "attendee_name": "John",
                "attendee_email": "",
            },
            follow_redirects=True,
        )
        assert b"both a name and an email" in resp.data

    def test_upload_roster_mails_a_clear_name_and_holds_the_rest(self, logged_in_client, app):
        from werkzeug.datastructures import MultiDict
        from models.database import get_pending_confirmations, get_tasks_by_meeting

        c, _ = logged_in_client
        resp = c.post(
            "/upload",
            data=MultiDict([
                ("meeting_title", "Roster check"),
                ("text_input", "John will deploy the schema tomorrow. Sara will send the notes by Friday. Priya will write the recap."),
                ("attendee_name", "John Smith"),
                ("attendee_email", "john@example.com"),
                ("attendee_name", "Sara Khan"),
                ("attendee_email", "sara.khan@example.com"),
                ("attendee_name", "Sara Ali"),
                ("attendee_email", "sara.ali@example.com"),
            ]),
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Confirm who receives this task" in resp.data
        db = app.config["CLARIO_DB_PATH"]
        tasks = get_tasks_by_meeting(1, db)
        by_name = {task["assigned_to"]: task for task in tasks}
        assert by_name["John Smith"]["assigned_email"] == "john@example.com"
        pending = get_pending_confirmations(1, db)
        held = {item["assignee_name"] for item in pending}
        assert "Sara" in held
        assert "Priya" in held
        sara = next(item for item in pending if item["assignee_name"] == "Sara")
        assert {person["email"] for person in sara["candidates"]} == {
            "sara.khan@example.com",
            "sara.ali@example.com",
        }


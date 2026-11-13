"""Follow-up mail: assignee tasks in the body, summary PDF for everyone."""

import json
import os
import tempfile
from unittest.mock import patch

from models.database import (
    get_meeting_emails,
    init_db,
    insert_meeting,
    insert_participants,
    insert_tasks,
    insert_user,
)
from services.notifications import build_summary_pdf, send_invite_emails, send_meeting_followups


def _db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    return path


def test_summary_pdf_contains_the_brief():
    pdf = build_summary_pdf(
        "Planning",
        "The team chose React Native.",
        [{"text": "Use React Native."}],
        [{"assigned_to": "Sara", "description": "Design the logo", "due_date": "Friday"}],
    )
    assert pdf.startswith(b"%PDF")
    assert b"Design the logo" in pdf
    assert b"Use React Native." in pdf


def test_followups_email_assignee_and_owner(tmp_path):
    db = _db()
    try:
        owner_id = insert_user("ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_participants(meeting_id, [
            {"name": "Sara Khan", "email": "sara@example.com"},
        ], db_path=db)
        insert_tasks(meeting_id, [{
            "description": "Sara will design the logo.",
            "assigned_to": "Sara",
            "due_date": "Friday",
            "status": "pending",
            "keyword": "will",
            "priority": "Low",
        }], db_path=db)

        sent = []

        class FakeSMTP:
            def __init__(self, host, port, timeout=None):
                sent.append(("connect", host, port))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def starttls(self):
                sent.append("tls")

            def login(self, user, password):
                sent.append(("login", user))

            def send_message(self, message):
                sent.append(message)

        with patch("services.notifications.BREVO_API_KEY", ""), \
             patch("services.notifications.RESEND_API_KEY", ""), \
             patch("services.notifications.SMTP_HOST", "smtp.example.com"), \
             patch("services.notifications.SMTP_FROM", "clario@example.com"), \
             patch("services.notifications.SMTP_USER", "clario@example.com"), \
             patch("services.notifications.SMTP_PASSWORD", "secret"), \
             patch("services.notifications.EMAIL_DIR", str(tmp_path)), \
             patch("services.notifications.smtplib.SMTP", FakeSMTP):
            send_meeting_followups({
                "summary": "The team chose React Native.",
                "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
            })

        messages = [item for item in sent if hasattr(item, "get")]
        recipients = sorted(item["To"] for item in messages)
        assert recipients == ["ada@example.com", "sara@example.com"]
        sara = next(item for item in messages if item["To"] == "sara@example.com")
        assert "design the logo" in sara.get_body(preferencelist=("plain",)).get_content()
        assert "meeting-summary.pdf" in sara.as_string()
        saved = list(tmp_path.rglob("*.pdf"))
        assert len(saved) == 2
        rows = get_meeting_emails(meeting_id, db)
        assert {row["recipient"] for row in rows} == {"ada@example.com", "sara@example.com"}
        assert all(row["status"] == "sent" for row in rows)
    finally:
        os.remove(db)


def test_followups_ignore_results_without_a_meeting():
    send_meeting_followups({"summary": "orphan"})


def test_invite_sends_over_https_when_brevo_is_set(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["key"] = request.get_header("Api-key")
        return _Response()

    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "host@example.com")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr("services.notifications.urllib.request.urlopen", _urlopen)

    results = send_invite_emails(
        ["guest@example.com"],
        "https://clario.example/live/abc",
        "Planning",
        "Ada",
    )
    assert results[0]["status"] == "sent"
    assert captured["url"] == "https://api.brevo.com/v3/smtp/email"
    assert captured["key"] == "brevo-test"
    assert captured["body"]["to"] == [{"email": "guest@example.com"}]
    assert "https://clario.example/live/abc" in captured["body"]["textContent"]


def test_brevo_without_a_sender_explains_the_render_setting(monkeypatch, tmp_path):
    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "")
    monkeypatch.setattr("services.notifications.SMTP_USER", "")
    monkeypatch.delenv("CLARIO_SMTP_FROM", raising=False)
    monkeypatch.delenv("CLARIO_SMTP_USER", raising=False)
    monkeypatch.setattr("services.notifications.EMAIL_DIR", str(tmp_path))
    monkeypatch.setattr("services.notifications.DEBUG", False)
    results = send_invite_emails(["guest@example.com"], "https://x/live/t", "Planning", "Ada")
    assert results[0]["status"] == "failed"
    assert "CLARIO_SMTP_FROM" in results[0]["detail"]


def test_sender_address_accepts_a_display_name(monkeypatch):
    from services.notifications import _sender_address

    monkeypatch.setenv("CLARIO_SMTP_FROM", "Clario <Host@Example.com>")
    assert _sender_address() == "Host@Example.com"


def test_invite_without_transport_names_the_https_setup(monkeypatch, tmp_path):
    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_HOST", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", str(tmp_path))
    monkeypatch.setattr("services.notifications.DEBUG", False)
    results = send_invite_emails(["guest@example.com"], "https://x/live/t", "Planning", "Ada")
    assert results[0]["status"] == "saved"
    assert "CLARIO_BREVO_API_KEY" in results[0]["detail"]

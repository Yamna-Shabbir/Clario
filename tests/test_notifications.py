"""Follow-up mail: assignee tasks in the body, summary PDF for everyone."""

import base64
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
        ada = next(item for item in messages if item["To"] == "ada@example.com")
        sara_body = sara.get_body(preferencelist=("plain",)).get_content()
        assert "design the logo" in sara_body
        assert "ada (ada@example.com)" in sara_body
        assert "ada@example.com" in str(sara["From"]).lower()
        assert "sara@example.com" not in str(sara["From"]).lower()
        assert "clario@example.com" not in str(sara["From"]).lower()
        assert "ada@example.com" in str(sara["Reply-To"]).lower()
        assert str(ada["From"]) == str(sara["From"])
        assert sara["Subject"].startswith("Your tasks")
        assert ada["Subject"].startswith("Summary of")
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


class _HttpBody:
    def __init__(self, raw: bytes):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _brevo_urlopen(senders, captured):
    def _urlopen(request, timeout=0):
        if request.full_url.rstrip("/").endswith("/v3/senders"):
            captured["gets"] = captured.get("gets", 0) + 1
            return _HttpBody(json.dumps({"senders": senders}).encode())
        body = json.loads(request.data.decode())
        captured["posts"] = captured.get("posts", 0) + 1
        captured.setdefault("bodies", []).append(body)
        captured["url"] = request.full_url
        captured["body"] = body
        captured["key"] = request.get_header("Api-key")
        return _HttpBody(b"{}")

    return _urlopen


def test_invite_sends_over_https_when_brevo_is_set(monkeypatch):
    captured = {}
    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "fixed@example.com")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr(
        "services.notifications.urllib.request.urlopen",
        _brevo_urlopen([{"email": "host@example.com", "active": True, "name": "Other"}], captured),
    )

    results = send_invite_emails(
        ["guest@example.com"],
        "https://clario.example/live/abc",
        "Planning",
        "Ada",
        host_email="host@example.com",
    )
    assert results[0]["status"] == "sent"
    assert captured["url"] == "https://api.brevo.com/v3/smtp/email"
    assert captured["key"] == "brevo-test"
    assert captured["gets"] == 1
    assert captured["body"]["sender"] == {"name": "Ada", "email": "host@example.com"}
    assert captured["body"]["replyTo"] == {"email": "host@example.com", "name": "Ada"}
    assert captured["body"]["to"] == [{"email": "guest@example.com"}]
    assert captured["body"]["sender"]["email"] != "guest@example.com"
    assert captured["body"]["sender"]["email"] != "fixed@example.com"
    assert "https://clario.example/live/abc" in captured["body"]["textContent"]


def test_invite_sends_from_the_host_when_brevo_authorizes_them(monkeypatch):
    captured = {}
    monkeypatch.setenv("CLARIO_SMTP_FROM", "verified@example.com")
    monkeypatch.delenv("CLARIO_SMTP_USER", raising=False)
    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "verified@example.com")
    monkeypatch.setattr("services.notifications.SMTP_USER", "")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr(
        "services.notifications.urllib.request.urlopen",
        _brevo_urlopen(
            [{"email": "ada.user@example.com", "active": True, "name": "Stored name"}],
            captured,
        ),
    )

    results = send_invite_emails(
        ["guest@example.com"],
        "https://clario.example/live/abc",
        "Planning",
        "Ada",
        host_email="Ada User <ada.user@example.com>",
    )
    assert results[0]["status"] == "sent"
    assert captured["body"]["sender"] == {"name": "Ada", "email": "ada.user@example.com"}
    assert captured["body"]["replyTo"] == {"email": "ada.user@example.com", "name": "Ada"}
    assert "Ada (ada.user@example.com)" in captured["body"]["textContent"]
    assert "verified@example.com" not in json.dumps(captured["body"])


def test_invite_refuses_a_host_brevo_has_not_verified(monkeypatch):
    captured = {}
    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "verified@example.com")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr(
        "services.notifications.urllib.request.urlopen",
        _brevo_urlopen(
            [{"email": "ada.user@example.com", "active": False, "name": "Ada"}],
            captured,
        ),
    )

    results = send_invite_emails(
        ["guest@example.com"],
        "https://clario.example/live/abc",
        "Planning",
        "Ada",
        host_email="ada.user@example.com",
    )
    assert results[0]["status"] == "failed"
    assert "not an active sender" in results[0]["detail"]
    assert "did not substitute" in results[0]["detail"]
    assert captured.get("posts", 0) == 0


def test_invite_without_a_host_address_does_not_substitute_a_sender(monkeypatch):
    def _urlopen(request, timeout=0):
        raise AssertionError(request.full_url)

    monkeypatch.setenv("CLARIO_SMTP_FROM", "verified@example.com")
    monkeypatch.delenv("CLARIO_SMTP_USER", raising=False)
    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "verified@example.com")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr("services.notifications.urllib.request.urlopen", _urlopen)

    results = send_invite_emails(
        ["guest@example.com"], "https://x/live/t", "Planning", "Ada", host_email="not-an-email",
    )
    assert results[0]["status"] == "failed"
    assert "did not substitute" in results[0]["detail"]


def test_brevo_without_a_host_explains_that_no_fixed_sender_is_used(monkeypatch, tmp_path):
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
    assert "meeting-host" in results[0]["detail"]
    assert "CLARIO_SMTP_FROM" not in results[0]["detail"]


def test_followups_send_summary_and_tasks_from_an_authorized_host(tmp_path, monkeypatch):
    db = _db()
    captured = {}
    try:
        owner_id = insert_user("Ada", "ada@example.com", "hash", db_path=db)
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
            "assigned_email": "sara@example.com",
        }], db_path=db)
        monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
        monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
        monkeypatch.setattr("services.notifications.SMTP_FROM", "fixed@example.com")
        monkeypatch.setattr("services.notifications.EMAIL_DIR", str(tmp_path))
        monkeypatch.setattr("services.notifications.DEBUG", False)
        monkeypatch.setattr(
            "services.notifications.urllib.request.urlopen",
            _brevo_urlopen([{"email": "ada@example.com", "active": True}], captured),
        )
        send_meeting_followups({
            "summary": "The team chose React Native.",
            "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
        })
    finally:
        os.remove(db)

    assert captured["posts"] == 2
    subjects = set()
    for body in captured["bodies"]:
        assert body["sender"] == {"name": "Ada", "email": "ada@example.com"}
        assert body["replyTo"] == {"email": "ada@example.com", "name": "Ada"}
        assert body["sender"]["email"] != body["to"][0]["email"] or body["to"][0]["email"] == "ada@example.com"
        assert body["to"][0]["email"] in {"ada@example.com", "sara@example.com"}
        assert body["sender"]["email"] != "fixed@example.com"
        assert body["attachment"][0]["name"] == "meeting-summary.pdf"
        subjects.add(body["subject"])
    assert subjects == {"Your tasks from Planning", "Summary of Planning"}
    sara = next(body for body in captured["bodies"] if body["to"][0]["email"] == "sara@example.com")
    assert sara["sender"]["email"] == "ada@example.com"
    assert list(tmp_path.rglob("*.pdf"))


def test_followups_keep_the_pdf_when_the_host_is_not_a_brevo_sender(tmp_path, monkeypatch):
    db = _db()
    captured = {}
    try:
        owner_id = insert_user("Ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_participants(meeting_id, [
            {"name": "Sara Khan", "email": "sara@example.com"},
        ], db_path=db)
        monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
        monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
        monkeypatch.setattr("services.notifications.SMTP_FROM", "fixed@example.com")
        monkeypatch.setattr("services.notifications.EMAIL_DIR", str(tmp_path))
        monkeypatch.setattr(
            "services.notifications.urllib.request.urlopen",
            _brevo_urlopen([{"email": "fixed@example.com", "active": True}], captured),
        )
        send_meeting_followups({
            "summary": "Notes.",
            "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
        })
        rows = get_meeting_emails(meeting_id, db)
    finally:
        os.remove(db)

    assert rows
    assert all(row["status"] == "failed" for row in rows)
    assert all("ada@example.com" in row["detail"] for row in rows)
    assert all("did not substitute" in row["detail"] for row in rows)
    assert captured.get("posts", 0) == 0
    assert list(tmp_path.rglob("*.pdf"))


def test_sender_address_accepts_a_display_name(monkeypatch):
    from services.notifications import _sender_address

    monkeypatch.setenv("CLARIO_SMTP_FROM", "Clario <Host@Example.com>")
    assert _sender_address() == "Host@Example.com"


def _decode_gmail_raw(raw: str) -> bytes:
    padded = raw + ("=" * (-len(raw) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def test_invite_sends_through_the_hosts_gmail(monkeypatch):
    captured = {}

    def _urlopen(request, timeout=0):
        if request.full_url.endswith("/userinfo"):
            return _HttpBody(json.dumps({"email": "ada@gmail.com"}).encode())
        body = json.loads(request.data.decode())
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["raw"] = body["raw"]
        return _HttpBody(b'{"id":"m1"}')

    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "fixed@example.com")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr("services.notifications._host_gmail_can_send", lambda user_id, db_path: True)
    monkeypatch.setattr(
        "services.integrations.google_meet.get_valid_access_token",
        lambda user_id, db_path=None: "host-token",
    )
    monkeypatch.setattr("services.notifications.urllib.request.urlopen", _urlopen)

    results = send_invite_emails(
        ["guest@example.com"],
        "https://clario.example/live/abc",
        "Planning",
        "Ada",
        host_email="ada@gmail.com",
        host_user_id=7,
    )
    assert results[0]["status"] == "sent"
    assert captured["url"] == "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    assert captured["auth"] == "Bearer host-token"
    raw = _decode_gmail_raw(captured["raw"])
    assert b"ada@gmail.com" in raw.lower()
    assert b"guest@example.com" in raw.lower()
    assert b"fixed@example.com" not in raw.lower()
    assert b"https://clario.example/live/abc" in raw


def test_invite_refuses_a_different_connected_google_account(monkeypatch):
    def _urlopen(request, timeout=0):
        if request.full_url.endswith("/userinfo"):
            return _HttpBody(json.dumps({"email": "other@gmail.com"}).encode())
        raise AssertionError(request.full_url)

    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr("services.notifications._host_gmail_can_send", lambda user_id, db_path: True)
    monkeypatch.setattr(
        "services.integrations.google_meet.get_valid_access_token",
        lambda user_id, db_path=None: "host-token",
    )
    monkeypatch.setattr("services.notifications.urllib.request.urlopen", _urlopen)

    results = send_invite_emails(
        ["guest@example.com"],
        "https://x/live/t",
        "Planning",
        "Ada",
        host_email="ada@gmail.com",
        host_user_id=7,
    )
    assert results[0]["status"] == "failed"
    assert "other@gmail.com" in results[0]["detail"]
    assert "ada@gmail.com" in results[0]["detail"]
    assert "did not substitute" in results[0]["detail"]


def test_gmail_smtp_does_not_send_as_a_different_login(monkeypatch, tmp_path):
    db = _db()
    called = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            called.append(args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            return None

        def login(self, user, password):
            return None

        def send_message(self, message):
            called.append(message)

    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setattr("services.notifications.SMTP_USER", "other@gmail.com")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "other@gmail.com")
    monkeypatch.setattr("services.notifications.SMTP_PASSWORD", "secret")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", str(tmp_path))
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr("services.notifications.smtplib.SMTP", FakeSMTP)

    try:
        owner_id = insert_user("Ada", "ada@gmail.com", "hash", db_path=db)
        results = send_invite_emails(
            ["guest@example.com"],
            "https://x/live/t",
            "Planning",
            "Ada",
            host_email="ada@gmail.com",
            host_user_id=owner_id,
            db_path=db,
        )
    finally:
        os.remove(db)
    assert results[0]["status"] == "saved"
    assert "logged in" in results[0]["detail"]
    assert called == []


def test_gmail_smtp_sends_when_the_login_is_the_host(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            return None

        def login(self, user, password):
            sent.append(user)

        def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "")
    monkeypatch.setattr("services.notifications.RESEND_API_KEY", "")
    monkeypatch.setattr("services.notifications.SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setattr("services.notifications.SMTP_USER", "ada@gmail.com")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "ada@gmail.com")
    monkeypatch.setattr("services.notifications.SMTP_PASSWORD", "secret")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", "")
    monkeypatch.setattr("services.notifications.DEBUG", False)
    monkeypatch.setattr("services.notifications.smtplib.SMTP", FakeSMTP)

    results = send_invite_emails(
        ["guest@example.com"],
        "https://x/live/t",
        "Planning",
        "Ada",
        host_email="ada@gmail.com",
    )
    assert results[0]["status"] == "sent"
    message = next(item for item in sent if hasattr(item, "get"))
    assert "ada@gmail.com" in str(message["From"]).lower()
    assert sent[0] == "ada@gmail.com"


def test_sends_as_self_ignores_a_shared_provider_key(monkeypatch):
    from services.notifications import sends_as_self

    monkeypatch.setattr("services.notifications.BREVO_API_KEY", "brevo-test")
    monkeypatch.setattr("services.notifications._host_gmail_can_send", lambda user_id, db_path: False)
    monkeypatch.setattr("services.notifications._is_gmail_smtp", lambda: False)
    assert sends_as_self(4, "unused") is False
    assert sends_as_self(None) is False


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

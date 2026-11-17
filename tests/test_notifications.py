"""Follow-up mail: assignee tasks in the body, summary PDF for everyone."""

import base64
import json
import os
import tempfile
from unittest.mock import patch

from models.database import (
    get_meeting_emails,
    init_db,
    insert_decisions,
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


def test_report_summary_is_a_brief_not_the_transcript():
    from services.notifications import compose_report_summary

    text = compose_report_summary(
        [{"content": "We will do project meeting on Thursday."}],
        [{
            "assigned_to": "Yamna",
            "description": "Do task number 1 and 2 by Thursday",
            "due_date": "2026-11-19",
        }],
    )
    assert "turned on the meeting" not in text.lower()
    assert "agreed" in text.lower()
    assert "project meeting" in text.lower()
    assert "Yamna will do task number 1 and 2 by Thursday" in text
    assert "19 November 2026" in text


def test_followups_use_the_edited_task(tmp_path):
    from models.database import set_task_mail_approval, update_task_for_owner, get_tasks_by_meeting

    db = _db()
    try:
        owner_id = insert_user("ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_tasks(meeting_id, [{
            "description": "Send the original notes.",
            "assigned_to": "Ada",
            "due_date": "Friday",
            "status": "pending",
            "keyword": "will",
            "priority": "Low",
        }], db_path=db)
        task_id = get_tasks_by_meeting(meeting_id, db)[0]["id"]
        assert update_task_for_owner(
            meeting_id, task_id, owner_id,
            description="Send the revised notes.",
            assigned_to="Ada",
            due_date="Monday",
            priority="High",
            db_path=db,
        )
        assert set_task_mail_approval(meeting_id, task_id, owner_id, True, db)
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

            def login(self, *args):
                return None

            def send_message(self, message):
                sent.append(message)

        with patch("services.notifications.BREVO_API_KEY", ""), \
             patch("services.notifications.RESEND_API_KEY", ""), \
             patch("services.notifications.SMTP_HOST", "smtp.example.com"), \
             patch("services.notifications.SMTP_FROM", "ada@example.com"), \
             patch("services.notifications.SMTP_USER", "ada@example.com"), \
             patch("services.notifications.SMTP_PASSWORD", "secret"), \
             patch("services.notifications.EMAIL_DIR", str(tmp_path)), \
             patch("services.notifications.smtplib.SMTP", FakeSMTP):
            send_meeting_followups({
                "summary": "Yumnashabbir5 Okay Mamuna, I have turned on the meeting.",
                "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
            })
        plain = sent[0].get_body(preferencelist=("plain",)).get_content()
        pdf = next(tmp_path.rglob("*.pdf")).read_bytes()
        assert "Send the revised notes." in plain
        assert "Send the original notes." not in plain
        assert "turned on the meeting" not in plain
        assert b"revised notes" in pdf
        assert b"turned on the meeting" not in pdf
        assert b"Key points" not in pdf
        assert b"Deadline" not in pdf
    finally:
        os.remove(db)


def test_summary_pdf_contains_the_brief():
    pdf = build_summary_pdf(
        "Planning",
        "The team chose React Native.",
        [{"text": "Use React Native."}],
        [{"assigned_to": "Sara", "description": "Design the logo", "due_date": "Friday"}],
    )
    assert pdf.startswith(b"%PDF")
    assert b"The team chose React Native." in pdf
    assert b"Design the logo" not in pdf
    assert b"Use React Native." not in pdf


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


def test_ten_people_get_their_own_tasks_and_everyone_gets_the_report(tmp_path):
    """A named instruction is mailed to that person. An unnamed one is not guessed."""
    from services.nlp.task_verification import verify_meeting_tasks

    db = _db()
    names = ["Ada", "Bo", "Cy", "De", "Ev", "Fa", "Gi", "Hu", "Io", "Jo"]
    people = [{"name": name, "email": f"{name.lower()}@example.com"} for name in names]
    transcript = "\n".join([
        "Ada (ada@example.com): The meeting has started and I am recording.",
        "Ada (ada@example.com): Do you have any tasks?",
        "Ada (ada@example.com): Bo, I want you to send the notes by Friday and we will use chat for updates.",
        "Ada (ada@example.com): Cy, please check the budget by Wednesday.",
        "De (de@example.com): I will finish the report by Friday.",
        "Bo (bo@example.com): Yes, do the outline by Wednesday.",
        "Ada (ada@example.com): Ok, now I am ending the meeting.",
    ])
    try:
        checked = verify_meeting_tasks(
            transcript, participants=people, meeting_date="2026-11-15",
        )
        assert len(checked["tasks"]) == 3
        assert len(checked["decisions"]) == 1
        assert checked["decisions"][0]["text"] == "Use chat for updates"
        assert any(
            item.get("kind") == "unclear_assignee" and "outline" in (item.get("description") or "").lower()
            for item in checked["review"]
        )
        assert not any("outline" in task["description"].lower() for task in checked["tasks"])

        owner_id = insert_user("Ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_participants(meeting_id, people, db_path=db)
        insert_tasks(meeting_id, checked["tasks"], db_path=db)
        insert_decisions(meeting_id, checked["decisions"], db_path=db)

        sent = []

        class FakeSMTP:
            def __init__(self, host, port, timeout=None):
                return None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def starttls(self):
                return None

            def login(self, user, password):
                return None

            def send_message(self, message):
                sent.append(message)

        with patch("services.notifications.BREVO_API_KEY", ""), \
             patch("services.notifications.RESEND_API_KEY", ""), \
             patch("services.notifications.SMTP_HOST", "smtp.example.com"), \
             patch("services.notifications.SMTP_FROM", "ada@example.com"), \
             patch("services.notifications.SMTP_USER", "ada@example.com"), \
             patch("services.notifications.SMTP_PASSWORD", "secret"), \
             patch("services.notifications.EMAIL_DIR", str(tmp_path)), \
             patch("services.notifications.smtplib.SMTP", FakeSMTP):
            send_meeting_followups({
                "summary": "Use chat for updates. Bo will send the notes. Cy will check the budget.",
                "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
            })

        messages = [item for item in sent if hasattr(item, "get")]
        by_to = {item["To"]: item for item in messages}
        assert set(by_to) == {person["email"] for person in people}
        bo = by_to["bo@example.com"].get_body(preferencelist=("plain",)).get_content()
        cy = by_to["cy@example.com"].get_body(preferencelist=("plain",)).get_content()
        de = by_to["de@example.com"].get_body(preferencelist=("plain",)).get_content()
        ev = by_to["ev@example.com"].get_body(preferencelist=("plain",)).get_content()
        assert by_to["bo@example.com"]["Subject"].startswith("Your tasks")
        assert "notes" in bo.lower() and "budget" not in bo.lower()
        assert "budget" in cy.lower() and "notes" not in cy.lower()
        assert "report" in de.lower()
        assert "outline" not in bo.lower() and "outline" not in cy.lower()
        assert by_to["ev@example.com"]["Subject"].startswith("Summary of")
        assert "attached PDF" in ev
        assert all("meeting-summary.pdf" in item.as_string() for item in messages)
        rows = get_meeting_emails(meeting_id, db)
        assert {row["recipient"] for row in rows} == {person["email"] for person in people}
        assert all(row["status"] == "sent" for row in rows)
    finally:
        os.remove(db)


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


def test_assignee_mail_includes_the_assignment_and_is_not_repeated(tmp_path):
    db = _db()
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            return None

        def login(self, user, password):
            return None

        def send_message(self, message):
            sent.append(message)

    try:
        owner_id = insert_user("Ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_participants(meeting_id, [
            {"name": "Sara Khan", "email": "sara@example.com"},
        ], db_path=db)
        insert_tasks(meeting_id, [{
            "description": "Sara will design the logo.",
            "assigned_to": "Sara Khan",
            "assigned_email": "sara@example.com",
            "assigned_by": "Ada",
            "due_date": "Friday 3pm",
            "status": "pending",
            "keyword": "will",
            "priority": "High",
            "evidence_time": "00:01:10",
        }], db_path=db)
        with patch("services.notifications.BREVO_API_KEY", ""), \
             patch("services.notifications.RESEND_API_KEY", ""), \
             patch("services.notifications.SMTP_HOST", "smtp.example.com"), \
             patch("services.notifications.SMTP_FROM", "clario@example.com"), \
             patch("services.notifications.SMTP_USER", "clario@example.com"), \
             patch("services.notifications.SMTP_PASSWORD", "secret"), \
             patch("services.notifications.EMAIL_DIR", ""), \
             patch("services.notifications.smtplib.SMTP", FakeSMTP):
            payload = {
                "summary": "Notes.",
                "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
            }
            send_meeting_followups(payload)
            first = len(sent)
            send_meeting_followups(payload)
        sara = next(item for item in sent if item["To"] == "sara@example.com")
        body = sara.get_body(preferencelist=("plain",)).get_content()
        assert "Sara will design the logo." in body
        assert "Assigned to: Sara Khan" in body
        assert "Assigned by: Ada" in body
        assert "Planning" in body
        assert "Friday 3pm" in body
        assert "00:01:10" in body
        assert "Priority: High" in body
        assert "Status: pending" in body
        assert f"/result/{meeting_id}" in body
        assert "ada@example.com" in str(sara["From"]).lower()
        assert len(sent) == first
        rows = get_meeting_emails(meeting_id, db)
        assert all(row["status"] == "sent" for row in rows)
    finally:
        os.remove(db)


def test_a_spoken_sentence_is_not_used_as_an_email_address(tmp_path):
    db = _db()
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

        def login(self, *args):
            return None

        def send_message(self, message):
            sent.append(str(message["To"]))

    try:
        owner_id = insert_user("Ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_tasks(meeting_id, [{
            "description": "Please send the notes by Friday.",
            "assigned_to": "Please send the notes by Friday.",
            "due_date": "Friday",
            "status": "pending",
            "keyword": "please",
            "priority": "Low",
        }], db_path=db)
        with patch("services.notifications.BREVO_API_KEY", ""), \
             patch("services.notifications.RESEND_API_KEY", ""), \
             patch("services.notifications.SMTP_HOST", "smtp.example.com"), \
             patch("services.notifications.SMTP_FROM", "ada@example.com"), \
             patch("services.notifications.SMTP_USER", "ada@example.com"), \
             patch("services.notifications.SMTP_PASSWORD", "secret"), \
             patch("services.notifications.EMAIL_DIR", ""), \
             patch("services.notifications.smtplib.SMTP", FakeSMTP):
            send_meeting_followups({
                "summary": "Notes.",
                "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
            })
        assert sent == ["ada@example.com"]
    finally:
        os.remove(db)


def test_followups_wait_until_every_task_is_confirmed(tmp_path):
    db = _db()
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

        def login(self, *args):
            return None

        def send_message(self, message):
            sent.append(message)

    try:
        owner_id = insert_user("Ada", "ada@example.com", "hash", db_path=db)
        meeting_id = insert_meeting(owner_id, title="Planning", db_path=db)
        insert_tasks(meeting_id, [{
            "description": "Send the notes.",
            "assigned_to": "Ada",
            "due_date": "Friday",
            "status": "pending",
            "keyword": "will",
            "priority": "High",
        }], db_path=db)
        payload = {
            "summary": "Notes were assigned.",
            "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
            "require_approval": True,
        }
        with patch("services.notifications.BREVO_API_KEY", ""), \
             patch("services.notifications.RESEND_API_KEY", ""), \
             patch("services.notifications.SMTP_HOST", "smtp.example.com"), \
             patch("services.notifications.SMTP_FROM", "ada@example.com"), \
             patch("services.notifications.SMTP_USER", "ada@example.com"), \
             patch("services.notifications.SMTP_PASSWORD", "secret"), \
             patch("services.notifications.EMAIL_DIR", str(tmp_path)), \
             patch("services.notifications.smtplib.SMTP", FakeSMTP):
            send_meeting_followups(payload)
            assert sent == []
            rows = get_meeting_emails(meeting_id, db)
            assert rows[0]["status"] == "waiting"
            from models.database import set_task_mail_approval, get_tasks_by_meeting
            task_id = get_tasks_by_meeting(meeting_id, db)[0]["id"]
            assert set_task_mail_approval(meeting_id, task_id, owner_id, True, db)
            payload["require_approval"] = False
            send_meeting_followups(payload)
        message = sent[0]
        plain = message.get_body(preferencelist=("plain",)).get_content()
        rich = message.get_body(preferencelist=("html",)).get_content()
        assert "Task | Assignee | Deadline | Priority" in plain
        assert "Friday" in plain
        assert "Attendees" not in plain and "Attendees" not in rich
        assert "listed below" not in plain and "listed below" not in rich
        assert "Key points" not in plain and "Key points" not in rich
        assert "Notes were assigned." not in plain and "Notes were assigned." not in rich
        assert "<table" in rich and "Deadline" in rich
        assert "<h2" in rich and "font-family:Arial" in rich
        pdf = build_summary_pdf(
            "Planning", "The group agreed that chat will be used for updates. Ada will send the notes.",
            [{"content": "Use chat for updates."}],
            [{"assigned_to": "Ada", "description": "Send the notes.", "due_date": "Friday"}],
            participants=[{"name": "Ada", "email": "ada@example.com"}],
            when="15 November 2026",
            meeting_time="17:04",
        )
        assert b"Key points" not in pdf
        assert b"Clario" in pdf
        assert b"Ada will send the notes." in pdf
        assert b"Use chat for updates." not in pdf
        assert b"ada@example.com" in pdf
        assert b"15 November 2026" in pdf
    finally:
        os.remove(db)

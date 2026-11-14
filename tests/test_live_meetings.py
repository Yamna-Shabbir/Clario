"""Live rooms, shared-name confirmation, and the task board."""

import os
from io import BytesIO

from werkzeug.security import generate_password_hash

from app import create_app
from models.database import (
    get_meeting_emails,
    get_tasks_by_meeting,
    get_tasks_visible_to_user,
    init_db,
    insert_meeting,
    insert_tasks,
    insert_user,
    replace_meeting_participants,
    resolve_task_confirmation,
)
from services.live import format_speaker_transcript, speaker_lines
from services.nlp.participant_matching import ambiguous_matches, match
from services.notifications import send_meeting_followups


def test_speaker_transcript_keeps_name_and_email():
    turns = speaker_lines(
        [
            {"audio_path": "a.webm", "display_name": "Yamna", "email": "yamna@example.com"},
            {"audio_path": "b.webm", "display_name": "Lead", "email": "lead@example.com"},
        ],
        lambda path: "I will send the notes." if path == "a.webm" else "",
    )
    text = format_speaker_transcript(turns)
    assert text == "Yamna (yamna@example.com): I will send the notes."


def test_shared_first_name_is_ambiguous_and_holds_mail(tmp_path, monkeypatch):
    people = [
        {"name": "Sara Ali", "email": "sara.a@example.com"},
        {"name": "Sara Noor", "email": "sara.b@example.com"},
    ]
    assert match("Sara", people) is None
    choices = ambiguous_matches("Sara", people)
    assert {item["email"] for item in choices} == {"sara.a@example.com", "sara.b@example.com"}

    db = str(tmp_path / "clario.db")
    init_db(db)
    owner_id = insert_user("host", "host@example.com", "hash", db_path=db)
    meeting_id = insert_meeting(owner_id, title="Planning", db_path=db, status="completed")
    replace_meeting_participants(meeting_id, [
        {"name": "Sara Ali", "email": "sara.a@example.com", "source": "live"},
        {"name": "Sara Noor", "email": "sara.b@example.com", "source": "live"},
    ], db)
    insert_tasks(meeting_id, [{
        "description": "Sara will send the notes by Friday.",
        "assigned_to": "Sara",
        "due_date": "Friday",
        "status": "pending",
        "keyword": "will",
        "priority": "Low",
    }], db)
    monkeypatch.setattr("services.notifications.SMTP_HOST", "")
    monkeypatch.setattr("services.notifications.SMTP_FROM", "")
    monkeypatch.setattr("services.notifications.EMAIL_DIR", str(tmp_path))

    send_meeting_followups({
        "summary": "Notes were assigned.",
        "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
    })
    rows = get_meeting_emails(meeting_id, db)
    assert rows[0]["status"] == "waiting"
    assert not list(tmp_path.rglob("*.pdf"))

    task_id = get_tasks_by_meeting(meeting_id, db)[0]["id"]
    assert resolve_task_confirmation(meeting_id, task_id, "sara.b@example.com", db)
    send_meeting_followups({
        "summary": "Notes were assigned.",
        "stats": {"meeting_id": meeting_id, "user_id": owner_id, "db_path": db},
    })
    rows = get_meeting_emails(meeting_id, db)
    assert any(row["status"] == "saved" and row["recipient"] == "sara.b@example.com" for row in rows)


def test_live_room_records_named_audio_and_task_board(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    upload = str(tmp_path / "uploads")
    os.makedirs(upload, exist_ok=True)
    app = create_app(test_config={
        "db_path": db,
        "upload_dir": upload,
        "secret_key": "test-secret",
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    init_db(db)
    host_id = insert_user("Yamna", "yamna@example.com", generate_password_hash("Password1!"), db_path=db)
    guest_id = insert_user("Lead", "lead@example.com", generate_password_hash("Password1!"), db_path=db)
    client = app.test_client()
    monkeypatch.setattr("services.notifications.send_invite_emails", lambda *args, **kwargs: [
        {"recipient": "lead@example.com", "status": "sent", "detail": "Sent."},
    ])

    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    started = client.post("/live/start", data={"title": "Demo", "invites": "lead@example.com"}, follow_redirects=False)
    assert started.status_code == 302
    room_path = started.headers["Location"]
    assert "/live/" in room_path
    page = client.get(room_path)
    assert b"Join link" in page.data
    assert b"lead@example.com" in page.data or b"Copy link" in page.data

    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "Lead"
        sess["email"] = "lead@example.com"
    joined = client.get(room_path)
    assert joined.status_code == 200
    token = room_path.rstrip("/").split("/")[-1]
    request = client.post(
        f"/live/{token}/request",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert request.status_code == 200
    assert request.get_json()["registered"] is True
    assert request.get_json()["status"] == "pending"
    waiting_poll = client.get(f"/live/{token}/admission")
    assert waiting_poll.get_json()["status"] == "pending"
    blocked_signal = client.post(
        f"/live/{token}/signal",
        data={"to_user_id": host_id, "kind": "offer", "payload": '{"type":"offer","sdp":"v=0"}'},
    )
    assert blocked_signal.status_code == 403
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    admitted = client.post(
        f"/live/{token}/admit",
        data={"user_id": guest_id},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert admitted.status_code == 200
    assert admitted.get_json()["ok"] is True
    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "Lead"
        sess["email"] = "lead@example.com"
    admitted_poll = client.get(f"/live/{token}/admission")
    assert admitted_poll.get_json()["status"] == "admitted"
    roster = client.get(f"/live/{token}/roster")
    assert roster.get_json()["my_admission"] == "admitted"
    offered = client.post(
        f"/live/{token}/signal",
        data={"to_user_id": host_id, "kind": "offer", "payload": '{"type":"offer","sdp":"v=0"}'},
    )
    assert offered.status_code == 200
    assert offered.get_json()["ok"] is True
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    host_boot = client.get(f"/live/{token}/signal?bootstrap=1")
    assert host_boot.status_code == 200
    assert host_boot.get_json()["signals"] == []
    assert host_boot.get_json()["cursor"] >= offered.get_json()["id"]
    host_poll = client.get(f"/live/{token}/signal?since=0")
    kinds = [row["kind"] for row in host_poll.get_json()["signals"]]
    assert "offer" in kinds
    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "Lead"
        sess["email"] = "lead@example.com"
    chat_post = client.post(f"/live/{token}/chat", data={"message": "Hello everyone"})
    assert chat_post.status_code == 200
    assert chat_post.get_json()["ok"] is True
    msgs = client.get(f"/live/{token}/messages")
    assert msgs.get_json()["messages"][-1]["message"] == "Hello everyone"
    saved = client.post(
        f"/live/{token}/recording",
        data={"audio": (BytesIO(b"RIFFdemo"), "speaker.webm")},
        content_type="multipart/form-data",
    )
    assert saved.status_code == 200
    assert saved.get_json()["ok"] is True

    captured = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("services.jobs.submit_pipeline_job", fake_submit)
    monkeypatch.setattr(
        "services.live.transcribe_speaker_audio",
        lambda path: "Lead will prepare the slides by tomorrow.",
    )
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    ended = client.post(f"/live/{token}/end")
    assert ended.status_code == 302
    assert "Lead (lead@example.com):" in captured["text_input"]
    assert {person["email"] for person in captured["participants"]} == {
        "lead@example.com",
        "yamna@example.com",
    }

    meeting_id = int(ended.headers["Location"].rstrip("/").split("/")[-1])
    insert_tasks(meeting_id, [
        {
            "description": "Lead will prepare the slides by tomorrow.",
            "assigned_to": "Lead",
            "assigned_email": "lead@example.com",
            "due_date": "tomorrow",
            "status": "pending",
            "keyword": "will",
            "priority": "Low",
        },
        {
            "description": "Yamna will send the summary.",
            "assigned_to": "Yamna",
            "assigned_email": "yamna@example.com",
            "due_date": "Friday",
            "status": "pending",
            "keyword": "will",
            "priority": "Low",
        },
    ], db)
    visible = get_tasks_visible_to_user(guest_id, "lead@example.com", db)
    assert len(visible) == 2

    monkeypatch.setattr("utils.visualization.generate_status_pie_chart", lambda *args, **kwargs: "")
    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "Lead"
        sess["email"] = "lead@example.com"
    board = client.get("/tasks")
    assert b"prepare the slides" in board.data
    assert b"Yamna will send the summary" in board.data
    lead_task = next(task for task in visible if task["assigned_email"] == "lead@example.com")
    updated = client.post(f"/tasks/{lead_task['id']}/status", data={"status": "doing"})
    assert updated.status_code == 302
    assert get_tasks_by_meeting(meeting_id, db)[0]["status"] in ("doing", "pending")
    statuses = {task["assigned_email"]: task["status"] for task in get_tasks_by_meeting(meeting_id, db)}
    assert statuses["lead@example.com"] == "doing"

    blocked = client.post(f"/tasks/{next(task['id'] for task in get_tasks_by_meeting(meeting_id, db) if task['assigned_email'] == 'yamna@example.com')}/status", data={"status": "completed"})
    assert blocked.status_code == 302
    statuses = {task["assigned_email"]: task["status"] for task in get_tasks_by_meeting(meeting_id, db)}
    assert statuses["yamna@example.com"] == "pending"


def test_end_after_ending_claim_saves_the_meeting(tmp_path, monkeypatch):
    """The End button claims the room, then posts /end. That second step must not 404."""
    db = str(tmp_path / "app.db")
    upload = str(tmp_path / "uploads")
    os.makedirs(upload, exist_ok=True)
    app = create_app(test_config={
        "db_path": db,
        "upload_dir": upload,
        "secret_key": "test-secret",
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    init_db(db)
    host_id = insert_user("Yamna", "yamna@example.com", generate_password_hash("Password1!"), db_path=db)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    started = client.post("/live/start", data={"title": "Demo"}, follow_redirects=False)
    token = started.headers["Location"].rstrip("/").split("/")[-1]
    assert client.get(f"/live/{token}").status_code == 200
    saved = client.post(
        f"/live/{token}/recording",
        data={"audio": (BytesIO(b"RIFFdemo"), "speaker.webm")},
        content_type="multipart/form-data",
    )
    assert saved.status_code == 200
    monkeypatch.setattr("services.live.transcribe_speaker_audio", lambda path: "Yamna will send the notes.")
    monkeypatch.setattr("services.jobs.submit_pipeline_job", lambda **kwargs: None)
    claimed = client.post(f"/live/{token}/ending")
    assert claimed.status_code == 200
    assert claimed.get_json()["status"] == "ending"
    still_open = client.get(f"/live/{token}")
    assert still_open.status_code == 200
    opened = client.get(f"/live/{token}/end")
    assert opened.status_code == 302
    assert opened.headers["Location"].rstrip("/").endswith(f"/live/{token}")
    ended = client.post(f"/live/{token}/end")
    assert ended.status_code == 302
    assert "/result/" in ended.headers["Location"]


def test_guest_microphone_uploads_after_the_host_starts_ending(tmp_path, monkeypatch):
    """The other person's final upload happens after the room is already ending."""
    db = str(tmp_path / "app.db")
    upload = str(tmp_path / "uploads")
    os.makedirs(upload, exist_ok=True)
    app = create_app(test_config={
        "db_path": db,
        "upload_dir": upload,
        "secret_key": "test-secret",
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
    })
    init_db(db)
    host_id = insert_user("Yamna", "yamna@example.com", generate_password_hash("Password1!"), db_path=db)
    guest_id = insert_user("Aisha", "aisha@example.com", generate_password_hash("Password1!"), db_path=db)
    client = app.test_client()
    captured = {}

    def speak(path):
        if f"u{host_id}_" in os.path.basename(path):
            return "Yamna will ask for the task."
        return "Aisha will send the notes by Friday."

    monkeypatch.setattr("services.live.transcribe_speaker_audio", speak)
    monkeypatch.setattr("services.jobs.submit_pipeline_job", lambda **kwargs: captured.update(kwargs))
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    started = client.post("/live/start", data={"title": "Demo"}, follow_redirects=False)
    token = started.headers["Location"].rstrip("/").split("/")[-1]
    assert client.get(f"/live/{token}").status_code == 200
    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "Aisha"
        sess["email"] = "aisha@example.com"
    assert client.get(f"/live/{token}").status_code == 200
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    assert client.post(
        f"/live/{token}/admit",
        data={"user_id": guest_id},
        headers={"X-Requested-With": "XMLHttpRequest"},
    ).status_code == 200
    host_audio = client.post(
        f"/live/{token}/recording",
        data={"audio": (BytesIO(b"RIFFhost"), "speaker.webm"), "role": "speaker"},
        content_type="multipart/form-data",
    )
    assert host_audio.status_code == 200
    assert client.post(f"/live/{token}/ending").status_code == 200
    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "Aisha"
        sess["email"] = "aisha@example.com"
    guest_audio = client.post(
        f"/live/{token}/recording",
        data={"audio": (BytesIO(b"RIFFguest"), "speaker.webm"), "role": "speaker"},
        content_type="multipart/form-data",
    )
    assert guest_audio.status_code == 200
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "Yamna"
        sess["email"] = "yamna@example.com"
    ended = client.post(f"/live/{token}/end")
    assert ended.status_code == 302
    text = captured["text_input"]
    assert "Yamna (yamna@example.com):" in text
    assert "Aisha (aisha@example.com):" in text
    assert "send the notes" in text

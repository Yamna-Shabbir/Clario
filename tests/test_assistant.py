"""Assistant retrieval, follow-ups, evidence, permissions, and errors."""

import json

from werkzeug.security import generate_password_hash

from models.database import (
    get_user_by_email,
    insert_decisions,
    insert_meeting,
    insert_meeting_recordings,
    insert_summary,
    insert_tasks,
    insert_transcript,
    insert_user,
)
from services.assistant.chat import (
    HELP_EMPTY,
    MEETING_EMPTY,
    MODEL_NOTE,
    READONLY,
    answer_question,
)


def _db(app):
    return app.config["CLARIO_DB_PATH"]


def _seed_hiring(app, user_id):
    db = _db(app)
    meeting_id = insert_meeting(
        user_id,
        title="Hiring",
        meeting_date="2026-11-13",
        status="completed",
        db_path=db,
    )
    insert_transcript(
        meeting_id,
        "Ada: I will send the offer letter by Friday.",
        db,
    )
    insert_summary(meeting_id, "The team agreed to send an offer letter.", db)
    insert_tasks(meeting_id, [{
        "description": "Send the offer letter",
        "assigned_to": "Ada",
        "assigned_email": "ada@example.com",
        "due_date": "2026-11-20",
        "status": "pending",
        "priority": "High",
        "source_text": "I will send the offer letter by Friday.",
        "evidence_speaker": "Ada",
        "evidence_start": 12.5,
        "evidence_end": 16.0,
    }], db)
    insert_decisions(meeting_id, [{
        "text": "Send the offer this week",
        "source": "We will send the offer this week.",
        "evidence_speaker": "Ada",
        "evidence_start": 20.0,
    }], db)
    insert_meeting_recordings(meeting_id, [{
        "user_id": user_id,
        "speaker_name": "Full meeting",
        "audio_path": "uploads/hiring-full.wav",
        "kind": "meeting",
        "status": "saved",
    }], db)
    return meeting_id


def test_help_uses_documentation_and_does_not_invent(app):
    found = answer_question(
        "How do I start a meeting?",
        user_id=1,
        email="test@example.com",
        db_path=_db(app),
    )
    assert found["route"] == "help"
    assert "Start meeting" in found["content"]
    assert "docs/assistant-help.md" in found["content"]
    assert "Slack" not in found["content"]

    missing = answer_question(
        "How do I connect Slack?",
        user_id=1,
        email="test@example.com",
        db_path=_db(app),
    )
    assert missing["content"] == HELP_EMPTY
    assert missing["sources"] == []


def test_meeting_answer_includes_evidence(logged_in_client, app):
    client, creds = logged_in_client
    user = get_user_by_email(creds["email"], _db(app))
    meeting_id = _seed_hiring(app, user["id"])
    created = client.post("/assistant/conversations", json={})
    assert created.status_code == 200
    conversation_id = created.get_json()["id"]
    response = client.post(
        f"/assistant/conversations/{conversation_id}/messages",
        json={"message": "What tasks came out of Hiring?"},
    )
    assert response.status_code == 200
    message = response.get_json()["message"]
    assert "Send the offer letter" in message["content"]
    assert "2026-11-20" in message["content"]
    assert message["evidence"]
    item = message["evidence"][0]
    assert item["speaker"] == "Ada"
    assert "offer letter" in item["excerpt"]
    assert item["timestamp"] == "0:12"
    assert item["meeting_url"].endswith(f"/result/{meeting_id}")
    assert f"/meeting/{meeting_id}/recording/" in item["audio_url"]
    assert item["seek"] == 12.5


def test_followup_uses_the_same_meeting(logged_in_client, app):
    client, creds = logged_in_client
    user = get_user_by_email(creds["email"], _db(app))
    _seed_hiring(app, user["id"])
    conversation_id = client.post("/assistant/conversations", json={}).get_json()["id"]
    first = client.post(
        f"/assistant/conversations/{conversation_id}/messages",
        json={"message": "What tasks came out of Hiring?"},
    )
    assert first.status_code == 200
    second = client.post(
        f"/assistant/conversations/{conversation_id}/messages",
        json={"message": "When is it due?"},
    )
    body = second.get_json()["message"]["content"]
    assert "2026-11-20" in body
    assert "Hiring" in body


def test_other_user_cannot_see_meetings_or_history(logged_in_client, app):
    client, creds = logged_in_client
    user = get_user_by_email(creds["email"], _db(app))
    _seed_hiring(app, user["id"])
    conversation_id = client.post("/assistant/conversations", json={}).get_json()["id"]
    client.post(
        f"/assistant/conversations/{conversation_id}/messages",
        json={"message": "What tasks came out of Hiring?"},
    )
    outsider = app.test_client()
    insert_user("outsider", "outsider@example.com", generate_password_hash("OtherPass123!"), _db(app))
    other = get_user_by_email("outsider@example.com", _db(app))
    with outsider.session_transaction() as sess:
        sess["user_id"] = other["id"]
        sess["username"] = "outsider"
        sess["email"] = "outsider@example.com"
    denied = outsider.get(f"/assistant/conversations/{conversation_id}")
    assert denied.status_code == 404
    own = outsider.post("/assistant/conversations", json={})
    reply = outsider.post(
        f"/assistant/conversations/{own.get_json()['id']}/messages",
        json={"message": "What tasks came out of Hiring?"},
    )
    assert "Send the offer letter" not in reply.get_json()["message"]["content"]
    assert MEETING_EMPTY in reply.get_json()["message"]["content"]


def test_assignee_sees_task_without_the_transcript(app):
    owner = insert_user("owner", "owner@example.com", generate_password_hash("OwnerPass123!"), _db(app))
    meeting_id = _seed_hiring(app, owner)
    assignee = insert_user("ada", "ada@example.com", generate_password_hash("AdaPass123!"), _db(app))
    hidden = answer_question(
        "What did the transcript say in Hiring?",
        user_id=assignee,
        email="ada@example.com",
        db_path=_db(app),
    )
    assert "secret" not in hidden["content"].lower()
    assert "I will send the offer letter by Friday." not in hidden["content"]
    assert hidden["evidence"] == []
    visible = answer_question(
        "What tasks are still pending?",
        user_id=assignee,
        email="ada@example.com",
        db_path=_db(app),
    )
    assert "Send the offer letter" in visible["content"]
    assert meeting_id


def test_unavailable_and_readonly(app):
    empty = answer_question(
        "What did we decide about the lunar rover?",
        user_id=1,
        email="nobody@example.com",
        db_path=_db(app),
    )
    assert empty["content"] == MEETING_EMPTY
    blocked = answer_question(
        "Delete the task from Hiring",
        user_id=1,
        email="nobody@example.com",
        db_path=_db(app),
    )
    assert blocked["content"] == READONLY


def test_model_error_falls_back_to_notes(app, monkeypatch):
    monkeypatch.setattr("config.ASSISTANT_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr(
        "services.assistant.chat.complete",
        lambda messages, timeout=30: (_ for _ in ()).throw(RuntimeError("down")),
    )
    reply = answer_question(
        "How do I start a meeting?",
        user_id=1,
        email="test@example.com",
        db_path=_db(app),
    )
    assert "Start meeting" in reply["content"]
    assert MODEL_NOTE in reply["content"]


def test_model_wording_outside_the_notes_is_dropped(app, monkeypatch):
    monkeypatch.setattr("config.ASSISTANT_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr(
        "services.assistant.chat.complete",
        lambda messages, timeout=30: "Connect Slack and send the offer letter.",
    )
    reply = answer_question(
        "How do I start a meeting?",
        user_id=1,
        email="test@example.com",
        db_path=_db(app),
    )
    assert "Slack" not in reply["content"]
    assert "Start meeting" in reply["content"]


def test_page_and_validation(logged_in_client):
    client, _creds = logged_in_client
    page = client.get("/assistant")
    assert page.status_code == 200
    assert b"How do I start a meeting?" in page.data
    assert b"History" in page.data
    anonymous = client.application.test_client()
    assert anonymous.get("/assistant").status_code == 302
    conversation_id = client.post("/assistant/conversations", json={}).get_json()["id"]
    empty = client.post(
        f"/assistant/conversations/{conversation_id}/messages",
        json={"message": "   "},
    )
    assert empty.status_code == 400
    missing = client.get("/assistant/conversations/99999")
    assert missing.status_code == 404


def test_message_requires_csrf(tmp_dirs):
    from app import create_app

    flask_app = create_app(test_config={
        "db_path": tmp_dirs["db_path"],
        "upload_dir": tmp_dirs["upload_dir"],
        "graph_dir": tmp_dirs["graph_dir"],
        "TESTING": True,
        "WTF_CSRF_ENABLED": True,
    })
    csrf_client = flask_app.test_client()
    db = flask_app.config["CLARIO_DB_PATH"]
    user_id = insert_user("csrf", "csrf@example.com", generate_password_hash("CsrfPass123!"), db)
    with csrf_client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = "csrf"
        sess["email"] = "csrf@example.com"
    response = csrf_client.post(
        "/assistant/conversations",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert response.status_code == 400

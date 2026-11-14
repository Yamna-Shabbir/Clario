"""Saved live recordings, access control, and transcription retry."""

import os
from io import BytesIO

from models.database import (
    add_live_member,
    add_live_recording,
    create_live_room,
    get_live_recordings,
    get_live_room_by_token,
    get_meeting_recordings,
    get_meeting_status,
    get_meetings_by_user,
    get_tasks_by_meeting,
    init_db,
    insert_meeting,
    insert_meeting_recordings,
    insert_user,
    replace_meeting_participants,
)
from services.live import finish_live_room, meeting_highlights


def _room_with_track(db_path, upload_dir, *, speech="Aisha Khan, please send the notes."):
    host_id = insert_user("host", "host@example.com", "hash", db_path=db_path)
    guest_id = insert_user("guest", "guest@example.com", "hash", db_path=db_path)
    room_id = create_live_room(host_id, "Planning", "room-token", db_path)
    add_live_member(room_id, host_id, "Host", "host@example.com", db_path, admission_status="admitted")
    add_live_member(room_id, guest_id, "Aisha Khan", "guest@example.com", db_path, admission_status="admitted")
    path = os.path.join(upload_dir, "guest.webm")
    with open(path, "wb") as handle:
        handle.write(b"not-empty-audio")
    add_live_recording(room_id, guest_id, path, db_path)
    return host_id, guest_id, path


def test_empty_upload_does_not_replace_a_saved_track(tmp_path):
    db = str(tmp_path / "clario.db")
    init_db(db)
    host_id = insert_user("host", "host@example.com", "hash", db_path=db)
    room_id = create_live_room(host_id, "Planning", "tok", db)
    add_live_member(room_id, host_id, "Host", "host@example.com", db, admission_status="admitted")
    good = tmp_path / "good.webm"
    good.write_bytes(b"audio")
    add_live_recording(room_id, host_id, str(good), db)
    empty = tmp_path / "empty.webm"
    empty.write_bytes(b"")
    try:
        add_live_recording(room_id, host_id, str(empty), db)
        raised = False
    except ValueError:
        raised = True
    assert raised
    rows = get_live_recordings(room_id, db)
    assert rows[0]["audio_path"] == str(good)
    assert good.exists()


def test_failed_transcription_keeps_the_recording(tmp_path):
    db = str(tmp_path / "clario.db")
    init_db(db)
    upload = tmp_path / "uploads"
    upload.mkdir()
    host_id, _, path = _room_with_track(db, str(upload))
    room = get_live_room_by_token("room-token", db)

    def fail(_path):
        raise RuntimeError("asr down")

    meeting_id = finish_live_room(room, db_path=db, graph_dir=str(tmp_path), transcribe=fail)
    status = get_meeting_status(meeting_id, host_id, db)
    assert status["status"] == "failed"
    assert "kept" in (status["error"] or "").lower()
    assert os.path.isfile(path)
    recordings = get_meeting_recordings(meeting_id, db)
    assert len(recordings) == 1
    assert recordings[0]["status"] == "saved"

    again = finish_live_room(get_live_room_by_token("room-token", db), db_path=db, transcribe=fail)
    assert again == meeting_id
    assert len(get_meeting_recordings(meeting_id, db)) == 1


def test_retry_transcribes_without_a_second_copy(client, tmp_dirs, monkeypatch):
    db = tmp_dirs["db_path"]
    host_id, guest_id, path = _room_with_track(db, tmp_dirs["upload_dir"])
    room = get_live_room_by_token("room-token", db)

    def fail(_path):
        raise RuntimeError("asr down")

    meeting_id = finish_live_room(room, db_path=db, graph_dir=tmp_dirs["graph_dir"], transcribe=fail)
    monkeypatch.setattr(
        "services.live.transcribe_speaker_audio",
        lambda _path: "Aisha Khan, please send the notes and Bilal Noor, please review the budget.",
    )
    with client.session_transaction() as sess:
        sess["user_id"] = guest_id
        sess["username"] = "guest"
        sess["email"] = "guest@example.com"
    response = client.post(f"/meeting/{meeting_id}/retry-transcription")
    assert response.status_code == 302
    assert len(get_meeting_recordings(meeting_id, db)) == 1
    tasks = get_tasks_by_meeting(meeting_id, db)
    assignees = {task["assigned_to"] for task in tasks}
    assert "Aisha Khan" in assignees
    assert "Bilal Noor" in assignees
    assert all(task["meeting_id"] == meeting_id for task in tasks)
    assert os.path.isfile(path)


def test_participant_can_play_and_stranger_cannot(client, tmp_dirs):
    db = tmp_dirs["db_path"]
    owner = insert_user("owner", "owner@example.com", "hash", db_path=db)
    guest = insert_user("guest", "guest@example.com", "hash", db_path=db)
    stranger = insert_user("stranger", "stranger@example.com", "hash", db_path=db)
    meeting_id = insert_meeting(owner, title="Planning", db_path=db, status="completed")
    replace_meeting_participants(meeting_id, [
        {"name": "Guest", "email": "guest@example.com", "source": "live"},
    ], db)
    audio = os.path.join(tmp_dirs["upload_dir"], "track.webm")
    with open(audio, "wb") as handle:
        handle.write(b"audio-bytes")
    insert_meeting_recordings(meeting_id, [{
        "user_id": guest,
        "speaker_name": "Guest",
        "speaker_email": "guest@example.com",
        "audio_path": audio,
        "byte_size": 11,
        "status": "saved",
    }], db)
    recording_id = get_meeting_recordings(meeting_id, db)[0]["id"]

    with client.session_transaction() as sess:
        sess["user_id"] = guest
        sess["email"] = "guest@example.com"
    played = client.get(f"/meeting/{meeting_id}/recording/{recording_id}")
    assert played.status_code == 200
    assert played.data == b"audio-bytes"
    assert played.mimetype == "audio/webm"
    listed = get_meetings_by_user(guest, db)
    assert meeting_id in {row["id"] for row in listed}

    with client.session_transaction() as sess:
        sess["user_id"] = stranger
        sess["email"] = "stranger@example.com"
    denied = client.get(f"/meeting/{meeting_id}/recording/{recording_id}")
    assert denied.status_code == 404
    page = client.get(f"/result/{meeting_id}")
    assert page.status_code == 404


def test_recording_route_rejects_an_empty_file(client, tmp_dirs):
    db = tmp_dirs["db_path"]
    host_id = insert_user("host", "host@example.com", "hash", db_path=db)
    create_live_room(host_id, "Planning", "live-tok", db)
    with client.session_transaction() as sess:
        sess["user_id"] = host_id
        sess["username"] = "host"
        sess["email"] = "host@example.com"
    client.get("/live/live-tok")
    saved = client.post(
        "/live/live-tok/recording",
        data={"audio": (BytesIO(b"hello-audio"), "speaker.webm")},
        content_type="multipart/form-data",
    )
    assert saved.status_code == 200
    empty = client.post(
        "/live/live-tok/recording",
        data={"audio": (BytesIO(b""), "speaker.webm")},
        content_type="multipart/form-data",
    )
    assert empty.status_code == 400
    room = get_live_room_by_token("live-tok", db)
    rows = get_live_recordings(room["id"], db)
    assert len(rows) == 1
    assert os.path.getsize(rows[0]["audio_path"]) > 0


def test_recordings_page_plays_saved_audio_for_participants_only(client, tmp_dirs):
    db = tmp_dirs["db_path"]
    owner = insert_user("owner", "owner@example.com", "hash", db_path=db)
    guest = insert_user("guest", "guest@example.com", "hash", db_path=db)
    stranger = insert_user("stranger", "stranger@example.com", "hash", db_path=db)
    meeting_id = insert_meeting(owner, title="Planning notes", db_path=db, status="completed")
    replace_meeting_participants(meeting_id, [
        {"name": "Guest", "email": "guest@example.com", "source": "live"},
    ], db)
    audio = os.path.join(tmp_dirs["upload_dir"], "library.webm")
    with open(audio, "wb") as handle:
        handle.write(b"audio-bytes")
    insert_meeting_recordings(meeting_id, [{
        "user_id": guest,
        "speaker_name": "Guest",
        "speaker_email": "guest@example.com",
        "audio_path": audio,
        "byte_size": 11,
        "status": "saved",
    }], db)

    with client.session_transaction() as sess:
        sess["user_id"] = guest
        sess["email"] = "guest@example.com"
    page = client.get("/recordings")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "Planning notes" in body
    assert "Play meeting" in body
    assert f"/meeting/{meeting_id}/recording/" in body

    with client.session_transaction() as sess:
        sess["user_id"] = stranger
        sess["email"] = "stranger@example.com"
    hidden = client.get("/recordings")
    assert "Planning notes" not in hidden.get_data(as_text=True)
    assert "No recordings yet" in hidden.get_data(as_text=True)


def test_highlights_come_only_from_the_transcript():
    found = meeting_highlights(
        "The budget review covered the archive plan in detail. Will the archive be ready?"
    )
    assert found["questions"] == ["Will the archive be ready?"]
    assert found["points"]
    assert "archive plan" in found["points"][0]

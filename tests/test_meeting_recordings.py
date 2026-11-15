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
from services.live import finish_live_room, meeting_highlights, tracks_to_transcribe


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


def test_a_mix_does_not_replace_the_speaker_track(tmp_path):
    db = str(tmp_path / "clario.db")
    init_db(db)
    host_id = insert_user("host", "host@example.com", "hash", db_path=db)
    room_id = create_live_room(host_id, "Planning", "tok", db)
    add_live_member(room_id, host_id, "Host", "host@example.com", db, admission_status="admitted")
    voice = tmp_path / "voice.webm"
    voice.write_bytes(b"voice")
    whole = tmp_path / "whole.webm"
    whole.write_bytes(b"everyone")
    add_live_recording(room_id, host_id, str(voice), db, kind="speaker")
    add_live_recording(room_id, host_id, str(whole), db, kind="mix")
    rows = get_live_recordings(room_id, db)
    kinds = {row["kind"]: row["audio_path"] for row in rows}
    assert kinds["speaker"] == str(voice)
    assert kinds["mix"] == str(whole)
    assert voice.exists() and whole.exists()


def test_the_speaker_is_stored_as_the_assigner():
    from services.nlp.task_verification import verify_meeting_tasks

    checked = verify_meeting_tasks(
        "Hina (hina@example.com): Omar, please send the notes by Friday."
    )
    tasks = checked["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["assigned_to"] == "Omar"
    assert tasks[0]["assigned_by"] == "Hina"
    assert tasks[0]["assigned_by_email"] == "hina@example.com"


def test_a_shorter_upload_does_not_replace_a_longer_recording(tmp_path):
    db = str(tmp_path / "clario.db")
    init_db(db)
    host_id = insert_user("host", "host@example.com", "hash", db_path=db)
    room_id = create_live_room(host_id, "Planning", "tok", db)
    add_live_member(room_id, host_id, "Host", "host@example.com", db, admission_status="admitted")
    long_file = tmp_path / "long.webm"
    long_file.write_bytes(b"long-audio-bytes")
    short_file = tmp_path / "short.webm"
    short_file.write_bytes(b"short")
    add_live_recording(room_id, host_id, str(long_file), db)
    add_live_recording(room_id, host_id, str(short_file), db)
    rows = get_live_recordings(room_id, db)
    assert len(rows) == 1
    assert rows[0]["audio_path"] == str(long_file)
    assert long_file.exists()


def test_both_microphones_are_saved_and_transcribed_in_order(tmp_path):
    from models.database import get_meeting_participants, get_speaker_turns

    db = str(tmp_path / "clario.db")
    init_db(db)
    upload = tmp_path / "uploads"
    upload.mkdir()
    host_id, guest_id, guest_path = _room_with_track(db, str(upload), speech="unused")
    host_path = os.path.join(str(upload), "host.webm")
    with open(host_path, "wb") as handle:
        handle.write(b"host-audio")
    room = get_live_room_by_token("room-token", db)
    add_live_recording(room["id"], host_id, host_path, db)

    def speak(path):
        if os.path.basename(path) == "host.webm":
            return "The plan was described."
        return "The notes will be sent."

    meeting_id = finish_live_room(room, db_path=db, graph_dir=str(tmp_path), transcribe=speak)
    turns = get_speaker_turns(meeting_id, db)
    assert [turn["speaker_name"] for turn in turns] == ["Aisha Khan", "Host"]
    assert [turn["content"] for turn in turns] == [
        "The notes will be sent.",
        "The plan was described.",
    ]
    people = get_meeting_participants(meeting_id, db)
    assert people
    assert all(person["left_at"] for person in people)
    recordings = get_meeting_recordings(meeting_id, db)
    assert {row["user_id"] for row in recordings} == {host_id, guest_id}
    again = finish_live_room(get_live_room_by_token("room-token", db), db_path=db, transcribe=speak)
    assert again == meeting_id
    assert len(get_meeting_recordings(meeting_id, db)) == 2


def test_a_missing_microphone_uses_the_meeting_mix(tmp_path):
    from models.database import get_accessible_meeting_detail, get_speaker_turns

    db = str(tmp_path / "clario.db")
    init_db(db)
    upload = tmp_path / "uploads"
    upload.mkdir()
    host_id, guest_id, path = _room_with_track(db, str(upload))
    room = get_live_room_by_token("room-token", db)
    from models.database import get_connection
    with get_connection(db) as conn:
        conn.execute(
            "DELETE FROM live_recordings WHERE room_id = ? AND user_id = ?",
            (room["id"], guest_id),
        )
        conn.commit()
    host_path = os.path.join(str(upload), "host-only.webm")
    mix_path = os.path.join(str(upload), "everyone.webm")
    with open(host_path, "wb") as handle:
        handle.write(b"host-only-audio")
    with open(mix_path, "wb") as handle:
        handle.write(b"everyone-in-the-room")
    add_live_recording(room["id"], host_id, host_path, db, kind="speaker")
    add_live_recording(room["id"], host_id, mix_path, db, kind="mix")

    def speak(audio_path):
        if os.path.basename(audio_path) == "everyone.webm":
            return "Aisha Khan will send the budget by Friday."
        return "I asked who will send the budget."

    meeting_id = finish_live_room(
        room, db_path=db, graph_dir=str(tmp_path), transcribe=speak,
    )
    turns = get_speaker_turns(meeting_id, db)
    assert [turn["speaker_name"] for turn in turns] == ["Host", "Aisha Khan"]
    assert turns[1]["speaker_email"] == "guest@example.com"
    text = (get_accessible_meeting_detail(meeting_id, host_id, db) or {}).get("transcript") or ""
    assert "Aisha Khan will send the budget" in text
    assert "I asked who will send the budget" in text
    status = get_meeting_status(meeting_id, host_id, db)
    assert status["status"] != "partial"


def test_email_labels_are_not_repeated_and_other_speech_keeps_its_speaker():
    from services.live import attribute_unheard_speech, format_speaker_transcript
    from services.nlp.task_verification import verify_meeting_tasks

    text = format_speaker_transcript([
        {"speaker_name": "ada.user5@example.com", "speaker_email": "ada.user5@example.com", "content": "Give me any task."},
        {"speaker_name": "noor.user1@example.com", "speaker_email": "noor.user1@example.com", "content": "I will make the tea by Friday."},
    ])
    assert text == (
        "ada.user5@example.com: Give me any task.\n"
        "noor.user1@example.com: I will make the tea by Friday."
    )
    turns = attribute_unheard_speech(
        "Give me any task. I will make the tea by Friday.",
        [{"content": "Give me any task."}],
        [{"display_name": "noor.user1@example.com", "email": "noor.user1@example.com"}],
    )
    assert turns[0]["speaker_email"] == "noor.user1@example.com"
    assert "make the tea" in turns[0]["content"]
    assert "Give me any task" not in turns[0]["content"]
    people = [
        {"name": "ada.user5@example.com", "email": "ada.user5@example.com"},
        {"name": "noor.user1@example.com", "email": "noor.user1@example.com"},
    ]
    checked = verify_meeting_tasks(
        "ada.user5@example.com: Noor, please make the tea by Friday.\n"
        "noor.user1@example.com: I will make the tea by Friday.",
        participants=people,
    )
    tasks = checked["tasks"]
    assert tasks
    given = next(task for task in tasks if task.get("assigned_by_email") == "ada.user5@example.com")
    assert given["assigned_email"] == "noor.user1@example.com"


def test_tracks_skip_the_mix_when_every_microphone_arrived():
    saved = [
        {"kind": "speaker", "id": 1, "speaker_email": "a@example.com", "byte_size": 10},
        {"kind": "mix", "id": 2, "byte_size": 50},
        {"kind": "speaker", "id": 3, "speaker_email": "b@example.com", "byte_size": 12},
    ]
    chosen = tracks_to_transcribe(saved, missing_names=[])
    assert [row["id"] for row in chosen] == [1, 3]
    covered = tracks_to_transcribe(saved, missing_names=["Aisha"])
    assert [row["id"] for row in covered] == [1, 3, 2]


def test_a_missing_microphone_is_partial_not_complete(tmp_path):
    db = str(tmp_path / "clario.db")
    init_db(db)
    upload = tmp_path / "uploads"
    upload.mkdir()
    host_id, guest_id, path = _room_with_track(db, str(upload))
    room = get_live_room_by_token("room-token", db)
    from models.database import get_connection
    with get_connection(db) as conn:
        conn.execute(
            "DELETE FROM live_recordings WHERE room_id = ? AND user_id = ?",
            (room["id"], guest_id),
        )
        conn.commit()
    host_path = os.path.join(str(upload), "host-only.webm")
    with open(host_path, "wb") as handle:
        handle.write(b"host-only-audio")
    add_live_recording(room["id"], host_id, host_path, db)
    meeting_id = finish_live_room(
        room, db_path=db, graph_dir=str(tmp_path), transcribe=lambda _path: "The plan was described.",
    )
    status = get_meeting_status(meeting_id, host_id, db)
    assert status["status"] == "partial"
    assert "Aisha Khan" in (status["error"] or "")
    assert os.path.isfile(host_path)


def test_full_meeting_wav_keeps_both_voices_on_one_timeline(tmp_path):
    import struct
    import wave
    from services.audio_mix import build_meeting_mix

    def write_wav(path, samples):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(struct.pack("<" + ("h" * len(samples)), *samples))

    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    write_wav(first, [10000] * 8000)
    write_wav(second, [12000] * 8000)
    db = str(tmp_path / "clario.db")
    init_db(db)
    owner = insert_user("ada", "ada@example.com", "hash", db_path=db)
    meeting_id = insert_meeting(owner, title="Live", db_path=db)
    rows = [
        {
            "speaker_name": "Ada",
            "speaker_email": "ada@example.com",
            "audio_path": str(first),
            "byte_size": first.stat().st_size,
            "status": "saved",
            "kind": "speaker",
            "created_at": "2026-11-15T00:00:00",
        },
        {
            "speaker_name": "Sara",
            "speaker_email": "sara@example.com",
            "audio_path": str(second),
            "byte_size": second.stat().st_size,
            "status": "saved",
            "kind": "speaker",
            "created_at": "2026-11-15T00:00:01",
        },
    ]
    insert_meeting_recordings(meeting_id, rows, db)
    path = build_meeting_mix(meeting_id, rows, db)
    assert path and os.path.isfile(path)
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        duration = handle.getnframes() / handle.getframerate()
    samples = struct.unpack("<" + ("h" * (len(frames) // 2)), frames)
    assert duration == 2
    assert samples[100] != 0
    assert samples[8000 + 100] != 0
    assert max(abs(sample) for sample in samples) <= 32767
    stored = get_meeting_recordings(meeting_id, db)
    full = next(row for row in stored if row["kind"] == "meeting")
    assert full["speaker_name"] == "Full meeting"
    assert {row["kind"] for row in stored if row["kind"] == "speaker"} == {"speaker"}
    assert build_meeting_mix(meeting_id, stored, db) is None


def test_three_microphones_mix_without_clipping_or_dropping_a_voice(tmp_path):
    import struct
    import wave
    from services.audio_mix import build_meeting_mix

    def write_wav(path, samples):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(struct.pack("<" + ("h" * len(samples)), *samples))

    paths = []
    for name, level in (("a", 4000), ("b", 20000), ("c", 8000)):
        path = tmp_path / f"{name}.wav"
        write_wav(path, [level] * 4000)
        paths.append(path)
    db = str(tmp_path / "clario.db")
    init_db(db)
    owner = insert_user("ada", "ada@example.com", "hash", db_path=db)
    meeting_id = insert_meeting(owner, title="Three", db_path=db)
    rows = []
    for index, path in enumerate(paths):
        rows.append({
            "speaker_name": f"Person {index}",
            "speaker_email": f"p{index}@example.com",
            "audio_path": str(path),
            "byte_size": path.stat().st_size,
            "status": "saved",
            "kind": "speaker",
            "created_at": f"2026-11-15T00:00:0{index}",
        })
    path = build_meeting_mix(meeting_id, rows, db)
    assert path
    with wave.open(path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        duration = handle.getnframes() / handle.getframerate()
    samples = struct.unpack("<" + ("h" * (len(frames) // 2)), frames)
    assert duration == 2.5
    assert samples[100] != 0
    assert samples[8000 + 100] != 0
    assert samples[16000 + 100] != 0
    assert max(abs(sample) for sample in samples) <= 32767


def test_a_status_remark_creates_no_task_or_decision():
    from services.nlp.task_verification import verify_meeting_tasks

    checked = verify_meeting_tasks("Now it's recording.")
    assert checked["tasks"] == []
    assert checked["decisions"] == []
    assert all((item.get("assigned_to") or "") != "Now" for item in checked["review"])


def test_highlights_come_only_from_the_transcript():
    found = meeting_highlights(
        "The budget review covered the archive plan in detail. Will the archive be ready?"
    )
    assert found["questions"] == ["Will the archive be ready?"]
    assert found["points"]
    assert "archive plan" in found["points"][0]

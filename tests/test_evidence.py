"""Evidence stays on the spoken line, survives a status change, and stays private."""

import os

from models.database import (
    get_decisions_by_meeting,
    get_tasks_by_meeting,
    get_user_by_email,
    init_db,
    insert_decisions,
    insert_meeting,
    insert_meeting_recordings,
    insert_tasks,
    set_assignee_task_status,
)
from services.live import evidence_playback
from services.nlp.task_verification import (
    attach_recorded_evidence,
    verify_meeting_tasks,
)


def test_each_item_keeps_its_own_line_and_times():
    result = verify_meeting_tasks(
        "Hina (hina@example.com): [00:01:12] Please send the notes by Friday.\n"
        "Omar (omar@example.com): We will use chat for updates.",
        participants=[
            {"name": "Hina", "email": "hina@example.com"},
            {"name": "Omar", "email": "omar@example.com"},
        ],
        meeting_date="2026-11-15",
        segments=[
            {
                "speaker": "Hina",
                "speaker_email": "hina@example.com",
                "text": "Please send the notes by Friday.",
                "original": "Please send the notes by Friday.",
                "timestamp": "00:01:12",
                "started_at": 72.0,
                "ended_at": 80.0,
                "turn_id": 4,
            },
            {
                "speaker": "Omar",
                "speaker_email": "omar@example.com",
                "text": "We will use chat for updates.",
                "original": "We will use chat for updates.",
                "started_at": 90.0,
                "ended_at": 96.0,
                "turn_id": 5,
            },
        ],
    )
    notes = next(task for task in result["tasks"] if "notes" in task["description"].lower())
    assert notes["source_text"] == "Please send the notes by Friday."
    assert notes["evidence_speaker"] == "Hina"
    assert notes["evidence_start"] == 72.0
    assert notes["evidence_end"] == 80.0
    assert notes["speaker_turn_id"] == 4
    assert notes["evidence_start"] != 90.0
    decision = result["decisions"][0]
    assert decision["evidence_speaker"] == "Omar"
    assert decision["evidence_start"] == 90.0
    assert decision["evidence_end"] == 96.0
    assert decision["speaker_turn_id"] == 5
    assert "notes" not in (decision.get("source") or "").lower()


def test_a_line_without_a_time_does_not_receive_one():
    result = verify_meeting_tasks(
        "Hina: Omar, please send the notes by Friday.",
        participants=[
            {"name": "Hina", "email": "hina@example.com"},
            {"name": "Omar", "email": "omar@example.com"},
        ],
    )
    task = result["tasks"][0]
    assert task.get("evidence_start") is None
    assert task.get("evidence_end") is None


def test_recorded_turns_attach_only_when_the_words_and_speaker_match():
    segments = [
        {"speaker": "Hina", "text": "Please send the notes.", "original": "Please send the notes."},
        {"speaker": "Omar", "text": "Please send the notes.", "original": "Please send the notes."},
    ]
    attach_recorded_evidence(segments, [
        {"id": 1, "speaker_name": "Omar", "content": "Please send the notes.", "started_at": 40, "ended_at": 48},
        {"id": 2, "speaker_name": "Hina", "content": "Please send the notes.", "started_at": 10, "ended_at": 18},
    ])
    assert segments[0]["started_at"] == 10
    assert segments[0]["turn_id"] == 2
    assert segments[1]["started_at"] == 40
    assert segments[1]["turn_id"] == 1


def test_saved_evidence_remains_after_a_status_change_and_meeting_delete(tmp_path):
    db = str(tmp_path / "evidence.db")
    init_db(db)
    owner = 1
    from models.database import insert_user
    owner = insert_user("Hina", "hina@example.com", "hash", db_path=db)
    meeting_id = insert_meeting(owner, title="Planning", db_path=db)
    insert_tasks(meeting_id, [{
        "description": "Send the notes",
        "assigned_to": "Omar",
        "assigned_email": "omar@example.com",
        "assigned_by": "Hina",
        "source_text": "Please send the notes.",
        "evidence_speaker": "Hina",
        "evidence_email": "hina@example.com",
        "evidence_start": 12.5,
        "evidence_end": 18.0,
        "speaker_turn_id": 3,
        "evidence_time": "0:12–0:18",
    }], db_path=db)
    insert_decisions(meeting_id, [{
        "text": "Use chat for updates",
        "source": "We will use chat for updates.",
        "evidence_speaker": "Omar",
        "evidence_start": 20.0,
        "evidence_end": 24.0,
        "speaker_turn_id": 4,
    }], db_path=db)
    task_id = get_tasks_by_meeting(meeting_id, db)[0]["id"]
    set_assignee_task_status(task_id, "omar@example.com", "doing", db)
    kept = get_tasks_by_meeting(meeting_id, db)[0]
    assert kept["status"] == "doing"
    assert kept["source_text"] == "Please send the notes."
    assert kept["evidence_speaker"] == "Hina"
    assert kept["evidence_start"] == 12.5
    assert kept["evidence_end"] == 18.0
    assert kept["speaker_turn_id"] == 3
    decision = get_decisions_by_meeting(meeting_id, db)[0]
    assert decision["source_text"] == "We will use chat for updates."
    assert decision["evidence_start"] == 20.0

    from models.database import get_connection
    with get_connection(db) as conn:
        conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        conn.commit()
    assert get_tasks_by_meeting(meeting_id, db) == []
    assert get_decisions_by_meeting(meeting_id, db) == []


def test_playback_uses_the_full_recording_and_skips_a_missing_time():
    recordings = [
        {"id": 8, "kind": "meeting", "status": "saved"},
        {"id": 2, "kind": "speaker", "status": "saved", "speaker_email": "hina@example.com"},
    ]
    play = evidence_playback(recordings, {"evidence_start": 12.5})
    assert play == {"recording_id": 8, "start": 12.5}
    assert evidence_playback(recordings, {"evidence_start": None}) is None
    several = [
        {"id": 2, "kind": "speaker", "status": "saved"},
        {"id": 3, "kind": "speaker", "status": "saved"},
    ]
    assert evidence_playback(several, {"evidence_start": 12.5}) is None


def test_result_page_offers_evidence_and_blocks_other_accounts(logged_in_client, tmp_dirs):
    client, creds = logged_in_client
    db = tmp_dirs["db_path"]
    owner = get_user_by_email(creds["email"], db)
    meeting_id = insert_meeting(owner["id"], title="Planning", db_path=db)
    insert_tasks(meeting_id, [
        {
            "description": "Send the notes",
            "assigned_to": "Omar",
            "assigned_email": "omar@example.com",
            "source_text": "Please send the notes by Friday.",
            "evidence_speaker": "Hina",
            "evidence_start": 12.5,
            "evidence_end": 18.0,
            "evidence_time": "0:12–0:18",
        },
        {
            "description": "Check the budget",
            "assigned_to": "Hina",
            "source_text": "Please check the budget.",
            "evidence_speaker": "Omar",
        },
    ], db_path=db)
    insert_decisions(meeting_id, [{
        "text": "Use chat for updates",
        "source": "We will use chat for updates.",
        "evidence_speaker": "Omar",
        "evidence_start": 30,
        "evidence_end": 34,
        "evidence_time": "0:30–0:34",
    }], db_path=db)
    audio = os.path.join(tmp_dirs["upload_dir"], "full.wav")
    with open(audio, "wb") as handle:
        handle.write(b"RIFFdemo")
    insert_meeting_recordings(meeting_id, [{
        "speaker_name": "Full meeting",
        "audio_path": audio,
        "byte_size": 8,
        "status": "saved",
        "kind": "meeting",
    }], db_path=db)
    page = client.get(f"/result/{meeting_id}")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert html.count("View Evidence") == 3
    assert "Please send the notes by Friday." in html
    assert "We will use chat for updates." in html
    assert "Hina" in html
    assert "0:12–0:18" in html
    assert "No timestamp for this line" in html
    assert html.count("Play Evidence") == 2
    assert 'data-start="12.5"' in html
    assert 'data-evidence-play' in html
    assert "Open full recording" in html

    outsider = client.application.test_client()
    outsider.post("/register", data={
        "username": "other",
        "email": "other@example.com",
        "password": "OtherPass1!",
        "confirm": "OtherPass1!",
    })
    denied = outsider.get("/result/" + str(meeting_id))
    assert denied.status_code == 404
    from models.database import get_meeting_recordings
    recording_id = get_meeting_recordings(meeting_id, db)[0]["id"]
    audio_denied = outsider.get(f"/meeting/{meeting_id}/recording/{recording_id}")
    assert audio_denied.status_code == 404
    audio_ok = client.get(f"/meeting/{meeting_id}/recording/{recording_id}")
    assert audio_ok.status_code == 200

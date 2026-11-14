"""Meeting-level task evidence, speaker context, and review flags."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from models.database import (
    get_task_reviews,
    get_tasks_by_meeting,
    init_db,
    insert_meeting,
    insert_tasks,
    insert_user,
    replace_task_reviews,
)
from services.nlp.task_detection import detect_tasks
from services.nlp.task_verification import (
    prepare_meeting_turns,
    segment_transcript,
    verify_meeting_tasks,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "meeting_actions.json"
MULTI = Path(__file__).resolve().parent / "fixtures" / "multilingual_meetings.json"


def _meeting(meeting_id: str) -> dict:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return next(item for item in data["meetings"] if item["id"] == meeting_id)


def _multi(meeting_id: str) -> dict:
    data = json.loads(MULTI.read_text(encoding="utf-8"))
    return next(item for item in data["meetings"] if item["id"] == meeting_id)


def _verified(meeting: dict) -> dict:
    turns = prepare_meeting_turns(meeting["transcript"])
    return verify_meeting_tasks(meeting["transcript"], segments=turns)


def test_noun_phrase_is_not_a_task():
    tasks = detect_tasks("So far, 25 test cases have been run and 20 passed.")
    assert tasks == []


def test_yes_is_not_treated_as_a_person():
    tasks = detect_tasks("Yes, the API update is still open.")
    assert tasks == []


def test_speaker_and_timestamp_stay_with_the_commitment():
    meeting = _meeting("english-standup")
    turns = segment_transcript(meeting["transcript"])
    result = verify_meeting_tasks(meeting["transcript"], segments=turns)
    login = next(task for task in result["tasks"] if "login bug" in task["description"].lower())
    assert login["assigned_to"] == "Omar"
    assert login["evidence_time"] == "00:01:22"
    assert "I can fix the login bug before Friday." in login["source_text"]
    assert "00:01:22" not in result["transcript"]


def test_later_deadline_replaces_the_earlier_one():
    meeting = _meeting("english-standup")
    turns = segment_transcript(meeting["transcript"])
    result = verify_meeting_tasks(meeting["transcript"], segments=turns)
    specs = [task for task in result["tasks"] if "api spec" in task["description"].lower()]
    assert len(specs) == 1
    assert "thursday" in specs[0]["description"].lower()
    assert any(item["kind"] == "superseded" for item in result["review"])


def test_reply_keeps_both_sentences_as_evidence():
    meeting = _meeting("mixed-followup")
    turns = segment_transcript(meeting["transcript"])
    result = verify_meeting_tasks(meeting["transcript"], segments=turns)
    login = next(task for task in result["tasks"] if "login page" in task["description"].lower())
    assert login["assigned_to"] == "Ali"
    assert login["source_text"] == "I can do that tomorrow."
    assert "login page" in login["source_context"].lower()
    assert "tomorrow" in (login["description"] + login["due_date"]).lower()


def test_unnamed_imperative_is_reviewed_not_assigned_to_a_guess():
    result = verify_meeting_tasks("Please send the meeting notes to everyone.")
    assert result["tasks"] == []
    assert result["review"]
    assert result["review"][0]["assigned_to"] == "Unassigned"
    assert "nobody is named" in result["review"][0]["reason"].lower()


def test_unnamed_soft_commitment_is_only_a_review():
    result = verify_meeting_tasks("Someone ought to glance at the logs eventually.")
    assert result["tasks"] == []
    assert result["review"][0]["kind"] == "possible_miss"
    assert result["review"][0]["assigned_to"] == "Unassigned"


def test_timestamped_name_header_sets_speaker():
    transcript = "[00:04:10] Sara:\nI'll finish the page by Friday."
    turns = prepare_meeting_turns(transcript)
    result = verify_meeting_tasks(transcript, segments=turns)
    assert result["tasks"][0]["assigned_to"] == "Sara"


def test_directed_please_waits_for_acceptance():
    transcript = (
        "Ayesha:\nOmar, API update aaj shaam tak bhej dena.\n\n"
        "Omar:\nTheek hai, main bhej dunga."
    )
    turns = prepare_meeting_turns(transcript)
    result = verify_meeting_tasks(transcript, segments=turns)
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["assigned_to"] == "Omar"
    assert "evening" in result["tasks"][0]["description"].lower()


def test_chatter_and_past_status_are_not_tasks():
    meeting = _meeting("english-standup")
    turns = segment_transcript(meeting["transcript"])
    result = verify_meeting_tasks(meeting["transcript"], segments=turns)
    blob = " ".join(task["description"] for task in result["tasks"]).lower()
    assert "budget" not in blob
    assert "how is the login bug" not in blob
    assert any("release" in (item.get("text") or "") for item in result["decisions"])


def test_reviews_round_trip_in_sqlite():
    folder = Path(__file__).resolve().parents[1] / "data" / "_review_test"
    folder.mkdir(parents=True, exist_ok=True)
    db_path = str(folder / "clario.db")
    for suffix in ("", "-wal", "-shm"):
        stale = Path(db_path + suffix)
        if stale.exists():
            stale.unlink()
    init_db(db_path)
    user_id = insert_user("ada", "ada@example.com", "hash", db_path)
    meeting_id = insert_meeting(user_id, "Standup", db_path=db_path)
    insert_tasks(meeting_id, [{
        "description": "Omar will fix the login bug before Friday.",
        "assigned_to": "Omar",
        "due_date": "Friday",
        "source_text": "I can fix the login bug before Friday.",
        "evidence_time": "00:01:22",
        "source_context": "",
    }], db_path)
    replace_task_reviews(meeting_id, [{
        "kind": "unclear_assignee",
        "description": "Please send the notes.",
        "assigned_to": "Unassigned",
        "due_date": "Not specified",
        "source_text": "Please send the notes.",
        "reason": "A commitment is here, but nobody is named.",
    }], db_path)
    stored = get_tasks_by_meeting(meeting_id, db_path)
    reviews = get_task_reviews(meeting_id, db_path)
    assert stored[0]["source_text"] == "I can fix the login bug before Friday."
    assert stored[0]["evidence_time"] == "00:01:22"
    assert reviews[0]["kind"] == "unclear_assignee"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        left = conn.execute(
            "SELECT COUNT(*) FROM task_reviews WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()[0]
    assert left == 0


def test_urdu_named_please_is_kept_without_a_reply():
    result = _verified(_multi("urdu-script-assignments"))
    notes = [
        task for task in result["tasks"]
        if "notes" in task["description"].lower() and "monday" in task["description"].lower()
    ]
    assert len(notes) == 1
    assert notes[0]["assigned_to"] == "Kamran"
    assert notes[0]["evidence_time"] == "00:10:33"
    assert "نوٹس" in notes[0]["source_text"]
    api = [
        task for task in result["tasks"]
        if "api" in task["description"].lower() and "friday" in task["description"].lower()
    ]
    assert len(api) == 1
    assert api[0]["assigned_to"] == "Kamran"
    assert "بھیج دوں گا" in api[0]["source_text"]


def test_urdu_budget_deadline_and_slides_request():
    result = _verified(_multi("urdu-budget-slides"))
    budget = [
        task for task in result["tasks"]
        if task["assigned_to"] == "Saad" and "budget" in task["description"].lower()
        and "wednesday" in task["description"].lower()
    ]
    slides = [
        task for task in result["tasks"]
        if task["assigned_to"] == "Saad" and "slides" in task["description"].lower()
        and "sunday" in task["description"].lower()
    ]
    assert len(budget) == 1
    assert "بجٹ" not in budget[0]["description"]
    assert "بھیج دوں گا" in budget[0]["source_text"]
    assert len(slides) == 1
    assert slides[0]["evidence_time"] == "00:30:16"
    assert "سلائیڈز" in slides[0]["source_text"]
    assert all("ڈمو" not in task["source_text"] for task in result["tasks"])


def test_urdu_short_acceptance_keeps_the_requested_update():
    result = _verified(_multi("urdu-indirect-deadline"))
    login = [
        task for task in result["tasks"]
        if "login" in task["description"].lower() and "thursday" in task["description"].lower()
    ]
    update = [
        task for task in result["tasks"]
        if task["assigned_to"] == "Bilal"
        and "client" in task["description"].lower()
        and "update" in task["description"].lower()
    ]
    assert len(login) == 1
    assert login[0]["assigned_to"] == "Bilal"
    assert len(update) == 1
    assert "بھیج دوں گا" in update[0]["source_text"]
    assert update[0]["evidence_time"] == "01:00:36"
    assert len(result["tasks"]) == 2


def test_urdu_agreement_words_are_not_assignees():
    text = (
        "بالکل، عایشہ آپ dashboard میں task status کا filter add کریں۔ "
        "اس کی deadline پر medium priority رکھیں۔\n"
        "ٹھیک ہے، علی آپ authentication bug fix کریں۔ "
        "Duplicate records والا کام بدھ تک مکمل کر دیں۔\n"
        "ٹھیک ہے۔ میں updates پیر شام تک بھیج دوں گی۔"
    )
    result = verify_meeting_tasks(
        text,
        segments=prepare_meeting_turns(text, translate=False),
    )
    for task in result["tasks"]:
        assert "بالکل" not in task["assigned_to"]
        assert "ٹھیک" not in task["assigned_to"]
        assert "عایشہ" not in task["description"]
        assert "ٹھیک" not in task["description"]
    ayesha = [task for task in result["tasks"] if task["assigned_to"] == "Ayesha"]
    assert any(
        "filter" in task["description"].lower() and "dashboard" in task["description"].lower()
        for task in ayesha
    )
    ali = [task for task in result["tasks"] if task["assigned_to"] == "Ali"]
    assert any("bug" in task["description"].lower() for task in ali)
    assert any(
        "wednesday" in task["description"].lower()
        or "wednesday" in (task.get("due_date") or "").lower()
        for task in ali
    )
    mentioned = result["tasks"] + result["review"]
    monday = [
        item for item in mentioned
        if "monday" in (item.get("description") or "").lower()
        or "monday" in (item.get("due_date") or "").lower()
    ]
    assert monday
    assert all("ٹھیک" not in (item.get("assigned_to") or "") for item in monday)


def _pending(tasks, assignee, fragment):
    matches = [
        task for task in tasks
        if task["assigned_to"] == assignee and fragment in task["description"].lower()
    ]
    assert len(matches) == 1
    assert matches[0]["status"] == "pending"
    return matches[0]


def test_one_request_keeps_every_requirement_for_that_person():
    transcript = (
        "Lina Ahmed, can you please send me the summary of this meeting and Omar, "
        "can you please check if this meeting is archived and future users can access this."
    )
    result = verify_meeting_tasks(transcript)
    assert len(result["tasks"]) == 2
    summary = _pending(result["tasks"], "Lina Ahmed", "meeting summary")
    access = _pending(result["tasks"], "Omar", "archived")
    assert summary["description"] == "Send the meeting summary"
    assert "future users" in access["description"].lower()
    assert "access it" in access["description"].lower()
    assert access["description"].lower().startswith("check whether")
    assert summary["source_text"] == transcript
    assert access["source_text"] == transcript


def test_one_sentence_assigns_different_work_to_two_people():
    transcript = (
        "Host:\n"
        "Hina Qureshi, can you please send the meeting notes and Farah, "
        "can you please check whether the recording is saved and later readers can open it. "
        "Please both of you do this task."
    )
    result = verify_meeting_tasks(transcript, segments=prepare_meeting_turns(transcript))
    notes = _pending(result["tasks"], "Hina Qureshi", "notes")
    recording = _pending(result["tasks"], "Farah", "recording")
    assert "can you please" not in notes["description"].lower()
    assert "can you please" not in recording["description"].lower()
    assert "farah" not in notes["description"].lower()
    assert "hina" not in recording["description"].lower()
    assert "both of you" not in notes["description"].lower()
    assert "both of you" not in recording["description"].lower()
    assert len(result["tasks"]) == 2
    assert all(task["assigned_to"] != "Team" for task in result["tasks"])
    assert "send the meeting notes" in notes["source_text"].lower()
    assert "Host" in result["transcript"]


def test_one_sentence_can_give_one_person_two_tasks():
    transcript = "Nadia, please draft the agenda and send the calendar invite."
    result = verify_meeting_tasks(transcript)
    draft = _pending(result["tasks"], "Nadia", "agenda")
    invite = _pending(result["tasks"], "Nadia", "invite")
    assert "please" not in draft["description"].lower()
    assert draft["source_text"] == transcript
    assert invite["source_text"] == transcript
    assert len(result["tasks"]) == 2


def test_one_sentence_can_name_three_assignees():
    transcript = (
        "Adeel, please book the room, Sana, please send the agenda, "
        "and Noor, please share the notes."
    )
    result = verify_meeting_tasks(transcript)
    assert len(result["tasks"]) == 3
    _pending(result["tasks"], "Adeel", "room")
    _pending(result["tasks"], "Sana", "agenda")
    _pending(result["tasks"], "Noor", "notes")
    assert {task["assigned_to"] for task in result["tasks"]} == {"Adeel", "Sana", "Noor"}


def test_a_request_without_a_name_is_not_given_a_guessed_assignee():
    result = verify_meeting_tasks("Please archive the signed contract with the vendor.")
    assert result["tasks"] == []
    assert result["review"]
    assert result["review"][0]["assigned_to"] == "Unassigned"


def test_a_group_reminder_does_not_duplicate_named_tasks():
    transcript = (
        "Hina, please send the meeting notes and Farah, please check the recording. "
        "Please both of you do this task."
    )
    result = verify_meeting_tasks(transcript)
    assert len(result["tasks"]) == 2
    assert {task["assigned_to"] for task in result["tasks"]} == {"Hina", "Farah"}
    blob = " ".join(task["description"] for task in result["tasks"]).lower()
    assert "both of you" not in blob
    assert "do this task" not in blob


def test_roman_urdu_splits_two_named_requests():
    transcript = "Hina, notes bhej dena aur Farah, recording check kar lo."
    result = verify_meeting_tasks(transcript, segments=prepare_meeting_turns(transcript, translate=False))
    assert len(result["tasks"]) == 2
    _pending(result["tasks"], "Hina", "notes")
    _pending(result["tasks"], "Farah", "recording")
    assert all(task["source_text"] == transcript for task in result["tasks"])


def test_urdu_script_splits_two_named_requests():
    transcript = "احمد، نوٹس بھیج دیں اور سارہ، بجٹ چیک کر دیں۔"
    result = verify_meeting_tasks(
        transcript,
        segments=prepare_meeting_turns(transcript, translate=False),
    )
    assert len(result["tasks"]) == 2
    descriptions = " ".join(task["description"] for task in result["tasks"]).lower()
    assert "notes" in descriptions
    assert "budget" in descriptions
    assert len({task["assigned_to"] for task in result["tasks"]}) == 2
    assert all("اور" not in task["description"] for task in result["tasks"])


def test_split_tasks_are_stored_as_separate_rows(tmp_path):
    transcript = (
        "Adeel, please book the room, Sana, please send the agenda, "
        "and Noor, please share the notes."
    )
    result = verify_meeting_tasks(transcript)
    db_path = str(tmp_path / "clario.db")
    init_db(db_path)
    user_id = insert_user("ada", "ada@example.com", "hash", db_path=db_path)
    meeting_id = insert_meeting(user_id, "Split", db_path=db_path)
    insert_tasks(meeting_id, result["tasks"], db_path=db_path)
    stored = get_tasks_by_meeting(meeting_id, db_path)
    assert len(stored) == 3
    assert {row["assigned_to"] for row in stored} == {"Adeel", "Sana", "Noor"}
    assert all(row["status"] == "pending" for row in stored)

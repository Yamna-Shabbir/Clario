"""Tests for the report's preprocessing, utterance labels, and decision graph."""

import os
import tempfile

from models.database import (
    get_decisions_by_meeting,
    get_tasks_by_meeting,
    init_db,
    insert_decisions,
    insert_meeting,
    insert_user,
)
from services.knowledge_graph.builder import build_decision_graph
from services.nlp.utterances import (
    LABEL_DECISION,
    LABEL_DISCUSSION,
    LABEL_TASK,
    classify_utterances,
)
from services.preprocessing.cleaning import clean_transcript


SAMPLE = (
    "Um, we'll use React Native for this. "
    "Sara will design the logo. "
    "The deadline might shift. [00:01:02] [noise]"
)


def test_clean_transcript_drops_fillers_noise_and_timestamps():
    cleaned = clean_transcript(SAMPLE)
    assert "um" not in cleaned.lower().split()
    assert "[noise]" not in cleaned.lower()
    assert "00:01:02" not in cleaned
    assert "React Native" in cleaned
    assert "Sara will design the logo." in cleaned


def test_utterance_labels_match_report_examples():
    labels = {item["text"]: item["label"] for item in classify_utterances(clean_transcript(SAMPLE))}
    assert labels["we'll use React Native for this."] == LABEL_DECISION
    assert labels["Sara will design the logo."] == LABEL_TASK
    assert labels["The deadline might shift."] == LABEL_DISCUSSION


def test_decision_graph_links_person_task_deadline_and_meeting():
    graph = build_decision_graph(
        tasks=[{
            "description": "Sara will design the logo.",
            "assigned_to": "Sara",
            "due_date": "Friday",
        }],
        decisions=[{"text": "We'll use React Native for this.", "confidence": 0.8}],
        meeting_label="Meeting 1",
    )
    relations = {(edge["source"], edge["relation"], edge["target"]) for edge in graph["edges"]}
    types = {node["type"] for node in graph["nodes"]}
    assert types == {"Meeting", "Person", "Task", "Decision", "Deadline"}
    assert ("person-sara", "ASSIGNED_TO", "task-1") in relations
    assert ("task-1", "DEADLINE", "deadline-1") in relations
    assert ("task-1", "BELONGS_TO", "meeting") in relations
    assert ("decision-1", "BELONGS_TO", "meeting") in relations
    decision = next(node for node in graph["nodes"] if node["type"] == "Decision")
    assert decision["source"] == "We'll use React Native for this."
    assert decision["confidence"] == 0.8


def test_pasted_speaker_email_and_decision_are_stored(monkeypatch, tmp_path):
    """Name (email) lines stay matchable after the transcript is rendered."""
    monkeypatch.setattr("services.pipeline._get_nlp", lambda: None)
    monkeypatch.setattr("services.pipeline._get_summariser", lambda: (None, None))
    monkeypatch.setattr("services.pipeline.STATIC_GRAPH_DIR", str(tmp_path))
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        init_db(path)
        user_id = insert_user("ada", "ada@example.com", "hash", db_path=path)
        meeting_id = insert_meeting(user_id, title="Kickoff", db_path=path)
        from services.pipeline import run_clario_pipeline

        result = run_clario_pipeline(
            text_input=(
                "Sarah (sarah@example.com): I will send the launch checklist by Friday.\n"
                "Sarah (sarah@example.com): We decided to ship the dashboard on Monday.\n"
            ),
            meeting_id=meeting_id,
            user_id=user_id,
            db_path=path,
        )
        assert result["tasks"][0]["assigned_email"] == "sarah@example.com"
        stored = get_tasks_by_meeting(meeting_id, path)
        assert stored[0]["assigned_email"] == "sarah@example.com"
        assert len(stored[0]["due_date"]) == 10 and stored[0]["due_date"][4] == "-"
        decisions = get_decisions_by_meeting(meeting_id, path)
        assert [row["content"] for row in decisions] == [
            "We decided to ship the dashboard on Monday."
        ]
    finally:
        os.remove(path)


def test_roster_rows_skip_blanks_and_reject_a_half_filled_person():
    from services.preprocessing.participants import roster_from_pairs

    people, error = roster_from_pairs(
        ["John Smith", "", "Sara"],
        [" John@Example.com ", "", ""],
    )
    assert error == "Enter both a name and an email for each person."
    people, error = roster_from_pairs(["John Smith", ""], ["John@Example.com", ""])
    assert error == ""
    assert people == [{
        "name": "John Smith",
        "email": "john@example.com",
        "source": "upload",
    }]


def test_unmatched_roster_name_waits_for_the_uploader():
    from services.nlp.participant_matching import pending_name_collisions

    roster = [
        {"name": "John Smith", "email": "john@example.com"},
        {"name": "Sara Khan", "email": "sara@example.com"},
    ]
    held = pending_name_collisions(
        [{
            "id": 4,
            "assigned_to": "Priya",
            "assigned_email": None,
            "description": "Priya will write the recap.",
        }],
        roster,
    )
    assert held[0]["task_id"] == 4
    assert {person["email"] for person in held[0]["candidates"]} == {
        "john@example.com",
        "sara@example.com",
    }


def test_mentioned_emails_are_participants():
    from services.preprocessing.participants import mentioned_participants

    found = mentioned_participants(
        "Yamna Shabbir (yumnashabbir5@gmail.com) will send the summary. "
        "Project Lead (sp23-bcs-127@cuilahore.edu.pk) will prepare the slides."
    )
    emails = {person["email"] for person in found}
    assert emails == {"yumnashabbir5@gmail.com", "sp23-bcs-127@cuilahore.edu.pk"}


def test_task_detection_without_spacy():
    from unittest.mock import patch

    from services.nlp.task_detection import detect_tasks

    with patch("services.nlp.task_detection.load_nlp", return_value=None):
        tasks = detect_tasks("Sara will design the logo by Friday.")
    assert len(tasks) == 1
    assert tasks[0]["assigned_to"] == "Sara"
    assert "logo" in tasks[0]["description"]


def test_decisions_round_trip_in_sqlite():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        init_db(path)
        user_id = insert_user("ada", "ada@example.com", "hash", db_path=path)
        meeting_id = insert_meeting(user_id, title="Planning", db_path=path)
        insert_decisions(meeting_id, [{
            "text": "We'll use React Native for this.",
            "confidence": 0.8,
        }], db_path=path)
        rows = get_decisions_by_meeting(meeting_id, db_path=path)
        assert rows[0]["content"] == "We'll use React Native for this."
        assert rows[0]["confidence"] == 0.8
    finally:
        os.remove(path)

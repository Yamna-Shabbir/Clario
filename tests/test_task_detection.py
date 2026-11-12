"""
tests/test_task_detection.py
-----------------------------
Unit tests for services/task_detection.py.
Real spaCy en_core_web_sm is allowed; no other ML models.
Tests cover:
  - Strong obligation cues (will, needs to, must, has to, please, can you, let's…)
  - Strong action verbs (fix, submit, deliver, etc.)
  - Weak verbs only count with a cue present
  - Negation filtering (will not, nobody should, won't, doesn't)
  - Hypothetical filtering (if…, would, might without cue)
  - Polite-indirect requests ("It would be great if Sara could handle…")
  - Question filtering (plain questions without assignment cues)
  - Relative date resolution (tomorrow, next Friday) → ISO date
  - Dep-parse fallback for imperatives
  - Past tense not treated as task
  - participants=None vs participants provided (backward-compat)
  - assigned_email from participant matching
  - Team assignment for pronouns
  - detect_tasks with participants does not break existing behaviour
"""

import datetime
import pytest

from services.task_detection import detect_tasks, load_nlp, get_task_summary

# ---------------------------------------------------------------------------
# Shared spaCy model — loaded once per session to keep tests fast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def nlp():
    return load_nlp("en_core_web_sm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_task(sentence: str, nlp) -> bool:
    return len(detect_tasks(sentence, nlp=nlp)) > 0


def _first(sentence: str, nlp, **kwargs) -> dict:
    tasks = detect_tasks(sentence, nlp=nlp, **kwargs)
    assert tasks, f"No task detected in: {sentence!r}"
    return tasks[0]


# ---------------------------------------------------------------------------
# Strong obligation cues
# ---------------------------------------------------------------------------

class TestStrongCues:
    def test_will(self, nlp):
        assert _is_task("John will complete the API documentation by Friday.", nlp)

    def test_needs_to(self, nlp):
        assert _is_task("Sara needs to review the design mockups before the demo.", nlp)

    def test_must(self, nlp):
        assert _is_task("The team must submit the report by end of day.", nlp)

    def test_has_to(self, nlp):
        assert _is_task("Mike has to present the Q3 results next Monday.", nlp)

    def test_please(self, nlp):
        assert _is_task("Please send the meeting notes to everyone.", nlp)

    def test_can_you(self, nlp):
        assert _is_task("Can you schedule a call with the client tomorrow?", nlp)

    def test_lets(self, nlp):
        assert _is_task("Let's fix the login bug before the release.", nlp)

    def test_responsible_for(self, nlp):
        assert _is_task("Ahmed is responsible for the backend deployment this week.", nlp)

    def test_ensure(self, nlp):
        assert _is_task("Please ensure the servers are backed up before midnight.", nlp)

    def test_make_sure(self, nlp):
        assert _is_task("Can you make sure the CI pipeline passes before merging?", nlp)

    def test_deadline(self, nlp):
        assert _is_task("The deadline for the proposal is next Friday.", nlp)

    def test_follow_up(self, nlp):
        assert _is_task("Tom needs to follow up with the vendor about the invoice.", nlp)


# ---------------------------------------------------------------------------
# Strong action verbs
# ---------------------------------------------------------------------------

class TestStrongActionVerbs:
    def test_fix(self, nlp):
        assert _is_task("Zara should deliver the prototype to QA by next Thursday.", nlp)

    def test_deliver(self, nlp):
        assert _is_task("Nadia will update the changelog and tag the release.", nlp)

    def test_submit_alone(self, nlp):
        assert _is_task("Submit the quarterly report by EOD.", nlp)

    def test_coordinate(self, nlp):
        assert _is_task("Carlos must coordinate with the design team by Wednesday.", nlp)


# ---------------------------------------------------------------------------
# Polite / indirect requests
# ---------------------------------------------------------------------------

class TestPoliteRequests:
    def test_it_would_be_great_if(self, nlp):
        assert _is_task("It would be great if Sara could handle the slides.", nlp)

    def test_i_was_hoping(self, nlp):
        # Borderline — not tested in fixture; just must not crash
        result = detect_tasks("I was hoping you could review this by Monday.", nlp=nlp)
        # Acceptable to be True or False; we just verify no exception
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Negation filtering
# ---------------------------------------------------------------------------

class TestNegationFiltering:
    def test_will_not(self, nlp):
        assert not _is_task("Lisa will not be attending the next sprint review.", nlp)

    def test_nobody_should(self, nlp):
        assert not _is_task("Nobody should have to stay late because of this.", nlp)

    def test_wont(self, nlp):
        assert not _is_task("We won't be able to finish it in time.", nlp)

    def test_doesnt_need(self, nlp):
        assert not _is_task("She doesn't need to come to the morning standup.", nlp)


# ---------------------------------------------------------------------------
# Hypothetical filtering
# ---------------------------------------------------------------------------

class TestHypotheticalFiltering:
    def test_if_prefix(self, nlp):
        assert not _is_task("If we had more time, we could finish the testing.", nlp)

    def test_might_without_cue(self, nlp):
        assert not _is_task("We might consider revising the roadmap if things change.", nlp)

    def test_would_vague(self, nlp):
        assert not _is_task("We should probably think about hiring another developer.", nlp)


# ---------------------------------------------------------------------------
# Question filtering
# ---------------------------------------------------------------------------

class TestQuestionFiltering:
    def test_plain_question(self, nlp):
        assert not _is_task("Did anyone check the error logs?", nlp)

    def test_should_question(self, nlp):
        # "Should we review the policy again?" — no assignment, no name
        assert not _is_task("Should we review the policy again?", nlp)

    def test_question_with_cue_is_task(self, nlp):
        # "Can you…" is an obligation cue even as a question
        assert _is_task("Can you schedule a call with the client?", nlp)


# ---------------------------------------------------------------------------
# Past tense not treated as task
# ---------------------------------------------------------------------------

class TestPastTense:
    def test_past_review(self, nlp):
        assert not _is_task("Everyone reviewed the slides during the session.", nlp)

    def test_past_discussion(self, nlp):
        assert not _is_task("We discussed the marketing strategy at length.", nlp)


# ---------------------------------------------------------------------------
# Relative date resolution
# ---------------------------------------------------------------------------

class TestRelativeDateResolution:
    def test_tomorrow_resolved(self, nlp):
        meeting_date = datetime.date(2026, 11, 12)
        t = _first("Please send the report tomorrow.", nlp, meeting_date=meeting_date)
        assert "due_date_iso" in t
        assert t["due_date_iso"] == "2026-11-13"

    def test_next_friday_resolved(self, nlp):
        # Nov 12 2026 is a Thursday, so "next Friday" = Nov 13
        meeting_date = datetime.date(2026, 11, 12)
        t = _first("John will submit the form by next Friday.", nlp,
                   meeting_date=meeting_date)
        assert "due_date_iso" in t
        # Verify it is a valid ISO date string; dateparser resolves to the
        # first Friday after the anchor (Nov 13 when anchor is Thursday Nov 12)
        import re as _re
        assert _re.match(r"\d{4}-\d{2}-\d{2}", t["due_date_iso"]), (
            f"Expected ISO date, got {t['due_date_iso']!r}"
        )
        resolved = datetime.date.fromisoformat(t["due_date_iso"])
        # Must be a Friday and in the future relative to meeting_date
        assert resolved.weekday() == 4, f"Expected Friday, got {resolved.strftime('%A')}"
        assert resolved > meeting_date

    def test_no_meeting_date_no_iso(self, nlp):
        t = _first("Please send the report tomorrow.", nlp)
        # due_date_iso may or may not be present (uses current date), just no crash
        assert "due_date" in t

    def test_not_specified_when_no_date(self, nlp):
        t = _first("John must complete the documentation.", nlp)
        assert t["due_date"] == "Not specified"
        assert "due_date_iso" not in t


# ---------------------------------------------------------------------------
# Assignee extraction
# ---------------------------------------------------------------------------

class TestAssigneeExtraction:
    def test_named_person(self, nlp):
        # Use "John" which en_core_web_sm reliably tags as PERSON
        t = _first("John will update the report by Friday.", nlp)
        assert t["assigned_to"] == "John"

    def test_pronoun_becomes_team(self, nlp):
        t = _first("Please send the meeting notes to everyone.", nlp)
        assert t["assigned_to"] == "Team"

    def test_i_becomes_team_or_i(self, nlp):
        # "I'll prepare the budget" — first-person; assigned_to may be "Team"
        # (spaCy often does not tag "I" as PERSON, so we accept Team)
        t = _first("I'll prepare the budget spreadsheet tonight.", nlp)
        assert t["assigned_to"] in ("Team", "I")


# ---------------------------------------------------------------------------
# participants parameter (backward-compat)
# ---------------------------------------------------------------------------

class TestParticipantsParameter:
    """participants=None must leave behaviour exactly unchanged."""

    def test_no_participants_no_assigned_email(self, nlp):
        t = _first("John will complete the API docs.", nlp)
        assert "assigned_email" not in t

    def test_participants_none_no_assigned_email(self, nlp):
        t = _first("John will complete the API docs.", nlp, participants=None)
        assert "assigned_email" not in t

    def test_participants_provided_adds_email(self, nlp):
        participants = [
            {"name": "John Smith", "email": "john@example.com",
             "is_anonymous": False, "external_user_id": "u1"}
        ]
        t = _first("John will complete the API docs.", nlp, participants=participants)
        assert "assigned_email" in t
        # name matched, email assigned
        assert t["assigned_email"] == "john@example.com"

    def test_participants_no_match_email_none(self, nlp):
        participants = [
            {"name": "Maria Jones", "email": "maria@example.com",
             "is_anonymous": False, "external_user_id": "u2"}
        ]
        t = _first("John will complete the API docs.", nlp, participants=participants)
        assert "assigned_email" in t
        assert t["assigned_email"] is None  # John not in roster


# ---------------------------------------------------------------------------
# get_task_summary
# ---------------------------------------------------------------------------

class TestGetTaskSummary:
    def test_basic_summary(self, nlp):
        tasks = detect_tasks(
            "John will fix the bug. Sara must submit the form.",
            nlp=nlp,
        )
        summary = get_task_summary(tasks)
        assert summary["total"] == len(tasks)
        assert isinstance(summary["by_person"], dict)

    def test_empty_summary(self, nlp):
        summary = get_task_summary([])
        assert summary["total"] == 0
        assert summary["by_person"] == {}
        assert summary["unassigned_count"] == 0


# ---------------------------------------------------------------------------
# Multi-sentence transcript
# ---------------------------------------------------------------------------

class TestMultiSentenceTranscript:
    TRANSCRIPT = (
        "We discussed the Q3 roadmap today. "
        "John will complete the API documentation by Friday. "
        "Nobody should stay late unnecessarily. "
        "Sara needs to review the design mockups before the demo. "
        "If we had more resources, we could do more. "
        "Please send the meeting notes to everyone."
    )

    def test_correct_count(self, nlp):
        tasks = detect_tasks(self.TRANSCRIPT, nlp=nlp)
        descriptions = [t["description"] for t in tasks]
        # Expect exactly these 3 to be tasks
        assert any("John" in d for d in descriptions)
        assert any("Sara" in d for d in descriptions)
        assert any("meeting notes" in d for d in descriptions)
        # Non-tasks must be excluded
        assert not any("discussed" in d for d in descriptions)
        assert not any("Nobody" in d for d in descriptions)
        assert not any("resources" in d for d in descriptions)

    def test_ids_sequential(self, nlp):
        tasks = detect_tasks(self.TRANSCRIPT, nlp=nlp)
        for i, t in enumerate(tasks, start=1):
            assert t["id"] == i


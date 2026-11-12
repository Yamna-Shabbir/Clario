"""
tests/test_integrations_import_matching.py
-------------------------------------------
Tests for Part D of Google Meet integration:
- Participant name matching hierarchy (services/participant_matching.py)
    * Exact matching (case / whitespace / punctuation insensitive)
    * Unique first-name matching
    * Fuzzy matching via SequenceMatcher (ratio >= 0.8 with unique best candidate)
    * Ambiguity handling (returns None)
    * Pronouns and 'Team' never match
- Task detection with and without participants parameter
- rematch_meeting_tasks (enforces ownership, updates only matching tasks)
- POST /integrations/google/import:
    * Login enforcement
    * Meeting ownership check (abort 404 for other user's meeting)
    * Idempotent import (replace participants, no duplicate rows)
    * Session prefill when no meeting_id supplied -> upload links participants
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from werkzeug.security import generate_password_hash

from models.database import (
    get_connection,
    get_meeting_detail,
    get_meeting_participants,
    get_tasks_by_meeting,
    init_db,
    insert_meeting,
    insert_tasks,
    insert_user,
    replace_meeting_participants,
)
from services.nlp.participant_matching import match, rematch_meeting_tasks
from services.nlp.task_detection import detect_tasks


TEST_FERNET_KEY = Fernet.generate_key()


def _set_encryption_key(monkeypatch):
    monkeypatch.setenv("CLARIO_TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())
    import config
    monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())


def _login(client, username="importuser", email="importuser@example.com"):
    client.post(
        "/register",
        data={
            "username": username,
            "email": email,
            "password": "Password123!",
            "confirm": "Password123!",
        },
        follow_redirects=True,
    )
    import routes.integrations as _r_integ
    from models.database import get_user_by_email
    user = get_user_by_email(email, db_path=_r_integ.DB_PATH)
    return user["id"]


# =============================================================================
# 1. Matcher Unit Tests
# =============================================================================

class TestParticipantMatcher:

    def test_exact_match_normalized(self):
        participants = [
            {"name": "Alice M. Smith", "email": "alice@example.com"},
            {"name": "Bob Jones", "email": "bob@example.com"},
        ]
        # Exact match with case, punctuation, whitespace differences
        res1 = match("alice m smith", participants)
        assert res1 is not None
        assert res1["name"] == "Alice M. Smith"

        res2 = match("  ALICE M. SMITH! ", participants)
        assert res2 is not None
        assert res2["name"] == "Alice M. Smith"

    def test_unique_first_name_match(self):
        participants = [
            {"name": "Sarah Connor", "email": "sarah@resistance.org"},
            {"name": "John Connor", "email": "john@resistance.org"},
        ]
        # "Sarah" is single token, unique first name in roster
        res = match("Sarah", participants)
        assert res is not None
        assert res["name"] == "Sarah Connor"
        assert res["email"] == "sarah@resistance.org"

    def test_ambiguous_first_name_returns_none(self):
        participants = [
            {"name": "Sarah Connor", "email": "sarah1@example.com"},
            {"name": "Sarah Jenkins", "email": "sarah2@example.com"},
        ]
        # Ambiguous: two Sarahs
        res = match("Sarah", participants)
        assert res is None

    def test_fuzzy_matching(self):
        participants = [
            {"name": "Alexander Hamilton", "email": "alex@example.com"},
            {"name": "Thomas Jefferson", "email": "thomas@example.com"},
        ]
        # "Alexandr" ratio with "alexander" >= 0.8
        res = match("Alexandr", participants)
        assert res is not None
        assert res["name"] == "Alexander Hamilton"

    def test_ambiguous_fuzzy_returns_none(self):
        participants = [
            {"name": "Alec Baldwin", "email": "alec@example.com"},
            {"name": "Alek Wek", "email": "alek@example.com"},
        ]
        # "Alex" fuzzy ratio with "alec" and "alek" is tied
        res = match("Alex", participants)
        assert res is None

    def test_no_match_returns_none(self):
        participants = [
            {"name": "Bruce Wayne", "email": "bruce@wayne.com"},
        ]
        assert match("Clark Kent", participants) is None

    @pytest.mark.parametrize("pronoun", [
        "he", "she", "it", "they", "them", "we", "i", "you", "him", "her",
        "someone", "anyone", "everyone", "nobody", "himself", "herself",
    ])
    def test_pronouns_never_match(self, pronoun):
        participants = [
            {"name": "Theodore", "email": "theo@example.com"},
            {"name": "Hermione", "email": "hermione@example.com"},
        ]
        assert match(pronoun, participants) is None
        assert match(pronoun.upper(), participants) is None

    @pytest.mark.parametrize("group_term", ["Team", "team", "THE TEAM", "All", "Unknown"])
    def test_team_and_generic_terms_never_match(self, group_term):
        participants = [
            {"name": "Tim Burton", "email": "tim@example.com"},
            {"name": "Team Lead", "email": "lead@example.com"},
        ]
        assert match(group_term, participants) is None

    def test_empty_inputs_return_none(self):
        assert match("", [{"name": "Alice"}]) is None
        assert match("   ", [{"name": "Alice"}]) is None
        assert match("Alice", []) is None
        assert match("Alice", None) is None


# =============================================================================
# 2. detect_tasks with participants Tests
# =============================================================================

class TestDetectTasksWithParticipants:

    def test_detect_tasks_without_participants_unchanged(self, real_nlp):
        text = "Bob will prepare the presentation slides by Friday."
        tasks = detect_tasks(text, nlp_model=real_nlp, participants=None)
        assert len(tasks) == 1
        assert tasks[0]["assigned_to"] == "Bob"
        assert "assigned_email" not in tasks[0]

    def test_detect_tasks_with_participants_matches_assignee_and_email(self, real_nlp):
        text = "Bob will prepare the presentation slides by Friday."
        roster = [
            {"name": "Bob Dylan", "email": "bob@dylan.com"},
            {"name": "Alice Cooper", "email": "alice@cooper.com"},
        ]
        tasks = detect_tasks(text, nlp_model=real_nlp, participants=roster)
        assert len(tasks) == 1
        assert tasks[0]["assigned_to"] == "Bob Dylan"
        assert tasks[0]["assigned_email"] == "bob@dylan.com"

    def test_detect_tasks_with_participants_unmatched_stays_original(self, real_nlp):
        text = "Charlie will send out the invitations today."
        roster = [
            {"name": "Alice Cooper", "email": "alice@cooper.com"},
        ]
        tasks = detect_tasks(text, nlp_model=real_nlp, participants=roster)
        assert len(tasks) == 1
        assert tasks[0]["assigned_to"] == "Charlie"
        assert tasks[0]["assigned_email"] is None


# =============================================================================
# 3. rematch_meeting_tasks Unit Tests
# =============================================================================

class TestRematchMeetingTasks:

    def test_rematch_updates_only_matching_tasks(self, app):
        import routes.integrations as _r_integ
        db_path = _r_integ.DB_PATH

        uid = insert_user("rematch1", "rematch1@example.com", generate_password_hash("p"), db_path=db_path)
        mid = insert_meeting(uid, title="Sprint Review", db_path=db_path)

        # Pre-seed tasks: task 1 has assigned_to="Sarah", task 2 has assigned_to="Unknown"
        insert_tasks(
            mid,
            [
                {"description": "Prepare slides", "assigned_to": "Sarah", "due_date": "Friday", "priority": "High"},
                {"description": "Follow up with client", "assigned_to": "Unknown", "due_date": "Monday", "priority": "Low"},
            ],
            db_path=db_path,
        )

        # Seed meeting participants
        replace_meeting_participants(
            mid,
            [
                {"name": "Sarah Connor", "email": "sarah@cyberdyne.com", "source": "google_meet"},
            ],
            db_path=db_path,
        )

        updated_count = rematch_meeting_tasks(mid, uid, db_path=db_path)
        assert updated_count == 1

        tasks = get_tasks_by_meeting(mid, db_path=db_path)
        # Sarah should be updated to full canonical name + email
        t_sarah = next(t for t in tasks if t["description"] == "Prepare slides")
        assert t_sarah["assigned_to"] == "Sarah Connor"
        assert t_sarah["assigned_email"] == "sarah@cyberdyne.com"

        # Unknown remains untouched
        t_unk = next(t for t in tasks if t["description"] == "Follow up with client")
        assert t_unk["assigned_to"] == "Unknown"
        assert t_unk["assigned_email"] is None

    def test_rematch_unowned_meeting_returns_zero(self, app):
        import routes.integrations as _r_integ
        db_path = _r_integ.DB_PATH

        user_a = insert_user("usera", "usera@example.com", generate_password_hash("p"), db_path=db_path)
        user_b = insert_user("userb", "userb@example.com", generate_password_hash("p"), db_path=db_path)
        mid = insert_meeting(user_a, title="User A Private Meeting", db_path=db_path)

        # User B attempts to rematch user A's meeting
        res = rematch_meeting_tasks(mid, user_b, db_path=db_path)
        assert res == 0


# =============================================================================
# 4. POST /integrations/google/import Route Tests
# =============================================================================

class TestGoogleImportRoute:

    def test_import_requires_login(self, client):
        resp = client.post("/integrations/google/import", data={"external_meeting_id": "conferenceRecords/rec1"})
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]

    def test_missing_external_meeting_id_redirects(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client)

        resp = client.post("/integrations/google/import", data={}, follow_redirects=False)
        assert resp.status_code in (301, 302)
        assert "/google/meetings" in resp.headers["Location"]

    def test_import_into_unowned_meeting_returns_404(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        uid_current = _login(client, username="alice_importer", email="alice_imp@example.com")

        import routes.integrations as _r_integ
        db_path = _r_integ.DB_PATH

        # Create meeting owned by another user (bob)
        uid_bob = insert_user("bob_owner", "bob_owner@example.com", generate_password_hash("p"), db_path=db_path)
        bob_meeting_id = insert_meeting(uid_bob, title="Bob Meeting", db_path=db_path)

        # Mock get_participants to succeed
        fake_participants = [{"name": "Attendee One", "email": None, "source": "google_meet"}]
        with patch("routes.integrations.get_participants", return_value=fake_participants):
            resp = client.post(
                "/integrations/google/import",
                data={
                    "external_meeting_id": "conferenceRecords/secret",
                    "meeting_id": str(bob_meeting_id),
                },
            )

        assert resp.status_code == 404

    def test_import_into_existing_meeting_stores_participants_and_rematches(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        uid = _login(client, username="clario_owner", email="clario_owner@example.com")

        import routes.integrations as _r_integ
        db_path = _r_integ.DB_PATH

        mid = insert_meeting(uid, title="Q4 Roadmap", db_path=db_path)
        insert_tasks(
            mid,
            [{"description": "Design mockups", "assigned_to": "David", "due_date": "Next week", "priority": "Medium"}],
            db_path=db_path,
        )

        fake_participants = [
            {
                "name": "David Gilmour",
                "email": "david@pinkfloyd.com",
                "external_user_id": "users/david123",
                "is_anonymous": False,
                "joined_at": "2026-11-12T10:00:00Z",
                "left_at": "2026-11-12T11:00:00Z",
                "participant_type": "signed_in",
            }
        ]

        with patch("routes.integrations.get_participants", return_value=fake_participants):
            resp = client.post(
                "/integrations/google/import",
                data={
                    "external_meeting_id": "conferenceRecords/conf_q4",
                    "meeting_id": str(mid),
                },
                follow_redirects=False,
            )

        assert resp.status_code in (301, 302)
        assert f"/result/{mid}" in resp.headers["Location"]

        # Verify DB: participants stored, meeting platform updated
        parts = get_meeting_participants(mid, db_path=db_path)
        assert len(parts) == 1
        assert parts[0]["name"] == "David Gilmour"
        assert parts[0]["email"] == "david@pinkfloyd.com"

        detail = get_meeting_detail(mid, uid, db_path=db_path)
        assert detail["platform"] == "google_meet"
        assert detail["external_meeting_id"] == "conferenceRecords/conf_q4"

        # Verify tasks were rematched: "David" -> "David Gilmour"
        tasks = get_tasks_by_meeting(mid, db_path=db_path)
        assert tasks[0]["assigned_to"] == "David Gilmour"
        assert tasks[0]["assigned_email"] == "david@pinkfloyd.com"

    def test_import_idempotent_no_duplicate_rows(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        uid = _login(client, username="idempotent_user", email="idempotent@example.com")

        import routes.integrations as _r_integ
        db_path = _r_integ.DB_PATH

        mid = insert_meeting(uid, title="Standup", db_path=db_path)
        fake_participants = [
            {"name": "User 1", "email": "u1@example.com"},
            {"name": "User 2", "email": "u2@example.com"},
        ]

        with patch("routes.integrations.get_participants", return_value=fake_participants):
            # First import
            client.post(
                "/integrations/google/import",
                data={"external_meeting_id": "conferenceRecords/rec_dup", "meeting_id": str(mid)},
            )
            # Second import of same conference
            client.post(
                "/integrations/google/import",
                data={"external_meeting_id": "conferenceRecords/rec_dup", "meeting_id": str(mid)},
            )

        parts = get_meeting_participants(mid, db_path=db_path)
        assert len(parts) == 2  # Not duplicated to 4

    def test_import_without_meeting_id_prefills_session_for_upload(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, username="new_uploader", email="new_upload@example.com")

        fake_participants = [
            {"name": "Roger Waters", "email": "roger@pinkfloyd.com"},
        ]

        with patch("routes.integrations.get_participants", return_value=fake_participants):
            resp = client.post(
                "/integrations/google/import",
                data={"external_meeting_id": "conferenceRecords/rec_new"},
                follow_redirects=False,
            )

        # Redirects to /upload
        assert resp.status_code in (301, 302)
        assert "/upload" in resp.headers["Location"]

        # Verify session was populated
        with client.session_transaction() as sess:
            assert "prefill_participants" in sess
            assert sess["prefill_participants"] == fake_participants
            assert sess["prefill_external_meeting_id"] == "conferenceRecords/rec_new"
            assert sess["prefill_platform"] == "google_meet"

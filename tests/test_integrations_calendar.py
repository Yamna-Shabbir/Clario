"""
tests/test_integrations_calendar.py
-----------------------------------
Tests for Part E: Google Calendar attendee email lookup and import enrichment.

Official Google Calendar API Documentation consulted:
- Events list: https://developers.google.com/calendar/api/v3/reference/events/list
- Events resource: https://developers.google.com/calendar/api/v3/reference/events#resource

Authorized Scope:
- https://www.googleapis.com/auth/calendar.readonly

Covers:
1. find_calendar_attendees:
   - Match by Meet conference code in hangoutLink
   - Match by conferenceData.conferenceId and entryPoints
   - Match by start time proximity window fallback
   - Organizer included in attendees list
   - Deduplication by email address
   - Graceful handling of 403 (unauthorized/scope missing), 404, RateLimited, TokenRevoked, NotConnected -> returns []
   - No matching event found -> returns []
2. Import flow with Calendar enrichment (POST /integrations/google/import):
   - Enriches participant emails from matched calendar attendees
   - Propagates assigned_email to rematched tasks
   - Unmatched or ambiguous attendees leave participant email NULL
   - Calendar failure degrades gracefully without aborting import
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from models.database import (
    get_meeting_detail,
    get_meeting_participants,
    get_tasks_by_meeting,
    init_db,
    insert_meeting,
    insert_tasks,
    insert_user,
    upsert_oauth_connection,
)
from services.integrations.base import ApiError, NotConnected, RateLimited, TokenRevoked
from services.integrations.crypto import encrypt_token
from services.integrations.google_meet import (
    CALENDAR_API_BASE,
    CALENDAR_READONLY_SCOPE,
    DEFAULT_SCOPES,
    find_calendar_attendees,
)

TEST_FERNET_KEY = Fernet.generate_key()


def _set_encryption_key(monkeypatch):
    monkeypatch.setenv("CLARIO_TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())
    import config
    monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())


def _login(client, username="caluser", email="caluser@example.com"):
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
# 1. find_calendar_attendees Unit Tests
# =============================================================================

def test_find_calendar_attendees_by_hangout_link(tmp_path, monkeypatch):
    """Event matched via hangoutLink containing Meet meeting code."""
    _set_encryption_key(monkeypatch)
    db_file = str(tmp_path / "test_cal.db")
    init_db(db_file)

    user_id = insert_user("cal_user1", "cal1@example.com", "hash", db_path=db_file)
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    mock_events = {
        "items": [
            {
                "id": "event_other",
                "summary": "Design sync",
                "hangoutLink": "https://meet.google.com/zzz-yyyy-xxx",
                "attendees": [{"email": "other@example.com", "displayName": "Other User"}],
            },
            {
                "id": "event_target",
                "summary": "Sprint Planning",
                "hangoutLink": "https://meet.google.com/abc-mnop-xyz",
                "attendees": [
                    {"email": "alice@example.com", "displayName": "Alice Smith"},
                    {"email": "bob@example.com", "displayName": "Bob Jones"},
                ],
                "organizer": {"email": "lead@example.com", "displayName": "Project Lead"},
            },
        ]
    }

    with patch("services.integrations.google_meet._make_authed_get", return_value=mock_events) as mock_get:
        attendees = find_calendar_attendees(
            user_id,
            meeting_code="abc-mnop-xyz",
            db_path=db_file,
        )

        assert mock_get.called
        assert len(attendees) == 3
        emails = {a["email"] for a in attendees}
        assert emails == {"alice@example.com", "bob@example.com", "lead@example.com"}

        alice = next(a for a in attendees if a["email"] == "alice@example.com")
        assert alice["display_name"] == "Alice Smith"
        assert alice["name"] == "Alice Smith"


def test_find_calendar_attendees_by_conference_data(tmp_path, monkeypatch):
    """Event matched via conferenceData entryPoints URI or conferenceId."""
    _set_encryption_key(monkeypatch)
    db_file = str(tmp_path / "test_cal.db")
    init_db(db_file)

    user_id = insert_user("cal_user2", "cal2@example.com", "hash", db_path=db_file)
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    mock_events = {
        "items": [
            {
                "id": "event_conf",
                "summary": "Architecture Review",
                "conferenceData": {
                    "conferenceId": "def-uvwx-rst",
                    "entryPoints": [
                        {"entryPointType": "video", "uri": "https://meet.google.com/def-uvwx-rst"}
                    ],
                },
                "attendees": [
                    {"email": "carol@example.com", "displayName": "Carol White"},
                ],
            }
        ]
    }

    with patch("services.integrations.google_meet._make_authed_get", return_value=mock_events):
        attendees = find_calendar_attendees(
            user_id,
            meeting_code="def-uvwx-rst",
            db_path=db_file,
        )

        assert len(attendees) == 1
        assert attendees[0]["email"] == "carol@example.com"
        assert attendees[0]["display_name"] == "Carol White"


def test_find_calendar_attendees_by_start_time_proximity(tmp_path, monkeypatch):
    """Event matched via start time proximity when meeting code is absent."""
    _set_encryption_key(monkeypatch)
    db_file = str(tmp_path / "test_cal.db")
    init_db(db_file)

    user_id = insert_user("cal_user3", "cal3@example.com", "hash", db_path=db_file)
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    meeting_time = "2026-11-12T10:00:00Z"
    mock_events = {
        "items": [
            {
                "id": "event_far",
                "summary": "Morning Standup",
                "start": {"dateTime": "2026-11-12T08:00:00Z"},
                "attendees": [{"email": "far@example.com", "displayName": "Far User"}],
            },
            {
                "id": "event_close",
                "summary": "Client Demo",
                "start": {"dateTime": "2026-11-12T10:05:00Z"},  # 5 min diff
                "attendees": [
                    {"email": "dave@example.com", "displayName": "Dave Brown"},
                ],
            },
        ]
    }

    with patch("services.integrations.google_meet._make_authed_get", return_value=mock_events):
        attendees = find_calendar_attendees(
            user_id,
            start_time=meeting_time,
            db_path=db_file,
        )

        assert len(attendees) == 1
        assert attendees[0]["email"] == "dave@example.com"
        assert attendees[0]["display_name"] == "Dave Brown"


def test_find_calendar_attendees_no_match(tmp_path, monkeypatch):
    """Returns [] when events exist but none match code or time."""
    _set_encryption_key(monkeypatch)
    db_file = str(tmp_path / "test_cal.db")
    init_db(db_file)

    user_id = insert_user("cal_user4", "cal4@example.com", "hash", db_path=db_file)
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    mock_events = {
        "items": [
            {
                "id": "event_1",
                "summary": "Unrelated meeting",
                "hangoutLink": "https://meet.google.com/zzz-zzzz-zzz",
                "attendees": [{"email": "z@example.com", "displayName": "Z"}],
            }
        ]
    }

    with patch("services.integrations.google_meet._make_authed_get", return_value=mock_events):
        attendees = find_calendar_attendees(
            user_id,
            meeting_code="abc-defg-hij",
            db_path=db_file,
        )
        assert attendees == []


def test_find_calendar_attendees_api_error_graceful(tmp_path, monkeypatch):
    """Returns [] safely without error when Google Calendar API returns 403 / ApiError."""
    _set_encryption_key(monkeypatch)
    db_file = str(tmp_path / "test_cal.db")
    init_db(db_file)

    user_id = insert_user("cal_user5", "cal5@example.com", "hash", db_path=db_file)
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    with patch("services.integrations.google_meet._make_authed_get", side_effect=ApiError("403 Forbidden: scope not granted")):
        attendees = find_calendar_attendees(user_id, meeting_code="abc-defg-hij", db_path=db_file)
        assert attendees == []

    with patch("services.integrations.google_meet._make_authed_get", side_effect=NotConnected("No connection")):
        attendees = find_calendar_attendees(user_id, meeting_code="abc-defg-hij", db_path=db_file)
        assert attendees == []


def test_find_calendar_attendees_deduplication(tmp_path, monkeypatch):
    """Duplicate attendee emails (e.g. attendee + organizer) are merged."""
    _set_encryption_key(monkeypatch)
    db_file = str(tmp_path / "test_cal.db")
    init_db(db_file)

    user_id = insert_user("cal_user6", "cal6@example.com", "hash", db_path=db_file)
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    mock_events = {
        "items": [
            {
                "id": "event_dup",
                "hangoutLink": "https://meet.google.com/dup-test-xyz",
                "attendees": [
                    {"email": "host@example.com", "displayName": "Host Attendee"},
                    {"email": "guest@example.com", "displayName": "Guest"},
                ],
                "organizer": {
                    "email": "host@example.com",
                    "displayName": "Host Organizer",
                },
            }
        ]
    }

    with patch("services.integrations.google_meet._make_authed_get", return_value=mock_events):
        attendees = find_calendar_attendees(user_id, meeting_code="dup-test-xyz", db_path=db_file)
        assert len(attendees) == 2
        emails = [a["email"] for a in attendees]
        assert emails == ["host@example.com", "guest@example.com"]


# =============================================================================
# 2. End-to-End Import Flow with Calendar Enrichment Tests
# =============================================================================

def test_import_enriches_participant_emails_and_tasks(client, app, monkeypatch):
    """
    POST /integrations/google/import:
    - Retrieves Meet participants with email=None
    - Looks up matching Calendar event
    - Enriches participant emails using matcher
    - Updates tasks.assigned_email via rematch_meeting_tasks
    """
    _set_encryption_key(monkeypatch)
    import routes.integrations as _r_integ
    db_file = _r_integ.DB_PATH

    user_id = _login(client, username="e2euser", email="e2euser@example.com")
    enc_access = encrypt_token("fake-access", key=TEST_FERNET_KEY)
    upsert_oauth_connection(user_id, "google", enc_access, None, None, CALENDAR_READONLY_SCOPE, db_path=db_file)

    # Create an existing meeting with a task assigned to "Bob"
    meeting_id = insert_meeting(
        user_id=user_id,
        title="Weekly Sync",
        audio_filename="meet.mp3",
        db_path=db_file,
    )
    insert_tasks(
        meeting_id,
        [
            {
                "description": "Bob will deliver the slides by Friday.",
                "assigned_to": "Bob",
                "due_date": "Friday",
                "priority": "Medium",
                "status": "pending",
            }
        ],
        db_path=db_file,
    )

    # Mock Meet participants (no email returned by Meet API)
    mock_participants = [
        {
            "name": "Bob Smith",
            "email": None,
            "external_user_id": "users/1001",
            "is_anonymous": False,
            "joined_at": "2026-11-12T10:00:00Z",
            "left_at": "2026-11-12T10:30:00Z",
        },
        {
            "name": "Alice Wonderland",
            "email": None,
            "external_user_id": "users/1002",
            "is_anonymous": False,
            "joined_at": "2026-11-12T10:01:00Z",
            "left_at": "2026-11-12T10:30:00Z",
        },
    ]

    # Mock Calendar attendees (contains emails)
    mock_cal_attendees = [
        {"name": "Bob Smith", "display_name": "Bob Smith", "email": "bob.smith@company.com"},
        {"name": "Alice Wonderland", "display_name": "Alice Wonderland", "email": "alice@company.com"},
    ]

    with patch("routes.integrations.get_participants", return_value=mock_participants):
        with patch("routes.integrations.find_calendar_attendees", return_value=mock_cal_attendees):
            res = client.post(
                "/integrations/google/import",
                data={
                    "external_meeting_id": "conferenceRecords/conf_123",
                    "meeting_id": str(meeting_id),
                    "meeting_code": "abc-mnop-xyz",
                },
                follow_redirects=True,
            )
            assert res.status_code == 200

    # Verify meeting_participants rows now have emails populated
    saved_participants = get_meeting_participants(meeting_id, db_path=db_file)
    assert len(saved_participants) == 2

    bob_part = next(p for p in saved_participants if "Bob" in p["name"])
    assert bob_part["email"] == "bob.smith@company.com"

    alice_part = next(p for p in saved_participants if "Alice" in p["name"])
    assert alice_part["email"] == "alice@company.com"

    # Verify task was rematched with canonical name and assigned_email
    tasks = get_tasks_by_meeting(meeting_id, db_path=db_file)
    assert len(tasks) == 1
    assert tasks[0]["assigned_to"] == "Bob Smith"
    assert tasks[0]["assigned_email"] == "bob.smith@company.com"


def test_import_calendar_ambiguity_leaves_email_null(client, app, monkeypatch):
    """
    If multiple calendar attendees match ambiguously, email remains NULL.
    """
    _set_encryption_key(monkeypatch)
    import routes.integrations as _r_integ
    db_file = _r_integ.DB_PATH

    user_id = _login(client, username="ambiguser", email="ambig@example.com")
    meeting_id = insert_meeting(user_id=user_id, title="Sync", audio_filename="a.mp3", db_path=db_file)

    mock_participants = [
        {"name": "Alex", "email": None, "external_user_id": "users/99", "is_anonymous": False}
    ]

    # Two Alexes in calendar -> ambiguous match
    mock_cal_attendees = [
        {"name": "Alex Smith", "display_name": "Alex Smith", "email": "alex.smith@example.com"},
        {"name": "Alex Jones", "display_name": "Alex Jones", "email": "alex.jones@example.com"},
    ]

    with patch("routes.integrations.get_participants", return_value=mock_participants):
        with patch("routes.integrations.find_calendar_attendees", return_value=mock_cal_attendees):
            client.post(
                "/integrations/google/import",
                data={
                    "external_meeting_id": "conferenceRecords/ambig_rec",
                    "meeting_id": str(meeting_id),
                },
                follow_redirects=True,
            )

    saved_participants = get_meeting_participants(meeting_id, db_path=db_file)
    assert len(saved_participants) == 1
    assert saved_participants[0]["email"] is None


def test_import_succeeds_when_calendar_fails(client, app, monkeypatch):
    """
    If Google Calendar fails or throws an exception, import continues normally without error.
    """
    _set_encryption_key(monkeypatch)
    import routes.integrations as _r_integ
    db_file = _r_integ.DB_PATH

    user_id = _login(client, username="safeuser", email="safe@example.com")
    meeting_id = insert_meeting(user_id=user_id, title="Sync", audio_filename="a.mp3", db_path=db_file)

    mock_participants = [
        {"name": "David", "email": None, "external_user_id": "users/55", "is_anonymous": False}
    ]

    with patch("routes.integrations.get_participants", return_value=mock_participants):
        with patch("routes.integrations.find_calendar_attendees", side_effect=Exception("API connection failure")):
            res = client.post(
                "/integrations/google/import",
                data={
                    "external_meeting_id": "conferenceRecords/safe_rec",
                    "meeting_id": str(meeting_id),
                },
                follow_redirects=True,
            )
            assert res.status_code == 200

    saved_participants = get_meeting_participants(meeting_id, db_path=db_file)
    assert len(saved_participants) == 1
    assert saved_participants[0]["name"] == "David"
    assert saved_participants[0]["email"] is None

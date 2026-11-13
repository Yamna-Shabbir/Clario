"""
tests/test_integrations_meet_api.py
------------------------------------
Tests for reading Google Meet conference records and participants.

Coverage:
 - list_recent_meetings:
     - Normalized fields, newest first
     - Pagination across conference records
     - Space lookup to fetch meetingCode
     - Graceful handling when space lookup fails
     - Empty list handling
     - 401 response triggers auto-refresh and retry
     - 429 response triggers bounded retry and raises RateLimited
     - 403 and 404 raise ApiError
     - NotConnected error when no OAuth row exists
 - get_participants:
     - Normalized fields: name, email (None), external_user_id, is_anonymous, joined_at, left_at
     - Anonymous and phone participant handling
     - De-duplication and session merging of signed-in participants (earliest join, latest leave)
     - Merging nested participantSessions
     - Pagination
     - NotConnected error
 - Routes:
     - GET /integrations/google/meetings (auth required, 200 list, not connected, token revoked, api error)
     - GET /integrations/google/participants (auth required, 200 list, validation of record parameter)

ALL external HTTP calls are mocked — no real network traffic.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from services.integrations.base import (
    ApiError,
    NotConnected,
    RateLimited,
    TokenRevoked,
)


TEST_FERNET_KEY = Fernet.generate_key()
FAKE_SCOPE = "https://www.googleapis.com/auth/meetings.space.readonly"


def _set_encryption_key(monkeypatch):
    """Inject a consistent Fernet key for token encryption/decryption in tests."""
    monkeypatch.setenv("CLARIO_TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())
    import config
    monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", TEST_FERNET_KEY.decode())


def _set_google_creds(monkeypatch):
    """Inject dummy Google credentials."""
    import config
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(config, "OAUTH_REDIRECT_BASE", "http://127.0.0.1:5000")
    monkeypatch.setenv("CLARIO_GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("CLARIO_GOOGLE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("CLARIO_OAUTH_REDIRECT_BASE", "http://127.0.0.1:5000")


def _login(client, app):
    """Register and log in a user for route testing."""
    creds = {
        "username": "meetuser",
        "email": "meetuser@example.com",
        "password": "Password123!",
    }
    client.post(
        "/register",
        data={
            "username": creds["username"],
            "email": creds["email"],
            "password": creds["password"],
            "confirm": creds["password"],
        },
        follow_redirects=True,
    )
    import routes.integrations as _r_integ
    from models.database import get_user_by_email
    user = get_user_by_email(creds["email"], db_path=_r_integ.DB_PATH)
    return user["id"], creds["username"], creds["email"]


def _seed_connection(user_id, db_path, monkeypatch):
    """Seed an active oauth connection."""
    _set_encryption_key(monkeypatch)
    from services.integrations.crypto import encrypt_token
    from models.database import upsert_oauth_connection

    at_enc = encrypt_token("ya29.sample_valid_token")
    rt_enc = encrypt_token("1//sample_valid_refresh")
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

    upsert_oauth_connection(
        user_id=user_id,
        provider="google",
        access_token_enc=at_enc,
        refresh_token_enc=rt_enc,
        expires_at=expires_at,
        scope=FAKE_SCOPE,
        db_path=db_path,
    )


# =============================================================================
# 1. list_recent_meetings unit tests
# =============================================================================

class TestListRecentMeetings:

    def test_raises_not_connected_when_no_oauth_row(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        from services.integrations.google_meet import list_recent_meetings
        import routes.integrations as _r_integ

        with pytest.raises(NotConnected):
            list_recent_meetings(9999, db_path=_r_integ.DB_PATH)

    def test_returns_normalized_list_with_pagination_and_space_code(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import list_recent_meetings
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet1", "meet1@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        # Page 1 has 2 records and nextPageToken
        resp_page1 = MagicMock()
        resp_page1.status_code = 200
        resp_page1.json.return_value = {
            "conferenceRecords": [
                {
                    "name": "conferenceRecords/rec1",
                    "startTime": "2026-11-10T10:00:00Z",
                    "endTime": "2026-11-10T11:00:00Z",
                    "space": "spaces/sp1",
                },
                {
                    "name": "conferenceRecords/rec2",
                    "startTime": "2026-11-11T14:00:00Z",
                    "endTime": "2026-11-11T14:45:00Z",
                    "space": "spaces/sp2",
                },
            ],
            "nextPageToken": "page2_token",
        }

        # Page 2 has 1 record, no nextPageToken
        resp_page2 = MagicMock()
        resp_page2.status_code = 200
        resp_page2.json.return_value = {
            "conferenceRecords": [
                {
                    "name": "conferenceRecords/rec3",
                    "startTime": "2026-11-09T08:00:00Z",
                    "endTime": "2026-11-09T09:00:00Z",
                    "space": "spaces/sp3",
                }
            ]
        }

        # Space responses
        resp_sp1 = MagicMock()
        resp_sp1.status_code = 200
        resp_sp1.json.return_value = {"meetingCode": "aaa-bbbb-ccc"}

        resp_sp2 = MagicMock()
        resp_sp2.status_code = 200
        resp_sp2.json.return_value = {"meetingCode": "ddd-eeee-fff"}

        resp_sp3 = MagicMock()
        resp_sp3.status_code = 200
        resp_sp3.json.return_value = {"meetingCode": "ggg-hhhh-iii"}

        def _mock_get(url, *args, **kwargs):
            if "pageToken=page2_token" in str(kwargs.get("params", {})) or kwargs.get("params", {}).get("pageToken") == "page2_token":
                return resp_page2
            elif "conferenceRecords" in url and "spaces" not in url:
                return resp_page1
            elif "spaces/sp1" in url:
                return resp_sp1
            elif "spaces/sp2" in url:
                return resp_sp2
            elif "spaces/sp3" in url:
                return resp_sp3
            return MagicMock(status_code=404, reason="Not Found")

        with patch("requests.get", side_effect=_mock_get):
            meetings = list_recent_meetings(uid, days=30, db_path=_r_integ.DB_PATH)

        assert len(meetings) == 3
        # Should be sorted newest first: rec2 (Nov 11) > rec1 (Nov 10) > rec3 (Nov 9)
        assert meetings[0]["external_id"] == "conferenceRecords/rec2"
        assert meetings[0]["meeting_code"] == "ddd-eeee-fff"
        assert meetings[1]["external_id"] == "conferenceRecords/rec1"
        assert meetings[1]["meeting_code"] == "aaa-bbbb-ccc"
        assert meetings[2]["external_id"] == "conferenceRecords/rec3"
        assert meetings[2]["meeting_code"] == "ggg-hhhh-iii"

    def test_space_code_failure_gracefully_sets_none(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import list_recent_meetings
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet2", "meet2@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp_records = MagicMock()
        resp_records.status_code = 200
        resp_records.json.return_value = {
            "conferenceRecords": [
                {
                    "name": "conferenceRecords/rec404",
                    "startTime": "2026-11-12T12:00:00Z",
                    "endTime": "2026-11-12T13:00:00Z",
                    "space": "spaces/broken",
                }
            ]
        }

        resp_space_fail = MagicMock()
        resp_space_fail.status_code = 404
        resp_space_fail.reason = "Space Not Found"
        resp_space_fail.json.return_value = {"error": {"message": "Space Not Found"}}

        def _mock_get(url, *args, **kwargs):
            if "conferenceRecords" in url and "spaces" not in url:
                return resp_records
            return resp_space_fail

        with patch("requests.get", side_effect=_mock_get):
            meetings = list_recent_meetings(uid, days=30, db_path=_r_integ.DB_PATH)

        assert len(meetings) == 1
        assert meetings[0]["external_id"] == "conferenceRecords/rec404"
        assert meetings[0]["meeting_code"] is None

    def test_empty_conference_records(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import list_recent_meetings
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet3", "meet3@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"conferenceRecords": []}

        with patch("requests.get", return_value=resp):
            meetings = list_recent_meetings(uid, days=30, db_path=_r_integ.DB_PATH)

        assert meetings == []

    def test_401_triggers_refresh_and_retry(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import list_recent_meetings
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet4", "meet4@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.reason = "Unauthorized"
        resp_401.json.return_value = {"error": "invalid_token"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "conferenceRecords": [
                {
                    "name": "conferenceRecords/retried",
                    "startTime": "2026-11-12T10:00:00Z",
                    "space": "",
                }
            ]
        }

        # Mock token refresh POST
        resp_refresh = MagicMock()
        resp_refresh.status_code = 200
        resp_refresh.json.return_value = {
            "access_token": "ya29.new_token_after_refresh",
            "expires_in": 3600,
        }

        calls = [resp_401, resp_ok]

        with patch("requests.get", side_effect=lambda *a, **kw: calls.pop(0)):
            with patch("requests.post", return_value=resp_refresh):
                meetings = list_recent_meetings(uid, days=30, db_path=_r_integ.DB_PATH)

        assert len(meetings) == 1
        assert meetings[0]["external_id"] == "conferenceRecords/retried"

    def test_429_retries_and_raises_rate_limited(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import list_recent_meetings
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet5", "meet5@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.reason = "Rate Limit Exceeded"
        resp_429.json.return_value = {"error": {"message": "Quota exceeded"}}

        # Patch time.sleep so test runs instantaneously
        with patch("requests.get", return_value=resp_429):
            with patch("time.sleep", return_value=None):
                with pytest.raises(RateLimited):
                    list_recent_meetings(uid, days=30, db_path=_r_integ.DB_PATH)

    def test_403_raises_api_error(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import list_recent_meetings
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet6", "meet6@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.reason = "Forbidden"

        with patch("requests.get", return_value=resp_403):
            with pytest.raises(ApiError) as exc_info:
                list_recent_meetings(uid, days=30, db_path=_r_integ.DB_PATH)
            assert "forbidden" in str(exc_info.value).lower()


# =============================================================================
# 2. get_participants unit tests
# =============================================================================

class TestGetParticipants:

    def test_raises_not_connected_when_no_oauth_row(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        from services.integrations.google_meet import get_participants
        import routes.integrations as _r_integ

        with pytest.raises(NotConnected):
            get_participants(9999, "conferenceRecords/any", db_path=_r_integ.DB_PATH)

    def test_returns_normalized_and_merged_participants(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import get_participants
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet7", "meet7@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "participants": [
                # Signed-in user session 1
                {
                    "name": "conferenceRecords/r1/participants/p1",
                    "earliestStartTime": "2026-11-12T10:00:00Z",
                    "latestEndTime": "2026-11-12T10:30:00Z",
                    "signedinUser": {
                        "user": "users/user_alice",
                        "displayName": "Alice Cooper",
                    },
                },
                # Same signed-in user session 2 (e.g. rejoined later)
                {
                    "name": "conferenceRecords/r1/participants/p2",
                    "earliestStartTime": "2026-11-12T10:45:00Z",
                    "latestEndTime": "2026-11-12T11:15:00Z",
                    "signedinUser": {
                        "user": "users/user_alice",
                        "displayName": "Alice Cooper",
                    },
                },
                # Anonymous user
                {
                    "name": "conferenceRecords/r1/participants/p3",
                    "earliestStartTime": "2026-11-12T10:05:00Z",
                    "latestEndTime": "2026-11-12T10:40:00Z",
                    "anonymousUser": {
                        "displayName": "External Client",
                    },
                },
                # Phone user
                {
                    "name": "conferenceRecords/r1/participants/p4",
                    "earliestStartTime": "2026-11-12T10:10:00Z",
                    "latestEndTime": "2026-11-12T10:50:00Z",
                    "phoneUser": {
                        "displayName": "+1 555-***-1234",
                    },
                },
            ]
        }

        with patch("requests.get", return_value=resp):
            participants = get_participants(uid, "conferenceRecords/r1", db_path=_r_integ.DB_PATH)

        # Alice should be merged into 1 participant; 1 anon; 1 phone => total 3
        assert len(participants) == 3

        alice = next(p for p in participants if p["external_user_id"] == "users/user_alice")
        assert alice["name"] == "Alice Cooper"
        assert alice["email"] is None
        assert alice["is_anonymous"] is False
        assert alice["participant_type"] == "signed_in"
        # Earliest join is 10:00, latest left is 11:15
        assert alice["joined_at"] == "2026-11-12T10:00:00Z"
        assert alice["left_at"] == "2026-11-12T11:15:00Z"

        anon = next(p for p in participants if p["participant_type"] == "anonymous")
        assert anon["name"] == "External Client"
        assert anon["is_anonymous"] is True
        assert anon["external_user_id"] is None
        assert anon["email"] is None

        phone = next(p for p in participants if p["participant_type"] == "phone")
        assert phone["name"] == "+1 555-***-1234"
        assert phone["is_anonymous"] is True
        assert phone["external_user_id"] is None

    def test_nested_participant_sessions_merged(self, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _set_google_creds(monkeypatch)
        import routes.integrations as _r_integ
        from services.integrations.google_meet import get_participants
        from models.database import insert_user
        from werkzeug.security import generate_password_hash

        uid = insert_user("meet8", "meet8@example.com", generate_password_hash("p"), db_path=_r_integ.DB_PATH)
        _seed_connection(uid, _r_integ.DB_PATH, monkeypatch)

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "participants": [
                {
                    "name": "conferenceRecords/r1/participants/p1",
                    "signedinUser": {
                        "user": "users/user_bob",
                        "displayName": "Bob Dylan",
                    },
                    "participantSessions": [
                        {"startTime": "2026-11-12T09:00:00Z", "endTime": "2026-11-12T09:30:00Z"},
                        {"startTime": "2026-11-12T09:40:00Z", "endTime": "2026-11-12T10:10:00Z"},
                    ],
                }
            ]
        }

        with patch("requests.get", return_value=resp):
            participants = get_participants(uid, "conferenceRecords/r1", db_path=_r_integ.DB_PATH)

        assert len(participants) == 1
        bob = participants[0]
        assert bob["joined_at"] == "2026-11-12T09:00:00Z"
        assert bob["left_at"] == "2026-11-12T10:10:00Z"


# =============================================================================
# 3. Route tests: GET /integrations/google/meetings
# =============================================================================

class TestGoogleMeetingsRoute:

    def test_requires_login(self, client):
        resp = client.get("/integrations/google/meetings")
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]

    def test_not_connected_redirects_to_index(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        with patch("routes.integrations.list_recent_meetings", side_effect=NotConnected("No connection")):
            resp = client.get("/integrations/google/meetings", follow_redirects=False)

        assert resp.status_code in (301, 302)
        assert "/integrations" in resp.headers["Location"]

    def test_token_revoked_redirects_to_index(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        with patch("routes.integrations.list_recent_meetings", side_effect=TokenRevoked("Revoked")):
            resp = client.get("/integrations/google/meetings", follow_redirects=False)

        assert resp.status_code in (301, 302)
        assert "/integrations" in resp.headers["Location"]

    def test_200_renders_meetings_list_and_duration(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        fake_meetings = [
            {
                "external_id": "conferenceRecords/xyz123",
                "start_time": "2026-11-12T10:00:00Z",
                "end_time": "2026-11-12T10:45:00Z",
                "meeting_code": "abc-defg-hij",
                "space_name": "spaces/sp123",
                "participant_count": 4,
            }
        ]

        with patch("routes.integrations.list_recent_meetings", return_value=fake_meetings):
            resp = client.get("/integrations/google/meetings")

        assert resp.status_code == 200
        text = resp.get_data(as_text=True)
        assert "abc-defg-hij" in text
        assert "45 min" in text
        assert "4" in text
        assert "View participants" in text
        assert "Import" in text

    def test_empty_meetings_renders_friendly_message(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        with patch("routes.integrations.list_recent_meetings", return_value=[]):
            resp = client.get("/integrations/google/meetings")

        assert resp.status_code == 200
        text = resp.get_data(as_text=True)
        assert "No recent meetings found" in text


# =============================================================================
# 4. Route tests: GET /integrations/google/participants
# =============================================================================

class TestGoogleParticipantsRoute:

    def test_requires_login(self, client):
        resp = client.get("/integrations/google/participants?record=conferenceRecords/rec1")
        assert resp.status_code in (301, 302)
        assert "/login" in resp.headers["Location"]

    def test_invalid_record_name_redirects(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        resp = client.get("/integrations/google/participants?record=invalid_name", follow_redirects=False)
        assert resp.status_code in (301, 302)
        assert "/google/meetings" in resp.headers["Location"]

    def test_200_renders_participants_table(self, client, app, monkeypatch):
        _set_encryption_key(monkeypatch)
        _login(client, app)

        fake_parts = [
            {
                "name": "Sarah Connor",
                "email": None,
                "external_user_id": "users/sarah",
                "is_anonymous": False,
                "joined_at": "2026-11-12T10:00:00Z",
                "left_at": "2026-11-12T10:50:00Z",
                "participant_type": "signed_in",
            },
            {
                "name": "Anonymous Guest",
                "email": None,
                "external_user_id": None,
                "is_anonymous": True,
                "joined_at": "2026-11-12T10:05:00Z",
                "left_at": None,
                "participant_type": "anonymous",
            },
        ]

        with patch("routes.integrations.get_participants", return_value=fake_parts):
            resp = client.get("/integrations/google/participants?record=conferenceRecords/rec1")

        assert resp.status_code == 200
        text = resp.get_data(as_text=True)
        assert "Sarah Connor" in text
        assert "Signed in" in text
        assert "Anonymous Guest" in text
        assert "Active" in text
        assert "users/sarah" in text

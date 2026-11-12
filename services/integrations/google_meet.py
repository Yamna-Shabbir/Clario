"""
services/integrations/google_meet.py
------------------------------------
Google OAuth 2.0 web-server flow implementation for Google Meet integration.

Official Google documentation pages consulted:
- "Using OAuth 2.0 for Web Server Applications":
  https://developers.google.com/identity/protocols/oauth2/web-server
- "OAuth 2.0 Token Revocation":
  https://developers.google.com/identity/protocols/oauth2/web-server#tokenrevoke

Security & Privacy:
- Never logs, prints, or exposes authorization codes, client secrets, access tokens,
  or refresh tokens.
- Tokens are encrypted at rest using Fernet before saving in SQLite.
- All outbound HTTP requests specify explicit timeouts.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import requests

try:
    import config
except ImportError:
    config = None

from models.database import (
    DB_PATH,
    delete_oauth_connection,
    get_oauth_connection,
    upsert_oauth_connection,
)
from services.integrations.base import (
    ApiError,
    NotConnected,
    RateLimited,
    TokenRevoked,
)
from services.integrations.crypto import (
    decrypt_token,
    encrypt_token,
)

PROVIDER_SLUG = "google"

# Current official Google OAuth 2.0 endpoints
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
REVOKE_URI = "https://oauth2.googleapis.com/revoke"

# Scopes authorized on the user's consent screen
CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"

DEFAULT_SCOPES: List[str] = [
    "https://www.googleapis.com/auth/meetings.space.readonly",
    CALENDAR_READONLY_SCOPE,
]

REQUEST_TIMEOUT_SECONDS = 15

MEET_API_BASE = "https://meet.googleapis.com/v2"
CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


def _get_client_id() -> str:
    if config is not None:
        val = getattr(config, "GOOGLE_CLIENT_ID", "")
        if val:
            return val
    return os.environ.get("CLARIO_GOOGLE_CLIENT_ID", "")


def _get_client_secret() -> str:
    if config is not None:
        val = getattr(config, "GOOGLE_CLIENT_SECRET", "")
        if val:
            return val
    return os.environ.get("CLARIO_GOOGLE_CLIENT_SECRET", "")


def _get_default_redirect_uri() -> str:
    base = "http://127.0.0.1:5000"
    if config is not None:
        base = getattr(config, "OAUTH_REDIRECT_BASE", base) or base
    base = os.environ.get("CLARIO_OAUTH_REDIRECT_BASE", base).rstrip("/")
    return f"{base}/integrations/google/callback"


def get_authorization_url(
    state: str,
    redirect_uri: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> str:
    """
    Construct the Google OAuth 2.0 authorization URL.

    Parameters
    ----------
    state        : Cryptographically random anti-forgery state string.
    redirect_uri : Callback URI (defaults to config OAUTH_REDIRECT_BASE + /integrations/google/callback).
    scopes       : List of approved Google API scopes.

    Returns
    -------
    str : Full authorization redirect URL.
    """
    client_id = _get_client_id()
    if not client_id:
        raise ApiError("Google Client ID is not configured.")

    target_redirect = redirect_uri or _get_default_redirect_uri()
    scope_list = scopes or DEFAULT_SCOPES
    scope_str = " ".join(scope_list)

    params = {
        "client_id": client_id,
        "redirect_uri": target_redirect,
        "response_type": "code",
        "scope": scope_str,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "include_granted_scopes": "true",
    }
    return f"{AUTH_URI}?{urlencode(params)}"


def exchange_code_for_tokens(
    code: str,
    redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Exchange an authorization code for access and refresh tokens.

    Parameters
    ----------
    code         : Short-lived authorization code from the Google redirect.
    redirect_uri : Redirect URI used in the initial request.

    Returns
    -------
    dict with keys: access_token, refresh_token (optional), expires_in, scope, token_type
    """
    client_id = _get_client_id()
    client_secret = _get_client_secret()
    if not client_id or not client_secret:
        raise ApiError("Google OAuth client credentials are not configured.")

    payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri or _get_default_redirect_uri(),
        "grant_type": "authorization_code",
    }

    try:
        response = requests.post(
            TOKEN_URI,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ApiError("Failed to reach Google token endpoint.") from None

    if response.status_code == 429:
        raise RateLimited("Google token endpoint rate limit exceeded.")

    try:
        data = response.json()
    except Exception:
        raise ApiError(f"Invalid response from Google token endpoint (HTTP {response.status_code}).")

    if response.status_code != 200:
        err_msg = data.get("error_description") or data.get("error") or "Unknown error"
        raise ApiError(f"Google token exchange failed: {err_msg}")

    if "access_token" not in data:
        raise ApiError("Google token exchange response did not contain an access_token.")

    return data


def refresh_access_token(
    refresh_token: str,
) -> Dict[str, Any]:
    """
    Request a fresh access token from Google using an existing refresh token.

    Parameters
    ----------
    refresh_token : Decrypted plain refresh token string.

    Returns
    -------
    dict with keys: access_token, expires_in, scope (optional), token_type
    """
    client_id = _get_client_id()
    client_secret = _get_client_secret()
    if not client_id or not client_secret:
        raise ApiError("Google OAuth client credentials are not configured.")

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    try:
        response = requests.post(
            TOKEN_URI,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        raise ApiError("Failed to reach Google token endpoint during refresh.") from None

    if response.status_code == 429:
        raise RateLimited("Google token refresh rate limit exceeded.")

    try:
        data = response.json()
    except Exception:
        raise ApiError(f"Invalid response from Google token endpoint (HTTP {response.status_code}).")

    if response.status_code != 200:
        err_code = data.get("error", "")
        err_desc = data.get("error_description", "")
        if err_code == "invalid_grant":
            raise TokenRevoked(
                "Google OAuth authorization has expired or been revoked. Please reconnect."
            )
        raise ApiError(f"Google token refresh failed: {err_desc or err_code}")

    if "access_token" not in data:
        raise ApiError("Google token refresh response did not contain an access_token.")

    return data


def revoke_token(token: str) -> bool:
    """
    Best-effort revocation of an OAuth token with Google.

    Parameters
    ----------
    token : Decrypted plain access or refresh token string.

    Returns
    -------
    bool : True if revocation succeeded (HTTP 200), False otherwise.
    """
    if not token:
        return False
    try:
        response = requests.post(
            REVOKE_URI,
            data={"token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


def get_valid_access_token(
    user_id: int,
    db_path: str = DB_PATH,
) -> str:
    """
    Fetch an active, unexpired Google access token for a user.

    If the current access token has expired (or will expire in the next 5 minutes),
    it is automatically refreshed using the stored refresh token, re-encrypted,
    and updated in the database.

    Parameters
    ----------
    user_id : Current authenticated user id.
    db_path : SQLite database path.

    Returns
    -------
    str : Valid plain decrypted access token.

    Raises
    ------
    NotConnected : User has no Google OAuth connection in the database.
    TokenRevoked : Google rejected the refresh token (invalid_grant); deletes connection.
    RateLimited  : Google rate limit encountered.
    ApiError     : Network or unexpected API error.
    """
    conn = get_oauth_connection(user_id, PROVIDER_SLUG, db_path=db_path)
    if not conn:
        raise NotConnected("Google account is not connected.")

    access_token_enc = conn.get("access_token_enc")
    if not access_token_enc:
        raise NotConnected("Stored Google connection has no access token.")

    current_access_token = decrypt_token(access_token_enc)

    # Check expiration (with a 5-minute safety buffer)
    expires_at_str = conn.get("expires_at")
    is_expired = True

    if expires_at_str:
        try:
            # Handle ISO string with potential Z or timezone offsets
            normalized = expires_at_str.replace("Z", "+00:00")
            expires_dt = datetime.fromisoformat(normalized)
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            if now_utc < (expires_dt - timedelta(minutes=5)):
                is_expired = False
        except Exception:
            is_expired = True

    if not is_expired:
        return current_access_token

    # Token is expired or expiring soon; refresh it
    refresh_token_enc = conn.get("refresh_token_enc")
    if not refresh_token_enc:
        delete_oauth_connection(user_id, PROVIDER_SLUG, db_path=db_path)
        raise TokenRevoked("No refresh token stored. Please reconnect your Google account.")

    plain_refresh_token = decrypt_token(refresh_token_enc)

    try:
        new_token_data = refresh_access_token(plain_refresh_token)
    except TokenRevoked:
        # Invalid grant: delete stored connection and alert user
        delete_oauth_connection(user_id, PROVIDER_SLUG, db_path=db_path)
        raise

    new_access_token = new_token_data["access_token"]
    new_access_enc = encrypt_token(new_access_token)

    # Optional new refresh token
    new_refresh = new_token_data.get("refresh_token")
    if new_refresh:
        new_refresh_enc = encrypt_token(new_refresh)
    else:
        new_refresh_enc = refresh_token_enc

    expires_in_sec = int(new_token_data.get("expires_in", 3600))
    new_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in_sec)).isoformat()
    new_scope = new_token_data.get("scope") or conn.get("scope")

    upsert_oauth_connection(
        user_id=user_id,
        provider=PROVIDER_SLUG,
        access_token_enc=new_access_enc,
        refresh_token_enc=new_refresh_enc,
        expires_at=new_expires_at,
        scope=new_scope,
        db_path=db_path,
    )

    return new_access_token


# ---------------------------------------------------------------------------
# Meet API helpers
# ---------------------------------------------------------------------------

def _make_authed_get(
    url: str,
    params: Dict,
    user_id: int,
    db_path: str,
    *,
    _retry: bool = True,
) -> Dict:
    """
    Make an authenticated GET request to the Google Meet API.

    Handles 401 (auto-refresh once), 403, 404, 429 (bounded backoff),
    and other non-200 responses. Never logs tokens.

    Parameters
    ----------
    url      : Full URL to request.
    params   : Query parameters dict.
    user_id  : Authenticated user id.
    db_path  : SQLite database path.
    _retry   : Internal flag; set False to disable the 401-refresh retry.

    Returns
    -------
    dict : Parsed JSON response body.

    Raises
    ------
    NotConnected : No OAuth connection found.
    TokenRevoked : Google rejected the refresh token.
    RateLimited  : HTTP 429 after bounded backoff.
    ApiError     : Any other non-200 response.
    """
    token = get_valid_access_token(user_id, db_path)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException:
        raise ApiError("Failed to reach Google Meet API.") from None

    # --- 401: try to refresh and retry once ---
    if response.status_code == 401 and _retry:
        conn = get_oauth_connection(user_id, PROVIDER_SLUG, db_path=db_path)
        if not conn or not conn.get("refresh_token_enc"):
            raise NotConnected("No refresh token available.")
        plain_refresh = decrypt_token(conn["refresh_token_enc"])
        new_data = refresh_access_token(plain_refresh)
        new_at = new_data["access_token"]
        new_at_enc = encrypt_token(new_at)
        expires_in = int(new_data.get("expires_in", 3600))
        new_expires = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        upsert_oauth_connection(
            user_id=user_id,
            provider=PROVIDER_SLUG,
            access_token_enc=new_at_enc,
            refresh_token_enc=conn.get("refresh_token_enc"),
            expires_at=new_expires,
            scope=conn.get("scope"),
            db_path=db_path,
        )
        # Retry with the new token directly (don't go through get_valid_access_token)
        retry_headers = {
            "Authorization": f"Bearer {new_at}",
            "Accept": "application/json",
        }
        try:
            response = requests.get(url, params=params, headers=retry_headers, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException:
            raise ApiError("Failed to reach Google Meet API on retry.") from None
        if response.status_code == 401:
            raise ApiError("Authentication failed after token refresh. Please reconnect your Google account.")

    # --- 403 ---
    if response.status_code == 403:
        raise ApiError("Access forbidden: check Google Meet API is enabled and scopes are correct.")

    # --- 404 ---
    if response.status_code == 404:
        raise ApiError("Resource not found.")

    # --- 429: bounded exponential backoff (max 2 retries) ---
    if response.status_code == 429:
        for sleep_secs in (1, 2):
            time.sleep(sleep_secs)
            try:
                response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException:
                raise ApiError("Failed to reach Google Meet API.") from None
            if response.status_code != 429:
                break
        else:
            raise RateLimited("Google Meet API rate limit exceeded. Please try again later.")
        if response.status_code == 429:
            raise RateLimited("Google Meet API rate limit exceeded. Please try again later.")

    # --- Other non-200 ---
    if response.status_code != 200:
        try:
            err_body = response.json()
            err_msg = (
                err_body.get("error", {}).get("message", "")
                or err_body.get("message", "")
                or response.reason
                or "Unknown error"
            )
        except Exception:
            err_msg = response.reason or "Unknown error"
        raise ApiError(f"Meet API error {response.status_code}: {err_msg}")

    try:
        return response.json()
    except Exception:
        raise ApiError(f"Invalid JSON response from Google Meet API (HTTP {response.status_code}).") from None


def list_recent_meetings(
    user_id: int,
    days: int = 30,
    db_path: str = DB_PATH,
) -> List[Dict]:
    """
    List recent Google Meet conference records where the user is the organizer.

    Parameters
    ----------
    user_id : Authenticated user id.
    days    : Number of days back to query (default 30).
    db_path : SQLite database path.

    Returns
    -------
    List of meeting dicts sorted by start_time descending:
        external_id, start_time, end_time, meeting_code, space_name, participant_count

    Raises
    ------
    NotConnected : User has no Google OAuth connection.
    TokenRevoked : Google rejected the refresh token.
    RateLimited  : Google API rate limit hit.
    ApiError     : Unexpected API error.
    """
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    # RFC3339 format with Z suffix
    since_str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    filter_str = f'start_time>="{since_str}"'

    meetings: List[Dict] = []
    page_token: Optional[str] = None
    max_pages = 10
    page_count = 0

    while page_count < max_pages:
        params: Dict[str, Any] = {"filter": filter_str, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token

        data = _make_authed_get(
            f"{MEET_API_BASE}/conferenceRecords",
            params,
            user_id,
            db_path,
        )

        records = data.get("conferenceRecords", [])
        for record in records:
            # Fetch space details for meetingCode
            space_name = record.get("space", "")
            meeting_code = None
            if space_name:
                try:
                    space_data = _make_authed_get(
                        f"{MEET_API_BASE}/{space_name}",
                        {},
                        user_id,
                        db_path,
                    )
                    meeting_code = space_data.get("meetingCode")
                except ApiError:
                    meeting_code = None

            meetings.append({
                "external_id": record["name"],
                "start_time": record["startTime"],
                "end_time": record.get("endTime"),
                "meeting_code": meeting_code,
                "space_name": space_name,
                "participant_count": None,
            })

        page_token = data.get("nextPageToken")
        page_count += 1
        if not page_token:
            break

    # Sort newest first by start_time
    meetings.sort(key=lambda m: m["start_time"], reverse=True)
    return meetings


def get_participants(
    user_id: int,
    external_id: str,
    db_path: str = DB_PATH,
) -> List[Dict]:
    """
    Fetch participants for a Google Meet conference record.

    Parameters
    ----------
    user_id     : Authenticated user id.
    external_id : Conference record name (e.g. "conferenceRecords/abc123").
    db_path     : SQLite database path.

    Returns
    -------
    List of participant dicts sorted by joined_at ascending:
        name, email, external_user_id, is_anonymous, joined_at, left_at, participant_type

    Raises
    ------
    NotConnected : User has no Google OAuth connection.
    TokenRevoked : Google rejected the refresh token.
    RateLimited  : Google API rate limit hit.
    ApiError     : Unexpected API error.
    """
    all_participants: List[Dict] = []
    page_token: Optional[str] = None
    max_pages = 10
    page_count = 0

    while page_count < max_pages:
        params: Dict[str, Any] = {"pageSize": 250}
        if page_token:
            params["pageToken"] = page_token

        data = _make_authed_get(
            f"{MEET_API_BASE}/{external_id}/participants",
            params,
            user_id,
            db_path,
        )

        for p in data.get("participants", []):
            if "signedinUser" in p:
                su = p["signedinUser"]
                normalized = {
                    "name": su.get("displayName", ""),
                    "email": None,
                    "external_user_id": su.get("user"),
                    "is_anonymous": False,
                    "joined_at": p.get("earliestStartTime"),
                    "left_at": p.get("latestEndTime"),
                    "participant_type": "signed_in",
                }
            elif "anonymousUser" in p:
                au = p["anonymousUser"]
                normalized = {
                    "name": au.get("displayName", ""),
                    "email": None,
                    "external_user_id": None,
                    "is_anonymous": True,
                    "joined_at": p.get("earliestStartTime"),
                    "left_at": p.get("latestEndTime"),
                    "participant_type": "anonymous",
                }
            elif "phoneUser" in p:
                pu = p["phoneUser"]
                normalized = {
                    "name": pu.get("displayName", ""),
                    "email": None,
                    "external_user_id": None,
                    "is_anonymous": True,
                    "joined_at": p.get("earliestStartTime"),
                    "left_at": p.get("latestEndTime"),
                    "participant_type": "phone",
                }
            else:
                # Unknown participant type; skip
                continue

            # Merge nested participantSessions if present
            sessions = p.get("participantSessions") or p.get("sessions") or []
            if sessions:
                s_starts = [s.get("startTime") for s in sessions if s.get("startTime")]
                s_ends = [s.get("endTime") for s in sessions if s.get("endTime")]
                has_active = any(s.get("endTime") is None for s in sessions)
                if s_starts:
                    earliest_s = min(s_starts)
                    if not normalized["joined_at"] or earliest_s < normalized["joined_at"]:
                        normalized["joined_at"] = earliest_s
                if has_active:
                    normalized["left_at"] = None
                elif s_ends:
                    latest_e = max(s_ends)
                    if not normalized["left_at"] or latest_e > normalized["left_at"]:
                        normalized["left_at"] = latest_e

            all_participants.append(normalized)

        page_token = data.get("nextPageToken")
        page_count += 1
        if not page_token:
            break

    # De-duplicate signed-in users by external_user_id
    # Keep earliest joined_at and latest left_at
    deduped: List[Dict] = []
    seen_signed_in: Dict[str, Dict] = {}

    for p in all_participants:
        if p["participant_type"] == "signed_in" and p["external_user_id"]:
            uid = p["external_user_id"]
            if uid in seen_signed_in:
                existing = seen_signed_in[uid]
                # Keep earliest joined_at
                if p["joined_at"] and (not existing["joined_at"] or p["joined_at"] < existing["joined_at"]):
                    existing["joined_at"] = p["joined_at"]
                # Keep latest left_at; if any session is active (None), overall left_at is None (active)
                if existing["left_at"] is None or p["left_at"] is None:
                    existing["left_at"] = None
                elif p["left_at"] > existing["left_at"]:
                    existing["left_at"] = p["left_at"]
            else:
                seen_signed_in[uid] = dict(p)
        else:
            # Anonymous / phone users cannot be de-duped
            deduped.append(p)

    deduped.extend(seen_signed_in.values())

    # Sort by joined_at ascending (None last)
    deduped.sort(key=lambda p: (p["joined_at"] is None, p["joined_at"] or ""))
    return deduped


def find_calendar_attendees(
    user_id: int,
    meeting_code_or_start_time_window: Any = None,
    *,
    meeting_code: Optional[str] = None,
    start_time: Optional[str] = None,
    time_window_minutes: int = 60,
    db_path: str = DB_PATH,
) -> List[Dict[str, Optional[str]]]:
    """
    Find Google Calendar attendees for an imported Google Meet meeting.

    Queries the Google Calendar API v3 events.list endpoint using the authorized
    'https://www.googleapis.com/auth/calendar.readonly' scope, matches the calendar
    event corresponding to the Meet conference (by meeting code or start time window),
    and returns normalized attendee records.

    Official Google Calendar API Documentation:
    - Events list: https://developers.google.com/calendar/api/v3/reference/events/list
    - Events resource: https://developers.google.com/calendar/api/v3/reference/events#resource

    Parameters
    ----------
    user_id : int
        Authenticated user ID.
    meeting_code_or_start_time_window : Any
        Flexible parameter accepting a meeting code string (e.g. 'abc-mnop-xyz'),
        an ISO start time string, a datetime object, or a dict with meeting metadata.
    meeting_code : Optional[str]
        Explicit Google Meet code (e.g. 'abc-mnop-xyz').
    start_time : Optional[str]
        Explicit start time (ISO 8601 string or datetime).
    time_window_minutes : int
        Search tolerance window around the meeting start time (default 60 min).
    db_path : str
        SQLite database path.

    Returns
    -------
    List[Dict[str, Optional[str]]]
        List of dicts: {'display_name': str or None, 'email': str, 'name': str}.
        Returns empty list if no matching event is found, or on any API/permission error.
    """
    import re

    target_code = meeting_code
    target_start = start_time

    if meeting_code_or_start_time_window is not None:
        if isinstance(meeting_code_or_start_time_window, dict):
            target_code = (
                target_code
                or meeting_code_or_start_time_window.get("meeting_code")
                or meeting_code_or_start_time_window.get("code")
            )
            target_start = (
                target_start
                or meeting_code_or_start_time_window.get("start_time")
                or meeting_code_or_start_time_window.get("start")
            )
        elif isinstance(meeting_code_or_start_time_window, datetime):
            if not target_start:
                target_start = meeting_code_or_start_time_window.isoformat()
        elif isinstance(meeting_code_or_start_time_window, str):
            val = meeting_code_or_start_time_window.strip()
            if "T" in val or (":" in val and "-" in val and len(val) >= 16):
                if not target_start:
                    target_start = val
            else:
                if not target_code:
                    target_code = val

    norm_code = ""
    if target_code:
        clean_code = str(target_code).split("/")[-1].split("?")[0]
        norm_code = re.sub(r"[^a-zA-Z0-9]", "", clean_code).lower()

    ref_dt = None
    if target_start:
        if isinstance(target_start, datetime):
            ref_dt = target_start
        elif isinstance(target_start, str):
            try:
                clean_ts = target_start.replace("Z", "+00:00")
                ref_dt = datetime.fromisoformat(clean_ts)
            except Exception:
                ref_dt = None

    if ref_dt:
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.replace(tzinfo=timezone.utc)
        # Search +/- 7 days around the meeting
        t_min = (ref_dt - timedelta(days=7)).isoformat().replace("+00:00", "Z")
        t_max = (ref_dt + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    else:
        # Default: last 30 days to 2 days ahead
        now = datetime.now(timezone.utc)
        t_min = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")
        t_max = (now + timedelta(days=2)).isoformat().replace("+00:00", "Z")

    params = {
        "singleEvents": "true",
        "orderBy": "startTime",
        "timeMin": t_min,
        "timeMax": t_max,
        "maxResults": "50",
    }

    try:
        data = _make_authed_get(
            f"{CALENDAR_API_BASE}/calendars/primary/events",
            params=params,
            user_id=user_id,
            db_path=db_path,
        )
    except (NotConnected, TokenRevoked, RateLimited, ApiError, Exception):
        return []

    items = data.get("items", []) if isinstance(data, dict) else []
    if not items:
        return []

    matched_event = None

    # 1. Match by Meet conference code
    if norm_code:
        for event in items:
            hangout = event.get("hangoutLink") or ""
            if norm_code in re.sub(r"[^a-zA-Z0-9]", "", hangout).lower():
                matched_event = event
                break
            conf_data = event.get("conferenceData") or {}
            conf_id = conf_data.get("conferenceId") or ""
            if norm_code in re.sub(r"[^a-zA-Z0-9]", "", conf_id).lower():
                matched_event = event
                break
            for ep in conf_data.get("entryPoints", []):
                ep_uri = ep.get("uri") or ""
                if norm_code in re.sub(r"[^a-zA-Z0-9]", "", ep_uri).lower():
                    matched_event = event
                    break
            if matched_event:
                break
            loc = event.get("location") or ""
            if norm_code in re.sub(r"[^a-zA-Z0-9]", "", loc).lower():
                matched_event = event
                break

    # 2. Fallback: match by start time proximity if not matched by code
    if not matched_event and ref_dt:
        best_diff = None
        best_event = None
        tolerance_seconds = max(time_window_minutes * 60, 900)  # at least 15 min

        for event in items:
            evt_start = event.get("start", {})
            dt_str = evt_start.get("dateTime") or evt_start.get("date")
            if not dt_str:
                continue
            try:
                evt_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if evt_dt.tzinfo is None:
                    evt_dt = evt_dt.replace(tzinfo=timezone.utc)
                diff = abs((evt_dt - ref_dt).total_seconds())
                if diff <= tolerance_seconds:
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_event = event
            except Exception:
                continue
        matched_event = best_event

    if not matched_event:
        return []

    # Extract attendees & organizer
    attendees_out: List[Dict[str, Optional[str]]] = []
    seen_emails = set()

    for att in matched_event.get("attendees", []):
        email = att.get("email")
        if not email or not email.strip():
            continue
        email_clean = email.strip()
        email_key = email_clean.lower()
        if email_key in seen_emails:
            continue
        seen_emails.add(email_key)

        disp = (att.get("displayName") or "").strip() or None
        attendees_out.append({
            "display_name": disp,
            "email": email_clean,
            "name": disp or email_clean.split("@")[0].replace(".", " ").title(),
        })

    organizer = matched_event.get("organizer", {})
    org_email = organizer.get("email")
    if org_email and org_email.strip():
        org_email_clean = org_email.strip()
        org_key = org_email_clean.lower()
        if org_key not in seen_emails:
            seen_emails.add(org_key)
            org_disp = (organizer.get("displayName") or "").strip() or None
            attendees_out.append({
                "display_name": org_disp,
                "email": org_email_clean,
                "name": org_disp or org_email_clean.split("@")[0].replace(".", " ").title(),
            })

    return attendees_out


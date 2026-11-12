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
DEFAULT_SCOPES: List[str] = [
    "https://www.googleapis.com/auth/meetings.space.readonly",
]

REQUEST_TIMEOUT_SECONDS = 15


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


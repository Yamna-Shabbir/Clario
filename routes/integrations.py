"""
routes/integrations.py
----------------------
Blueprint handling platform integrations:
- GET  /integrations                  : Integration status overview
- GET  /integrations/google/connect   : Initiates Google OAuth 2.0 flow
- GET  /integrations/google/callback  : Validates state and exchanges code
- POST /integrations/google/disconnect: Revokes token and removes stored connection
"""

import secrets
from datetime import datetime, timedelta, timezone
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import DB_PATH
import config
from models.database import (
    delete_oauth_connection,
    get_oauth_connection,
    upsert_oauth_connection,
)
from routes.utils import current_user, login_required
from services.integrations.base import ApiError, RateLimited
from services.integrations.crypto import decrypt_token, encrypt_token
from services.integrations.google_meet import (
    PROVIDER_SLUG,
    exchange_code_for_tokens,
    get_authorization_url,
    revoke_token,
)

integrations_bp = Blueprint("integrations", __name__, url_prefix="/integrations")


@integrations_bp.route("", methods=["GET"])
@login_required
def index():
    """Render the integrations management status page."""
    user_id = session["user_id"]
    google_conn = get_oauth_connection(user_id, PROVIDER_SLUG, db_path=DB_PATH)

    client_id = getattr(config, "GOOGLE_CLIENT_ID", "")
    client_secret = getattr(config, "GOOGLE_CLIENT_SECRET", "")
    is_configured = bool(client_id and client_secret)

    scope_list = []
    if google_conn and google_conn.get("scope"):
        scope_list = [s.strip() for s in google_conn["scope"].split() if s.strip()]

    return render_template(
        "integrations/index.html",
        user=current_user(),
        google_conn=google_conn,
        google_configured=is_configured,
        scope_list=scope_list,
    )


@integrations_bp.route("/google/connect", methods=["GET"])
@login_required
def google_connect():
    """Start the Google OAuth 2.0 authorization code flow."""
    client_id = getattr(config, "GOOGLE_CLIENT_ID", "")
    client_secret = getattr(config, "GOOGLE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        flash(
            "Google OAuth is not configured on this server. "
            "Please configure CLARIO_GOOGLE_CLIENT_ID and CLARIO_GOOGLE_CLIENT_SECRET in .env.",
            "error",
        )
        return redirect(url_for("integrations.index"))

    # Generate cryptographically secure random state to prevent CSRF in the OAuth flow
    state = secrets.token_urlsafe(32)
    session["google_oauth_state"] = state

    try:
        auth_url = get_authorization_url(state=state)
    except ApiError as exc:
        flash(str(exc), "error")
        return redirect(url_for("integrations.index"))

    return redirect(auth_url)


@integrations_bp.route("/google/callback", methods=["GET"])
@login_required
def google_callback():
    """Handle callback redirect from Google's authorization server."""
    # 1. Check for error returned by Google
    error = request.args.get("error")
    if error:
        if error == "access_denied":
            flash("Google authorization was cancelled or denied.", "warning")
        else:
            flash(f"Google authorization failed: {error}", "error")
        return redirect(url_for("integrations.index"))

    # 2. Validate anti-forgery state parameter
    expected_state = session.pop("google_oauth_state", None)
    received_state = request.args.get("state")

    if not expected_state or not received_state or not secrets.compare_digest(expected_state, received_state):
        flash("Invalid or expired OAuth state parameter. Please try connecting again.", "error")
        return redirect(url_for("integrations.index"))

    # 3. Verify authorization code is present
    code = request.args.get("code")
    if not code:
        flash("Authorization code missing from Google response.", "error")
        return redirect(url_for("integrations.index"))

    # 4. Exchange authorization code for tokens
    try:
        token_data = exchange_code_for_tokens(code)
    except RateLimited as exc:
        flash(str(exc), "warning")
        return redirect(url_for("integrations.index"))
    except (ApiError, Exception) as exc:
        flash(f"Could not complete Google authorization: {str(exc)}", "error")
        return redirect(url_for("integrations.index"))

    # 5. Encrypt tokens and persist connection
    access_token_plain = token_data["access_token"]
    refresh_token_plain = token_data.get("refresh_token")

    access_token_enc = encrypt_token(access_token_plain)
    refresh_token_enc = encrypt_token(refresh_token_plain) if refresh_token_plain else None

    expires_in_sec = int(token_data.get("expires_in", 3600))
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in_sec)).isoformat()
    scope = token_data.get("scope", "")

    user_id = session["user_id"]
    upsert_oauth_connection(
        user_id=user_id,
        provider=PROVIDER_SLUG,
        access_token_enc=access_token_enc,
        refresh_token_enc=refresh_token_enc,
        expires_at=expires_at,
        scope=scope,
        db_path=DB_PATH,
    )

    flash("Google Meet connected successfully! 🎉", "success")
    return redirect(url_for("integrations.index"))


@integrations_bp.route("/google/disconnect", methods=["POST"])
@login_required
def google_disconnect():
    """Revoke tokens with Google (best-effort) and delete connection from DB."""
    user_id = session["user_id"]
    conn = get_oauth_connection(user_id, PROVIDER_SLUG, db_path=DB_PATH)

    if not conn:
        flash("Google account is not currently connected.", "warning")
        return redirect(url_for("integrations.index"))

    # Best-effort token revocation
    token_to_revoke = None
    if conn.get("refresh_token_enc"):
        try:
            token_to_revoke = decrypt_token(conn["refresh_token_enc"])
        except Exception:
            token_to_revoke = None

    if not token_to_revoke and conn.get("access_token_enc"):
        try:
            token_to_revoke = decrypt_token(conn["access_token_enc"])
        except Exception:
            token_to_revoke = None

    if token_to_revoke:
        revoke_token(token_to_revoke)

    delete_oauth_connection(user_id, PROVIDER_SLUG, db_path=DB_PATH)
    flash("Google Meet disconnected successfully.", "info")
    return redirect(url_for("integrations.index"))


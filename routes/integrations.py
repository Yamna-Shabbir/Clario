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
    abort,
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
    get_meeting_detail,
    get_meetings_by_user,
    get_oauth_connection,
    replace_meeting_participants,
    update_meeting_platform,
    upsert_oauth_connection,
)
from routes.utils import current_user, login_required
from services.integrations.base import ApiError, NotConnected, RateLimited, TokenRevoked
from services.integrations.crypto import decrypt_token, encrypt_token
from services.integrations.google_meet import (
    MEET_API_BASE,
    PROVIDER_SLUG,
    _make_authed_get,
    exchange_code_for_tokens,
    find_calendar_attendees,
    get_authorization_url,
    get_participants,
    list_recent_meetings,
    revoke_token,
)
from services.participant_matching import match, rematch_meeting_tasks

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


@integrations_bp.route("/google/meetings", methods=["GET"])
@login_required
def google_meetings():
    """List recent Google Meet conference records for the logged-in user."""
    user_id = session["user_id"]

    try:
        meetings = list_recent_meetings(user_id, days=30, db_path=DB_PATH)
    except NotConnected:
        flash("Connect your Google account first.", "warning")
        return redirect(url_for("integrations.index"))
    except TokenRevoked:
        flash("Your Google token has expired. Please reconnect.", "warning")
        return redirect(url_for("integrations.index"))
    except RateLimited:
        flash("Google API rate limit hit. Please try again later.", "warning")
        return render_template(
            "integrations/meetings.html",
            meetings=[],
            user=current_user(),
        )
    except ApiError as exc:
        flash(str(exc), "error")
        return render_template(
            "integrations/meetings.html",
            meetings=[],
            user=current_user(),
        )

    # Compute duration_minutes for each meeting
    meetings_list = []
    for m in meetings:
        entry = dict(m)
        if m.get("start_time") and m.get("end_time"):
            try:
                start = datetime.fromisoformat(m["start_time"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(m["end_time"].replace("Z", "+00:00"))
                entry["duration_minutes"] = int((end - start).total_seconds() / 60)
            except Exception:
                entry["duration_minutes"] = None
        else:
            entry["duration_minutes"] = None
        meetings_list.append(entry)

    user_meetings = get_meetings_by_user(user_id, db_path=DB_PATH)

    return render_template(
        "integrations/meetings.html",
        meetings=meetings_list,
        user_meetings=user_meetings,
        user=current_user(),
    )


@integrations_bp.route("/google/participants", methods=["GET"])
@login_required
def google_participants():
    """Show participants for a specific Google Meet conference record."""
    user_id = session["user_id"]
    record = request.args.get("record", "")

    if not record.startswith("conferenceRecords/"):
        flash("Invalid meeting record identifier.", "error")
        return redirect(url_for("integrations.google_meetings"))

    try:
        participants = get_participants(user_id, record, db_path=DB_PATH)
    except NotConnected:
        flash("Connect your Google account first.", "warning")
        return redirect(url_for("integrations.index"))
    except TokenRevoked:
        flash("Your Google token has expired. Please reconnect.", "warning")
        return redirect(url_for("integrations.index"))
    except RateLimited:
        flash("Google API rate limit hit. Please try again later.", "warning")
        return render_template(
            "integrations/participants.html",
            participants=[],
            record=record,
            user=current_user(),
        )
    except ApiError as exc:
        flash(str(exc), "error")
        return render_template(
            "integrations/participants.html",
            participants=[],
            record=record,
            user=current_user(),
        )

    user_meetings = get_meetings_by_user(user_id, db_path=DB_PATH)
    meeting_code = request.args.get("meeting_code", "").strip()
    start_time = request.args.get("start_time", "").strip()

    return render_template(
        "integrations/participants.html",
        participants=participants,
        record=record,
        user_meetings=user_meetings,
        meeting_code=meeting_code,
        start_time=start_time,
        user=current_user(),
    )


@integrations_bp.route("/google/import", methods=["POST"])
@login_required
def google_import():
    """
    Import participants from a Google Meet conference into a Clario meeting.

    Form parameters:
    - external_meeting_id (required): conferenceRecords/{id}
    - meeting_id (optional): Clario meeting ID to import into.

    Behavior:
    - If meeting_id is given: verifies user ownership (abort 404 if unowned).
      Stores participants (idempotently via replace_meeting_participants),
      sets meetings.platform='google_meet' and meetings.external_meeting_id,
      and rematches tasks of the meeting.
    - If no meeting_id is given: pre-fills participant list in session and redirects
      to the upload page.
    """
    user_id = session["user_id"]
    external_meeting_id = request.form.get("external_meeting_id", "").strip()
    raw_meeting_id = request.form.get("meeting_id")

    if not external_meeting_id:
        flash("External meeting record identifier is required.", "error")
        return redirect(url_for("integrations.google_meetings"))

    try:
        participants = get_participants(user_id, external_meeting_id, db_path=DB_PATH)
    except NotConnected:
        flash("Connect your Google account first.", "warning")
        return redirect(url_for("integrations.index"))
    except TokenRevoked:
        flash("Your Google token has expired. Please reconnect.", "warning")
        return redirect(url_for("integrations.index"))
    except RateLimited:
        flash("Google API rate limit hit. Please try again later.", "warning")
        return redirect(url_for("integrations.google_meetings"))
    except ApiError as exc:
        flash(f"Could not retrieve participants: {str(exc)}", "error")
        return redirect(url_for("integrations.google_meetings"))

    # Part E: Attempt to enrich participant emails via Google Calendar event attendees
    meeting_code = request.form.get("meeting_code", "").strip()
    start_time = request.form.get("start_time", "").strip()
    try:
        if not meeting_code:
            try:
                rec = _make_authed_get(f"{MEET_API_BASE}/{external_meeting_id}", {}, user_id, db_path=DB_PATH)
                start_time = start_time or rec.get("startTime", "")
                space_name = rec.get("space", "")
                if space_name:
                    space_data = _make_authed_get(f"{MEET_API_BASE}/{space_name}", {}, user_id, db_path=DB_PATH)
                    meeting_code = space_data.get("meetingCode", "")
            except Exception:
                pass

        cal_attendees = find_calendar_attendees(
            user_id,
            meeting_code=meeting_code,
            start_time=start_time,
            db_path=DB_PATH,
        )
        if cal_attendees:
            for p in participants:
                if not p.get("email") and p.get("name"):
                    matched_cal = match(p["name"], cal_attendees)
                    if not matched_cal:
                        # Fallback: check exact email username or full email address
                        p_norm = p["name"].strip().lower()
                        email_matches = [
                            ca for ca in cal_attendees
                            if ca.get("email") and (
                                p_norm == ca["email"].strip().lower()
                                or p_norm == ca["email"].strip().lower().split("@")[0]
                            )
                        ]
                        if len(email_matches) == 1:
                            matched_cal = email_matches[0]
                    if matched_cal and matched_cal.get("email"):
                        p["email"] = matched_cal["email"]
    except Exception:
        # Graceful degradation: never fail the import if calendar lookup fails
        pass

    # Case 1: Import into existing Clario meeting
    if raw_meeting_id and str(raw_meeting_id).strip():
        try:
            meeting_id = int(raw_meeting_id)
        except (ValueError, TypeError):
            flash("Invalid meeting ID provided.", "error")
            return redirect(url_for("integrations.google_meetings"))

        meeting = get_meeting_detail(meeting_id, user_id, db_path=DB_PATH)
        if not meeting:
            abort(404)

        # Idempotent storage: replace existing participants without duplicating
        replace_meeting_participants(meeting_id, participants, db_path=DB_PATH)
        update_meeting_platform(meeting_id, platform="google_meet", external_meeting_id=external_meeting_id, db_path=DB_PATH)

        # Rematch tasks
        rematched = rematch_meeting_tasks(meeting_id, user_id, db_path=DB_PATH)

        msg = f"Imported {len(participants)} participant(s) into '{meeting.get('title', 'Meeting')}'."
        if rematched > 0:
            msg += f" {rematched} action item(s) updated with participant names."
        flash(msg, "success")
        return redirect(url_for("meeting.result", meeting_id=meeting_id))

    # Case 2: No meeting_id -> Prefill session for the next upload
    session["prefill_participants"] = participants
    session["prefill_external_meeting_id"] = external_meeting_id
    session["prefill_platform"] = "google_meet"
    flash(
        f"Imported {len(participants)} participant(s) from Google Meet. "
        "Upload audio or paste a transcript to create and link the meeting.",
        "info",
    )
    return redirect(url_for("meeting.upload"))



"""
routes/live.py
--------------
Start a Clario meeting, share the join link, and record each participant.
"""

import json
import logging
import os
import re
import sqlite3
from typing import List

from flask import (
    Blueprint, abort, flash, jsonify, redirect, render_template,
    request, session, url_for,
)
from werkzeug.utils import secure_filename

from config import DB_PATH, UPLOAD_DIR
from models.database import (
    add_live_member,
    add_live_message,
    add_live_recording,
    add_live_signal,
    create_live_room,
    get_live_member,
    get_live_members,
    get_live_messages,
    get_live_recordings,
    get_live_signals_for_user,
    latest_live_signal_id,
    get_live_room_by_token,
    init_db,
    set_live_member_admission,
)
from routes.utils import current_user, login_required
from services.live import finish_live_room, safe_recording_path
from services.notifications import mail_is_configured

live_bp = Blueprint("live", __name__)
logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_AUDIO_EXT = {".webm", ".ogg", ".wav", ".mp3", ".m4a", ".flac"}


def _parse_emails(raw: str) -> List[str]:
    found = []
    seen = set()
    for piece in re.split(r"[\s,;]+", raw or ""):
        email = piece.strip().lower()
        if not _EMAIL_RE.match(email) or email in seen:
            continue
        seen.add(email)
        found.append(email)
    return found


def _join_url(token: str) -> str:
    return url_for("live.room", token=token, _external=True)


def _is_host(room: dict, user_id: int) -> bool:
    return int(room.get("host_user_id") or 0) == int(user_id)


def _normalize_admission(row: dict, host_user_id: int) -> str:
    if int(row.get("user_id") or 0) == int(host_user_id):
        return "admitted"
    status = (row.get("admission_status") or "pending").strip().lower()
    if status not in ("pending", "admitted", "denied"):
        return "pending"
    # Legacy rows were migrated as ``admitted`` without host action.
    if status == "admitted" and not row.get("host_admitted_at"):
        return "pending"
    return status


def _fix_legacy_guest_admission(room_id: int, user_id: int, host_user_id: int,
                                db_path: str) -> None:
    """Persist pending when DB says admitted but the host never admitted them."""
    if int(user_id) == int(host_user_id):
        return
    row = get_live_member(room_id, user_id, db_path)
    if not row:
        return
    if (row.get("admission_status") or "").strip().lower() == "admitted" and not row.get(
        "host_admitted_at"
    ):
        set_live_member_admission(room_id, user_id, "pending", db_path)


def _register_join_request(room: dict, user_id: int, db_path: str) -> str:
    """Ensure the visitor is in ``live_members`` and waiting for host approval."""
    host_id = int(room["host_user_id"])
    is_host = _is_host(room, user_id)
    prior = get_live_member(room["id"], user_id, db_path)
    if prior and prior.get("admission_status") == "denied" and not is_host:
        set_live_member_admission(room["id"], user_id, "pending", db_path)
    add_live_member(
        room["id"],
        user_id,
        session.get("username") or session.get("email") or "Participant",
        session.get("email") or "",
        db_path,
        keep_existing_name=True,
        admission_status="admitted" if is_host else "pending",
    )
    if is_host:
        set_live_member_admission(room["id"], user_id, "admitted", db_path)
    else:
        _fix_legacy_guest_admission(room["id"], user_id, host_id, db_path)
        row = get_live_member(room["id"], user_id, db_path)
        if row and _normalize_admission(row, host_id) != "admitted":
            set_live_member_admission(room["id"], user_id, "pending", db_path)
    return _admission_status(room, user_id, db_path)


def _admission_status(room: dict, user_id: int, db_path: str) -> str:
    if _is_host(room, user_id):
        return "admitted"
    row = get_live_member(int(room["id"]), int(user_id), db_path)
    if not row:
        return "pending"
    return _normalize_admission(row, int(room["host_user_id"]))


def _partition_members(rows: List[dict], host_user_id: int) -> tuple:
    """Split live members into admitted (in meeting) and waiting-room pending."""
    admitted, waiting = [], []
    for row in rows:
        status = _normalize_admission(row, host_user_id)
        if status == "admitted":
            admitted.append(row)
        elif status == "pending":
            waiting.append(row)
    return admitted, waiting


def _require_admitted(room: dict, user_id: int, db_path: str) -> bool:
    return _admission_status(room, user_id, db_path) == "admitted"


def _member_payload(member: dict, recorded_ids: set, session_user_id: int,
                    host_user_id: int) -> dict:
    uid = member["user_id"]
    return {
        "user_id": uid,
        "display_name": member["display_name"],
        "email": member.get("email") or "",
        "recorded": uid in recorded_ids,
        "is_self": uid == session_user_id,
        "is_host": uid == host_user_id,
        "admission_status": _normalize_admission(member, host_user_id),
    }


@live_bp.route("/live/start", methods=["GET", "POST"])
@login_required
def start():
    """Create a meeting room and optionally email the join link."""
    if request.method == "POST":
        title = request.form.get("title", "").strip() or "Live meeting"
        invites = _parse_emails(request.form.get("invites", ""))
        init_db(DB_PATH)
        from secrets import token_urlsafe
        token = token_urlsafe(18)
        try:
            create_live_room(session["user_id"], title, token, DB_PATH)
        except sqlite3.Error as exc:
            logger.exception("Could not create live room")
            flash(f"The meeting could not be created. {exc}", "error")
            return redirect(url_for("live.start"))
        link = _join_url(token)
        if invites:
            from services.notifications import send_invite_emails
            results = send_invite_emails(
                invites, link, title, session.get("username") or "A teammate",
                host_email=session.get("email") or "",
            )
            sent = sum(1 for item in results if item["status"] == "sent")
            failed = next((item for item in results if item["status"] == "failed"), None)
            held = next((item for item in results if item["status"] != "sent"), None)
            if sent:
                flash(f"Join link emailed to {sent} people.", "success")
            if failed:
                flash(
                    failed.get("detail")
                    or "The invite email could not be sent. Copy the link from the meeting page.",
                    "error",
                )
            elif not sent and held:
                flash(
                    held.get("detail")
                    or "Invite email was not delivered. Copy the join link from the meeting page.",
                    "info",
                )
        return redirect(url_for("live.room", token=token))
    return render_template(
        "live/start.html",
        user=current_user(),
        mail_configured=mail_is_configured(),
    )


@live_bp.route("/live/<token>")
@login_required
def room(token: str):
    """Join a live meeting. The visitor's account becomes their speaker name."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found:
        abort(404)
    # "ending" has no meeting yet: the host still needs this page to finish saving.
    if found.get("meeting_id") or found["status"] not in {"live", "ending", "saving"}:
        from models.database import user_can_access_meeting
        if found.get("meeting_id") and user_can_access_meeting(
            int(found["meeting_id"]), session["user_id"], DB_PATH,
        ):
            return redirect(url_for("meeting.result", meeting_id=found["meeting_id"]))
        flash("This meeting has ended. Open Tasks to see what was assigned.", "info")
        return redirect(url_for("tasks.index"))
    is_host = _is_host(found, session["user_id"])
    host_id = int(found["host_user_id"])
    admission = _register_join_request(found, session["user_id"], DB_PATH)
    all_rows = get_live_members(found["id"], DB_PATH)
    if is_host:
        for row in all_rows:
            _fix_legacy_guest_admission(found["id"], int(row["user_id"]), host_id, DB_PATH)
        all_rows = get_live_members(found["id"], DB_PATH)
    members, waiting = _partition_members(all_rows, host_id)
    if not is_host:
        waiting = []
    recordings = get_live_recordings(found["id"], DB_PATH)
    recorded_ids = {row["user_id"] for row in recordings}
    me = get_live_member(found["id"], session["user_id"], DB_PATH) or {}
    in_meeting = admission == "admitted"
    return render_template(
        "live/room.html",
        user=current_user(),
        room=found,
        members=members,
        waiting_members=waiting,
        recorded_ids=recorded_ids,
        my_name=me.get("display_name") or session.get("username") or "",
        already_recorded=session["user_id"] in recorded_ids,
        join_url=_join_url(token),
        is_host=is_host,
        admission_status=admission,
        in_meeting=admission == "admitted",
        mail_configured=mail_is_configured(),
    )


@live_bp.route("/live/<token>/invite", methods=["POST"])
@login_required
def invite(token: str):
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["host_user_id"] != session["user_id"]:
        abort(404)
    if found["status"] != "live":
        flash("This meeting has already ended.", "error")
        return redirect(url_for("live.room", token=token))
    invites = _parse_emails(request.form.get("invites", ""))
    if not invites:
        flash("Enter at least one email address.", "error")
        return redirect(url_for("live.room", token=token))
    from services.notifications import send_invite_emails
    results = send_invite_emails(
        invites,
        _join_url(token),
        found.get("title") or "Live meeting",
        session.get("username") or "A teammate",
        host_email=session.get("email") or "",
    )
    sent = [item["recipient"] for item in results if item["status"] == "sent"]
    failed = next((item for item in results if item["status"] == "failed"), None)
    held = next((item for item in results if item["status"] != "sent"), None)
    if sent:
        flash(f"Link sent to {', '.join(sent)}.", "success")
    if failed:
        flash(
            failed.get("detail")
            or "Email could not be delivered. Copy the join link from this page.",
            "error",
        )
    elif not sent and held:
        flash(
            held.get("detail")
            or "Email could not be delivered. Copy the join link from this page.",
            "info",
        )
    return redirect(url_for("live.room", token=token))


@live_bp.route("/live/<token>/name", methods=["POST"])
@login_required
def set_name(token: str):
    """The name spoken in the meeting stays tied to this account's email."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        abort(404)
    name = re.sub(r"\s+", " ", request.form.get("display_name", "")).strip()
    if not name or len(name) > 80:
        flash("Enter the name people will use for you in this meeting.", "error")
        return redirect(url_for("live.room", token=token))
    add_live_member(
        found["id"],
        session["user_id"],
        name,
        session.get("email") or "",
        DB_PATH,
    )
    flash("Your name is saved for this meeting.", "success")
    return redirect(url_for("live.room", token=token))


@live_bp.route("/live/<token>/recording", methods=["POST"])
@login_required
def recording(token: str):
    """Store the current user's microphone recording for this room."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    # Guests upload when the host has already moved the room to "ending".
    if not found or found["status"] not in {"live", "ending", "saving"} or found.get("meeting_id"):
        return jsonify({"ok": False, "error": "This meeting is not live."}), 404
    if not get_live_member(found["id"], session["user_id"], DB_PATH):
        return jsonify({"ok": False, "error": "Join the meeting before recording."}), 403
    if not _require_admitted(found, session["user_id"], DB_PATH):
        return jsonify({"ok": False, "error": "The host has not admitted you yet."}), 403
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify({"ok": False, "error": "No audio was uploaded."}), 400
    safe_name = secure_filename(audio.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in _AUDIO_EXT:
        mime = (audio.mimetype or "").lower()
        ext = ".ogg" if "ogg" in mime else ".webm"
    from services.preprocessing.remote_transcription import audio_upload_allowed
    from utils.storage import FREE_TIER_AUDIO_MESSAGE, upload_block_reason
    if not audio_upload_allowed():
        return jsonify({"ok": False, "error": FREE_TIER_AUDIO_MESSAGE}), 403
    space_error = upload_block_reason(UPLOAD_DIR, request.content_length or 0)
    if space_error:
        return jsonify({"ok": False, "error": space_error}), 507
    path = safe_recording_path(UPLOAD_DIR, token, session["user_id"], ext)
    audio.save(path)
    role = (request.form.get("role") or "speaker").strip().lower()
    try:
        add_live_recording(
            found["id"], session["user_id"], path, DB_PATH,
            kind="mix" if role == "mix" else "speaker",
        )
    except ValueError as exc:
        if os.path.isfile(path):
            os.remove(path)
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "status": "saved"})


@live_bp.route("/live/<token>/messages")
@login_required
def messages(token: str):
    """Poll chat messages for the live room."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        return jsonify({"ok": False, "error": "Meeting not live."}), 404
    if not _require_admitted(found, session["user_id"], DB_PATH):
        return jsonify({"ok": False, "error": "Not admitted yet."}), 403
    since = request.args.get("since", type=int) or 0
    rows = get_live_messages(found["id"], since_id=since, db_path=DB_PATH)
    return jsonify({"ok": True, "messages": rows})


@live_bp.route("/live/<token>/chat", methods=["POST"])
@login_required
def chat(token: str):
    """Post a chat message in the live room."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        return jsonify({"ok": False, "error": "Meeting not live."}), 404
    me = get_live_member(found["id"], session["user_id"], DB_PATH)
    if not me:
        return jsonify({"ok": False, "error": "Join the meeting first."}), 403
    if not _require_admitted(found, session["user_id"], DB_PATH):
        return jsonify({"ok": False, "error": "Not admitted yet."}), 403
    payload = request.get_json(silent=True) or {}
    text = (payload.get("message") or request.form.get("message") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Message is empty."}), 400
    msg_id = add_live_message(
        found["id"],
        session["user_id"],
        me.get("display_name") or session.get("username") or "Participant",
        text,
        DB_PATH,
    )
    if not msg_id:
        return jsonify({"ok": False, "error": "Message could not be saved."}), 400
    return jsonify({"ok": True, "id": msg_id})


@live_bp.route("/live/<token>/roster")
@login_required
def roster(token: str):
    """Participant list and recording badges for live polling."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found:
        return jsonify({"ok": False}), 404
    if found["status"] not in {"live", "ending"}:
        return jsonify({
            "ok": True,
            "status": found["status"],
            "meeting_id": found.get("meeting_id"),
            "members": [],
        })
    if found["status"] == "ending":
        recordings = get_live_recordings(found["id"], DB_PATH)
        recorded_ids = {
            row["user_id"] for row in recordings
            if (row.get("kind") or "speaker") == "speaker"
        }
        host_id = int(found["host_user_id"])
        admitted_rows, _pending = _partition_members(get_live_members(found["id"], DB_PATH), host_id)
        members = [
            _member_payload(m, recorded_ids, session["user_id"], host_id) for m in admitted_rows
        ]
        return jsonify({
            "ok": True,
            "status": "ending",
            "upload_now": True,
            "members": members,
            "missing": [m["display_name"] for m in members if not m["recorded"]],
        })
    is_host = _is_host(found, session["user_id"])
    admission = _register_join_request(found, session["user_id"], DB_PATH)
    recordings = get_live_recordings(found["id"], DB_PATH)
    recorded_ids = {
        row["user_id"] for row in recordings
        if (row.get("kind") or "speaker") == "speaker"
    }
    host_id = int(found["host_user_id"])
    all_rows = get_live_members(found["id"], DB_PATH)
    if is_host:
        for row in all_rows:
            _fix_legacy_guest_admission(found["id"], int(row["user_id"]), host_id, DB_PATH)
        all_rows = get_live_members(found["id"], DB_PATH)
    admitted_rows, pending_rows = _partition_members(all_rows, host_id)
    members = [
        _member_payload(m, recorded_ids, session["user_id"], host_id) for m in admitted_rows
    ]
    waiting = []
    if is_host:
        waiting = [
            _member_payload(m, recorded_ids, session["user_id"], host_id) for m in pending_rows
        ]
    return jsonify({
        "ok": True,
        "status": "live",
        "title": found.get("title") or "Live meeting",
        "my_admission": admission,
        "members": members if is_host or admission == "admitted" else [],
        "waiting": waiting,
    })


@live_bp.route("/live/<token>/request", methods=["POST"])
@login_required
def request_join(token: str):
    """Guest asks to join; creates/updates their waiting-room row."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        return jsonify({"ok": False, "error": "Meeting not live."}), 404
    if _is_host(found, session["user_id"]):
        return jsonify({"ok": True, "status": "admitted", "registered": True})
    status = _register_join_request(found, session["user_id"], DB_PATH)
    return jsonify({"ok": True, "status": status, "registered": True})


@live_bp.route("/live/<token>/admission")
@login_required
def admission_status(token: str):
    """Poll waiting-room status for the current user."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        return jsonify({"ok": False, "error": "Meeting not live."}), 404
    status = _register_join_request(found, session["user_id"], DB_PATH)
    return jsonify({"ok": True, "status": status, "is_host": _is_host(found, session["user_id"])})


@live_bp.route("/live/<token>/admit", methods=["POST"])
@login_required
def admit(token: str):
    """Host admits someone from the waiting room."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or not _is_host(found, session["user_id"]):
        abort(404)
    user_id = request.form.get("user_id", type=int) or (request.get_json(silent=True) or {}).get("user_id")
    if not user_id:
        return jsonify({"ok": False, "error": "Missing participant."}), 400
    if user_id == session["user_id"]:
        return jsonify({"ok": True})
    if not set_live_member_admission(found["id"], user_id, "admitted", DB_PATH):
        return jsonify({"ok": False, "error": "Participant not found."}), 404
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"ok": True})
    flash("Participant admitted.", "success")
    return redirect(url_for("live.room", token=token))


@live_bp.route("/live/<token>/deny", methods=["POST"])
@login_required
def deny(token: str):
    """Host denies a waiting-room request."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or not _is_host(found, session["user_id"]):
        abort(404)
    user_id = request.form.get("user_id", type=int) or (request.get_json(silent=True) or {}).get("user_id")
    if not user_id:
        return jsonify({"ok": False, "error": "Missing participant."}), 400
    if not set_live_member_admission(found["id"], user_id, "denied", DB_PATH):
        return jsonify({"ok": False, "error": "Participant not found."}), 404
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"ok": True})
    flash("Participant was not admitted.", "info")
    return redirect(url_for("live.room", token=token))


@live_bp.route("/live/<token>/signal", methods=["GET", "POST"])
@login_required
def webrtc_signal(token: str):
    """Poll or post WebRTC signaling messages (offer, answer, ICE)."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        return jsonify({"ok": False, "error": "Meeting not live."}), 404
    if not get_live_member(found["id"], session["user_id"], DB_PATH):
        return jsonify({"ok": False, "error": "Join the meeting first."}), 403
    if not _require_admitted(found, session["user_id"], DB_PATH):
        return jsonify({"ok": False, "error": "Not admitted yet."}), 403

    if request.method == "GET":
        if request.args.get("bootstrap") == "1":
            cursor = latest_live_signal_id(found["id"], session["user_id"], DB_PATH)
            return jsonify({"ok": True, "signals": [], "cursor": cursor})
        since = request.args.get("since", type=int) or 0
        rows = get_live_signals_for_user(
            found["id"], session["user_id"], since_id=since, limit=200, db_path=DB_PATH,
        )
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = row["payload"]
            out.append({
                "id": row["id"],
                "from_user_id": row["from_user_id"],
                "kind": row["kind"],
                "payload": payload,
            })
        return jsonify({"ok": True, "signals": out})

    body = request.get_json(silent=True) or {}
    to_user_id = body.get("to_user_id") or request.form.get("to_user_id", type=int)
    kind = (body.get("kind") or request.form.get("kind") or "").strip().lower()
    if not to_user_id or not kind:
        return jsonify({"ok": False, "error": "Missing signal target or kind."}), 400
    recipient = get_live_member(found["id"], int(to_user_id), DB_PATH)
    if not recipient or _normalize_admission(recipient, found["host_user_id"]) != "admitted":
        return jsonify({"ok": False, "error": "Recipient is not in this meeting."}), 400
    raw_payload = body.get("payload") if "payload" in body else request.form.get("payload", "")
    if isinstance(raw_payload, (dict, list)):
        payload_str = json.dumps(raw_payload)
    else:
        payload_str = raw_payload or ""
    sig_id = add_live_signal(
        found["id"],
        session["user_id"],
        int(to_user_id),
        kind,
        payload_str,
        DB_PATH,
    )
    if not sig_id:
        return jsonify({"ok": False, "error": "Could not store signal."}), 400
    return jsonify({"ok": True, "id": sig_id})


@live_bp.route("/live/<token>/leave", methods=["POST"])
@login_required
def leave(token: str):
    """Record the time the current participant left. The host still ends the meeting."""
    from models.database import mark_live_member_left

    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["status"] != "live":
        return jsonify({"ok": False}), 404
    if not get_live_member(found["id"], session["user_id"], DB_PATH):
        return jsonify({"ok": False}), 404
    mark_live_member_left(found["id"], session["user_id"], DB_PATH)
    return jsonify({"ok": True})


@live_bp.route("/live/<token>/ending", methods=["POST"])
@login_required
def begin_end(token: str):
    """Ask every browser to upload its microphone before the meeting is saved."""
    from models.database import claim_live_room

    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["host_user_id"] != session["user_id"]:
        abort(404)
    if found["status"] == "live":
        claim_live_room(found["id"], DB_PATH)
    elif found["status"] not in {"ending", "ended"}:
        abort(404)
    return jsonify({"ok": True, "status": "ending"})


@live_bp.route("/live/<token>/end", methods=["GET", "POST"])
@login_required
def end(token: str):
    """Host ends the room and starts the normal meeting pipeline."""
    init_db(DB_PATH)
    found = get_live_room_by_token(token, DB_PATH)
    if not found or found["host_user_id"] != session["user_id"]:
        abort(404)
    if request.method == "GET":
        if found.get("meeting_id"):
            return redirect(url_for("meeting.result", meeting_id=found["meeting_id"]))
        if found["status"] in {"live", "ending", "saving"}:
            return redirect(url_for("live.room", token=token))
        abort(404)
    # The End button first posts to /ending, which moves the room to
    # "ending" before this form is submitted. Both states can still be saved.
    if found["status"] not in {"live", "ending"}:
        if found.get("meeting_id"):
            return redirect(url_for("meeting.result", meeting_id=found["meeting_id"]))
        abort(404)
    try:
        meeting_id = finish_live_room(found, db_path=DB_PATH)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("live.room", token=token))
    except Exception as exc:
        flash(f"The meeting could not be processed: {exc}", "error")
        return redirect(url_for("live.room", token=token))
    return redirect(url_for("meeting.result", meeting_id=meeting_id))

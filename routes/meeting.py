"""
routes/meeting.py
-----------------
Meeting upload and result routes.

POST /upload
  1. Save audio file (if provided)
  2. Transcribe   (Whisper)
  3. Summarise    (BART)
  4. Detect tasks (spaCy + regex)
  5. Build knowledge graph (NetworkX)
  6. Persist everything to DB under the authenticated user_id
  7. Redirect to /result/<meeting_id>

GET /result/<meeting_id>
  Render transcript + summary + tasks for one meeting (ownership enforced).
"""

import os
import pathlib
from uuid import uuid4
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, abort, flash, redirect, render_template,
    request, session, url_for,
)

from config import DB_PATH, UPLOAD_DIR, GRAPH_DIR
from models.database import (
    init_db,
    insert_meeting,
    insert_transcript,
    insert_summary,
    insert_tasks,
    get_meeting_detail,
    get_tasks_by_meeting,
    mark_task_complete,
    get_connection,
)
from services.pipeline import run_clario_pipeline
from routes.utils import login_required, current_user

_APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_GRAPHS_DIR = str(_APP_ROOT / "static" / "graphs")

meeting_bp = Blueprint("meeting", __name__)

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
_GRAPH_COLUMNS = (
    "graph_path",
    "bar_chart_path",
    "donut_chart_path",
    "status_chart_path",
)


def _contained_file(root: str, *parts: str):
    """Return a real file path only when every component stays inside root."""
    if not root or not parts:
        return None
    if any(not isinstance(part, str) or part in ("", ".", "..") for part in parts):
        return None
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, *parts))
    try:
        if os.path.commonpath([root_real, candidate]) != root_real:
            return None
    except ValueError:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


def _verified_audio_file(stored_name):
    """Resolve an uploaded audio name only when it is a single file in UPLOAD_DIR."""
    if not isinstance(stored_name, str):
        return None
    name = stored_name.replace("\\", "/").strip()
    if not name or "/" in name or ":" in name:
        return None
    return _contained_file(UPLOAD_DIR, name)


def _verified_graph_file(stored_path):
    """Resolve a chart path only when it is graphs/<file> inside STATIC_GRAPHS_DIR."""
    if not isinstance(stored_path, str):
        return None
    rel = stored_path.replace("\\", "/").strip().lstrip("/")
    parts = rel.split("/")
    if len(parts) != 2 or parts[0] != "graphs":
        return None
    return _contained_file(STATIC_GRAPHS_DIR, parts[1])


def _remove_owned_files(meeting: dict) -> None:
    """Delete audio and chart files recorded on this owned meeting."""
    verified = []
    audio = _verified_audio_file(meeting.get("audio_filename"))
    if audio:
        verified.append(audio)
    for column in _GRAPH_COLUMNS:
        graph = _verified_graph_file(meeting.get(column))
        if graph:
            verified.append(graph)
    for path in dict.fromkeys(verified):
        try:
            os.remove(path)
        except OSError as exc:
            print(f"  Could not remove meeting file: {exc}")


# ── Upload ─────────────────────────────────────────────────────────────────────
@meeting_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        user_id    = session["user_id"]
        meeting_title = request.form.get("meeting_title", "").strip() or "Untitled Meeting"
        text_input = request.form.get("text_input", "").strip()
        audio_file = request.files.get("audio_file")
        audio_path = None
        audio_name = None

        # ── Save audio file ────────────────────────────────────────────────────
        if audio_file and audio_file.filename:
            raw_filename = audio_file.filename
            safe_name = secure_filename(raw_filename)
            ext = os.path.splitext(safe_name)[1].lower()
            if not ext or ext not in ALLOWED_AUDIO:
                flash(
                    f"Unsupported format '{ext}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_AUDIO))}",
                    "error",
                )
                return render_template("meeting/upload.html", user=current_user())

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            # Save as u{user_id}_{uuid4().hex}{ext} so uploads never overwrite each other
            disk_filename = f"u{user_id}_{uuid4().hex}{ext}"
            audio_path = os.path.join(UPLOAD_DIR, disk_filename)
            audio_file.save(audio_path)
            # Keep original filename in meetings.audio_filename for display
            audio_name = raw_filename

        if not audio_path and not text_input:
            flash(
                "Please upload an audio file or paste a meeting transcript.",
                "error",
            )
            return render_template("meeting/upload.html", user=current_user())

        # ── Initialise DB + create meeting row ─────────────────────────────────
        init_db(DB_PATH)
        meeting_id = insert_meeting(
            user_id        = user_id,
            title          = meeting_title,
            audio_filename = audio_name,
            db_path        = DB_PATH,
        )

        # ── Check for prefilled participants from platform import ──────────────
        prefill_participants = session.pop("prefill_participants", None)
        prefill_external_id = session.pop("prefill_external_meeting_id", None)
        prefill_platform = session.pop("prefill_platform", None)

        if prefill_participants:
            from models.database import replace_meeting_participants, update_meeting_platform
            replace_meeting_participants(meeting_id, prefill_participants, DB_PATH)
            if prefill_platform and prefill_external_id:
                update_meeting_platform(meeting_id, prefill_platform, prefill_external_id, DB_PATH)

        # ── Run AI pipeline ────────────────────────────────────────────────────
        try:
            results = run_clario_pipeline(
                audio_path   = audio_path,
                text_input   = text_input or None,
                meeting_id   = meeting_id,
                user_id      = user_id,
                db_path      = DB_PATH,
                graph_dir    = GRAPH_DIR,
                participants = prefill_participants,
            )
        except Exception as exc:
            flash(f"Pipeline error: {exc}", "error")
            return render_template("meeting/upload.html", user=current_user())

        if not results:
            flash("Pipeline returned no output. Check your input.", "error")
            return render_template("meeting/upload.html", user=current_user())

        flash("Meeting processed successfully! 🎉", "success")
        return redirect(url_for("meeting.result", meeting_id=meeting_id))

    return render_template("meeting/upload.html", user=current_user())


# ── Result ─────────────────────────────────────────────────────────────────────
@meeting_bp.route("/result/<int:meeting_id>")
@login_required
def result(meeting_id: int):
    user_id = session["user_id"]

    init_db(DB_PATH)

    # Ownership-enforced fetch: abort(404) prevents meeting ID enumeration
    meeting = get_meeting_detail(meeting_id, user_id, DB_PATH)
    if not meeting:
        abort(404)

    tasks = get_tasks_by_meeting(meeting_id, DB_PATH)
    from models.database import get_meeting_participants
    participants = get_meeting_participants(meeting_id, DB_PATH)

    return render_template(
        "meeting/result.html",
        user         = current_user(),
        meeting      = meeting,
        tasks        = tasks,
        participants = participants,
    )


# ── Mark task complete ──────────────────────────────────────────────────────────────
@meeting_bp.route("/task/<int:task_id>/complete", methods=["POST"])
@login_required
def complete_task(task_id: int):
    """Mark a task as completed (ownership enforced)."""
    user_id = session["user_id"]

    # Verify the task belongs to a meeting owned by the current user
    init_db(DB_PATH)
    conn = get_connection(DB_PATH)
    row = conn.execute("""
        SELECT tk.id, m.id AS meeting_id
        FROM   tasks    tk
        JOIN   meetings m ON m.id = tk.meeting_id
        WHERE  tk.id = ? AND m.user_id = ?
    """, (task_id, user_id)).fetchone()
    conn.close()

    if not row:
        flash("Task not found or access denied.", "error")
        return redirect(url_for("dashboard.index"))

    mark_task_complete(task_id, DB_PATH)

    # ── Regenerate status pie chart to reflect updated pending/completed counts ──
    try:
        import pathlib
        from utils.visualization import generate_status_pie_chart
        from models.database import get_tasks_by_meeting

        meeting_id = row["meeting_id"]
        updated_tasks = get_tasks_by_meeting(meeting_id, DB_PATH)

        app_root = pathlib.Path(__file__).resolve().parent.parent
        status_path_abs = app_root / "static" / "graphs" / f"status_m{meeting_id}.png"
        status_path_abs.parent.mkdir(parents=True, exist_ok=True)

        generate_status_pie_chart(updated_tasks, output_path=str(status_path_abs))
        print(f"  Status pie chart regenerated for meeting {meeting_id}")
    except Exception as exc:
        print(f"  Status chart regeneration failed: {exc}")

    flash("Task marked as completed! ✅", "success")
    return redirect(url_for("meeting.result", meeting_id=row["meeting_id"]))


# ── Delete meeting ─────────────────────────────────────────────────────────────
@meeting_bp.route("/meeting/<int:meeting_id>/delete", methods=["POST"])
@login_required
def delete_meeting(meeting_id: int):
    """Delete one meeting owned by the current user.

    Transcripts, summaries, and tasks are removed by the foreign-key
    ON DELETE CASCADE rules. Audio and chart files are removed only when
    the stored path resolves inside the upload or static graph directory.
    This project has no CSRF extension; the POST is protected by the
    same session gate used for task completion.
    """
    user_id = session["user_id"]
    init_db(DB_PATH)

    meeting = get_meeting_detail(meeting_id, user_id, DB_PATH)
    if not meeting:
        flash("Meeting not found or access denied.", "error")
        return redirect(url_for("dashboard.index"))

    conn = get_connection(DB_PATH)
    try:
        cur = conn.execute(
            "DELETE FROM meetings WHERE id = ? AND user_id = ?",
            (meeting_id, user_id),
        )
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()

    if deleted != 1:
        flash("Meeting not found or access denied.", "error")
        return redirect(url_for("dashboard.index"))

    _remove_owned_files(meeting)
    flash("Meeting deleted.", "success")
    return redirect(url_for("dashboard.index"))

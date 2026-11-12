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

meeting_bp = Blueprint("meeting", __name__)

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


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

        # ── Run AI pipeline ────────────────────────────────────────────────────
        try:
            results = run_clario_pipeline(
                audio_path  = audio_path,
                text_input  = text_input or None,
                meeting_id  = meeting_id,
                user_id     = user_id,
                db_path     = DB_PATH,
                graph_dir   = GRAPH_DIR,
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

    return render_template(
        "meeting/result.html",
        user    = current_user(),
        meeting = meeting,
        tasks   = tasks,
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

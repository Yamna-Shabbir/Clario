"""
routes/tasks.py
---------------
Shared task board. Assignees update their own status. Everyone in the
meeting can see the other tasks and those statuses.
"""

import pathlib

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from config import DB_PATH
from models.database import (
    TASK_STATUSES,
    get_confirmations_for_user,
    get_connection,
    get_pending_confirmations,
    get_tasks_by_meeting,
    get_tasks_visible_to_user,
    init_db,
    resolve_task_confirmation,
    set_assignee_task_status,
)
from routes.utils import current_user, login_required

tasks_bp = Blueprint("tasks", __name__)

_STATUS_LABELS = {
    "pending": "Pending",
    "started": "Started",
    "doing": "Doing",
    "completed": "Completed",
}


def _refresh_status_chart(meeting_id: int) -> None:
    try:
        from utils.visualization import generate_status_pie_chart
        tasks = get_tasks_by_meeting(meeting_id, DB_PATH)
        app_root = pathlib.Path(__file__).resolve().parent.parent
        status_path = app_root / "static" / "graphs" / f"status_m{meeting_id}.png"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        generate_status_pie_chart(tasks, output_path=str(status_path))
    except Exception as exc:
        print(f"  Status chart regeneration failed: {exc}")


def _can_confirm(meeting_id: int, task_id: int, user_id: int, email: str) -> bool:
    pending = get_pending_confirmations(meeting_id, DB_PATH)
    row = next((item for item in pending if item["task_id"] == task_id), None)
    if not row:
        return False
    with get_connection(DB_PATH) as conn:
        meeting = conn.execute(
            "SELECT user_id FROM meetings WHERE id = ?",
            (meeting_id,),
        ).fetchone()
    if not meeting:
        return False
    assigner = (row.get("assigned_by_email") or "").strip().lower()
    return meeting["user_id"] == user_id or assigner == (email or "").strip().lower()


@tasks_bp.route("/tasks")
@login_required
def index():
    init_db(DB_PATH)
    email = session.get("email") or ""
    user_id = session["user_id"]
    tasks = get_tasks_visible_to_user(user_id, email, DB_PATH)
    confirmations = get_confirmations_for_user(user_id, email, DB_PATH)
    return render_template(
        "tasks/index.html",
        user=current_user(),
        tasks=tasks,
        confirmations=confirmations,
        statuses=TASK_STATUSES,
        status_labels=_STATUS_LABELS,
        my_email=email.strip().lower(),
    )


@tasks_bp.route("/tasks/<int:task_id>/status", methods=["POST"])
@login_required
def update_status(task_id: int):
    status = (request.form.get("status") or "").strip().lower()
    meeting_id = set_assignee_task_status(
        task_id, session.get("email") or "", status, DB_PATH,
    )
    if not meeting_id:
        flash("Only the person assigned to that task can change its status.", "error")
        return redirect(request.referrer or url_for("tasks.index"))
    _refresh_status_chart(meeting_id)
    flash(f"Task marked {_STATUS_LABELS.get(status, status)}.", "success")
    if request.form.get("return_to") == "result":
        return redirect(url_for("meeting.result", meeting_id=meeting_id))
    return redirect(url_for("tasks.index"))


@tasks_bp.route("/tasks/<int:task_id>/confirm", methods=["POST"])
@login_required
def confirm(task_id: int):
    """The assigner, or the meeting host, chooses which email a shared name means."""
    init_db(DB_PATH)
    email = (request.form.get("email") or "").strip().lower()
    with get_connection(DB_PATH) as conn:
        task = conn.execute(
            "SELECT meeting_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    if not task or not _can_confirm(task["meeting_id"], task_id, session["user_id"], session.get("email") or ""):
        flash("You cannot confirm that assignment.", "error")
        return redirect(url_for("tasks.index"))
    if not resolve_task_confirmation(task["meeting_id"], task_id, email, DB_PATH):
        flash("Choose one of the people who share that name.", "error")
        return redirect(url_for("tasks.index"))
    remaining = get_pending_confirmations(task["meeting_id"], DB_PATH)
    if not remaining:
        with get_connection(DB_PATH) as conn:
            owner = conn.execute(
                "SELECT user_id FROM meetings WHERE id = ?",
                (task["meeting_id"],),
            ).fetchone()
        owner_id = owner["user_id"] if owner else None
        if owner_id:
            from services.notifications import send_meeting_followups
            send_meeting_followups({
                "stats": {"meeting_id": task["meeting_id"], "user_id": owner_id, "db_path": DB_PATH},
            })
            flash("Assignment saved. Follow-up mail has been sent.", "success")
        else:
            flash("Assignment saved.", "success")
    else:
        flash("Assignment saved. Confirm the remaining shared names to send mail.", "success")
    if request.form.get("return_to") == "result":
        return redirect(url_for("meeting.result", meeting_id=task["meeting_id"]))
    return redirect(url_for("tasks.index"))

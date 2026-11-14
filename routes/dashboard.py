"""
routes/dashboard.py
-------------------
Dashboard — shows the authenticated user's meetings and statistics.
All queries are scoped to session["user_id"].
"""

from flask import render_template, session
from config import DB_PATH
from models.database import (
    get_meetings_by_user, get_pending_tasks,
    get_people_stats, init_db, get_connection,
    get_all_user_tasks
)
from routes.utils import login_required, current_user
from flask import Blueprint

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def index():
    user_id = session["user_id"]

    init_db(DB_PATH)

    meetings      = get_meetings_by_user(user_id, DB_PATH)
    pending_tasks = get_pending_tasks(user_id,    DB_PATH)
    people_stats  = get_people_stats(user_id,     DB_PATH)

    # Count ALL tasks for this user (not just pending) for the headline metric
    conn = get_connection(DB_PATH)
    total_row = conn.execute("""
        SELECT COUNT(*) AS cnt FROM tasks tk
        JOIN meetings m ON m.id = tk.meeting_id
        WHERE m.user_id = ?
    """, (user_id,)).fetchone()
    conn.close()
    total_tasks_count = total_row["cnt"] if total_row else 0

    all_tasks = get_all_user_tasks(user_id, DB_PATH)

    return render_template(
        "dashboard/index.html",
        user           = current_user(),
        meetings       = meetings,
        total_meetings = len(meetings),
        total_tasks    = total_tasks_count,
        pending_count  = len(pending_tasks),
        people_stats   = people_stats,
        all_tasks      = all_tasks,
    )

@dashboard_bp.route("/recordings")
@login_required
def recordings():
    """Saved meeting audio the current account is allowed to hear."""
    from models.database import list_saved_recordings

    init_db(DB_PATH)
    return render_template(
        "dashboard/recordings.html",
        user=current_user(),
        meetings=list_saved_recordings(session["user_id"], DB_PATH),
    )


@dashboard_bp.route("/comparison")
@login_required
def comparison():
    from models.database import get_comparison_stats
    user_id = session["user_id"]
    stats = get_comparison_stats(user_id, DB_PATH)
    
    return render_template(
        "dashboard/comparison.html",
        user=current_user(),
        stats=stats
    )

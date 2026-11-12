# models/__init__.py
# Public API for the models package — re-exports all database functions.

from models.database import (
    # Connection / schema
    init_db,
    get_connection,

    # Users
    insert_user,
    get_user_by_email,
    get_user_by_id,
    email_exists,
    username_exists,

    # Meetings
    insert_meeting,
    set_meeting_graphs,
    get_meetings_by_user,
    get_meeting_detail,

    # Transcripts
    insert_transcript,

    # Summaries
    insert_summary,

    # Tasks
    insert_tasks,
    get_tasks_by_meeting,
    get_pending_tasks,
    get_people_stats,
    mark_task_complete,

    # OAuth Connections
    upsert_oauth_connection,
    get_oauth_connection,
    delete_oauth_connection,

    # Participants
    insert_participants,
    get_meeting_participants,
)

__all__ = [
    "init_db", "get_connection",
    "insert_user", "get_user_by_email", "get_user_by_id",
    "email_exists", "username_exists",
    "insert_meeting", "set_meeting_graphs",
    "get_meetings_by_user", "get_meeting_detail",
    "insert_transcript",
    "insert_summary",
    "insert_tasks", "get_tasks_by_meeting",
    "get_pending_tasks", "get_people_stats", "mark_task_complete",
    "upsert_oauth_connection", "get_oauth_connection", "delete_oauth_connection",
    "insert_participants", "get_meeting_participants",
]

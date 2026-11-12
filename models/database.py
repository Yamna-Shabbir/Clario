"""
models/database.py
------------------
Full SQLite schema and all data-access functions for the Clario project.

Tables
------
  users        – registered accounts
  meetings     – one row per processed meeting, FK → users
  transcripts  – one row per meeting (1-to-1), FK → meetings
  summaries    – one row per meeting (1-to-1), FK → meetings
  tasks        – many rows per meeting,         FK → meetings
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import DB_PATH


# ── Connection ─────────────────────────────────────────────────────────────────

class ClosingConnection(sqlite3.Connection):
    """
    sqlite3.Connection subclass that automatically closes the connection
    when exiting a `with` context manager block (in addition to the default
    commit/rollback behavior).
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a configured SQLite connection with:
      - Row factory so results are dict-like
      - Foreign-key enforcement
      - WAL journal mode for concurrent reads
      - Auto-closing upon context-manager exit
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, factory=ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ── Schema ─────────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            -- ── Users ──────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now'))
            );

            -- ── Meetings ───────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS meetings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER NOT NULL,
                title               TEXT    DEFAULT 'Untitled Meeting',
                sentiment           TEXT    DEFAULT 'Neutral',
                meeting_date        TEXT    DEFAULT (date('now')),
                audio_filename      TEXT,
                graph_path          TEXT,
                bar_chart_path      TEXT,
                donut_chart_path    TEXT,
                status_chart_path   TEXT,
                platform            TEXT,
                external_meeting_id TEXT,
                status              TEXT    NOT NULL DEFAULT 'completed',
                error_message       TEXT,
                progress_stage      TEXT,
                created_at          TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            -- ── Transcripts (1-to-1 with meetings) ────────────────────────
            CREATE TABLE IF NOT EXISTS transcripts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL UNIQUE,
                content    TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Summaries (1-to-1 with meetings) ──────────────────────────
            CREATE TABLE IF NOT EXISTS summaries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL UNIQUE,
                content    TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Tasks (many-to-1 with meetings) ───────────────────────────
            CREATE TABLE IF NOT EXISTS tasks (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id     INTEGER NOT NULL,
                description    TEXT,
                assigned_to    TEXT    DEFAULT 'Team',
                due_date       TEXT    DEFAULT 'Not specified',
                status         TEXT    DEFAULT 'pending',
                keyword        TEXT,
                priority       TEXT    DEFAULT 'Low',
                assigned_email TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── OAuth Connections ─────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS oauth_connections (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                provider          TEXT    NOT NULL,
                access_token_enc  TEXT    NOT NULL,
                refresh_token_enc TEXT,
                expires_at        TEXT,
                scope             TEXT,
                created_at        TEXT    DEFAULT (datetime('now')),
                updated_at        TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, provider)
            );

            -- ── Meeting Participants ──────────────────────────────────────
            CREATE TABLE IF NOT EXISTS meeting_participants (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id  INTEGER NOT NULL,
                name        TEXT    NOT NULL,
                email       TEXT,
                source      TEXT    NOT NULL DEFAULT 'manual',
                external_id TEXT,
                joined_at   TEXT,
                left_at     TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
        
        # Safe migration for existing DBs (guarded by PRAGMA table_info)
        def _get_columns(table_name: str) -> set:
            cur = conn.execute(f"PRAGMA table_info({table_name})")
            return {row["name"] for row in cur.fetchall()}

        meeting_cols = _get_columns("meetings")
        if "title" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN title TEXT DEFAULT 'Untitled Meeting'")
        if "sentiment" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN sentiment TEXT DEFAULT 'Neutral'")
        if "platform" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN platform TEXT")
        if "external_meeting_id" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN external_meeting_id TEXT")
        if "status" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'")
        if "error_message" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN error_message TEXT")
        if "progress_stage" not in meeting_cols:
            conn.execute("ALTER TABLE meetings ADD COLUMN progress_stage TEXT")

        task_cols = _get_columns("tasks")
        if "assigned_email" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN assigned_email TEXT")
            
        conn.commit()
    print(f" DB ready: {db_path}")


# ── Users ──────────────────────────────────────────────────────────────────────

def insert_user(username: str, email: str, password_hash: str,
                db_path: str = DB_PATH) -> int:
    """Insert a new user and return the new row id."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
            (username.strip(), email.strip().lower(), password_hash),
        )
        conn.commit()
        return cur.lastrowid


def get_user_by_email(email: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def email_exists(email: str, db_path: str = DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone() is not None


def username_exists(username: str, db_path: str = DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username.strip(),)
        ).fetchone() is not None


# ── Meetings ───────────────────────────────────────────────────────────────────

def insert_meeting(user_id: int,
                   title: str = "Untitled Meeting",
                   sentiment: str = "Neutral",
                   meeting_date: Optional[str] = None,
                   audio_filename: Optional[str] = None,
                   db_path: str = DB_PATH,
                   *,
                   platform: Optional[str] = None,
                   external_meeting_id: Optional[str] = None,
                   status: str = "completed",
                   error_message: Optional[str] = None,
                   progress_stage: Optional[str] = None) -> int:
    """Insert a meeting row and return its id."""
    if not meeting_date:
        meeting_date = datetime.now().strftime("%Y-%m-%d")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO meetings 
               (user_id, title, sentiment, meeting_date, audio_filename, platform, external_meeting_id, status, error_message, progress_stage) 
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (user_id, title, sentiment, meeting_date, audio_filename, platform, external_meeting_id, status, error_message, progress_stage),
        )
        conn.commit()
        return cur.lastrowid


def update_meeting_status(meeting_id: int, status: str,
                          error_message: Optional[str] = None,
                          progress_stage: Optional[str] = None,
                          db_path: str = DB_PATH) -> None:
    """Update meeting status, error_message, and optional progress_stage."""
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE meetings 
               SET status = ?, 
                   error_message = ?, 
                   progress_stage = COALESCE(?, progress_stage) 
               WHERE id = ?""",
            (status, error_message, progress_stage, meeting_id),
        )
        conn.commit()


def update_meeting_progress(meeting_id: int, progress_stage: str,
                            db_path: str = DB_PATH) -> None:
    """Update meeting progress_stage."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE meetings SET progress_stage = ? WHERE id = ?",
            (progress_stage, meeting_id),
        )
        conn.commit()


def get_meeting_status(meeting_id: int, user_id: int,
                       db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Return status, stage, error for a meeting owned by user_id, or None if not found/unauthorized."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT status, progress_stage, error_message FROM meetings WHERE id = ? AND user_id = ?",
            (meeting_id, user_id),
        ).fetchone()
        if not row:
            return None
        return {
            "status": row["status"] or "completed",
            "stage": row["progress_stage"],
            "error": row["error_message"],
        }


def fail_interrupted_meetings(db_path: str = DB_PATH) -> int:
    """Mark any meetings stuck in 'queued' or 'processing' as failed on startup."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """UPDATE meetings 
               SET status = 'failed', 
                   error_message = 'Interrupted by server restart'
               WHERE status IN ('queued', 'processing')"""
        )
        conn.commit()
        return cur.rowcount


def set_meeting_graphs(meeting_id: int, graph_path: str,
                       bar_path: str, donut_path: str, status_path: str,
                       db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE meetings SET graph_path = ?, bar_chart_path = ?, donut_chart_path = ?, status_chart_path = ? WHERE id = ?",
            (graph_path, bar_path, donut_path, status_path, meeting_id),
        )
        conn.commit()


def update_meeting_sentiment(meeting_id: int, sentiment: str, db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("UPDATE meetings SET sentiment = ? WHERE id = ?", (sentiment, meeting_id))
        conn.commit()


def get_meetings_by_user(user_id: int,
                         db_path: str = DB_PATH) -> List[Dict]:
    """Return all meetings for a user, newest first, with task counts."""
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT  m.id,
                    m.title,
                    m.sentiment,
                    m.platform,
                    m.external_meeting_id,
                    m.meeting_date,
                    m.audio_filename,
                    m.graph_path,
                    m.bar_chart_path,
                    m.donut_chart_path,
                    m.status_chart_path,
                    m.status,
                    m.error_message,
                    m.progress_stage,
                    m.created_at,
                    t.content  AS transcript,
                    s.content  AS summary,
                    COUNT(tk.id) AS task_count
            FROM    meetings    m
            LEFT JOIN transcripts t  ON t.meeting_id = m.id
            LEFT JOIN summaries   s  ON s.meeting_id = m.id
            LEFT JOIN tasks       tk ON tk.meeting_id = m.id
            WHERE   m.user_id = ?
            GROUP BY m.id
            ORDER BY m.created_at DESC
        """, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def get_meeting_detail(meeting_id: int, user_id: int,
                       db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Fetch a single meeting with transcript + summary, enforcing user ownership.
    Returns None if not found or not owned by user_id.
    """
    with get_connection(db_path) as conn:
        row = conn.execute("""
            SELECT  m.*,
                    t.content AS transcript,
                    s.content AS summary
            FROM    meetings    m
            LEFT JOIN transcripts t ON t.meeting_id = m.id
            LEFT JOIN summaries   s ON s.meeting_id = m.id
            WHERE   m.id = ? AND m.user_id = ?
        """, (meeting_id, user_id)).fetchone()
        return dict(row) if row else None


# ── Transcripts ────────────────────────────────────────────────────────────────

def insert_transcript(meeting_id: int, content: str,
                      db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO transcripts (meeting_id, content) VALUES (?,?)",
            (meeting_id, content),
        )
        conn.commit()


# ── Summaries ──────────────────────────────────────────────────────────────────

def insert_summary(meeting_id: int, content: str,
                   db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO summaries (meeting_id, content) VALUES (?,?)",
            (meeting_id, content),
        )
        conn.commit()


# ── Tasks ──────────────────────────────────────────────────────────────────────

def insert_tasks(meeting_id: int, tasks: List[Dict],
                 db_path: str = DB_PATH) -> None:
    """Bulk-insert detected tasks for a meeting."""
    with get_connection(db_path) as conn:
        conn.executemany(
            """INSERT INTO tasks
               (meeting_id, description, assigned_to, due_date, status, keyword, priority, assigned_email)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    meeting_id,
                    t.get("description", ""),
                    t.get("assigned_to",  "Unknown"),
                    t.get("due_date",     "Not specified"),
                    t.get("status",       "pending"),
                    t.get("keyword",      ""),
                    t.get("priority",     "Low"),
                    t.get("assigned_email"),
                )
                for t in tasks
            ],
        )
        conn.commit()


def get_tasks_by_meeting(meeting_id: int,
                         db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE meeting_id = ? ORDER BY id",
            (meeting_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_pending_tasks(user_id: int,
                      db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT tk.* FROM tasks tk
            JOIN meetings m ON m.id = tk.meeting_id
            WHERE tk.status = 'pending' AND m.user_id = ?
        """, (user_id,)).fetchall()
        return [dict(r) for r in rows]

def get_all_user_tasks(user_id: int,
                      db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT tk.* FROM tasks tk
            JOIN meetings m ON m.id = tk.meeting_id
            WHERE m.user_id = ?
            ORDER BY tk.id DESC
        """, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def get_people_stats(user_id: int,
                     db_path: str = DB_PATH) -> List[Tuple[str, int]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT tk.assigned_to, COUNT(*) AS cnt
            FROM   tasks tk
            JOIN   meetings m ON m.id = tk.meeting_id
            WHERE  tk.assigned_to != 'Unknown' AND m.user_id = ?
            GROUP  BY tk.assigned_to
            ORDER  BY cnt DESC
        """, (user_id,)).fetchall()
        return [(r["assigned_to"], r["cnt"]) for r in rows]

def get_comparison_stats(user_id: int, db_path: str = DB_PATH) -> Dict:
    with get_connection(db_path) as conn:
        # Last 10 meetings
        meetings = conn.execute("""
            SELECT id, title, meeting_date, sentiment
            FROM meetings
            WHERE user_id = ?
            ORDER BY created_at ASC
            LIMIT 10
        """, (user_id,)).fetchall()
        
        meeting_ids = [m["id"] for m in meetings]
        meeting_labels = []
        for m in meetings:
            title = m["title"]
            if len(title) > 15:
                title = title[:15] + "..."
            meeting_labels.append(title)
        
        # Task counts per meeting
        tasks_data = []
        for mid in meeting_ids:
            cnt = conn.execute("SELECT COUNT(*) as cnt FROM tasks WHERE meeting_id = ?", (mid,)).fetchone()["cnt"]
            tasks_data.append(cnt)
            
        return {
            "labels": meeting_labels,
            "tasks_data": tasks_data,
            "sentiments": [m["sentiment"] for m in meetings]
        }


def mark_task_complete(task_id: int, db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'completed' WHERE id = ?", (task_id,)
        )
        conn.commit()


# ── OAuth Connections ─────────────────────────────────────────────────────────

def upsert_oauth_connection(
    user_id: int,
    provider: str,
    access_token_enc: str,
    refresh_token_enc: Optional[str] = None,
    expires_at: Optional[str] = None,
    scope: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Insert or update an OAuth connection for a user and provider."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO oauth_connections
               (user_id, provider, access_token_enc, refresh_token_enc, expires_at, scope, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(user_id, provider) DO UPDATE SET
                   access_token_enc = excluded.access_token_enc,
                   refresh_token_enc = COALESCE(excluded.refresh_token_enc, oauth_connections.refresh_token_enc),
                   expires_at = excluded.expires_at,
                   scope = excluded.scope,
                   updated_at = datetime('now')""",
            (user_id, provider.strip().lower(), access_token_enc, refresh_token_enc, expires_at, scope),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM oauth_connections WHERE user_id = ? AND provider = ?",
            (user_id, provider.strip().lower()),
        ).fetchone()
        return row["id"]


def get_oauth_connection(
    user_id: int,
    provider: str,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Return an OAuth connection for a user and provider, or None."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM oauth_connections WHERE user_id = ? AND provider = ?",
            (user_id, provider.strip().lower()),
        ).fetchone()
        return dict(row) if row else None


def delete_oauth_connection(
    user_id: int,
    provider: str,
    db_path: str = DB_PATH,
) -> bool:
    """Delete an OAuth connection for a user and provider. Returns True if deleted."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM oauth_connections WHERE user_id = ? AND provider = ?",
            (user_id, provider.strip().lower()),
        )
        conn.commit()
        return cur.rowcount > 0


# ── Meeting Participants ───────────────────────────────────────────────────────

def insert_participants(
    meeting_id: int,
    participants: List[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> None:
    """Bulk-insert participants for a meeting."""
    if not participants:
        return
    with get_connection(db_path) as conn:
        conn.executemany(
            """INSERT INTO meeting_participants
               (meeting_id, name, email, source, external_id, joined_at, left_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    meeting_id,
                    p.get("name", "Unknown"),
                    p.get("email"),
                    p.get("source", "manual"),
                    p.get("external_id"),
                    p.get("joined_at"),
                    p.get("left_at"),
                )
                for p in participants
            ],
        )
        conn.commit()


def get_meeting_participants(
    meeting_id: int,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Return all participants for a meeting ordered by id."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM meeting_participants WHERE meeting_id = ? ORDER BY id",
            (meeting_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def replace_meeting_participants(
    meeting_id: int,
    participants: List[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> None:
    """Replace all participants for a meeting with a new list (idempotent import)."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM meeting_participants WHERE meeting_id = ?", (meeting_id,))
        if participants:
            conn.executemany(
                """INSERT INTO meeting_participants
                   (meeting_id, name, email, source, external_id, joined_at, left_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        meeting_id,
                        p.get("name", "Unknown"),
                        p.get("email"),
                        p.get("source", "google_meet"),
                        p.get("external_id") or p.get("external_user_id"),
                        p.get("joined_at"),
                        p.get("left_at"),
                    )
                    for p in participants
                ],
            )
        conn.commit()


def update_meeting_platform(
    meeting_id: int,
    platform: str,
    external_meeting_id: str,
    db_path: str = DB_PATH,
) -> None:
    """Update platform and external meeting ID for a meeting."""
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE meetings
               SET platform = ?, external_meeting_id = ?
               WHERE id = ?""",
            (platform, external_meeting_id, meeting_id),
        )
        conn.commit()


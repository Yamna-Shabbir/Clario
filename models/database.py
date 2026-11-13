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

            -- ── Follow-up emails (many-to-1 with meetings) ────────────────
            CREATE TABLE IF NOT EXISTS meeting_emails (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                recipient  TEXT    NOT NULL,
                subject    TEXT,
                status     TEXT    NOT NULL,
                detail     TEXT,
                created_at TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Decisions (many-to-1 with meetings) ───────────────────────
            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id  INTEGER NOT NULL,
                content     TEXT    NOT NULL,
                confidence  REAL    DEFAULT 0.8,
                source_text TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Tasks (many-to-1 with meetings) ───────────────────────────
            CREATE TABLE IF NOT EXISTS tasks (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id        INTEGER NOT NULL,
                description       TEXT,
                assigned_to       TEXT    DEFAULT 'Team',
                due_date          TEXT    DEFAULT 'Not specified',
                status            TEXT    DEFAULT 'pending',
                keyword           TEXT,
                priority          TEXT    DEFAULT 'Low',
                assigned_email    TEXT,
                assigned_by_email TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Who said which part of a live meeting ─────────────────────
            CREATE TABLE IF NOT EXISTS meeting_speaker_turns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id    INTEGER NOT NULL,
                speaker_name  TEXT    NOT NULL,
                speaker_email TEXT,
                content       TEXT    NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Commitments a person should confirm before they are tasks ─
            CREATE TABLE IF NOT EXISTS task_reviews (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id     INTEGER NOT NULL,
                kind           TEXT    NOT NULL,
                description    TEXT,
                assigned_to    TEXT,
                due_date       TEXT,
                source_text    TEXT,
                source_context TEXT,
                evidence_time  TEXT,
                reason         TEXT    NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );

            -- ── Name collisions waiting for the assigner to choose ────────
            CREATE TABLE IF NOT EXISTS task_confirmations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id      INTEGER NOT NULL,
                task_id         INTEGER NOT NULL UNIQUE,
                assignee_name   TEXT    NOT NULL,
                candidates_json TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );

            -- ── Live rooms started inside Clario ──────────────────────────
            CREATE TABLE IF NOT EXISTS live_rooms (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                host_user_id INTEGER NOT NULL,
                title        TEXT    DEFAULT 'Live meeting',
                join_token   TEXT    NOT NULL UNIQUE,
                status       TEXT    NOT NULL DEFAULT 'live',
                meeting_id   INTEGER,
                created_at   TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (host_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS live_members (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id          INTEGER NOT NULL,
                user_id          INTEGER NOT NULL,
                display_name     TEXT    NOT NULL,
                email            TEXT    NOT NULL,
                admission_status TEXT    NOT NULL DEFAULT 'pending',
                joined_at        TEXT    DEFAULT (datetime('now')),
                UNIQUE(room_id, user_id),
                FOREIGN KEY (room_id) REFERENCES live_rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS live_recordings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id    INTEGER NOT NULL,
                user_id    INTEGER NOT NULL,
                audio_path TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (room_id) REFERENCES live_rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS live_messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id      INTEGER NOT NULL,
                user_id      INTEGER NOT NULL,
                display_name TEXT    NOT NULL,
                message      TEXT    NOT NULL,
                created_at   TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (room_id) REFERENCES live_rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS live_signals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id      INTEGER NOT NULL,
                from_user_id INTEGER NOT NULL,
                to_user_id   INTEGER NOT NULL,
                kind         TEXT    NOT NULL,
                payload      TEXT    NOT NULL,
                created_at   TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (room_id) REFERENCES live_rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (from_user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (to_user_id) REFERENCES users(id) ON DELETE CASCADE
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
        if "assigned_by_email" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN assigned_by_email TEXT")
        if "source_text" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN source_text TEXT")
        if "evidence_time" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN evidence_time TEXT")
        if "source_context" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN source_context TEXT")

        member_cols = _get_columns("live_members")
        if "admission_status" not in member_cols:
            conn.execute(
                "ALTER TABLE live_members ADD COLUMN admission_status "
                "TEXT NOT NULL DEFAULT 'admitted'",
            )
        if "host_admitted_at" not in member_cols:
            conn.execute(
                "ALTER TABLE live_members ADD COLUMN host_admitted_at TEXT",
            )

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
               (meeting_id, description, assigned_to, due_date, status, keyword,
                priority, assigned_email, source_text, evidence_time, source_context)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    meeting_id,
                    t.get("description", ""),
                    t.get("assigned_to",  "Unknown"),
                    t.get("due_date_iso") or t.get("due_date", "Not specified"),
                    t.get("status",       "pending"),
                    t.get("keyword",      ""),
                    t.get("priority",     "Low"),
                    t.get("assigned_email"),
                    t.get("source_text") or t.get("description") or "",
                    t.get("evidence_time") or "",
                    t.get("source_context") or "",
                )
                for t in tasks
            ],
        )
        conn.commit()


def insert_decisions(meeting_id: int, decisions: List[Dict],
                     db_path: str = DB_PATH) -> None:
    """Replace stored decisions for a meeting with *decisions*."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM decisions WHERE meeting_id = ?", (meeting_id,))
        conn.executemany(
            """INSERT INTO decisions (meeting_id, content, confidence, source_text)
               VALUES (?,?,?,?)""",
            [
                (
                    meeting_id,
                    d.get("text") or d.get("content") or "",
                    d.get("confidence", 0.8),
                    d.get("source") or d.get("text") or d.get("content") or "",
                )
                for d in decisions
                if (d.get("text") or d.get("content"))
            ],
        )
        conn.commit()


def replace_meeting_emails(meeting_id: int, rows: List[Dict],
                           db_path: str = DB_PATH) -> None:
    """Replace the follow-up email log for one meeting."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM meeting_emails WHERE meeting_id = ?", (meeting_id,))
        conn.executemany(
            """INSERT INTO meeting_emails (meeting_id, recipient, subject, status, detail)
               VALUES (?,?,?,?,?)""",
            [
                (
                    meeting_id,
                    row.get("recipient", ""),
                    row.get("subject", ""),
                    row.get("status", "skipped"),
                    row.get("detail", ""),
                )
                for row in rows
            ],
        )
        conn.commit()


def get_meeting_emails(meeting_id: int, db_path: str = DB_PATH) -> List[Dict]:
    """Return follow-up email attempts for a meeting, oldest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM meeting_emails WHERE meeting_id = ? ORDER BY id",
            (meeting_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_decisions_by_meeting(meeting_id: int,
                             db_path: str = DB_PATH) -> List[Dict]:
    """Return decisions stored for a meeting, in insertion order."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE meeting_id = ? ORDER BY id",
            (meeting_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def replace_task_reviews(meeting_id: int, reviews: List[Dict],
                         db_path: str = DB_PATH) -> None:
    """Replace the human-review list for one meeting."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM task_reviews WHERE meeting_id = ?", (meeting_id,))
        conn.executemany(
            """INSERT INTO task_reviews
               (meeting_id, kind, description, assigned_to, due_date,
                source_text, source_context, evidence_time, reason)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (
                    meeting_id,
                    item.get("kind") or "possible_miss",
                    item.get("description") or "",
                    item.get("assigned_to") or "",
                    item.get("due_date") or "Not specified",
                    item.get("source_text") or item.get("description") or "",
                    item.get("source_context") or "",
                    item.get("evidence_time") or "",
                    item.get("reason") or "Needs review",
                )
                for item in reviews
                if (item.get("description") or item.get("source_text"))
            ],
        )
        conn.commit()


def get_task_reviews(meeting_id: int, db_path: str = DB_PATH) -> List[Dict]:
    """Return review items for a meeting, in insertion order."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM task_reviews WHERE meeting_id = ? ORDER BY id",
            (meeting_id,),
        ).fetchall()
        return [dict(row) for row in rows]


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


TASK_STATUSES = ("pending", "started", "doing", "completed")


def set_assignee_task_status(task_id: int, email: str, status: str,
                             db_path: str = DB_PATH) -> Optional[int]:
    """Set a task status when the email is the assignee. Returns the meeting id."""
    if status not in TASK_STATUSES:
        return None
    cleaned = (email or "").strip().lower()
    if not cleaned:
        return None
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT id, meeting_id FROM tasks
               WHERE id = ? AND lower(assigned_email) = ?""",
            (task_id, cleaned),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (status, task_id),
        )
        conn.commit()
        return row["meeting_id"]


def get_tasks_visible_to_user(user_id: int, email: str,
                              db_path: str = DB_PATH) -> List[Dict]:
    """Tasks from meetings the user owns, joined, or was assigned in.

    Everyone in that meeting can see every task and its status.
    """
    cleaned = (email or "").strip().lower()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT tk.*, m.title AS meeting_title, m.meeting_date, m.user_id AS owner_id
            FROM tasks tk
            JOIN meetings m ON m.id = tk.meeting_id
            WHERE m.id IN (
                SELECT id FROM meetings WHERE user_id = ?
                UNION
                SELECT meeting_id FROM meeting_participants
                WHERE lower(COALESCE(email, '')) = ?
                UNION
                SELECT meeting_id FROM tasks
                WHERE lower(COALESCE(assigned_email, '')) = ?
            )
            ORDER BY tk.id DESC
            """,
            (user_id, cleaned, cleaned),
        ).fetchall()
        return [dict(r) for r in rows]


def insert_speaker_turns(meeting_id: int, turns: List[Dict[str, Any]],
                         db_path: str = DB_PATH) -> None:
    """Replace stored speaker turns for a meeting."""
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM meeting_speaker_turns WHERE meeting_id = ?",
            (meeting_id,),
        )
        conn.executemany(
            """INSERT INTO meeting_speaker_turns
               (meeting_id, speaker_name, speaker_email, content)
               VALUES (?, ?, ?, ?)""",
            [
                (
                    meeting_id,
                    turn.get("speaker_name") or "Speaker",
                    (turn.get("speaker_email") or "").strip().lower(),
                    turn.get("content") or "",
                )
                for turn in turns
                if (turn.get("content") or "").strip()
            ],
        )
        conn.commit()


def get_speaker_turns(meeting_id: int, db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM meeting_speaker_turns
               WHERE meeting_id = ? ORDER BY id""",
            (meeting_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def set_task_assigner(task_id: int, email: str, db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET assigned_by_email = ? WHERE id = ?",
            ((email or "").strip().lower(), task_id),
        )
        conn.commit()


def replace_pending_confirmations(meeting_id: int, rows: List[Dict[str, Any]],
                                  db_path: str = DB_PATH) -> None:
    """Replace the pending name-collision rows for a meeting."""
    import json
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM task_confirmations WHERE meeting_id = ? AND status = 'pending'",
            (meeting_id,),
        )
        for row in rows:
            conn.execute(
                """INSERT INTO task_confirmations
                   (meeting_id, task_id, assignee_name, candidates_json, status)
                   VALUES (?, ?, ?, ?, 'pending')
                   ON CONFLICT(task_id) DO UPDATE SET
                     assignee_name = excluded.assignee_name,
                     candidates_json = excluded.candidates_json,
                     status = 'pending'""",
                (
                    meeting_id,
                    row["task_id"],
                    row.get("assignee_name") or "",
                    json.dumps(row.get("candidates") or []),
                ),
            )
        conn.commit()


def get_pending_confirmations(meeting_id: int, db_path: str = DB_PATH) -> List[Dict]:
    import json
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, tk.description, tk.assigned_by_email
               FROM task_confirmations c
               JOIN tasks tk ON tk.id = c.task_id
               WHERE c.meeting_id = ? AND c.status = 'pending'
               ORDER BY c.id""",
            (meeting_id,),
        ).fetchall()
    parsed = []
    for row in rows:
        item = dict(row)
        try:
            item["candidates"] = json.loads(item.get("candidates_json") or "[]")
        except json.JSONDecodeError:
            item["candidates"] = []
        parsed.append(item)
    return parsed


def get_confirmations_for_user(user_id: int, email: str,
                               db_path: str = DB_PATH) -> List[Dict]:
    """Pending choices for the meeting owner or the person who assigned the task."""
    import json
    cleaned = (email or "").strip().lower()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, tk.description, tk.assigned_by_email, m.title AS meeting_title
               FROM task_confirmations c
               JOIN tasks tk ON tk.id = c.task_id
               JOIN meetings m ON m.id = c.meeting_id
               WHERE c.status = 'pending'
                 AND (m.user_id = ? OR lower(COALESCE(tk.assigned_by_email, '')) = ?)
               ORDER BY c.id""",
            (user_id, cleaned),
        ).fetchall()
    parsed = []
    for row in rows:
        item = dict(row)
        try:
            item["candidates"] = json.loads(item.get("candidates_json") or "[]")
        except json.JSONDecodeError:
            item["candidates"] = []
        parsed.append(item)
    return parsed


def resolve_task_confirmation(meeting_id: int, task_id: int, email: str,
                              db_path: str = DB_PATH) -> bool:
    """Point a task at one of the colliding people. Returns False if not allowed."""
    import json
    cleaned = (email or "").strip().lower()
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT candidates_json FROM task_confirmations
               WHERE meeting_id = ? AND task_id = ? AND status = 'pending'""",
            (meeting_id, task_id),
        ).fetchone()
        if not row:
            return False
        try:
            candidates = json.loads(row["candidates_json"] or "[]")
        except json.JSONDecodeError:
            candidates = []
        chosen = next(
            (c for c in candidates if (c.get("email") or "").strip().lower() == cleaned),
            None,
        )
        if not chosen:
            return False
        conn.execute(
            "UPDATE tasks SET assigned_to = ?, assigned_email = ? WHERE id = ?",
            (chosen.get("name") or cleaned, cleaned, task_id),
        )
        conn.execute(
            """UPDATE task_confirmations SET status = 'resolved'
               WHERE meeting_id = ? AND task_id = ?""",
            (meeting_id, task_id),
        )
        conn.commit()
        return True


def create_live_room(host_user_id: int, title: str, join_token: str,
                     db_path: str = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO live_rooms (host_user_id, title, join_token, status)
               VALUES (?, ?, ?, 'live')""",
            (host_user_id, title or "Live meeting", join_token),
        )
        conn.commit()
        return cur.lastrowid


def get_live_room_by_token(join_token: str, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM live_rooms WHERE join_token = ?",
            (join_token,),
        ).fetchone()
        return dict(row) if row else None


LIVE_ADMISSION_STATUSES = ("pending", "admitted", "denied")


def add_live_member(room_id: int, user_id: int, display_name: str, email: str,
                    db_path: str = DB_PATH, *, keep_existing_name: bool = False,
                    admission_status: str = "pending") -> None:
    """Add a person to a live room (waiting room defaults to ``pending``).

    ``keep_existing_name`` leaves a name the person already saved. The email
    is still refreshed from their account. Re-joining does not reset ``admitted``.
    """
    status = admission_status if admission_status in LIVE_ADMISSION_STATUSES else "pending"
    name_update = "" if keep_existing_name else "display_name = excluded.display_name,"
    with get_connection(db_path) as conn:
        conn.execute(
            f"""INSERT INTO live_members (room_id, user_id, display_name, email, admission_status)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(room_id, user_id) DO UPDATE SET
                 {name_update}
                 email = excluded.email,
                 admission_status = CASE
                   WHEN live_members.host_admitted_at IS NOT NULL THEN 'admitted'
                   WHEN excluded.admission_status = 'denied' THEN 'denied'
                   WHEN excluded.admission_status = 'pending' THEN 'pending'
                   WHEN excluded.admission_status = 'admitted' THEN 'admitted'
                   ELSE live_members.admission_status
                 END""",
            (room_id, user_id, display_name or email, (email or "").strip().lower(), status),
        )
        conn.commit()


def get_live_member(room_id: int, user_id: int, db_path: str = DB_PATH) -> Optional[Dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM live_members WHERE room_id = ? AND user_id = ?",
            (room_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def set_live_member_admission(room_id: int, user_id: int, status: str,
                              db_path: str = DB_PATH) -> bool:
    """Host sets waiting-room status for a participant."""
    if status not in LIVE_ADMISSION_STATUSES:
        return False
    with get_connection(db_path) as conn:
        if status == "admitted":
            cur = conn.execute(
                """UPDATE live_members
                   SET admission_status = ?, host_admitted_at = datetime('now')
                   WHERE room_id = ? AND user_id = ?""",
                (status, room_id, user_id),
            )
        else:
            cur = conn.execute(
                """UPDATE live_members
                   SET admission_status = ?, host_admitted_at = NULL
                   WHERE room_id = ? AND user_id = ?""",
                (status, room_id, user_id),
            )
        conn.commit()
        return cur.rowcount > 0


def get_live_members(room_id: int, db_path: str = DB_PATH, *,
                     admission: Optional[str] = None) -> List[Dict]:
    with get_connection(db_path) as conn:
        if admission and admission in LIVE_ADMISSION_STATUSES:
            rows = conn.execute(
                """SELECT * FROM live_members
                   WHERE room_id = ? AND admission_status = ?
                   ORDER BY id""",
                (room_id, admission),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM live_members WHERE room_id = ? ORDER BY id",
                (room_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def add_live_recording(room_id: int, user_id: int, audio_path: str,
                       db_path: str = DB_PATH) -> None:
    """Keep the newest recording for each person in the room."""
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM live_recordings WHERE room_id = ? AND user_id = ?",
            (room_id, user_id),
        )
        conn.execute(
            """INSERT INTO live_recordings (room_id, user_id, audio_path)
               VALUES (?, ?, ?)""",
            (room_id, user_id, audio_path),
        )
        conn.commit()


def get_live_recordings(room_id: int, db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT r.*, m.display_name, m.email
               FROM live_recordings r
               JOIN live_members m ON m.room_id = r.room_id AND m.user_id = r.user_id
               WHERE r.room_id = ?
               ORDER BY r.id""",
            (room_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_live_message(room_id: int, user_id: int, display_name: str, message: str,
                     db_path: str = DB_PATH) -> int:
    """Append a chat line for a live room."""
    cleaned = (message or "").strip()
    if not cleaned:
        return 0
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO live_messages (room_id, user_id, display_name, message)
               VALUES (?, ?, ?, ?)""",
            (room_id, user_id, display_name or "Participant", cleaned[:2000]),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_live_messages(room_id: int, *, since_id: int = 0,
                      limit: int = 100, db_path: str = DB_PATH) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, user_id, display_name, message, created_at
               FROM live_messages
               WHERE room_id = ? AND id > ?
               ORDER BY id ASC
               LIMIT ?""",
            (room_id, since_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def add_live_signal(room_id: int, from_user_id: int, to_user_id: int, kind: str,
                    payload: str, db_path: str = DB_PATH) -> int:
    """Store a WebRTC signaling message for a participant."""
    import json as _json

    if not kind or not to_user_id:
        return 0
    if not isinstance(payload, str):
        payload = _json.dumps(payload)
    if len(payload) > 65536:
        return 0
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO live_signals (room_id, from_user_id, to_user_id, kind, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (room_id, from_user_id, to_user_id, kind[:32], payload),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_live_signals_for_user(room_id: int, user_id: int, *, since_id: int = 0,
                              limit: int = 80, db_path: str = DB_PATH) -> List[Dict]:
    """Return signaling rows addressed to this user."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, from_user_id, to_user_id, kind, payload, created_at
               FROM live_signals
               WHERE room_id = ? AND to_user_id = ? AND id > ?
               ORDER BY id ASC
               LIMIT ?""",
            (room_id, user_id, since_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def latest_live_signal_id(room_id: int, user_id: int, db_path: str = DB_PATH) -> int:
    """Highest signaling row id already addressed to this participant."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT COALESCE(MAX(id), 0) AS max_id
               FROM live_signals
               WHERE room_id = ? AND to_user_id = ?""",
            (room_id, user_id),
        ).fetchone()
        return int(row["max_id"] if row else 0)


def end_live_room(room_id: int, meeting_id: int, db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE live_rooms
               SET status = 'ended', meeting_id = ?
               WHERE id = ?""",
            (meeting_id, room_id),
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


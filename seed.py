"""
seed.py
-------
Demo-data seeder for Clario.

WARNING: This script truncates ALL rows in the database and regenerates
         all graph/chart PNG files.  It is intentionally destructive and
         should only be run against a fresh development database.

Usage
-----
    python seed.py              # prompts for confirmation
    python seed.py --yes        # skips the prompt (CI / scripted usage)
    python seed.py --yes --force  # also runs when CLARIO_DEBUG is False

Guards
------
- Requires typing 'yes' at the prompt (or passing --yes).
- Refuses to run when CLARIO_DEBUG is False unless --force is supplied.
- Verifies that every path passed to shutil.rmtree is inside the project root.
"""

import argparse
import os
import pathlib
import shutil
import sys

# ── Force UTF-8 on Windows to avoid cp1252 errors with emoji in print() ──────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from werkzeug.security import generate_password_hash

# config is imported after the stdout fix so any logger output is clean.
from config import DB_PATH, UPLOAD_DIR, DEBUG
from models.database import (
    init_db, get_connection, insert_user, insert_meeting,
    insert_transcript, insert_summary, insert_tasks, set_meeting_graphs,
)
from utils.visualization import (
    build_knowledge_graph, generate_assignee_bar_chart,
    generate_priority_donut_chart, generate_status_pie_chart,
)

# ── Project root (absolute) ───────────────────────────────────────────────────
APP_ROOT = pathlib.Path(__file__).resolve().parent
STATIC_GRAPH_DIR = APP_ROOT / "static" / "graphs"


def _safe_rmtree(path: pathlib.Path) -> None:
    """
    Remove *path* recursively, but only if it is inside APP_ROOT.

    Raises
    ------
    RuntimeError if *path* resolves to a location outside the project root.
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(APP_ROOT)
    except ValueError:
        raise RuntimeError(
            f"Refusing to delete '{resolved}': path is outside the project "
            f"root '{APP_ROOT}'. Aborting."
        )
    if resolved.exists():
        shutil.rmtree(resolved)
        print(f"  Removed: {resolved}")
    else:
        print(f"  Skipped (not found): {resolved}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Clario database with demo data (DESTRUCTIVE)."
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow seeding even when CLARIO_DEBUG is False.",
    )
    return parser.parse_args()


def seed() -> None:
    args = _parse_args()

    # ── Production guard ──────────────────────────────────────────────────────
    if not DEBUG and not args.force:
        print(
            "ERROR: CLARIO_DEBUG is False (production mode). "
            "seed.py refuses to run to protect your data.\n"
            "If you really want to seed a production database, re-run with "
            "--force (and make sure you have a backup)."
        )
        sys.exit(1)

    # ── Confirmation prompt ───────────────────────────────────────────────────
    if not args.yes:
        print(
            "\n*** WARNING ***\n"
            "This will DELETE all rows from every database table (users, meetings,\n"
            "transcripts, summaries, tasks) and remove all generated graphs and\n"
            f"uploaded files under:\n"
            f"  {STATIC_GRAPH_DIR}\n"
            f"  {UPLOAD_DIR}\n"
            "\nThis action is IRREVERSIBLE.\n"
        )
        answer = input("Type 'yes' to continue, anything else to abort: ").strip()
        if answer.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\nCleaning up old data...")
    if os.path.exists(DB_PATH):
        try:
            conn = get_connection(DB_PATH)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "decisions" in tables:
                conn.execute("DELETE FROM decisions")
            if "task_confirmations" in tables:
                conn.execute("DELETE FROM task_confirmations")
            if "meeting_speaker_turns" in tables:
                conn.execute("DELETE FROM meeting_speaker_turns")
            if "live_recordings" in tables:
                conn.execute("DELETE FROM live_recordings")
            if "live_members" in tables:
                conn.execute("DELETE FROM live_members")
            if "live_rooms" in tables:
                conn.execute("DELETE FROM live_rooms")
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM summaries")
            conn.execute("DELETE FROM transcripts")
            conn.execute("DELETE FROM meetings")
            conn.execute("DELETE FROM users")
            conn.commit()
            conn.close()
            print("  Database tables truncated.")
        except Exception as e:
            print(f"  Warning: Could not clear database tables: {e}")

    _safe_rmtree(STATIC_GRAPH_DIR)
    STATIC_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    upload_path = pathlib.Path(UPLOAD_DIR).resolve()
    _safe_rmtree(upload_path)
    upload_path.mkdir(parents=True, exist_ok=True)

    # ── Schema ────────────────────────────────────────────────────────────────
    print("\nInitializing database schema...")
    init_db(DB_PATH)

    # ── Demo user ─────────────────────────────────────────────────────────────
    print("\nSeeding Demo User...")
    pwd_hash = generate_password_hash("Demo1234!")
    user_id = insert_user("DemoUser", "demo@clario.ai", pwd_hash)

    # ── Sample meeting ────────────────────────────────────────────────────────
    print("Seeding Sample Meeting 1: Q3 Planning...")
    m1_id = insert_meeting(
        user_id=user_id,
        title="Q3 Strategy Planning",
        sentiment="Positive",
        meeting_date="2026-05-02",
    )
    transcript_1 = (
        "Alright everyone, let's kick off the Q3 planning session. "
        "We have a lot of critical items to cover. "
        "John, you must finalize the Q3 marketing budget immediately, this is urgent. "
        "Sarah, please review the new landing page designs by this week. "
        "Mike will schedule a follow-up sync with the engineering team soon. "
        "Also, we need to make sure the client presentation is ready. "
        "Sarah needs to prepare the slides."
    )
    insert_transcript(m1_id, transcript_1)

    summary_1 = (
        "The team discussed Q3 planning and assigned critical marketing and design tasks. "
        "John is responsible for the budget, while Sarah will handle landing pages and "
        "presentation slides."
    )
    insert_summary(m1_id, summary_1)

    tasks_1 = [
        {
            "id": 1,
            "description": "John, you must finalize the Q3 marketing budget immediately, this is urgent.",
            "assigned_to": "John",
            "due_date": "Not specified",
            "keyword": "must",
            "priority": "High",
            "status": "pending",
        },
        {
            "id": 2,
            "description": "Sarah, please review the new landing page designs by this week.",
            "assigned_to": "Sarah",
            "due_date": "this week",
            "keyword": "please",
            "priority": "Medium",
            "status": "pending",
        },
        {
            "id": 3,
            "description": "Mike will schedule a follow-up sync with the engineering team soon.",
            "assigned_to": "Mike",
            "due_date": "soon",
            "keyword": "will",
            "priority": "Medium",
            "status": "completed",
        },
        {
            "id": 4,
            "description": "Sarah needs to prepare the slides.",
            "assigned_to": "Sarah",
            "due_date": "Not specified",
            "keyword": "needs to",
            "priority": "Low",
            "status": "pending",
        },
    ]
    insert_tasks(m1_id, tasks_1)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("Generating graphs for Meeting 1...")
    t1_graph = [dict(t, id=i + 1) for i, t in enumerate(tasks_1)]
    build_knowledge_graph(t1_graph, str(STATIC_GRAPH_DIR / f"graph_m{m1_id}.png"))
    generate_assignee_bar_chart(t1_graph, str(STATIC_GRAPH_DIR / f"bar_m{m1_id}.png"))
    generate_priority_donut_chart(t1_graph, str(STATIC_GRAPH_DIR / f"donut_m{m1_id}.png"))
    generate_status_pie_chart(t1_graph, str(STATIC_GRAPH_DIR / f"status_m{m1_id}.png"))

    set_meeting_graphs(
        m1_id,
        f"graphs/graph_m{m1_id}.png",
        f"graphs/bar_m{m1_id}.png",
        f"graphs/donut_m{m1_id}.png",
        f"graphs/status_m{m1_id}.png",
    )

    print("\nDatabase seeded successfully!")
    print(f"  Demo login: demo@clario.ai / Demo1234!")


if __name__ == "__main__":
    seed()

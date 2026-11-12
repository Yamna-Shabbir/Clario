"""
tests/test_database.py
----------------------
Unit tests for models/database.py.

Covers:
  - User CRUD (insert, fetch by email, fetch by id, duplicate rejection)
  - Meeting CRUD (insert, update graphs, update sentiment)
  - Transcript and summary (insert + replace semantics)
  - Task CRUD (bulk insert, fetch, mark complete)
  - Ownership scoping: user B cannot read user A's meeting via get_meeting_detail
  - Cascade deletes: deleting a meeting removes its transcript/summary/tasks
  - get_pending_tasks and get_people_stats respect user_id scoping
  - get_comparison_stats structure
"""

import pytest

from models.database import (
    # Schema
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
    update_meeting_sentiment,
    get_meetings_by_user,
    get_meeting_detail,
    # Transcripts / summaries
    insert_transcript,
    insert_summary,
    # Tasks
    insert_tasks,
    get_tasks_by_meeting,
    get_pending_tasks,
    get_all_user_tasks,
    get_people_stats,
    mark_task_complete,
    get_comparison_stats,
)


# ---------------------------------------------------------------------------
# Fixture: fresh in-memory-equivalent temp DB for every test
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_dirs):
    """
    Initialise the schema in a fresh temp SQLite file and return its path.
    ``tmp_dirs`` is provided by conftest.py.
    """
    path = tmp_dirs["db_path"]
    init_db(path)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_user(db, username="alice", email="alice@example.com", pw_hash="hashed"):
    return insert_user(username, email, pw_hash, db_path=db)


def _add_meeting(db, user_id, title="Test Meeting"):
    return insert_meeting(user_id=user_id, title=title, db_path=db)


def _sample_tasks():
    return [
        {
            "description": "Alice will finish the report.",
            "assigned_to": "Alice",
            "due_date":    "Friday",
            "status":      "pending",
            "keyword":     "will",
            "priority":    "Medium",
        },
        {
            "description": "Bob must review the design.",
            "assigned_to": "Bob",
            "due_date":    "Not specified",
            "status":      "pending",
            "keyword":     "must",
            "priority":    "High",
        },
    ]


# ---------------------------------------------------------------------------
# 1. Users
# ---------------------------------------------------------------------------

class TestUsers:
    def test_insert_and_fetch_by_email(self, db):
        uid = _add_user(db)
        user = get_user_by_email("alice@example.com", db_path=db)
        assert user is not None
        assert user["id"] == uid
        assert user["username"] == "alice"

    def test_email_lowercased_on_insert(self, db):
        insert_user("Bob", "BOB@EXAMPLE.COM", "h", db_path=db)
        assert get_user_by_email("bob@example.com", db_path=db) is not None

    def test_fetch_by_id(self, db):
        uid = _add_user(db)
        user = get_user_by_id(uid, db_path=db)
        assert user is not None
        assert user["email"] == "alice@example.com"

    def test_fetch_by_id_missing_returns_none(self, db):
        assert get_user_by_id(9999, db_path=db) is None

    def test_fetch_by_email_missing_returns_none(self, db):
        assert get_user_by_email("nobody@example.com", db_path=db) is None

    def test_email_exists_true(self, db):
        _add_user(db)
        assert email_exists("alice@example.com", db_path=db) is True

    def test_email_exists_false(self, db):
        assert email_exists("ghost@example.com", db_path=db) is False

    def test_username_exists_true(self, db):
        _add_user(db)
        assert username_exists("alice", db_path=db) is True

    def test_username_exists_false(self, db):
        assert username_exists("nobody", db_path=db) is False

    def test_duplicate_email_raises(self, db):
        _add_user(db)
        with pytest.raises(Exception):
            insert_user("alice2", "alice@example.com", "h", db_path=db)

    def test_duplicate_username_raises(self, db):
        _add_user(db)
        with pytest.raises(Exception):
            insert_user("alice", "other@example.com", "h", db_path=db)


# ---------------------------------------------------------------------------
# 2. Meetings
# ---------------------------------------------------------------------------

class TestMeetings:
    def test_insert_returns_int(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        assert isinstance(mid, int)
        assert mid > 0

    def test_get_meetings_by_user_empty(self, db):
        uid = _add_user(db)
        assert get_meetings_by_user(uid, db_path=db) == []

    def test_get_meetings_by_user_returns_own(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid, "My Meeting")
        rows = get_meetings_by_user(uid, db_path=db)
        assert len(rows) == 1
        assert rows[0]["id"] == mid
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["title"] == "My Meeting"

    def test_meetings_isolated_per_user(self, db):
        uid_a = insert_user("a", "a@a.com", "h", db_path=db)
        uid_b = insert_user("b", "b@b.com", "h", db_path=db)
        _add_meeting(db, uid_a)
        assert get_meetings_by_user(uid_b, db_path=db) == []

    def test_set_meeting_graphs(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        set_meeting_graphs(mid, "g.png", "b.png", "d.png", "s.png", db_path=db)
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["graph_path"] == "g.png"
        assert detail["bar_chart_path"] == "b.png"

    def test_update_sentiment(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        update_meeting_sentiment(mid, "Positive", db_path=db)
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["sentiment"] == "Positive"

    def test_get_meeting_detail_ownership_enforced(self, db):
        """get_meeting_detail must return None for a different user_id."""
        uid_a = insert_user("aa", "aa@aa.com", "h", db_path=db)
        uid_b = insert_user("bb", "bb@bb.com", "h", db_path=db)
        mid = _add_meeting(db, uid_a)
        # Owner can read it
        assert get_meeting_detail(mid, uid_a, db_path=db) is not None
        # Other user gets None
        assert get_meeting_detail(mid, uid_b, db_path=db) is None

    def test_get_meeting_detail_missing_id(self, db):
        uid = _add_user(db)
        assert get_meeting_detail(9999, uid, db_path=db) is None


# ---------------------------------------------------------------------------
# 3. Transcripts & Summaries
# ---------------------------------------------------------------------------

class TestTranscriptsAndSummaries:
    def test_insert_and_read_transcript(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_transcript(mid, "Hello world.", db_path=db)
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["transcript"] == "Hello world."

    def test_replace_transcript(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_transcript(mid, "First version.", db_path=db)
        insert_transcript(mid, "Updated version.", db_path=db)
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["transcript"] == "Updated version."

    def test_insert_and_read_summary(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_summary(mid, "A short summary.", db_path=db)
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["summary"] == "A short summary."

    def test_replace_summary(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_summary(mid, "V1 summary.", db_path=db)
        insert_summary(mid, "V2 summary.", db_path=db)
        detail = get_meeting_detail(mid, uid, db_path=db)
        assert detail["summary"] == "V2 summary."


# ---------------------------------------------------------------------------
# 4. Tasks
# ---------------------------------------------------------------------------

class TestTasks:
    def test_insert_and_fetch_tasks(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_tasks(mid, _sample_tasks(), db_path=db)
        tasks = get_tasks_by_meeting(mid, db_path=db)
        assert len(tasks) == 2

    def test_task_fields_stored_correctly(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_tasks(mid, _sample_tasks(), db_path=db)
        tasks = get_tasks_by_meeting(mid, db_path=db)
        alice_task = next(t for t in tasks if t["assigned_to"] == "Alice")
        assert alice_task["priority"] == "Medium"
        assert alice_task["status"] == "pending"
        assert alice_task["keyword"] == "will"

    def test_mark_task_complete(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_tasks(mid, _sample_tasks(), db_path=db)
        tasks = get_tasks_by_meeting(mid, db_path=db)
        task_id = tasks[0]["id"]
        mark_task_complete(task_id, db_path=db)
        tasks_after = get_tasks_by_meeting(mid, db_path=db)
        completed = next(t for t in tasks_after if t["id"] == task_id)
        assert completed["status"] == "completed"

    def test_get_pending_tasks_scoped_to_user(self, db):
        uid_a = insert_user("pa", "pa@a.com", "h", db_path=db)
        uid_b = insert_user("pb", "pb@b.com", "h", db_path=db)
        mid_a = _add_meeting(db, uid_a)
        mid_b = _add_meeting(db, uid_b)
        insert_tasks(mid_a, _sample_tasks(), db_path=db)
        insert_tasks(mid_b, _sample_tasks(), db_path=db)
        pending_a = get_pending_tasks(uid_a, db_path=db)
        pending_b = get_pending_tasks(uid_b, db_path=db)
        assert len(pending_a) == 2
        assert len(pending_b) == 2
        # No cross-contamination
        meeting_ids_a = {t["meeting_id"] for t in pending_a}
        meeting_ids_b = {t["meeting_id"] for t in pending_b}
        assert meeting_ids_a.isdisjoint(meeting_ids_b)

    def test_get_all_user_tasks(self, db):
        uid = _add_user(db)
        mid1 = _add_meeting(db, uid, "M1")
        mid2 = _add_meeting(db, uid, "M2")
        insert_tasks(mid1, _sample_tasks(), db_path=db)
        insert_tasks(mid2, _sample_tasks(), db_path=db)
        all_tasks = get_all_user_tasks(uid, db_path=db)
        assert len(all_tasks) == 4

    def test_get_people_stats(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_tasks(mid, _sample_tasks(), db_path=db)
        stats = get_people_stats(uid, db_path=db)
        names = [s[0] for s in stats]
        assert "Alice" in names
        assert "Bob" in names

    def test_get_people_stats_excludes_unknown(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_tasks(mid, [
            {"description": "Review docs.", "assigned_to": "Unknown",
             "due_date": "N/A", "status": "pending", "keyword": "review", "priority": "Low"},
        ], db_path=db)
        stats = get_people_stats(uid, db_path=db)
        names = [s[0] for s in stats]
        assert "Unknown" not in names


# ---------------------------------------------------------------------------
# 5. Cascade deletes
# ---------------------------------------------------------------------------

class TestCascadeDeletes:
    def test_delete_meeting_cascades_to_transcript(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_transcript(mid, "Some text.", db_path=db)
        # Delete meeting directly
        with get_connection(db) as conn:
            conn.execute("DELETE FROM meetings WHERE id = ?", (mid,))
            conn.commit()
        # Transcript must be gone
        with get_connection(db) as conn:
            row = conn.execute(
                "SELECT * FROM transcripts WHERE meeting_id = ?", (mid,)
            ).fetchone()
        assert row is None

    def test_delete_meeting_cascades_to_tasks(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        insert_tasks(mid, _sample_tasks(), db_path=db)
        with get_connection(db) as conn:
            conn.execute("DELETE FROM meetings WHERE id = ?", (mid,))
            conn.commit()
        tasks = get_tasks_by_meeting(mid, db_path=db)
        assert tasks == []

    def test_delete_user_cascades_to_meetings(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid)
        with get_connection(db) as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (uid,))
            conn.commit()
        with get_connection(db) as conn:
            row = conn.execute(
                "SELECT * FROM meetings WHERE id = ?", (mid,)
            ).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# 6. get_comparison_stats
# ---------------------------------------------------------------------------

class TestComparisonStats:
    def test_returns_correct_shape(self, db):
        uid = _add_user(db)
        for i in range(3):
            mid = _add_meeting(db, uid, f"Meeting {i}")
            insert_tasks(mid, _sample_tasks(), db_path=db)
        stats = get_comparison_stats(uid, db_path=db)
        assert "labels" in stats
        assert "tasks_data" in stats
        assert "sentiments" in stats
        assert len(stats["labels"]) == len(stats["tasks_data"]) == 3

    def test_task_counts_correct(self, db):
        uid = _add_user(db)
        mid = _add_meeting(db, uid, "Solo Meeting")
        insert_tasks(mid, _sample_tasks(), db_path=db)
        stats = get_comparison_stats(uid, db_path=db)
        assert stats["tasks_data"][0] == 2

    def test_empty_user_returns_empty_lists(self, db):
        uid = _add_user(db)
        stats = get_comparison_stats(uid, db_path=db)
        assert stats["labels"] == []
        assert stats["tasks_data"] == []


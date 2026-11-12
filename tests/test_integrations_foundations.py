"""
tests/test_integrations_foundations.py
---------------------------------------
Unit and migration tests for Google Meet / platform integration foundations:
- Fernet encryption and decryption helpers (services/integrations/crypto.py)
- Missing / invalid encryption key failure modes (no secret leakage)
- Idempotent database migrations (init_db) on new and existing databases
- Preserved existing data across schema migrations
- CRUD, constraints, and cascade deletes on oauth_connections & meeting_participants
"""

import os
import gc
import sqlite3
import tempfile
import pytest
from cryptography.fernet import Fernet

import config

from models.database import (
    init_db,
    get_connection,
    insert_user,
    insert_meeting,
    get_meeting_detail,
    insert_tasks,
    get_tasks_by_meeting,
    upsert_oauth_connection,
    get_oauth_connection,
    delete_oauth_connection,
    insert_participants,
    get_meeting_participants,
)
from services.integrations.crypto import (
    encrypt_token,
    decrypt_token,
)


# =============================================================================
# 1. Crypto & Fernet Helper Tests
# =============================================================================

class TestIntegrationsCrypto:
    """Tests for token encryption and decryption."""

    def test_encrypt_decrypt_round_trip_explicit_key(self):
        key = Fernet.generate_key()
        secret_token = "ya29.a0AfH6SMB_SAMPLE_ACCESS_TOKEN_STRING_12345"

        ciphertext = encrypt_token(secret_token, key=key)
        assert ciphertext != secret_token
        assert isinstance(ciphertext, str)

        decrypted = decrypt_token(ciphertext, key=key)
        assert decrypted == secret_token

    def test_encrypt_decrypt_via_env_var(self, monkeypatch):
        key = Fernet.generate_key().decode("utf-8")
        monkeypatch.setenv("CLARIO_TOKEN_ENCRYPTION_KEY", key)

        secret_token = "refresh-token-xyz-987654321"
        ciphertext = encrypt_token(secret_token)
        assert ciphertext != secret_token

        decrypted = decrypt_token(ciphertext)
        assert decrypted == secret_token

    def test_missing_key_raises_secret_free_error(self, monkeypatch):
        monkeypatch.setenv("CLARIO_TOKEN_ENCRYPTION_KEY", "")
        monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", "")

        with pytest.raises(ValueError) as excinfo:
            encrypt_token("some_token", key=None)
        msg = str(excinfo.value)
        assert "Token encryption key is missing" in msg
        assert "some_token" not in msg

        with pytest.raises(ValueError) as excinfo:
            decrypt_token("some_ciphertext", key=None)
        msg = str(excinfo.value)
        assert "Token encryption key is missing" in msg
        assert "some_ciphertext" not in msg

    def test_invalid_key_raises_secret_free_error(self):
        bad_key = "invalid-key-that-is-not-32-base64-bytes"
        with pytest.raises(ValueError) as excinfo:
            encrypt_token("some_token", key=bad_key)
        msg = str(excinfo.value)
        assert "Token encryption key is invalid" in msg
        assert bad_key not in msg
        assert "some_token" not in msg

        with pytest.raises(ValueError) as excinfo:
            decrypt_token("gAAAAABexample...", key=bad_key)
        msg = str(excinfo.value)
        assert "Token encryption key is invalid" in msg
        assert bad_key not in msg

    def test_corrupted_ciphertext_raises_secret_free_error(self):
        key = Fernet.generate_key()
        with pytest.raises(ValueError) as excinfo:
            decrypt_token("corrupted-not-a-fernet-token", key=key)
        msg = str(excinfo.value)
        assert "Token decryption failed" in msg

    def test_invalid_arguments_raise_value_error(self):
        key = Fernet.generate_key()
        with pytest.raises(ValueError) as excinfo:
            encrypt_token(None, key=key)
        assert "Cannot encrypt None" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            decrypt_token("", key=key)
        assert "Cannot decrypt empty" in str(excinfo.value)


# =============================================================================
# 2. Database Schema, Migration & Idempotency Tests
# =============================================================================

class TestIntegrationsSchemaAndMigrations:
    """Tests for schema updates, migrations idempotency, and data preservation."""

    @pytest.fixture()
    def tmp_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_integrations.db")
            yield db_path
            gc.collect()

    def test_init_db_runs_twice_without_error(self, tmp_db):
        """Calling init_db multiple times must be completely idempotent."""
        init_db(tmp_db)
        init_db(tmp_db)

        with get_connection(tmp_db) as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cur.fetchall()}
            assert "oauth_connections" in tables
            assert "meeting_participants" in tables
            assert "meetings" in tables
            assert "tasks" in tables

    def test_existing_database_migration_preserves_data(self, tmp_db):
        """
        Create a legacy schema database missing the new columns/tables,
        insert user data, run init_db(), and verify all data is untouched.
        """
        # 1. Create legacy database without new columns or new tables
        conn = sqlite3.connect(tmp_db)
        try:
            conn.executescript("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT DEFAULT 'Untitled Meeting',
                    sentiment TEXT DEFAULT 'Neutral',
                    meeting_date TEXT DEFAULT (date('now')),
                    audio_filename TEXT,
                    graph_path TEXT,
                    bar_chart_path TEXT,
                    donut_chart_path TEXT,
                    status_chart_path TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL,
                    description TEXT,
                    assigned_to TEXT DEFAULT 'Team',
                    due_date TEXT DEFAULT 'Not specified',
                    status TEXT DEFAULT 'pending',
                    keyword TEXT,
                    priority TEXT DEFAULT 'Low',
                    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                );
            """)
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                ("legacy_user", "legacy@example.com", "hash123"),
            )
            conn.execute(
                "INSERT INTO meetings (user_id, title, sentiment, meeting_date) VALUES (?, ?, ?, ?)",
                (1, "Legacy Budget Meeting", "Positive", "2026-11-01"),
            )
            conn.execute(
                "INSERT INTO tasks (meeting_id, description, assigned_to, due_date) VALUES (?, ?, ?, ?)",
                (1, "Send quarterly report", "Sarah", "Friday"),
            )
            conn.commit()
        finally:
            conn.close()

        # 2. Run the additive migration
        init_db(tmp_db)

        # 3. Verify existing rows remain perfectly intact
        with get_connection(tmp_db) as conn:
            user = conn.execute("SELECT * FROM users WHERE id = 1").fetchone()
            assert user["username"] == "legacy_user"
            assert user["email"] == "legacy@example.com"

            meeting = conn.execute("SELECT * FROM meetings WHERE id = 1").fetchone()
            assert meeting["title"] == "Legacy Budget Meeting"
            assert meeting["sentiment"] == "Positive"
            assert meeting["meeting_date"] == "2026-11-01"
            assert meeting["platform"] is None
            assert meeting["external_meeting_id"] is None

            task = conn.execute("SELECT * FROM tasks WHERE id = 1").fetchone()
            assert task["description"] == "Send quarterly report"
            assert task["assigned_to"] == "Sarah"
            assert task["due_date"] == "Friday"
            assert task["assigned_email"] is None

        # 4. Run migration again (second time) to verify idempotency on existing database
        init_db(tmp_db)

    def test_oauth_connections_crud_and_cascade(self, tmp_db):
        init_db(tmp_db)
        user_id = insert_user("oauth_user", "oauth@example.com", "pw", db_path=tmp_db)

        # Insert connection
        conn_id = upsert_oauth_connection(
            user_id=user_id,
            provider="google",
            access_token_enc="enc_access_token_1",
            refresh_token_enc="enc_refresh_token_1",
            expires_at="2026-11-12T13:30:00Z",
            scope="https://www.googleapis.com/auth/meetings.space.readonly",
            db_path=tmp_db,
        )
        assert conn_id > 0

        # Retrieve connection
        conn_record = get_oauth_connection(user_id, "google", db_path=tmp_db)
        assert conn_record is not None
        assert conn_record["provider"] == "google"
        assert conn_record["access_token_enc"] == "enc_access_token_1"
        assert conn_record["refresh_token_enc"] == "enc_refresh_token_1"
        assert conn_record["scope"] == "https://www.googleapis.com/auth/meetings.space.readonly"

        # Upsert with new access token (retains refresh token if not provided)
        upsert_oauth_connection(
            user_id=user_id,
            provider="google",
            access_token_enc="enc_access_token_updated",
            refresh_token_enc=None,
            expires_at="2026-11-12T14:30:00Z",
            db_path=tmp_db,
        )
        updated = get_oauth_connection(user_id, "google", db_path=tmp_db)
        assert updated["access_token_enc"] == "enc_access_token_updated"
        assert updated["refresh_token_enc"] == "enc_refresh_token_1"

        # Unique constraint on (user_id, provider)
        with get_connection(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM oauth_connections WHERE user_id = ? AND provider = 'google'",
                (user_id,),
            ).fetchone()["cnt"]
            assert count == 1

        # Delete connection
        deleted = delete_oauth_connection(user_id, "google", db_path=tmp_db)
        assert deleted is True
        assert get_oauth_connection(user_id, "google", db_path=tmp_db) is None

        # Re-insert and verify foreign-key cascade delete when user is deleted
        upsert_oauth_connection(
            user_id=user_id,
            provider="google",
            access_token_enc="token_enc",
            db_path=tmp_db,
        )
        with get_connection(tmp_db) as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()

        assert get_oauth_connection(user_id, "google", db_path=tmp_db) is None

    def test_meeting_participants_crud_and_cascade(self, tmp_db):
        init_db(tmp_db)
        user_id = insert_user("part_user", "part@example.com", "pw", db_path=tmp_db)
        meeting_id = insert_meeting(user_id, title="Sync", db_path=tmp_db)

        participants = [
            {
                "name": "Alice Smith",
                "email": "alice@example.com",
                "source": "google_meet",
                "external_id": "spaces/xyz/participants/1",
                "joined_at": "2026-11-12T10:00:00Z",
                "left_at": "2026-11-12T10:45:00Z",
            },
            {
                "name": "Bob Jones",
                "email": "bob@example.com",
                "source": "google_meet",
                "external_id": "spaces/xyz/participants/2",
            },
        ]
        insert_participants(meeting_id, participants, db_path=tmp_db)

        retrieved = get_meeting_participants(meeting_id, db_path=tmp_db)
        assert len(retrieved) == 2
        assert retrieved[0]["name"] == "Alice Smith"
        assert retrieved[0]["email"] == "alice@example.com"
        assert retrieved[0]["source"] == "google_meet"
        assert retrieved[1]["name"] == "Bob Jones"
        assert retrieved[1]["joined_at"] is None

        # Cascade delete on meeting removal
        with get_connection(tmp_db) as conn:
            conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            conn.commit()

        assert get_meeting_participants(meeting_id, db_path=tmp_db) == []

    def test_insert_meeting_platform_and_tasks_assigned_email(self, tmp_db):
        init_db(tmp_db)
        user_id = insert_user("meta_user", "meta@example.com", "pw", db_path=tmp_db)

        meeting_id = insert_meeting(
            user_id=user_id,
            title="Google Meet Standup",
            platform="google",
            external_meeting_id="spaces/aaa-bbb-ccc",
            db_path=tmp_db,
        )

        detail = get_meeting_detail(meeting_id, user_id, db_path=tmp_db)
        assert detail is not None
        assert detail["platform"] == "google"
        assert detail["external_meeting_id"] == "spaces/aaa-bbb-ccc"

        insert_tasks(
            meeting_id,
            [
                {
                    "description": "Submit code review",
                    "assigned_to": "Carol",
                    "assigned_email": "carol@example.com",
                    "due_date": "Tomorrow",
                    "priority": "High",
                }
            ],
            db_path=tmp_db,
        )

        tasks = get_tasks_by_meeting(meeting_id, db_path=tmp_db)
        assert len(tasks) == 1
        assert tasks[0]["assigned_to"] == "Carol"
        assert tasks[0]["assigned_email"] == "carol@example.com"

"""Free-tier download blocks, disk reserve, and non-destructive cleanup."""

import os
import sqlite3
from io import BytesIO

import pytest

from models.database import get_connection, init_db
from utils.storage import (
    FREE_TIER_AUDIO_MESSAGE,
    ModelDownloadBlocked,
    cleanup_safe_files,
    friendly_storage_error,
    upload_block_reason,
)


def test_connection_uses_wal_and_memory_temp(tmp_path):
    db_path = str(tmp_path / "pragmas.db")
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS keepers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO keepers (name) VALUES ('kept')")
    from utils.storage import checkpoint_database
    checkpoint_database(db_path)
    with get_connection(db_path) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM keepers").fetchone()[0] == "kept"


def test_cleanup_removes_only_incomplete_and_pycache(tmp_path):
    project = tmp_path / "proj"
    models = project / "data" / "hf-models" / "org--model"
    models.mkdir(parents=True)
    (models / "model.safetensors").write_bytes(b"weights")
    (models / "config.json").write_text("{}", encoding="utf-8")
    (models / "model.safetensors.incomplete").write_bytes(b"partial")
    database = project / "data" / "database"
    database.mkdir(parents=True)
    db_file = database / "clario.db"
    db_file.write_bytes(b"sqlite-bytes")
    upload = project / "uploads" / "clip.mp3"
    upload.parent.mkdir()
    upload.write_bytes(b"audio")
    chart = project / "static" / "graphs" / "graph_m1.png"
    chart.parent.mkdir(parents=True)
    chart.write_bytes(b"png")
    outbox = project / "data" / "outbox" / "meeting_1" / "person.pdf"
    outbox.parent.mkdir(parents=True)
    outbox.write_bytes(b"pdf")
    pycache = project / "services" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "mod.pyc").write_bytes(b"pyc")
    import utils.storage as storage
    original = storage.tempfile.gettempdir
    storage.tempfile.gettempdir = lambda: str(tmp_path)
    (tmp_path / "clario-partials").mkdir(exist_ok=True)
    (tmp_path / "clario-partials" / "chunk.incomplete").write_bytes(b"x")
    (tmp_path / "clario-partials" / "notes.txt").write_text("keep", encoding="utf-8")
    try:
        removed = cleanup_safe_files(project_root=str(project), models_dir=str(project / "data" / "hf-models"))
    finally:
        storage.tempfile.gettempdir = original

    assert any(path.endswith(".incomplete") for path in removed)
    assert (models / "model.safetensors").read_bytes() == b"weights"
    assert db_file.read_bytes() == b"sqlite-bytes"
    assert upload.read_bytes() == b"audio"
    assert chart.read_bytes() == b"png"
    assert outbox.read_bytes() == b"pdf"
    assert not pycache.exists()
    assert not (models / "model.safetensors.incomplete").exists()
    assert (tmp_path / "clario-partials" / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_disk_full_error_is_explained_without_deleting_rows(tmp_path):
    db_path = str(tmp_path / "full.db")
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES ('a', 'a@example.com', 'x')"
        )
    message = friendly_storage_error(sqlite3.OperationalError("database or disk is full"))
    assert "not deleted" in message
    with get_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_free_tier_blocks_snapshot_download(monkeypatch, tmp_path):
    import services.preprocessing.transcription as transcription

    monkeypatch.setattr("config.FREE_TIER", True)
    monkeypatch.setattr(transcription, "_hf_models_root", lambda: str(tmp_path))
    calls = []

    import sys
    from unittest.mock import MagicMock
    hub = MagicMock()
    hub.snapshot_download.side_effect = lambda *args, **kwargs: calls.append("download")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    with pytest.raises(ModelDownloadBlocked) as exc:
        transcription._ensure_local_hf_model("openai/whisper-small")
    assert "Paste the meeting transcript" in str(exc.value)
    assert calls == []
    assert not (tmp_path / "openai--whisper-small").exists()


def test_existing_weights_skip_the_download_even_on_the_free_plan(monkeypatch, tmp_path):
    import services.preprocessing.transcription as transcription

    folder = tmp_path / "openai--whisper-small"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"ok")
    (folder / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("config.FREE_TIER", True)
    monkeypatch.setattr(transcription, "_hf_models_root", lambda: str(tmp_path))
    assert transcription._ensure_local_hf_model("openai/whisper-small") == str(folder)


def test_paid_plan_blocks_download_when_space_is_low(monkeypatch, tmp_path):
    import services.preprocessing.transcription as transcription

    monkeypatch.setattr("config.FREE_TIER", False)
    monkeypatch.setattr("utils.storage.free_bytes", lambda path: 1024 * 1024)
    monkeypatch.setattr(transcription, "_hf_models_root", lambda: str(tmp_path))
    with pytest.raises(ModelDownloadBlocked) as exc:
        transcription._ensure_local_hf_model("openai/whisper-small")
    assert "Not enough free disk space" in str(exc.value)


def test_paid_plan_downloads_when_space_is_available(monkeypatch, tmp_path):
    import sys
    from unittest.mock import MagicMock
    import services.preprocessing.transcription as transcription

    monkeypatch.setattr("config.FREE_TIER", False)
    monkeypatch.setattr("utils.storage.free_bytes", lambda path: 20 * 1024 ** 3)
    monkeypatch.setattr(transcription, "_hf_models_root", lambda: str(tmp_path))
    hub = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    transcription._ensure_local_hf_model("openai/whisper-small")
    hub.snapshot_download.assert_called_once()


def test_free_tier_summarizer_does_not_download(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    import services.nlp.summarization as summarization

    monkeypatch.setattr(summarization, "_summarizer_unavailable", False)
    monkeypatch.setattr(summarization, "_summarizer_model", None)
    monkeypatch.setattr(summarization, "_summarizer_tokenizer", None)
    monkeypatch.setattr("config.FREE_TIER", True)
    transformers = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    try:
        model, tokenizer = summarization.load_summarizer("sshleifer/distilbart-cnn-12-6")
    finally:
        monkeypatch.setattr(summarization, "_summarizer_unavailable", False)
        monkeypatch.setattr(summarization, "_summarizer_model", None)
        monkeypatch.setattr(summarization, "_summarizer_tokenizer", None)
    assert model is None and tokenizer is None
    transformers.AutoTokenizer.from_pretrained.assert_not_called()


def test_upload_block_reason_names_the_reserve(monkeypatch):
    monkeypatch.setattr("utils.storage.free_bytes", lambda path: 1024)
    monkeypatch.setattr("config.DISK_RESERVE_MB", 64)
    message = upload_block_reason("uploads", incoming_bytes=0)
    assert "64 MB" in message
    assert "not deleted" in message


def test_pasted_transcript_still_runs_on_the_free_plan(logged_in_client, monkeypatch):
    monkeypatch.setattr("config.FREE_TIER", True)
    client, _creds = logged_in_client
    response = client.post(
        "/upload",
        data={"text_input": "Alice will finish the report by Friday.", "meeting_title": "Paste"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Paste" in response.data
    assert b"Audio transcription is not available" not in response.data


def test_free_tier_audio_is_refused_and_creates_no_meeting(logged_in_client, app, monkeypatch):
    monkeypatch.setattr("config.FREE_TIER", True)
    client, _creds = logged_in_client
    response = client.post(
        "/upload",
        data={"audio_file": (BytesIO(b"not-audio"), "standup.mp3"), "meeting_title": "Audio"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Paste the meeting transcript" in response.data
    with get_connection(app.config["CLARIO_DB_PATH"]) as conn:
        meetings = conn.execute(
            "SELECT COUNT(*) FROM meetings WHERE title = 'Audio'"
        ).fetchone()[0]
    assert meetings == 0
    assert FREE_TIER_AUDIO_MESSAGE


def test_upload_rejected_when_the_reserve_would_be_breached(logged_in_client, app, monkeypatch):
    monkeypatch.setattr("utils.storage.free_bytes", lambda path: 1024)
    client, _creds = logged_in_client
    before = _meeting_count(app)
    response = client.post(
        "/upload",
        data={"text_input": "Alice will finish the report by Friday.", "meeting_title": "Full disk"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Not enough free disk space" in response.data
    assert _meeting_count(app) == before


def _meeting_count(app) -> int:
    with get_connection(app.config["CLARIO_DB_PATH"]) as conn:
        return conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]

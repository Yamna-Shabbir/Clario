"""Remote laptop transcription and the local ASR service. No Whisper weights."""

import os
from io import BytesIO
from unittest.mock import MagicMock

import pytest
import requests

from models.database import get_meeting_detail, get_tasks_by_meeting, init_db, insert_meeting, insert_user
from services.preprocessing.local_asr import create_app, installed_asr_model_id
from services.preprocessing.remote_transcription import (
    RemoteTranscriptionError,
    transcribe_remote,
)


TOKEN = "local-transcription-test-token"


def _configure(monkeypatch, url="https://tunnel.example"):
    monkeypatch.setattr("config.LOCAL_TRANSCRIPTION_URL", url)
    monkeypatch.setattr("config.LOCAL_TRANSCRIPTION_TOKEN", TOKEN)


def _response(status, payload):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    return response


def test_remote_transcription_posts_audio_and_returns_text(monkeypatch, tmp_path):
    _configure(monkeypatch)
    audio = tmp_path / "standup.mp3"
    audio.write_bytes(b"not-real-audio")
    seen = {}

    def post(url, files, headers, timeout, allow_redirects):
        seen["url"] = url
        seen["headers"] = headers
        seen["timeout"] = timeout
        seen["redirects"] = allow_redirects
        name, handle, _mime = files["file"]
        seen["name"] = name
        seen["body"] = handle.read()
        return _response(200, {"text": "  Sara will design the logo by Friday.  "})

    monkeypatch.setattr("services.preprocessing.remote_transcription.requests.post", post)
    text = transcribe_remote(str(audio))
    assert text == "Sara will design the logo by Friday."
    assert seen["url"] == "https://tunnel.example/transcribe"
    assert seen["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert seen["timeout"][0] == 15
    assert seen["timeout"][1] > seen["timeout"][0]
    assert seen["redirects"] is False
    assert seen["body"] == b"not-real-audio"
    assert TOKEN not in text


def test_remote_transcription_reports_offline_without_the_token(monkeypatch, tmp_path):
    _configure(monkeypatch)
    audio = tmp_path / "standup.wav"
    audio.write_bytes(b"RIFFxxxx")

    def post(*_args, **_kwargs):
        raise requests.ConnectionError(f"connection failed token={TOKEN}")

    monkeypatch.setattr("services.preprocessing.remote_transcription.requests.post", post)
    with pytest.raises(RemoteTranscriptionError) as caught:
        transcribe_remote(str(audio))
    message = str(caught.value)
    assert "offline" in message
    assert TOKEN not in message


def test_remote_transcription_reports_timeout_without_the_token(monkeypatch, tmp_path):
    _configure(monkeypatch)
    audio = tmp_path / "standup.webm"
    audio.write_bytes(b"webm")

    def post(*_args, **_kwargs):
        raise requests.Timeout(f"read timed out {TOKEN}")

    monkeypatch.setattr("services.preprocessing.remote_transcription.requests.post", post)
    with pytest.raises(RemoteTranscriptionError) as caught:
        transcribe_remote(str(audio))
    message = str(caught.value)
    assert "timed out" in message
    assert TOKEN not in message


def test_remote_transcription_rejects_http_and_token_in_the_url(monkeypatch, tmp_path):
    audio = tmp_path / "standup.mp3"
    audio.write_bytes(b"audio")
    _configure(monkeypatch, url=f"http://tunnel.example/transcribe?access_token={TOKEN}")
    with pytest.raises(RemoteTranscriptionError) as caught:
        transcribe_remote(str(audio))
    assert TOKEN not in str(caught.value)


def test_local_service_rejects_a_bad_token_and_hides_it():
    app = create_app(token=TOKEN, transcribe=lambda _path: "should not run")
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b"audio-bytes"), "meeting.mp3")},
        headers={"Authorization": "Bearer wrong-token-value"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 401
    assert response.get_json()["error"] == "Unauthorized."
    assert TOKEN not in response.get_data(as_text=True)
    assert "wrong-token-value" not in response.get_data(as_text=True)


def test_local_service_rejects_unsupported_audio():
    called = {"count": 0}

    def transcribe(_path):
        called["count"] += 1
        return "nope"

    app = create_app(token=TOKEN, transcribe=transcribe)
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b"not audio"), "notes.exe")},
        headers={"Authorization": f"Bearer {TOKEN}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 415
    assert "Unsupported" in response.get_json()["error"]
    assert called["count"] == 0


def test_local_service_rejects_an_empty_file():
    app = create_app(token=TOKEN, transcribe=lambda _path: "unused")
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b""), "empty.wav")},
        headers={"Authorization": f"Bearer {TOKEN}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"]


def test_local_service_rejects_an_oversized_upload():
    app = create_app(token=TOKEN, transcribe=lambda _path: "unused", max_bytes=8)
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b"123456789"), "big.mp3")},
        headers={"Authorization": f"Bearer {TOKEN}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert TOKEN not in response.get_data(as_text=True)


def test_local_service_returns_a_transcript():
    app = create_app(
        token=TOKEN,
        transcribe=lambda path: f"heard {os.path.basename(path)}",
    )
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b"audio-bytes"), "meeting.m4a")},
        headers={"Authorization": f"Bearer {TOKEN}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["text"].startswith("heard ")
    assert TOKEN not in response.get_data(as_text=True)


def test_local_service_times_out(monkeypatch):
    def hang(_path):
        import time
        time.sleep(1.5)
        return "too late"

    app = create_app(token=TOKEN, transcribe=hang, process_timeout=0.2)
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b"audio-bytes"), "meeting.ogg")},
        headers={"Authorization": f"Bearer {TOKEN}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 504
    assert "timed out" in response.get_json()["error"]


def test_local_service_hides_transcription_failures():
    def explode(_path):
        raise RuntimeError(f"decoder blew up {TOKEN}")

    app = create_app(token=TOKEN, transcribe=explode)
    client = app.test_client()
    response = client.post(
        "/transcribe",
        data={"file": (BytesIO(b"audio-bytes"), "meeting.flac")},
        headers={"Authorization": f"Bearer {TOKEN}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Transcription failed." in body
    assert TOKEN not in body
    assert "decoder" not in body


def test_installed_model_skips_incomplete_checkpoints(monkeypatch, tmp_path):
    sukhan = tmp_path / "Loudness-ai--urdu-sukhan-asr-v1"
    small = tmp_path / "openai--whisper-small"
    sukhan.mkdir()
    small.mkdir()
    (small / "config.json").write_text("{}", encoding="utf-8")
    (small / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("config.HF_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("config.WHISPER_MODEL", "Loudness-ai/urdu-sukhan-asr-v1")
    monkeypatch.setattr("config.WHISPER_FALLBACK", "small")
    assert installed_asr_model_id() == "openai/whisper-small"


def test_manual_transcript_is_used_when_remote_transcription_fails(monkeypatch, tmp_path):
    _configure(monkeypatch)
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"audio")

    def explode(_path):
        raise RemoteTranscriptionError("The local transcription service is offline.")

    monkeypatch.setattr(
        "services.preprocessing.remote_transcription.transcribe_remote",
        explode,
    )
    monkeypatch.setattr(
        "services.pipeline._get_whisper",
        lambda: (_ for _ in ()).throw(AssertionError("in-process Whisper was loaded")),
    )
    db_path = str(tmp_path / "clario.db")
    init_db(db_path)
    user_id = insert_user("ada", "ada@example.com", "hash", db_path=db_path)
    meeting_id = insert_meeting(user_id, title="Fallback", db_path=db_path)
    from services.pipeline import run_clario_pipeline

    pasted = "Sara will design the logo by Friday."
    result = run_clario_pipeline(
        audio_path=str(audio),
        text_input=pasted,
        meeting_id=meeting_id,
        user_id=user_id,
        db_path=db_path,
        graph_dir=str(tmp_path / "graphs"),
    )
    assert pasted in result["transcript"]
    stored = get_meeting_detail(meeting_id, user_id, db_path)
    assert pasted in stored["transcript"]
    assert get_tasks_by_meeting(meeting_id, db_path)


def test_pipeline_stores_a_remote_transcript(monkeypatch, tmp_path):
    _configure(monkeypatch)
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"audio")
    spoken = "Sara will design the logo by Friday."

    def remote(path):
        assert path == str(audio)
        return spoken

    monkeypatch.setattr(
        "services.preprocessing.remote_transcription.transcribe_remote",
        remote,
    )
    monkeypatch.setattr(
        "services.pipeline._get_whisper",
        lambda: (_ for _ in ()).throw(AssertionError("in-process Whisper was loaded")),
    )
    db_path = str(tmp_path / "clario.db")
    init_db(db_path)
    user_id = insert_user("ada", "ada@example.com", "hash", db_path=db_path)
    meeting_id = insert_meeting(user_id, title="Remote", db_path=db_path)
    from services.pipeline import run_clario_pipeline

    result = run_clario_pipeline(
        audio_path=str(audio),
        meeting_id=meeting_id,
        user_id=user_id,
        db_path=db_path,
        graph_dir=str(tmp_path / "graphs"),
    )
    assert spoken in result["transcript"]
    stored = get_meeting_detail(meeting_id, user_id, db_path)
    assert spoken in stored["transcript"]
    tasks = get_tasks_by_meeting(meeting_id, db_path)
    assert tasks
    assert any("logo" in (task["description"] or "").lower() for task in tasks)


def test_free_tier_upload_uses_remote_transcription(logged_in_client, monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr("config.FREE_TIER", True)
    spoken = "Bob must review the design documents by Friday."

    def remote(_path):
        return spoken

    monkeypatch.setattr(
        "services.preprocessing.remote_transcription.transcribe_remote",
        remote,
    )
    client, _creds = logged_in_client
    response = client.post(
        "/upload",
        data={
            "audio_file": (BytesIO(b"not-real-audio"), "standup.mp3"),
            "meeting_title": "Hybrid",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    page = client.get(response.headers["Location"])
    assert page.status_code == 200
    assert b"Bob must review the design documents by Friday." in page.data
    assert b"Audio transcription is not available" not in page.data


def test_pasted_transcript_still_works_when_remote_transcription_is_configured(
    logged_in_client, monkeypatch,
):
    _configure(monkeypatch)
    monkeypatch.setattr("config.FREE_TIER", True)

    def remote(_path):
        raise AssertionError("pasted text should not call the laptop")

    monkeypatch.setattr(
        "services.preprocessing.remote_transcription.transcribe_remote",
        remote,
    )
    client, _creds = logged_in_client
    response = client.post(
        "/upload",
        data={
            "text_input": "Alice will finish the report by Friday.",
            "meeting_title": "Paste",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Alice will finish the report by Friday." in response.data


def test_free_tier_still_refuses_audio_without_a_remote_service(logged_in_client, monkeypatch):
    monkeypatch.setattr("config.FREE_TIER", True)
    monkeypatch.setattr("config.LOCAL_TRANSCRIPTION_URL", "")
    monkeypatch.setattr("config.LOCAL_TRANSCRIPTION_TOKEN", "")
    client, _creds = logged_in_client
    response = client.post(
        "/upload",
        data={"audio_file": (BytesIO(b"not-audio"), "standup.mp3"), "meeting_title": "Blocked"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert b"Paste the meeting transcript" in response.data

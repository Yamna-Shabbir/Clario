"""
Laptop transcription HTTP service.

Loads only the Whisper checkpoint already stored under CLARIO_HF_MODELS_DIR.
It does not import the meeting pipeline, spaCy, or the summarizer, and it
refuses to download weights.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, Optional

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from services.preprocessing.remote_transcription import PROCESS_TIMEOUT_SECONDS

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
MIN_TOKEN_LENGTH = 16

TranscribeFn = Callable[[str], str]


def tokens_match(provided: str, expected: str) -> bool:
    """Compare two secrets without leaking the expected value."""
    left = hashlib.sha256((provided or "").encode("utf-8")).digest()
    right = hashlib.sha256((expected or "").encode("utf-8")).digest()
    return hmac.compare_digest(left, right)


def bearer_token(header: str) -> str:
    """Return the bearer value, or an empty string when the scheme is wrong."""
    scheme, _, value = (header or "").partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return value.strip()


def installed_asr_model_id() -> str:
    """
    Pick a speech checkpoint that already has weights on disk.

    Preference is the configured Whisper model, then the configured fallback.
    Missing files are skipped. Nothing is downloaded.
    """
    from config import WHISPER_FALLBACK, WHISPER_MODEL
    from services.preprocessing.transcription import _hf_model_dir, _public_whisper_id

    ordered = []
    configured = (WHISPER_MODEL or "").strip()
    if configured:
        ordered.append(configured)
    fallback = (WHISPER_FALLBACK or "").strip()
    if fallback:
        ordered.append(_public_whisper_id(fallback))

    seen = set()
    for model_id in ordered:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        folder = _hf_model_dir(model_id)
        weights = os.path.join(folder, "model.safetensors")
        config_path = os.path.join(folder, "config.json")
        if os.path.isfile(weights) and os.path.isfile(config_path):
            return model_id
    raise RuntimeError(
        "No local Whisper weights were found. "
        "The transcription service will not download a model."
    )


def transcribe_with_installed_model(audio_path: str) -> str:
    """Transcribe with the on-disk Whisper model. Does not download weights."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from services.preprocessing.transcription import load_whisper, transcribe_audio

    model_id = installed_asr_model_id()
    model = load_whisper(model_id)
    return transcribe_audio(audio_path, model=model, model_size=model_id)


def create_app(
    *,
    token: str,
    transcribe: Optional[TranscribeFn] = None,
    max_bytes: Optional[int] = None,
    process_timeout: float = PROCESS_TIMEOUT_SECONDS,
) -> Flask:
    """
    Build the local transcription app.

    ``transcribe`` defaults to the installed Whisper model. Tests pass a fake.
    The app listens only when ``scripts/local_transcription_server.py`` runs it,
    and that entry point binds to localhost.
    """
    secret = (token or "").strip()
    if len(secret) < MIN_TOKEN_LENGTH:
        raise ValueError(
            f"CLARIO_LOCAL_TRANSCRIPTION_TOKEN must be at least {MIN_TOKEN_LENGTH} characters."
        )
    limit = max_bytes
    if limit is None:
        from config import MAX_CONTENT_LENGTH
        limit = MAX_CONTENT_LENGTH
    runner = transcribe or transcribe_with_installed_model
    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clario-local-asr")
    timeout = float(process_timeout) if process_timeout and process_timeout > 0 else float(
        PROCESS_TIMEOUT_SECONDS
    )

    app = Flask("clario_local_asr")
    app.config["MAX_CONTENT_LENGTH"] = max(1, int(limit))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({"error": "The recording is too large."}), 413

    @app.post("/transcribe")
    def transcribe_route():
        provided = bearer_token(request.headers.get("Authorization", ""))
        if not tokens_match(provided, secret):
            return jsonify({"error": "Unauthorized."}), 401

        upload = request.files.get("file") or request.files.get("audio")
        if upload is None or not upload.filename:
            return jsonify({"error": "No audio file was uploaded."}), 400
        safe_name = secure_filename(upload.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in ALLOWED_AUDIO:
            return jsonify({"error": "Unsupported audio format."}), 415

        if not lock.acquire(blocking=False):
            return jsonify({"error": "A transcription is already running."}), 503

        handle = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        path = handle.name
        handed_off = False
        try:
            upload.save(handle)
            handle.close()
            size = os.path.getsize(path)
            if size <= 0:
                return jsonify({"error": "The audio file is empty."}), 400
            if size > app.config["MAX_CONTENT_LENGTH"]:
                return jsonify({"error": "The recording is too large."}), 413

            def _work() -> str:
                try:
                    return runner(path)
                finally:
                    lock.release()
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            future = executor.submit(_work)
            handed_off = True
            try:
                text = future.result(timeout=timeout)
            except FuturesTimeout:
                return jsonify({"error": "Transcription timed out."}), 504
            except Exception:
                return jsonify({"error": "Transcription failed."}), 500
            if not isinstance(text, str):
                return jsonify({"error": "Transcription failed."}), 500
            return jsonify({"text": text.strip()})
        finally:
            try:
                handle.close()
            except OSError:
                pass
            if not handed_off:
                lock.release()
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    return app

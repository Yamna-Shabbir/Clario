"""
Forward meeting audio to a laptop transcription service.

The service is expected to sit behind a tunnel you start yourself
(Cloudflare Tunnel or ngrok). This module never logs the shared token
and never places it in a URL.
"""

from __future__ import annotations

import os
from typing import Tuple
from urllib.parse import urlparse

import requests

# The laptop may spend several minutes on one recording. The read timeout
# is longer than the service's own processing limit so a 504 can come back
# before the client gives up.
CONNECT_TIMEOUT_SECONDS = 15
PROCESS_TIMEOUT_SECONDS = 900
READ_TIMEOUT_SECONDS = 960


class RemoteTranscriptionError(RuntimeError):
    """User-facing failure that does not include the shared token."""


def remote_transcription_configured() -> bool:
    """True only when both the tunnel URL and the shared token are set."""
    url, token = _credentials()
    return bool(url and token)


def audio_upload_allowed() -> bool:
    """
    Free hosting refuses audio unless a laptop service is configured.

    Local development (free tier off) keeps loading Whisper in-process.
    A configured remote service does not download weights on the server.
    """
    import config

    if not config.FREE_TIER:
        return True
    return remote_transcription_configured()


def transcribe_remote(audio_path: str) -> str:
    """
    POST one audio file and return the transcript text.

    Raises
    ------
    RemoteTranscriptionError
        The service is offline, slow, unauthorized, or returned an error.
        The message is safe to store and show; it does not contain the token.
    FileNotFoundError
        ``audio_path`` is missing.
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    url, token = _credentials()
    if not url or not token:
        raise RemoteTranscriptionError(
            "Local transcription is not configured. Paste the meeting transcript instead."
        )
    endpoint = _endpoint(url)
    filename = os.path.basename(audio_path)
    try:
        with open(audio_path, "rb") as handle:
            response = requests.post(
                endpoint,
                files={"file": (filename, handle, "application/octet-stream")},
                headers={"Authorization": f"Bearer {token}"},
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=False,
            )
    except requests.Timeout:
        raise RemoteTranscriptionError(
            "The local transcription service timed out. "
            "Paste the meeting transcript, or try a shorter recording."
        ) from None
    except requests.ConnectionError:
        raise RemoteTranscriptionError(
            "The local transcription service is offline. "
            "Paste the meeting transcript, or start the laptop service and its secure tunnel."
        ) from None
    except requests.RequestException:
        raise RemoteTranscriptionError(
            "The local transcription service could not be reached. "
            "Paste the meeting transcript instead."
        ) from None

    return _transcript_from_response(response)


def _credentials() -> Tuple[str, str]:
    import config

    url = (getattr(config, "LOCAL_TRANSCRIPTION_URL", "") or "").strip()
    token = (getattr(config, "LOCAL_TRANSCRIPTION_TOKEN", "") or "").strip()
    return url, token


def _endpoint(url: str) -> str:
    """Return the transcribe URL, rejecting tokens stashed in the address."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme not in {"https", "http"} or not host:
        raise RemoteTranscriptionError("The local transcription URL is invalid.")
    if parsed.scheme == "http" and not loopback:
        raise RemoteTranscriptionError(
            "The local transcription URL must use HTTPS except on localhost."
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RemoteTranscriptionError(
            "The local transcription URL must not include a token, query, or password."
        )
    path = (parsed.path or "").rstrip("/")
    if not path.endswith("/transcribe"):
        path = f"{path}/transcribe"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _transcript_from_response(response: requests.Response) -> str:
    status = response.status_code
    if status in {301, 302, 303, 307, 308}:
        raise RemoteTranscriptionError(
            "The local transcription service returned a redirect. The audio was not sent on."
        )
    if status == 401:
        raise RemoteTranscriptionError(
            "The local transcription service rejected the request."
        )
    if status == 413:
        raise RemoteTranscriptionError(
            "The recording is too large for the local transcription service."
        )
    if status == 415:
        raise RemoteTranscriptionError(
            "The local transcription service does not support this audio format."
        )
    if status == 504:
        raise RemoteTranscriptionError(
            "The local transcription service timed out. "
            "Paste the meeting transcript, or try a shorter recording."
        )
    payload = _json_object(response)
    if status == 200:
        text = payload.get("text")
        if not isinstance(text, str):
            raise RemoteTranscriptionError(
                "The local transcription service returned an unexpected response."
            )
        return text.strip()
    raise RemoteTranscriptionError(
        "The local transcription service could not transcribe this recording. "
        "Paste the meeting transcript instead."
    )


def _json_object(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}

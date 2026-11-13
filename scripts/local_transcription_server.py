"""
Run Whisper on this computer for a Clario server that cannot load it.

Start the service, then publish localhost with a tunnel (Cloudflare Tunnel
or ngrok). Set the same token on this process and on Render:

    CLARIO_LOCAL_TRANSCRIPTION_TOKEN   shared secret (16+ characters)
    CLARIO_LOCAL_TRANSCRIPTION_URL     tunnel origin, on Render only

The process binds to 127.0.0.1. It loads the Whisper checkpoint already in
data/hf-models and does not download another copy. It does not load spaCy
or the summarizer.

    python scripts/local_transcription_server.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    token = os.environ.get("CLARIO_LOCAL_TRANSCRIPTION_TOKEN", "").strip()
    host = os.environ.get("CLARIO_LOCAL_TRANSCRIPTION_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    if host not in {"127.0.0.1", "::1"}:
        print(
            "Refusing to listen outside localhost. "
            "Publish this service with a secure tunnel instead.",
            file=sys.stderr,
        )
        return 1
    try:
        port = int(os.environ.get("CLARIO_LOCAL_TRANSCRIPTION_PORT", "8765"))
    except ValueError:
        print("CLARIO_LOCAL_TRANSCRIPTION_PORT must be a port number.", file=sys.stderr)
        return 1

    from services.preprocessing.local_asr import (
        create_app,
        installed_asr_model_id,
    )
    from services.preprocessing.transcription import load_whisper

    try:
        model_id = installed_asr_model_id()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Using installed ASR model {model_id}. Weights will not be downloaded.")
    load_whisper(model_id)
    try:
        app = create_app(token=token)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Local transcription listening on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

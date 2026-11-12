"""
services/transcription.py
--------------------------
Whisper-based audio-to-text transcription service.
Extracted from Clario Project notebook (cells 105, 107, 113-114).

FFmpeg note
-----------
Whisper requires FFmpeg to decode MP3/M4A/etc.  On systems where FFmpeg is not
on the system PATH (common on Windows), we fall back to the binary bundled with
``imageio-ffmpeg``.  The helper ``_ensure_ffmpeg_on_path()`` is called once at
module import time, so it is transparent to all callers.
"""

import os
import sys
from typing import Any, Dict, Optional

# Module-level model cache (loaded once per process)
_whisper_model = None


def _ensure_ffmpeg_on_path() -> None:
    """
    Make sure an ffmpeg executable is reachable on PATH.

    Strategy (in order):
    1. If 'ffmpeg' is already on PATH, do nothing.
    2. Try the binary bundled with imageio-ffmpeg (already installed as a
       transitive dep).  Prepend its containing directory to PATH.
    3. If neither is available, log a warning — Whisper will raise a
       descriptive error at transcription time.
    """
    import shutil

    if shutil.which("ffmpeg"):
        return  # already available

    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[transcription] FFmpeg resolved via imageio-ffmpeg: {ffmpeg_exe}")
    except Exception as exc:
        print(
            f"[transcription] WARNING: FFmpeg not found on PATH and imageio-ffmpeg "
            f"fallback failed ({exc}).  Audio transcription will likely fail.\n"
            "  Fix: install FFmpeg (https://ffmpeg.org/download.html) and add it to PATH.",
            file=sys.stderr,
        )


# Run once at import time so every call to transcribe_audio benefits.
_ensure_ffmpeg_on_path()


def _hf_models_root() -> str:
    """Project-drive folder for speech weights (avoids the small system cache)."""
    import config as cfg

    root = getattr(cfg, "HF_MODELS_DIR", None) or os.path.join(cfg.DATA_DIR, "hf-models")
    return root


def _whisper_download_root() -> str:
    root = os.path.join(_hf_models_root(), "whisper")
    os.makedirs(root, exist_ok=True)
    return root


def _load_openai_whisper(model_size: str):
    from services.native_compat import install
    install()
    import whisper
    root = _whisper_download_root()
    print(f"Loading Whisper '{model_size}' model from {root} ...")
    model = whisper.load_model(model_size, download_root=root)
    print("Whisper model loaded.")
    return model


_PUBLIC_WHISPER = {
    "tiny": "openai/whisper-tiny",
    "tiny.en": "openai/whisper-tiny.en",
    "base": "openai/whisper-base",
    "base.en": "openai/whisper-base.en",
    "small": "openai/whisper-small",
    "small.en": "openai/whisper-small.en",
    "medium": "openai/whisper-medium",
    "medium.en": "openai/whisper-medium.en",
    "large": "openai/whisper-large-v3",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo",
}


def _public_whisper_id(model_size: str) -> str:
    """Map a Whisper size name to the public Hugging Face checkpoint."""
    name = (model_size or "").strip()
    if "/" in name:
        return name
    return _PUBLIC_WHISPER.get(name, f"openai/whisper-{name}")


def _is_model_access_error(exc: BaseException) -> bool:
    """True when Hugging Face refused the gated speech model."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        piece in text
        for piece in (
            "gatedrepo",
            "403",
            "authorized list",
            "access to model",
            "cannot access gated",
        )
    )


def _hf_model_dir(model_id: str) -> str:
    return os.path.join(_hf_models_root(), model_id.replace("/", "--"))


def _ensure_local_hf_model(model_id: str) -> str:
    """Download the speech model onto the project drive, skipping extra formats."""
    folder = _hf_model_dir(model_id)
    weights = os.path.join(folder, "model.safetensors")
    config = os.path.join(folder, "config.json")
    if os.path.isfile(weights) and os.path.isfile(config):
        return folder
    from huggingface_hub import snapshot_download
    os.makedirs(folder, exist_ok=True)
    # Xet can buffer the whole checkpoint in memory and never write the file.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"Downloading '{model_id}' into {folder} ...")
    snapshot_download(
        model_id,
        local_dir=folder,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "normalizer.json",
            "model.safetensors",
        ],
    )
    return folder


def _load_huggingface_asr(model_id: str) -> _HuggingFaceAsr:
    from transformers import pipeline as hf_pipeline
    folder = _ensure_local_hf_model(model_id)
    print(f"Loading Hugging Face ASR '{model_id}' ...")
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=folder,
        chunk_length_s=30,
    )
    print("Hugging Face ASR loaded.")
    return _HuggingFaceAsr(pipe, model_id)


def load_whisper(model_size: str = "base"):
    """
    Load (or return cached) speech model.

    Hugging Face ids (contain ``/``) load via Transformers. When that load
    fails and ``CLARIO_WHISPER_FALLBACK`` is set, the public Whisper checkpoint
    named by the fallback is used instead. OpenAI size names load with
    openai-whisper only when that size is configured explicitly.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    name = (model_size or "base").strip()
    if "/" in name:
        try:
            _whisper_model = _load_huggingface_asr(name)
        except Exception as exc:
            _whisper_model = None
            from config import WHISPER_FALLBACK
            fallback = (WHISPER_FALLBACK or "").strip()
            public_id = _public_whisper_id(fallback) if fallback else ""
            if not public_id or public_id == name:
                raise RuntimeError(
                    f"Could not load {name}: {exc}. "
                    "Clario did not fall back to another Whisper checkpoint."
                ) from exc
            reason = (
                "not available to this Hugging Face account"
                if _is_model_access_error(exc)
                else f"load failed ({exc})"
            )
            print(
                f"[transcription] {name} {reason}. "
                f"Using configured fallback '{public_id}'."
            )
            try:
                _whisper_model = _load_huggingface_asr(public_id)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Could not load {name}: {exc}. "
                    f"Fallback {public_id} also failed: {fallback_exc}."
                ) from fallback_exc
            print(f"[transcription] Active ASR model: {public_id}")
            return _whisper_model
        print(f"[transcription] Active ASR model: {name}")
        return _whisper_model

    _whisper_model = _load_openai_whisper(name)
    try:
        setattr(_whisper_model, "name", name)
    except Exception:
        pass
    return _whisper_model


def is_english_only_whisper_model(model_name: str) -> bool:
    """English-only Whisper checkpoints end with ``.en`` and cannot handle Urdu."""
    return (model_name or "").strip().lower().endswith(".en")


def is_sukhan_model(model_name: str) -> bool:
    """Urdu Sukhan ASR writes Urdu in Urdu script and English in Latin letters."""
    return "sukhan" in (model_name or "").lower()


class _HuggingFaceAsr:
    """Whisper-compatible wrapper around a Transformers ASR pipeline."""

    def __init__(self, pipe: Any, model_name: str) -> None:
        self._pipe = pipe
        self.name = model_name

    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, str]:
        generate: Dict[str, Any] = {}
        task = kwargs.get("task") or "transcribe"
        if is_sukhan_model(self.name):
            # Model card: translation is out of scope; detect language unless forced.
            task = "transcribe"
        generate["task"] = task
        language = kwargs.get("language")
        if language:
            generate["language"] = language
        output = self._pipe(str(audio_path), generate_kwargs=generate)
        if isinstance(output, dict):
            text = output.get("text") or ""
        else:
            text = str(output or "")
        return {"text": text}


def whisper_transcribe_options(*, model_name: str = "") -> Dict[str, Any]:
    """Build keyword arguments for ``model.transcribe`` from config.

    Uses ``translate`` so Urdu speech becomes English; auto language detection
    supports English–Urdu code-switching on multilingual checkpoints.
    """
    from config import (
        WHISPER_INITIAL_PROMPT,
        WHISPER_LANGUAGE,
        WHISPER_TASK,
    )

    task = WHISPER_TASK if WHISPER_TASK in ("transcribe", "translate") else "translate"
    if is_english_only_whisper_model(model_name) or is_sukhan_model(model_name):
        task = "transcribe"
    opts: Dict[str, Any] = {"task": task, "fp16": False}
    lang = (WHISPER_LANGUAGE or "auto").lower()
    if lang not in ("", "auto", "detect"):
        opts["language"] = lang
    prompt = (WHISPER_INITIAL_PROMPT or "").strip()
    if prompt and not is_sukhan_model(model_name):
        opts["initial_prompt"] = prompt
    return opts


def transcribe_audio(audio_path: str,
                     model_size: str = "base",
                     model=None,
                     **whisper_kwargs: Any) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Parameters
    ----------
    audio_path : Absolute path to the audio file (mp3, wav, m4a, ...).
    model_size : Whisper model variant (used only if *model* is None).
    model      : Pre-loaded Whisper model instance (optional).
                 Pass this to avoid reloading the model on every call.

    Returns
    -------
    str - Transcribed text.

    Raises
    ------
    FileNotFoundError if *audio_path* does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model is None:
        model = load_whisper(model_size)

    resolved_size = getattr(model, "name", None) or model_size

    options = whisper_transcribe_options(model_name=resolved_size or model_size)
    options.update(whisper_kwargs)

    print(f"Transcribing: {audio_path} (task={options.get('task')})")
    result = model.transcribe(audio_path, **options)
    transcript = (result.get("text") or "").strip()

    from services.preprocessing.language import ensure_english_transcript

    if options.get("task") == "translate":
        transcript = ensure_english_transcript(
            transcript,
            threshold=0.05,
            enabled=True,
        )
    else:
        transcript = ensure_english_transcript(transcript)

    print(f"Transcription complete ({len(transcript)} chars).")
    return transcript


def save_transcript(transcript: str, output_path: str) -> None:
    """
    Persist a transcript string to a plain-text file.

    Parameters
    ----------
    transcript  : The transcribed text to save.
    output_path : Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(transcript)
    print(f"Transcript saved to: {output_path}")


def load_transcript(transcript_path: str) -> str:
    """
    Load a previously-saved transcript from disk.

    Parameters
    ----------
    transcript_path : Path to the .txt file.

    Returns
    -------
    str
    """
    with open(transcript_path, "r", encoding="utf-8") as fh:
        return fh.read()

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


def load_whisper(model_size: str = "base"):
    """
    Load (or return cached) Whisper model.

    Parameters
    ----------
    model_size : One of 'tiny', 'base', 'small', 'medium', 'large'.
                 'base' is a good balance of speed and accuracy.

    Returns
    -------
    whisper.model.Whisper
    """
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"Loading Whisper '{model_size}' model ...")
        _whisper_model = whisper.load_model(model_size)
        print("Whisper model loaded.")
    return _whisper_model


def transcribe_audio(audio_path: str,
                     model_size: str = "base",
                     model=None) -> str:
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

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()
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

"""
services/audio_mix.py
---------------------
Build one meeting recording from the saved microphone WAV files.

Samples are placed on a shared timeline using each track's created_at
when every track has one. The sum is scaled so the loudest sample stays
inside 16-bit range. This does not remove background noise or echo.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import wave
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional


def mix_wav_files(tracks: List[Dict[str, Any]]) -> bytes:
    """Mix 16-bit mono WAV files. ``offset_ms`` delays a track on the timeline."""
    pieces: List[List[int]] = []
    rate: Optional[int] = None
    for track in tracks:
        with wave.open(track["path"], "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Meeting mix accepts 16-bit mono WAV tracks.")
            if rate is None:
                rate = handle.getframerate()
            elif handle.getframerate() != rate:
                raise ValueError("Meeting mix tracks must share one sample rate.")
            frames = handle.readframes(handle.getnframes())
        count = len(frames) // 2
        samples = list(struct.unpack("<" + ("h" * count), frames)) if count else []
        samples = _balance(samples)
        pad = int(rate * max(int(track.get("offset_ms") or 0), 0) / 1000)
        pieces.append(([0] * pad) + samples)
    if not pieces or not rate:
        raise ValueError("There is no microphone audio to mix.")
    length = max(len(part) for part in pieces)
    mixed = [0] * length
    for part in pieces:
        for index, sample in enumerate(part):
            mixed[index] += int(sample)
    peak = max((abs(sample) for sample in mixed), default=0)
    if peak > 32767:
        scale = 32767 / peak
        mixed = [int(sample * scale) for sample in mixed]
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + ("h" * len(mixed)), *mixed))
    return buffer.getvalue()


def _balance(samples: List[int], target: int = 16000) -> List[int]:
    """Give each microphone a similar peak before the tracks are added together."""
    peak = max((abs(sample) for sample in samples), default=0)
    if peak <= 0 or peak == target:
        return samples
    scale = target / peak
    return [int(sample * scale) for sample in samples]


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def _pcm_wav(path: str) -> Optional[str]:
    """Return a 16-bit mono WAV path. A WAV that is already that shape is reused."""
    try:
        with wave.open(path, "rb") as handle:
            if handle.getnchannels() == 1 and handle.getsampwidth() == 2:
                return path
    except (wave.Error, EOFError, FileNotFoundError):
        pass
    exe = _ffmpeg_exe()
    if not exe:
        return None
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    result = subprocess.run(
        [exe, "-y", "-i", path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", handle.name],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.isfile(handle.name) or os.path.getsize(handle.name) == 0:
        try:
            os.remove(handle.name)
        except OSError:
            pass
        return None
    return handle.name


def _offsets_ms(rows: List[Dict[str, Any]]) -> List[int]:
    stamps = []
    for row in rows:
        raw = row.get("created_at") or ""
        try:
            stamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            return [0 for _ in rows]
    if any(row.get("created_at") in (None, "") for row in rows):
        return [0 for _ in rows]
    start = min(stamps)
    return [max(int((item - start).total_seconds() * 1000), 0) for item in stamps]


def build_meeting_mix(meeting_id: int, recordings: List[Dict[str, Any]],
                      db_path: str) -> Optional[str]:
    """Write one full-meeting WAV and store it. Existing mixes are left as they are.

    Returns the new path, or None when a full mix already exists or a
    microphone cannot be decoded. Browser mixes are not treated as this file.
    """
    if any(row.get("kind") == "meeting" for row in recordings):
        return None
    speakers = [
        row for row in recordings
        if (row.get("kind") or "speaker") == "speaker"
        and os.path.isfile(row.get("audio_path") or "")
    ]
    if not speakers:
        return None
    decoded: List[str] = []
    temporary: List[str] = []
    payload: Optional[bytes] = None
    try:
        for row in speakers:
            wav_path = _pcm_wav(row["audio_path"])
            if not wav_path:
                return None
            decoded.append(wav_path)
            if os.path.abspath(wav_path) != os.path.abspath(row["audio_path"]):
                temporary.append(wav_path)
        offsets = _offsets_ms(speakers)
        payload = mix_wav_files([
            {"path": wav_path, "offset_ms": offset}
            for wav_path, offset in zip(decoded, offsets)
        ])
    except ValueError:
        payload = None
    finally:
        for path in temporary:
            try:
                os.remove(path)
            except OSError:
                pass
    if not payload:
        return None
    folder = os.path.dirname(speakers[0]["audio_path"]) or "."
    path = os.path.join(folder, f"m{meeting_id}_full.wav")
    with open(path, "wb") as handle:
        handle.write(payload)
    with wave.open(path, "rb") as handle:
        duration = handle.getnframes() / float(handle.getframerate() or 1)
    from models.database import insert_meeting_recordings
    insert_meeting_recordings(meeting_id, [{
        "speaker_name": "Full meeting",
        "speaker_email": "",
        "audio_path": path,
        "byte_size": os.path.getsize(path),
        "duration_seconds": round(duration, 3),
        "status": "saved",
        "kind": "meeting",
    }], db_path)
    return path

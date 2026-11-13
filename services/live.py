"""
services/live.py
----------------
Live Clario meetings.

Each logged-in participant records their own microphone. The recording is
labelled with that account's name and email, then the tracks are transcribed
and joined into one transcript for the normal pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from config import DB_PATH, GRAPH_DIR, WHISPER_MODEL


def transcribe_speaker_audio(audio_path: str) -> str:
    """Transcribe one participant's recording.

    Uses the laptop service when it is configured. Otherwise Whisper stays
    lazily loaded in this process.
    """
    from services.preprocessing.remote_transcription import (
        remote_transcription_configured,
        transcribe_remote,
    )

    if remote_transcription_configured():
        return transcribe_remote(audio_path) or ""
    from services.pipeline import _get_whisper
    from services.preprocessing.transcription import transcribe_audio

    return transcribe_audio(
        audio_path,
        model=_get_whisper(),
        model_size=WHISPER_MODEL,
    ) or ""


def speaker_lines(recordings: List[Dict[str, Any]],
                  transcribe: Callable[[str], str]) -> List[Dict[str, str]]:
    """Turn recordings into speaker turns. Skips a track with no speech."""
    turns: List[Dict[str, str]] = []
    for recording in recordings:
        text = (transcribe(recording["audio_path"]) or "").strip()
        if not text:
            continue
        turns.append({
            "speaker_name": recording.get("display_name") or "Speaker",
            "speaker_email": (recording.get("email") or "").strip().lower(),
            "content": text,
        })
    return turns


def format_speaker_transcript(turns: List[Dict[str, str]]) -> str:
    """Prefix each turn so the name stays tied to the account email."""
    lines = []
    for turn in turns:
        name = turn.get("speaker_name") or "Speaker"
        email = (turn.get("speaker_email") or "").strip()
        label = f"{name} ({email})" if email else name
        lines.append(f"{label}: {turn.get('content', '').strip()}")
    return "\n".join(lines).strip()


def finish_live_room(room: Dict[str, Any], *,
                     db_path: str = DB_PATH,
                     graph_dir: str = GRAPH_DIR,
                     transcribe: Optional[Callable[[str], str]] = None) -> int:
    """Create the pipeline meeting for a live room and queue processing.

    Raises ValueError when nobody has a usable recording.
    """
    from models.database import (
        end_live_room,
        get_live_members,
        get_live_recordings,
        insert_meeting,
        insert_speaker_turns,
        replace_meeting_participants,
        update_meeting_platform,
    )
    from services.jobs import submit_pipeline_job

    recordings = get_live_recordings(int(room["id"]), db_path)
    turns = speaker_lines(recordings, transcribe or transcribe_speaker_audio)
    transcript = format_speaker_transcript(turns)
    if not transcript:
        raise ValueError("Record at least one person's microphone before ending the meeting.")

    meeting_id = insert_meeting(
        user_id=int(room["host_user_id"]),
        title=room.get("title") or "Live meeting",
        audio_filename="Live meeting recording",
        db_path=db_path,
        platform="live",
        external_meeting_id=room.get("join_token"),
        status="queued",
        progress_stage="queued",
    )
    members_as_participants = []
    seen = set()
    for member in get_live_members(int(room["id"]), db_path, admission="admitted"):
        email = (member.get("email") or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        members_as_participants.append({
            "name": member.get("display_name") or email,
            "email": email,
            "source": "live",
        })
    replace_meeting_participants(meeting_id, members_as_participants, db_path)
    update_meeting_platform(meeting_id, "live", room.get("join_token") or "", db_path)
    insert_speaker_turns(meeting_id, turns, db_path)
    end_live_room(int(room["id"]), meeting_id, db_path)
    submit_pipeline_job(
        meeting_id=meeting_id,
        user_id=int(room["host_user_id"]),
        text_input=transcript,
        db_path=db_path,
        graph_dir=graph_dir,
        participants=members_as_participants,
    )
    return meeting_id


def link_task_assigners(results: Dict[str, Any]) -> None:
    """Remember which speaker's turn contained each saved task."""
    stats = results.get("stats") or {}
    meeting_id = stats.get("meeting_id")
    if not meeting_id:
        return
    db_path = stats.get("db_path") or DB_PATH
    from models.database import get_speaker_turns, get_tasks_by_meeting, set_task_assigner

    turns = get_speaker_turns(int(meeting_id), db_path)
    if not turns:
        return
    for task in get_tasks_by_meeting(int(meeting_id), db_path):
        turn = _turn_for_task(task.get("description") or "", turns)
        if turn and turn.get("speaker_email"):
            set_task_assigner(int(task["id"]), turn["speaker_email"], db_path)


def _turn_for_task(description: str, turns: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    desc = (description or "").strip()
    if not desc:
        return None
    hits = []
    for turn in turns:
        text = turn.get("content") or ""
        if desc in text or (len(desc) > 24 and desc[:24] in text):
            hits.append(turn)
    if len(hits) == 1:
        return hits[0]
    return None


def safe_recording_path(upload_dir: str, token: str, user_id: int, extension: str) -> str:
    """Build a recording path that stays inside the live upload folder."""
    from uuid import uuid4

    folder = os.path.abspath(os.path.join(upload_dir, "live", token))
    os.makedirs(folder, exist_ok=True)
    filename = f"u{user_id}_{uuid4().hex}{extension}"
    path = os.path.abspath(os.path.join(folder, filename))
    if os.path.commonpath([folder, path]) != folder:
        raise ValueError("Recording path escaped the upload folder.")
    return path

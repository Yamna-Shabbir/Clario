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
import re
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


def saved_recording_rows(recordings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tracks that exist on disk and are not empty. Missing files are not saved."""
    rows = []
    for recording in recordings:
        path = recording.get("audio_path") or ""
        if not path or not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if size <= 0:
            continue
        kind = "mix" if recording.get("kind") == "mix" else "speaker"
        rows.append({
            "user_id": recording.get("user_id"),
            "speaker_name": "Whole meeting" if kind == "mix" else (recording.get("display_name") or "Speaker"),
            "speaker_email": (recording.get("email") or "").strip().lower(),
            "audio_path": path,
            "byte_size": size,
            "status": "saved",
            "kind": kind,
        })
    return rows


def choose_meeting_recordings(rows: List[Dict[str, Any]], host_user_id: int) -> List[Dict[str, Any]]:
    """Keep every microphone, plus one mix of the whole meeting.

    The host's mix is preferred because that browser hears every connected person.
    """
    speakers = [row for row in rows if row.get("kind") != "mix"]
    mixes = [row for row in rows if row.get("kind") == "mix"]
    mixes.sort(
        key=lambda row: (
            int(row.get("user_id") or 0) == int(host_user_id),
            int(row.get("byte_size") or 0),
        ),
        reverse=True,
    )
    if mixes:
        return speakers + [mixes[0]]
    return speakers


def meeting_participants_from_members(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Admitted accounts become meeting participants, with join and leave times."""
    people = []
    seen = set()
    for member in members:
        email = (member.get("email") or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        people.append({
            "name": member.get("display_name") or email,
            "email": email,
            "source": "live",
            "joined_at": member.get("joined_at") or "",
            "left_at": member.get("left_at") or "",
        })
    return people


def transcript_from_recordings(recordings: List[Dict[str, Any]],
                               transcribe: Callable[[str], str]) -> tuple:
    """Transcribe each saved track. A failed track does not delete the file."""
    turns: List[Dict[str, str]] = []
    errors: List[str] = []
    for recording in recordings:
        path = recording.get("audio_path") or ""
        try:
            text = (transcribe(path) or "").strip()
        except Exception as exc:
            errors.append(f"{recording.get('speaker_name') or 'Speaker'}: {exc}")
            continue
        if not text:
            continue
        turns.append({
            "speaker_name": recording.get("speaker_name") or "Speaker",
            "speaker_email": (recording.get("speaker_email") or "").strip().lower(),
            "content": text,
        })
    return format_speaker_transcript(turns), errors


def queue_live_analysis(meeting_id: int, user_id: int, *,
                        db_path: str = DB_PATH,
                        graph_dir: str = GRAPH_DIR,
                        transcribe: Optional[Callable[[str], str]] = None,
                        missing_names: Optional[List[str]] = None) -> None:
    """Transcribe saved tracks in the background, then run the meeting pipeline."""
    from services.jobs import CLARIO_SYNC_JOBS, get_executor, submit_pipeline_job

    def job() -> None:
        from models.database import (
            get_meeting_participants,
            get_meeting_recordings,
            insert_speaker_turns,
            update_meeting_status,
        )

        update_meeting_status(
            meeting_id,
            status="processing",
            progress_stage="transcribing",
            error_message=None,
            db_path=db_path,
        )
        saved = [
            row for row in get_meeting_recordings(meeting_id, db_path)
            if row.get("status") == "saved" and os.path.isfile(row.get("audio_path") or "")
        ]
        speakers = [row for row in saved if (row.get("kind") or "speaker") != "mix"]
        # Separate microphones keep who spoke. The mix is only transcribed
        # when nobody uploaded their own microphone.
        recordings = speakers or [row for row in saved if row.get("kind") == "mix"]
        recordings = sorted(
            recordings,
            key=lambda row: (row.get("created_at") or "", int(row.get("id") or 0)),
        )
        speak = transcribe or transcribe_speaker_audio
        turns: List[Dict[str, str]] = []
        errors: List[str] = []
        for recording in recordings:
            path = recording.get("audio_path") or ""
            label = recording.get("speaker_name") or "Speaker"
            try:
                text = (speak(path) or "").strip()
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                continue
            if not text:
                continue
            turns.append({
                "speaker_name": label,
                "speaker_email": (recording.get("speaker_email") or "").strip().lower(),
                "content": text,
            })
        transcript = format_speaker_transcript(turns)
        if not transcript:
            detail = " ".join(errors) or "No speech was transcribed."
            update_meeting_status(
                meeting_id,
                status="failed",
                progress_stage="failed",
                error_message=(
                    "Transcription failed. The recording was kept and can be retried. "
                    + detail
                )[:500],
                db_path=db_path,
            )
            return
        insert_speaker_turns(meeting_id, turns, db_path)
        participants = get_meeting_participants(meeting_id, db_path)
        submit_pipeline_job(
            meeting_id=meeting_id,
            user_id=user_id,
            text_input=transcript,
            db_path=db_path,
            graph_dir=graph_dir,
            participants=participants,
        )
        notes: List[str] = []
        if missing_names:
            notes.append("No microphone was saved for: " + ", ".join(missing_names) + ".")
        if errors:
            notes.append("Some microphones were not transcribed. The saved audio was kept. " + " ".join(errors))
        if notes:
            from models.database import get_meeting_status
            current = get_meeting_status(meeting_id, user_id, db_path) or {}
            if current.get("status") != "failed":
                update_meeting_status(
                    meeting_id,
                    status="partial",
                    progress_stage="partial",
                    error_message=" ".join(notes)[:500],
                    db_path=db_path,
                )

    if CLARIO_SYNC_JOBS:
        job()
        return
    get_executor().submit(job)


def finish_live_room(room: Dict[str, Any], *,
                     db_path: str = DB_PATH,
                     graph_dir: str = GRAPH_DIR,
                     transcribe: Optional[Callable[[str], str]] = None) -> int:
    """Save each person's recording, then queue transcription.

    Raises ValueError when nobody has a usable recording.
    A second call returns the meeting already created for this room.
    """
    from models.database import (
        claim_room_for_finish,
        end_live_room,
        get_live_members,
        get_live_recordings,
        get_live_room_by_token,
        insert_meeting,
        insert_meeting_recordings,
        release_live_room,
        replace_meeting_participants,
        update_meeting_platform,
    )

    existing = room.get("meeting_id")
    if room.get("status") == "ended" and existing:
        return int(existing)
    if room.get("status") in {"live", "ending"} and not claim_room_for_finish(int(room["id"]), db_path):
        current = get_live_room_by_token(room.get("join_token") or "", db_path) or room
        if current.get("meeting_id"):
            return int(current["meeting_id"])
        raise ValueError("This meeting is already being saved.")
    elif room.get("status") not in {"live", "ending"}:
        raise ValueError("This meeting is not live.")

    recordings = choose_meeting_recordings(
        saved_recording_rows(get_live_recordings(int(room["id"]), db_path)),
        int(room["host_user_id"]),
    )
    if not recordings:
        release_live_room(int(room["id"]), db_path)
        raise ValueError(
            "No complete microphone recording was saved. "
            "An empty or missing track is not marked as saved."
        )

    meeting_id = insert_meeting(
        user_id=int(room["host_user_id"]),
        title=room.get("title") or "Live meeting",
        audio_filename="Live meeting recording",
        db_path=db_path,
        platform="live",
        external_meeting_id=room.get("join_token"),
        status="queued",
        progress_stage="saved",
    )
    from models.database import mark_live_members_left
    mark_live_members_left(int(room["id"]), db_path)
    members = get_live_members(int(room["id"]), db_path, admission="admitted")
    participants = meeting_participants_from_members(members)
    recorded_ids = {
        int(row["user_id"]) for row in recordings if row.get("kind") != "mix" and row.get("user_id")
    }
    missing = [
        member.get("display_name") or member.get("email") or "Participant"
        for member in members
        if int(member["user_id"]) not in recorded_ids
    ]
    replace_meeting_participants(meeting_id, participants, db_path)
    update_meeting_platform(meeting_id, "live", room.get("join_token") or "", db_path)
    insert_meeting_recordings(meeting_id, recordings, db_path)
    end_live_room(int(room["id"]), meeting_id, db_path)
    queue_live_analysis(
        meeting_id,
        int(room["host_user_id"]),
        db_path=db_path,
        graph_dir=graph_dir,
        transcribe=transcribe,
        missing_names=missing,
    )
    return meeting_id


_DISCUSSION_CUE = re.compile(
    r"\b(please|will|must|need to|has to|action item|can you)\b",
    re.IGNORECASE,
)


def meeting_highlights(transcript: str) -> Dict[str, List[str]]:
    """Questions and longer remarks taken from the transcript itself.

    Nothing is added that the transcript does not already say.
    """
    questions: List[str] = []
    points: List[str] = []
    for raw in re.split(r"(?<=[.?!])\s+", transcript or ""):
        sentence = " ".join(raw.split()).strip()
        if len(sentence) < 12:
            continue
        if sentence.endswith("?"):
            if sentence not in questions:
                questions.append(sentence)
        elif not _DISCUSSION_CUE.search(sentence) and len(sentence.split()) >= 8:
            if sentence not in points:
                points.append(sentence)
    return {"questions": questions[:8], "points": points[:6]}


def retry_live_transcription(meeting_id: int, user_id: int, *,
                              db_path: str = DB_PATH,
                              graph_dir: str = GRAPH_DIR) -> Optional[str]:
    """Queue transcription again. Returns an error string when retry is refused."""
    from models.database import (
        get_accessible_meeting_detail,
        get_meeting_recordings,
        get_meeting_status,
        update_meeting_status,
        user_can_access_meeting,
    )

    if not user_can_access_meeting(meeting_id, user_id, db_path):
        return "Meeting not found."
    meeting = get_accessible_meeting_detail(meeting_id, user_id, db_path) or {}
    owner_id = int(meeting.get("user_id") or user_id)
    current = get_meeting_status(meeting_id, user_id, db_path) or {}
    if current.get("status") in {"queued", "processing"}:
        return "This meeting is already being processed."
    saved = [
        row for row in get_meeting_recordings(meeting_id, db_path)
        if row.get("status") == "saved" and os.path.isfile(row.get("audio_path") or "")
    ]
    if not saved:
        return "There is no saved recording to transcribe."
    update_meeting_status(
        meeting_id,
        status="queued",
        progress_stage="saved",
        error_message=None,
        db_path=db_path,
    )
    queue_live_analysis(meeting_id, owner_id, db_path=db_path, graph_dir=graph_dir)
    return None


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

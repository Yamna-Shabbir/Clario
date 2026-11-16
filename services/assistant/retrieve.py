"""Meeting facts the signed-in user is already allowed to see."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from models.database import (
    get_connection,
    get_decisions_by_meeting,
    get_meeting_participants,
    get_meeting_recordings,
    get_tasks_by_meeting,
    user_can_access_meeting,
)
from services.assistant.knowledge import tokenize
from services.live import evidence_playback


def visible_meeting_ids(user_id: int, email: str, db_path: str) -> List[int]:
    """Meetings this account owns, joined, was assigned in, or was admitted to."""
    cleaned = (email or "").strip().lower()
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id FROM meetings WHERE user_id = ?
            UNION
            SELECT meeting_id FROM meeting_participants
            WHERE lower(COALESCE(email, '')) = ? AND COALESCE(email, '') != ''
            UNION
            SELECT meeting_id FROM tasks
            WHERE lower(COALESCE(assigned_email, '')) = ? AND COALESCE(assigned_email, '') != ''
            UNION
            SELECT r.meeting_id FROM live_members lm
            JOIN live_rooms r ON r.id = lm.room_id
            WHERE lm.user_id = ? AND lm.admission_status = 'admitted'
              AND r.meeting_id IS NOT NULL
            """,
            (user_id, cleaned, cleaned, user_id),
        ).fetchall()
    return [int(row["id"]) for row in rows]


def _clock(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    if total < 0:
        return ""
    return f"{total // 60}:{total % 60:02d}"


def _excerpt(transcript: str, query: Sequence[str]) -> str:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", transcript or "")
    best = ""
    best_score = 0
    for piece in pieces:
        sentence = " ".join(piece.split()).strip()
        if len(sentence) < 8:
            continue
        score = len(set(tokenize(sentence)) & set(query))
        if score > best_score:
            best = sentence[:500]
            best_score = score
    return best if best_score else ""


def _bundle(meeting_id: int, user_id: int, db_path: str, query: Sequence[str]) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT m.id, m.title, m.meeting_date, m.status, m.user_id,
                      t.content AS transcript, s.content AS summary
               FROM meetings m
               LEFT JOIN transcripts t ON t.meeting_id = m.id
               LEFT JOIN summaries s ON s.meeting_id = m.id
               WHERE m.id = ?""",
            (meeting_id,),
        ).fetchone()
    if not row:
        return None
    opened = user_can_access_meeting(meeting_id, user_id, db_path)
    tasks = get_tasks_by_meeting(meeting_id, db_path)
    recordings = get_meeting_recordings(meeting_id, db_path) if opened else []
    decisions = get_decisions_by_meeting(meeting_id, db_path) if opened else []
    participants = get_meeting_participants(meeting_id, db_path) if opened else []
    evidence: List[Dict[str, Any]] = []
    for kind, items in (("task", tasks), ("decision", decisions)):
        for item in items:
            playback = evidence_playback(recordings, item) if opened else None
            start = item.get("evidence_start")
            evidence.append({
                "kind": kind,
                "meeting_id": meeting_id,
                "meeting_title": row["title"] or "Untitled Meeting",
                "description": item.get("description") or item.get("content") or "",
                "speaker": item.get("evidence_speaker") or "",
                "excerpt": (item.get("source_text") or item.get("content") or item.get("description") or "")[:500],
                "timestamp": _clock(start),
                "evidence_start": start,
                "recording_id": playback["recording_id"] if playback else None,
                "seek": playback["start"] if playback else None,
            })
    transcript = row["transcript"] or ""
    return {
        "id": int(row["id"]),
        "title": row["title"] or "Untitled Meeting",
        "meeting_date": row["meeting_date"] or "",
        "status": row["status"] or "",
        "opened": opened,
        "summary": (row["summary"] or "") if opened else "",
        "excerpt": _excerpt(transcript, query) if opened else "",
        "participants": [
            {"name": person.get("name") or "", "email": person.get("email") or ""}
            for person in participants
        ] if opened else [],
        "tasks": [
            {
                "description": task.get("description") or "",
                "assigned_to": task.get("assigned_to") or "",
                "due_date": task.get("due_date") or "",
                "status": task.get("status") or "",
                "priority": task.get("priority") or "",
            }
            for task in tasks
        ],
        "decisions": [
            {"content": decision.get("content") or ""}
            for decision in decisions
        ] if opened else [],
        "evidence": [item for item in evidence if item.get("excerpt")],
    }


def _score(bundle: Dict[str, Any], query: Sequence[str]) -> int:
    hay = " ".join([
        bundle.get("title") or "",
        bundle.get("summary") or "",
        bundle.get("excerpt") or "",
        " ".join(task.get("description") or "" for task in bundle.get("tasks") or []),
        " ".join(decision.get("content") or "" for decision in bundle.get("decisions") or []),
        " ".join(person.get("name") or "" for person in bundle.get("participants") or []),
    ])
    return len(set(tokenize(hay)) & set(query))


def retrieve_meetings(
    question: str,
    *,
    user_id: int,
    email: str,
    db_path: str,
    meeting_ids: Optional[Sequence[int]] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Rank meetings for this question.

    ``meeting_ids`` limits the search to meetings already in the conversation.
    Meetings the user cannot see are never loaded.
    """
    allowed = set(visible_meeting_ids(user_id, email, db_path))
    if meeting_ids is not None:
        chosen = [int(meeting_id) for meeting_id in meeting_ids if int(meeting_id) in allowed]
    else:
        chosen = list(allowed)
    query = tokenize(question)
    lowered = (question or "").lower()
    bundles = [
        bundle for meeting_id in chosen
        if (bundle := _bundle(meeting_id, user_id, db_path, query)) is not None
    ]
    if not bundles:
        return []
    latest = bool(re.search(r"\b(latest|last|recent|newest)\b", lowered))
    if meeting_ids is None and latest and not any(_score(bundle, query) for bundle in bundles):
        bundles.sort(key=lambda bundle: bundle["id"], reverse=True)
        return bundles[:1]
    if meeting_ids is not None:
        return bundles[: max(1, limit)]
    ranked = [bundle for bundle in bundles if _score(bundle, query)]
    broad = bool(re.search(r"\b(tasks?|decisions?|meetings?|pending|summary|transcript)\b", lowered))
    if not ranked and (broad or latest):
        bundles.sort(key=lambda bundle: bundle["id"], reverse=True)
        return bundles[: max(1, limit)]
    ranked.sort(key=lambda bundle: (_score(bundle, query), bundle["id"]), reverse=True)
    return ranked[: max(1, limit)]

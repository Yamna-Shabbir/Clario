"""
services/participant_matching.py
---------------------------------
Attendee name resolution and task assignment rematching.

Matching hierarchy:
1. Normalised exact match (case, whitespace, and punctuation insensitive).
2. Unique first-name match.
3. difflib SequenceMatcher fuzzy match (ratio >= 0.8 with a unique best candidate).

Rules:
- Never matches pronouns ("he", "she", "they", "we", etc.) or generic group terms like "Team".
- Returns None when ambiguous (multiple candidate matches) or when no match satisfies criteria.
- Rematches existing tasks for a meeting, updating only matching tasks and leaving others untouched.
"""

import difflib
import re
from typing import Any, Dict, List, Optional

from config import DB_PATH
from models.database import (
    get_connection,
    get_meeting_detail,
    get_meeting_participants,
    get_tasks_by_meeting,
)

# Pronouns and non-person designations that must NEVER be resolved to a participant
IGNORED_ASSIGNEES = {
    "team", "the team", "all", "everyone", "everybody", "unknown",
    "nobody", "someone", "somebody", "anyone", "anybody",
    "he", "she", "it", "they", "we", "i", "you", "me", "him", "her", "us", "them",
    "himself", "herself", "itself", "themselves", "myself", "yourself",
}


def _clean_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not s:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _first_name(s: str) -> str:
    """Extract normalized first name."""
    tokens = _clean_text(s).split()
    return tokens[0] if tokens else ""


def _classify_assignee(
    assignee_name: str,
    participants: Optional[List[Dict[str, Any]]],
) -> tuple:
    """Classify a name as unique, ambiguous, or unmatched.

    Returns ``("unique", participant)``, ``("ambiguous", [participants])``,
    or ``("none", [])``. The unique path is the same decision ``match``
    has always made.
    """
    if not assignee_name or not participants:
        return "none", []

    if "@" in assignee_name:
        email = assignee_name.strip().lower()
        email_matches = [
            p for p in participants
            if p and (p.get("email") or "").strip().lower() == email
        ]
        if len(email_matches) == 1:
            return "unique", email_matches[0]
        if len(email_matches) > 1:
            return "ambiguous", email_matches

    cleaned_assignee = _clean_text(assignee_name)
    if not cleaned_assignee or cleaned_assignee in IGNORED_ASSIGNEES:
        return "none", []

    valid_participants = [p for p in participants if p and p.get("name") and p.get("name").strip()]
    if not valid_participants:
        return "none", []

    exact_matches = [
        p for p in valid_participants
        if _clean_text(p["name"]) == cleaned_assignee
    ]
    if len(exact_matches) == 1:
        return "unique", exact_matches[0]
    if len(exact_matches) > 1:
        return "ambiguous", exact_matches

    assignee_tokens = cleaned_assignee.split()
    assignee_first = assignee_tokens[0]
    first_name_matches = []
    for p in valid_participants:
        p_tokens = _clean_text(p["name"]).split()
        p_first = p_tokens[0] if p_tokens else ""
        if len(assignee_tokens) == 1:
            if p_first == assignee_first:
                first_name_matches.append(p)
        elif len(assignee_tokens) > 1 and len(p_tokens) > 1:
            if p_first == assignee_first and p_tokens[1][0] == assignee_tokens[1][0]:
                first_name_matches.append(p)

    if len(first_name_matches) == 1:
        return "unique", first_name_matches[0]
    if len(first_name_matches) > 1:
        return "ambiguous", first_name_matches

    if len(assignee_tokens) == 1 and len(assignee_first) >= 3:
        local_matches = []
        for person in valid_participants:
            local = re.sub(
                r"[^a-z0-9]",
                "",
                (person.get("email") or "").split("@")[0].lower(),
            )
            if local and local == re.sub(r"[^a-z0-9]", "", assignee_first):
                local_matches.append(person)
        if len(local_matches) == 1:
            return "unique", local_matches[0]
        if len(local_matches) > 1:
            return "ambiguous", local_matches

    candidates = []
    for p in valid_participants:
        p_clean = _clean_text(p["name"])
        p_first = _first_name(p["name"])
        ratio_full = difflib.SequenceMatcher(None, cleaned_assignee, p_clean).ratio()
        ratio_first = (
            difflib.SequenceMatcher(None, cleaned_assignee, p_first).ratio()
            if len(assignee_tokens) == 1
            else 0.0
        )
        best_ratio = max(ratio_full, ratio_first)
        if best_ratio >= 0.8:
            candidates.append((best_ratio, p))

    if not candidates:
        return "none", []

    candidates.sort(key=lambda c: c[0], reverse=True)
    top_ratio, top_participant = candidates[0]
    if len(candidates) == 1:
        return "unique", top_participant

    second_ratio, _ = candidates[1]
    if top_ratio > second_ratio and (top_ratio - second_ratio) >= 0.04:
        return "unique", top_participant
    tied = [p for ratio, p in candidates if abs(ratio - top_ratio) < 0.04]
    return "ambiguous", tied or [p for _, p in candidates]


def match(
    assignee_name: str,
    participants: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Match an extracted assignee name to a meeting participant.

    Parameters
    ----------
    assignee_name : Raw name or string extracted by NER.
    participants  : List of participant dicts from meeting_participants.

    Returns
    -------
    Optional[Dict[str, Any]] : The matched participant dict or None.
    """
    kind, value = _classify_assignee(assignee_name, participants)
    if kind == "unique":
        return value
    return None


def ambiguous_matches(
    assignee_name: str,
    participants: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Return every participant who ties for this name.

    An empty list means the name is unmatched or already unique.
    Two or more means a person has to choose which email was meant.
    """
    kind, value = _classify_assignee(assignee_name, participants)
    if kind == "ambiguous":
        return list(value)
    return []


def _email_choices(people: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    choices = []
    seen = set()
    for person in people:
        email = (person.get("email") or "").strip().lower()
        if not email or email in seen:
            continue
        seen.add(email)
        choices.append({"name": person.get("name") or "", "email": email})
    return choices


def pending_name_collisions(
    tasks: List[Dict[str, Any]],
    participants: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Tasks the uploader must confirm before mail is sent.

    A shared or fuzzy name that fits two people is a tie. A person's name
    that fits nobody on the roster is also held, and the uploader picks
    which address they entered. Team and pronouns are never held.
    """
    roster = _email_choices(list(participants or []))
    collisions = []
    for task in tasks:
        if (task.get("assigned_email") or "").strip():
            continue
        assignee = (task.get("assigned_to") or "").strip()
        options = ambiguous_matches(assignee, participants)
        deduped = _email_choices(options)
        if len(deduped) >= 2:
            chosen = deduped
        elif (
            roster
            and assignee
            and _clean_text(assignee) not in IGNORED_ASSIGNEES
            and not options
        ):
            chosen = roster
        else:
            continue
        collisions.append({
            "task_id": task.get("id"),
            "assignee_name": assignee,
            "description": task.get("description") or "",
            "candidates": chosen,
        })
    return collisions


def rematch_meeting_tasks(
    meeting_id: int,
    user_id: int,
    db_path: str = DB_PATH,
) -> int:
    """
    Apply participant matching to already-saved tasks for a meeting.

    Only updates tasks whose assignee matches a participant.
    Leaves all other tasks untouched. Enforces meeting ownership.

    Parameters
    ----------
    meeting_id : Clario meeting ID.
    user_id    : Authenticated user ID.
    db_path    : Database path.

    Returns
    -------
    int : Count of updated tasks.
    """
    meeting = get_meeting_detail(meeting_id, user_id, db_path=db_path)
    if not meeting:
        return 0

    participants = get_meeting_participants(meeting_id, db_path=db_path)
    if not participants:
        return 0

    tasks = get_tasks_by_meeting(meeting_id, db_path=db_path)
    if not tasks:
        return 0

    updated_count = 0
    with get_connection(db_path) as conn:
        for t in tasks:
            current_assignee = t.get("assigned_to", "")
            current_email = t.get("assigned_email")

            matched = match(current_assignee, participants)
            if matched:
                canonical_name = matched.get("name")
                email = matched.get("email")

                if canonical_name != current_assignee or email != current_email:
                    conn.execute(
                        """UPDATE tasks
                           SET assigned_to = ?, assigned_email = ?
                           WHERE id = ?""",
                        (canonical_name, email, t["id"]),
                    )
                    updated_count += 1
        conn.commit()

    return updated_count

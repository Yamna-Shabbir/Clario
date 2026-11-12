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
    if not assignee_name or not participants:
        return None

    cleaned_assignee = _clean_text(assignee_name)
    if not cleaned_assignee or cleaned_assignee in IGNORED_ASSIGNEES:
        return None

    # Filter out participants without a valid name
    valid_participants = [p for p in participants if p and p.get("name") and p.get("name").strip()]
    if not valid_participants:
        return None

    # 1. Normalised exact match
    exact_matches = [
        p for p in valid_participants
        if _clean_text(p["name"]) == cleaned_assignee
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    elif len(exact_matches) > 1:
        # Ambiguous: multiple participants have the exact same normalized name
        return None

    # 2. Unique first-name match
    assignee_tokens = cleaned_assignee.split()
    assignee_first = assignee_tokens[0]

    # Only attempt first-name matching if assignee name is a single token or first names match
    first_name_matches = []
    for p in valid_participants:
        p_name = p["name"]
        p_tokens = _clean_text(p_name).split()
        p_first = p_tokens[0] if p_tokens else ""

        if len(assignee_tokens) == 1:
            if p_first == assignee_first:
                first_name_matches.append(p)
        elif len(assignee_tokens) > 1 and len(p_tokens) > 1:
            # First name matches and last initial matches (e.g., "Sarah C." matches "Sarah Connor")
            if p_first == assignee_first and p_tokens[1][0] == assignee_tokens[1][0]:
                first_name_matches.append(p)

    if len(first_name_matches) == 1:
        return first_name_matches[0]
    elif len(first_name_matches) > 1:
        # Ambiguous: multiple participants share the same first name
        return None

    # 3. difflib SequenceMatcher ratio >= 0.8 with a unique best candidate
    candidates = []
    for p in valid_participants:
        p_clean = _clean_text(p["name"])
        p_first = _first_name(p["name"])

        # Ratio against full name
        ratio_full = difflib.SequenceMatcher(None, cleaned_assignee, p_clean).ratio()

        # If assignee is a single word, also compare against the participant's first name
        ratio_first = (
            difflib.SequenceMatcher(None, cleaned_assignee, p_first).ratio()
            if len(assignee_tokens) == 1
            else 0.0
        )

        best_ratio = max(ratio_full, ratio_first)
        if best_ratio >= 0.8:
            candidates.append((best_ratio, p))

    if not candidates:
        return None

    # Sort descending by match ratio
    candidates.sort(key=lambda c: c[0], reverse=True)

    top_ratio, top_participant = candidates[0]
    if len(candidates) == 1:
        return top_participant

    second_ratio, _ = candidates[1]
    # Require a unique best score
    if top_ratio > second_ratio and (top_ratio - second_ratio) >= 0.04:
        return top_participant

    # Ambiguous fuzzy match
    return None


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

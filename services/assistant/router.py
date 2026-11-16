"""Choose application help, meeting data, or both for one question."""

from __future__ import annotations

import re
from typing import Dict, Optional, Set, Tuple

from services.assistant.knowledge import tokenize

_HELP = {
    "navigate", "navigation", "dashboard", "upload", "record", "recording",
    "transcribe", "transcription", "integration", "integrations", "google",
    "login", "sign", "feature", "features", "retry", "failed", "failure",
    "whisper", "paste", "sidebar", "menu", "trends", "chart", "charts",
    "report", "reports", "assistant", "consent", "waiting", "oauth",
    "disconnect", "connect", "gmail",
}
_MEETING = {
    "task", "tasks", "decision", "decisions", "deadline", "deadlines", "due",
    "summary", "summaries", "transcript", "participant", "participants",
    "meeting", "meetings", "assigned", "assignee", "priority", "pending",
    "said", "history", "action", "speaker", "evidence", "decide", "decided",
    "agreed", "completed", "done", "finished", "started", "doing",
}
_FOLLOW = {"that", "those", "this", "same", "earlier", "again", "above"}
_MUTATION = re.compile(
    r"\b(?:delete|remove|erase|edit|update|change|mark|set|complete|assign)\b"
    r".{0,48}\b(?:task|tasks|decision|decisions|status|assignee|meeting)\b",
    re.IGNORECASE,
)


def question_tokens(question: str) -> Set[str]:
    """Tokens plus a few stems the router treats as meeting or help words."""
    tokens = set(tokenize(question))
    if "meeting" in (question or "").lower() or "meetings" in (question or "").lower():
        tokens.add("meeting")
    return tokens


def is_mutation(question: str) -> bool:
    """True when the user asks the assistant to change stored meeting data."""
    return _MUTATION.search(question or "") is not None


def classify(question: str, previous: Optional[Dict] = None) -> Tuple[str, bool]:
    """
    Return ``(route, is_followup)``.

    ``route`` is ``help``, ``meetings``, or ``both``.
    A short question that points at the previous answer stays on that subject.
    """
    raw = (question or "").lower()
    tokens = question_tokens(question)
    help_hits = tokens & _HELP
    meeting_hits = tokens & _MEETING
    follow = bool(tokens & _FOLLOW) or bool(re.search(r"\bit\b", raw))
    previous_route = (previous or {}).get("route") or ""
    how_to = bool(re.search(r"\b(how do|how can|how to|where do|where can|where is)\b", raw))
    stored = bool(re.search(
        r"\b(my|our|latest|last|recent|due|assigned|said|pending|summary|transcript|deadline|who)\b",
        raw,
    ))
    if previous and follow and not how_to:
        route = "help" if previous_route == "help" else "meetings"
        if previous_route == "both":
            route = "both"
        return route, True
    if how_to and not stored:
        return "help", False
    if help_hits and meeting_hits:
        return "both", False
    if meeting_hits:
        return "meetings", False
    if help_hits:
        return "help", False
    if previous and previous_route in {"help", "meetings", "both"}:
        return previous_route, True
    return "help", False

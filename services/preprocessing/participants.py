"""
services/preprocessing/participants.py
--------------------------------------
Read "Name (email)" mentions out of a pasted transcript so follow-up mail
has an address even when Google Meet did not supply a roster.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_MENTION_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[<(]\s*"
    r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\s*[>)]"
)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def roster_from_pairs(names: List[str], emails: List[str]) -> tuple:
    """Turn parallel name and email fields into participant rows.

    Blank rows are ignored. A name without an email, or the reverse, is an
    error. The same address is kept once.
    """
    people: List[Dict[str, Any]] = []
    seen = set()
    count = max(len(names or []), len(emails or []))
    for index in range(count):
        name = (names[index] if index < len(names) else "").strip()
        email = (emails[index] if index < len(emails) else "").strip().lower()
        if not name and not email:
            continue
        if not name or not email:
            return [], "Enter both a name and an email for each person."
        if not _EMAIL_RE.match(email):
            return [], f"{email} is not an email address."
        if email in seen:
            continue
        seen.add(email)
        people.append({"name": name, "email": email, "source": "upload"})
    return people, ""


def mentioned_participants(text: str) -> List[Dict[str, Any]]:
    """Return unique people written as ``Name (email)`` or ``Name <email>``."""
    found: List[Dict[str, Any]] = []
    seen = set()
    for match in _MENTION_RE.finditer(text or ""):
        email = match.group(2).strip().lower()
        if email in seen:
            continue
        seen.add(email)
        found.append({
            "name": match.group(1).strip(),
            "email": email,
            "source": "transcript",
        })
    return found

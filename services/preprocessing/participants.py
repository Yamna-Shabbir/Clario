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

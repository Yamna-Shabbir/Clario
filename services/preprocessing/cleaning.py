"""
services/preprocessing/cleaning.py
----------------------------------
Transcript cleanup for Phase 1 (input and preprocessing).

Removes filler tokens, noise markers, and timestamp tags, then normalises a
few spoken contractions. Obligation cues such as "won't" and "let's" are
left intact so later task detection still sees them.
"""

from __future__ import annotations

import re

_FILLER_RE = re.compile(r"\b(?:uh+|um+|erm+|hmm+)\b[,.]?", re.IGNORECASE)
_NOISE_RE = re.compile(
    r"\[(?:noise|inaudible|laughter|crosstalk|silence|music)\]"
    r"|\((?:noise|inaudible|laughter|crosstalk|silence|music)\)",
    re.IGNORECASE,
)
_TIMESTAMP_RE = re.compile(
    r"[\[(]?\b\d{1,2}:\d{2}(?::\d{2})?\b[\])]?"
)
_CONTRACTIONS = (
    (re.compile(r"\bgonna\b", re.IGNORECASE), "going to"),
    (re.compile(r"\bwanna\b", re.IGNORECASE), "want to"),
    (re.compile(r"\bkinda\b", re.IGNORECASE), "kind of"),
)
_SPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def clean_transcript(text: str) -> str:
    """Return *text* with fillers, noise markers, and timestamps removed.

    Parameters
    ----------
    text : Raw transcript from Whisper or a pasted upload.

    Returns
    -------
    str
        Cleaned transcript. Empty or whitespace-only input returns "".
    """
    if not text:
        return ""

    cleaned = _NOISE_RE.sub(" ", text)
    cleaned = _TIMESTAMP_RE.sub(" ", cleaned)
    cleaned = _FILLER_RE.sub(" ", cleaned)
    for pattern, replacement in _CONTRACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()

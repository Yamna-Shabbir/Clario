"""
Detect Urdu / Arabic-script text and normalize transcripts to English.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# Arabic script blocks used by Urdu (and occasional Perso-Arabic punctuation).
_ARABIC_CHARS = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def arabic_script_ratio(text: str) -> float:
    """Share of non-whitespace characters in Arabic script (0.0–1.0)."""
    if not text or not text.strip():
        return 0.0
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    arabic = sum(1 for ch in chars if _ARABIC_CHARS.search(ch))
    return arabic / len(chars)


def latin_letter_ratio(text: str) -> float:
    """Share of non-whitespace characters that are Latin letters."""
    if not text or not text.strip():
        return 0.0
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    latin = sum(1 for ch in chars if _LATIN_LETTER.match(ch))
    return latin / len(chars)


def needs_english_normalization(text: str, *, threshold: float = 0.08) -> bool:
    """True when the text likely contains Urdu/Arabic script worth translating."""
    return arabic_script_ratio(text) >= threshold


def ensure_english_transcript(
    text: str,
    *,
    threshold: float = 0.08,
    translator: Optional[Callable[[str], str]] = None,
    enabled: bool = True,
) -> str:
    """
    Return English text for downstream NLP (summaries, tasks).

    Skips work when the transcript is already mostly Latin. When Arabic script
    remains (common with Whisper partial misses), runs optional translation.
    """
    cleaned = (text or "").strip()
    if not cleaned or not enabled:
        return cleaned
    if translator is None:
        from services.preprocessing.roman_urdu import utterance_to_english
        rewritten_lines = [
            utterance_to_english(line) if line.strip() else ""
            for line in cleaned.splitlines()
        ]
        rewritten = "\n".join(rewritten_lines).strip()
        if arabic_script_ratio(rewritten) < threshold:
            return rewritten
        cleaned = rewritten
    if not needs_english_normalization(cleaned, threshold=threshold):
        return cleaned
    if translator is None:
        from services.preprocessing.translation import translate_urdu_text_to_english
        translator = translate_urdu_text_to_english
    try:
        translated = translator(cleaned)
        if translated and translated.strip():
            return translated.strip()
    except Exception as exc:
        print(f"[language] Urdu→English translation failed: {exc}")
    return cleaned

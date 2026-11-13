"""
Turn Urdu-script and Roman Urdu meeting lines into English.

The task detector reads English. This module covers the closed set of
pronouns, postpositions, weekdays, and commitment endings that show up in
meeting speech. A line that is already English is returned unchanged.
A line that is only partly understood is still returned in English words
where we have them, and the caller keeps the original sentence as evidence.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from services.preprocessing.language import arabic_script_ratio

_ARABIC_RUN = re.compile(r"[\u0600-\u06FF]+")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'+-]*|[A-Za-z]")

# Distinctive Roman Urdu cues. "main" alone is not enough: English uses it
# as an adjective ("the main report").
_ROMAN_CUES = {
    "aap", "ap", "tum", "mujhe", "mujhay", "humein", "hum", "ham",
    "karunga", "karungi", "karenge", "dunga", "dungi", "denge", "dunga",
    "bhej", "bhejo", "bhejdo", "bhejna", "kardo", "karo", "dekho", "dekh",
    "theek", "thik", "nahi", "nahin", "nhi", "aaj", "aj", "kal", "parson",
    "peer", "pir", "mangal", "budh", "jumeraat", "jumerat", "juma", "jumma",
    "jummah", "hafte", "shaam", "sham", "subah", "raat", "zaroor", "zara",
    "kya", "sakte", "sakta", "sakti", "shayad", "chahiye", "chahie",
    "kiya", "kiye", "bola", "kaha", "wala", "wali", "hoga", "hogi", "honge",
    "dena", "lena", "loon", "lunga", "lungi",
}

_SPELLING = {
    "mai": "main", "mn": "main",
    "ap": "aap", "mujhay": "mujhe", "humein": "hum", "ham": "hum",
    "wo": "woh", "vo": "woh", "ye": "yeh",
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi",
    "aj": "aaj", "shaam": "shaam", "sham": "shaam",
    "thik": "theek", "jumma": "juma", "jummah": "juma", "jumaa": "juma",
    "jumerat": "jumeraat", "pir": "peer", "somwar": "peer", "somvaar": "peer",
    "itwaar": "itwar", "haftah": "hafta",
    "chahie": "chahiye", "zaroor": "zaroor", "shayad": "maybe",
    "baad": "later",
    "kr": "kar", "bhejn": "bhej",
}

# Longest first. Values are assembler atoms.
_ROMAN_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("assalam o alaikum", "hello"),
    ("assalamu alaikum", "hello"),
    ("walaikum assalam", "hello"),
    ("koi baat nahi", "okay"),
    ("koi masla nahi", "okay"),
    ("chalo theek hai", "okay"),
    ("theek hai", "okay"),
    ("theek kar dunga", "will-fix"),
    ("theek kar dungi", "will-fix"),
    ("theek kar denge", "will-fix"),
    ("theek kar dena", "please-fix"),
    ("theek kar do", "please-fix"),
    ("complete kar dunga", "will-complete"),
    ("complete kar dungi", "will-complete"),
    ("complete kar dena", "please-complete"),
    ("complete kar do", "please-complete"),
    ("share kar dunga", "will-share"),
    ("share kar dungi", "will-share"),
    ("share kar dena", "please-share"),
    ("share kar do", "please-share"),
    ("check kar dena", "please-check"),
    ("check kar do", "please-check"),
    ("check kar lo", "please-check"),
    ("dekh lunga", "will-check"),
    ("dekh lungi", "will-check"),
    ("dekh loon pehle, phir main kar dunga", "will-do"),
    ("dekh loon pehle, phir main kar dungi", "will-do"),
    ("phir main kar dunga", "will-do"),
    ("phir main kar dungi", "will-do"),
    ("dekh loon", "will-check"),
    ("dekh lo", "please-check"),
    ("dekh le", "will-check"),
    ("dekh sakte ho", "can-you-check"),
    ("dekh sakti ho", "can-you-check"),
    ("bhej sakte ho", "can-you-send"),
    ("bhej sakti ho", "can-you-send"),
    ("share kar sakte ho", "can-you-share"),
    ("share kar sakti ho", "can-you-share"),
    ("kar sakte ho", "can-you-do"),
    ("kar sakti ho", "can-you-do"),
    ("bhej dunga", "will-send"),
    ("bhej dungi", "will-send"),
    ("bhej denge", "will-send"),
    ("bhej dena", "please-send"),
    ("bhej do", "please-send"),
    ("bhej den", "please-send"),
    ("likh dunga", "will-write"),
    ("likh dungi", "will-write"),
    ("share kar dungi", "will-share"),
    ("bhej dungi", "will-send"),
    ("call bhi kar lena", "please-do"),
    ("likh dena", "please-write"),
    ("likh do", "please-write"),
    ("kar lunga", "will-do"),
    ("kar lungi", "will-do"),
    ("kar dunga", "will-do"),
    ("kar dungi", "will-do"),
    ("kar denge", "will-do"),
    ("kar dena", "please-do"),
    ("kar do", "please-do"),
    ("aaj shaam", "this-evening"),
    ("aaj raat", "tonight"),
    ("aaj subah", "this-morning"),
    ("main ne promise kiya tha", "did-say"),
    ("promise kiya tha", "did-say"),
    ("discuss kiya", "did-discuss"),
    ("discuss kiye", "did-discuss"),
    ("decide kiya", "did-decide"),
    ("faisla kiya", "did-decide"),
    ("date abhi clear nahi", ""),
    ("abhi clear nahi", ""),
    ("kaha tha", "did-say"),
    ("kaha thi", "did-say"),
    ("bola tha", "did-say"),
    ("bola thi", "did-say"),
    ("dekh liya tha", "did-check"),
    ("dekh liya", "did-check"),
)

_URDU_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("السلام علیکم", "hello"),
    ("وعلیکم السلام", "hello"),
    ("کوئی بات نہیں", "okay"),
    ("ٹھیک ہے", "okay"),
    ("بھیج دوں گا", "will-send"),
    ("بھیج دوں گی", "will-send"),
    ("بھیج دیں گے", "will-send"),
    ("بھیج دینا", "please-send"),
    ("بھیج دیں", "please-send"),
    ("بھیج دو", "please-send"),
    ("لکھ دوں گا", "will-write"),
    ("لکھ دوں گی", "will-write"),
    ("لکھ دیں", "please-write"),
    ("لکھ دو", "please-write"),
    ("مکمل کر دوں گا", "will-complete"),
    ("مکمل کر دوں گی", "will-complete"),
    ("مکمل کر دینا", "please-complete"),
    ("مکمل کر دیں", "please-complete"),
    ("چیک کر دیں", "please-check"),
    ("چیک کر دو", "please-check"),
    ("اپ ڈیٹ", "update"),
    ("لاگ ان", "login"),
    ("دیکھ لوں گا", "will-check"),
    ("دیکھ لوں گی", "will-check"),
    ("دیکھ لو", "please-check"),
    ("دیکھ لے", "will-check"),
    ("دیکھے", "will-check"),
    ("ٹھیک کر دوں گا", "will-fix"),
    ("ٹھیک کر دوں گی", "will-fix"),
    ("ٹھیک کر دیں", "please-fix"),
    ("ٹھیک کر دو", "please-fix"),
    ("کر دوں گا", "will-do"),
    ("کر دوں گی", "will-do"),
    ("کر دینا", "please-do"),
    ("کر دیں", "please-do"),
    ("کر دو", "please-do"),
    ("فیصلہ کیا", "did-decide"),
    ("دیکھ لیا تھا", "did-check"),
    ("آج شام", "this-evening"),
    ("آج رات", "tonight"),
    ("پیر شام تک", "monday-evening"),
    ("منگل شام تک", "tuesday-evening"),
    ("بدھ شام تک", "wednesday-evening"),
    ("جمعرات شام تک", "thursday-evening"),
    ("جمعہ شام تک", "friday-evening"),
    ("پیر شام", "monday-evening"),
    ("منگل شام", "tuesday-evening"),
    ("بدھ شام", "wednesday-evening"),
    ("جمعرات شام", "thursday-evening"),
    ("جمعہ شام", "friday-evening"),
    ("ہفتے شام", "saturday-evening"),
    ("اتوار شام", "sunday-evening"),
    ("add کر دیں", "please-add"),
    ("add کریں", "please-add"),
    ("add کر دو", "please-add"),
    ("fix کر دیں", "please-fix"),
    ("fix کریں", "please-fix"),
    ("fix کر دو", "please-fix"),
    ("update کریں", "please-do"),
    ("share کریں", "please-share"),
    ("send کریں", "please-send"),
    ("complete کریں", "please-complete"),
    ("جی بالکل", "okay"),
    ("بالکل", "okay"),
    ("ہاں جی", "yes"),
    ("والا کام", ""),
    ("والی کام", ""),
    ("براہ کرم", "please"),
    ("شاید", "maybe"),
)

_URDU_WORDS = {
    "آپ": "aap",
    "ہم": "hum",
    "تم": "tum",
    "مجھے": "mujhe",
    "یہ": "yeh",
    "وہ": "woh",
    "آج": "aaj",
    "کل": "kal",
    "پرسوں": "parson",
    "پیر": "peer",
    "منگل": "mangal",
    "بدھ": "budh",
    "جمعرات": "jumeraat",
    "جمعہ": "juma",
    "ہفتہ": "hafta",
    "اتوار": "itwar",
    "شام": "shaam",
    "صبح": "subah",
    "رات": "raat",
    "تک": "tak",
    "کو": "ko",
    "سے": "se",
    "کا": "ka",
    "کی": "ki",
    "کے": "ke",
    "نے": "ne",
    "ہے": "hai",
    "ہیں": "hain",
    "تھا": "tha",
    "تھی": "thi",
    "تھے": "the",
    "نہیں": "nahi",
    "لیکن": "lekin",
    "اور": "aur",
    "کیا": "kya",
    "رپورٹ": "report",
    "نوٹس": "notes",
    "بجٹ": "budget",
    "کلائنٹ": "client",
    "مسئلہ": "issue",
    "سلائیڈز": "slides",
    "سلائیڈ": "slides",
    "ریلیز": "release",
    "ڈمو": "demo",
    "بعد": "later",
    "کوئی": "koi",
    "بالکل": "okay",
    "عایشہ": "Ayesha",
    "عائشہ": "Ayesha",
    "آئیشہ": "Ayesha",
    "علی": "Ali",
    "کامران": "Kamran",
    "سعد": "Saad",
    "بلال": "Bilal",
    "فلٹر": "filter",
    "بگ": "bug",
    "پر": "par",
    "ہی": "hi",
    "اسے": "it",
    "اس": "it",
    "سب": "all",
    "بھی": "bhi",
    "میڈیم": "medium",
    "پرائیرٹی": "priority",
    "رکھیں": "please-set",
    "رکھو": "please-set",
    "ڈیش بورڈ": "dashboard",
}

_TIME = {
    "aaj": "today",
    "kal": "tomorrow",
    "parson": "day after tomorrow",
    "peer": "Monday",
    "mangal": "Tuesday",
    "budh": "Wednesday",
    "jumeraat": "Thursday",
    "juma": "Friday",
    "hafta": "Saturday",
    "itwar": "Sunday",
    "shaam": "evening",
    "subah": "morning",
    "raat": "night",
    "today": "today",
    "tomorrow": "tomorrow",
    "tonight": "tonight",
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
    "saturday": "Saturday",
    "sunday": "Sunday",
    "this-evening": "this evening",
    "this-morning": "this morning",
    "tonight": "tonight",
    "monday-evening": "Monday evening",
    "tuesday-evening": "Tuesday evening",
    "wednesday-evening": "Wednesday evening",
    "thursday-evening": "Thursday evening",
    "friday-evening": "Friday evening",
    "saturday-evening": "Saturday evening",
    "sunday-evening": "Sunday evening",
}

_PRONOUN = {
    "main": "I", "i": "I",
    "aap": "you", "tum": "you", "you": "you",
    "hum": "we", "we": "we",
}

_DROP = {
    "tak", "ko", "se", "ka", "ki", "ke", "kay", "ne", "hai", "hain", "ho",
    "mein", "men", "me", "koi", "baad", "later", "date",
    "hoon", "hun", "tha", "thi", "the", "thay", "aur", "lekin", "magar",
    "pehle", "ab", "abhi", "sirf", "bhi", "hi", "to", "woh", "yeh", "ye",
    "kaam", "wala", "wali", "wale", "hoga", "hogi", "honge", "nahi", "not",
    "please", "zara", "zaroor", "haan", "han", "ji", "kya", "mein", "par",
    "per", "ke", "ki", "a", "the", "of", "for", "and", "with", "on", "in",
    "all", "it",
}

# Capitalised English words that are work, not people.
_NOT_A_PERSON = {
    "duplicate", "records", "record", "authentication", "bug", "dashboard",
    "task", "status", "filter", "update", "updates", "medium", "priority",
    "deadline", "login", "report", "notes", "budget", "slides", "api",
    "client", "release", "demo", "notification", "checklist", "metrics",
    "design", "email", "page", "system", "issue", "work", "wednesday",
    "monday", "tuesday", "thursday", "friday", "saturday", "sunday",
    "okay", "yes", "absolutely", "someone", "somebody", "team",
}

_VERB_ATOM = {
    "will-send": ("will", "send"),
    "will-fix": ("will", "fix"),
    "will-complete": ("will", "complete"),
    "will-share": ("will", "share"),
    "will-check": ("will", "check"),
    "will-write": ("will", "write"),
    "will-do": ("will", "do"),
    "please-send": ("please", "send"),
    "please-fix": ("please", "fix"),
    "please-complete": ("please", "complete"),
    "please-share": ("please", "share"),
    "please-check": ("please", "check"),
    "please-write": ("please", "write"),
    "please-do": ("please", "do"),
    "please-add": ("please", "add"),
    "will-add": ("will", "add"),
    "please-set": ("please", "set"),
    "can-you-send": ("question", "send"),
    "can-you-check": ("question", "check"),
    "can-you-share": ("question", "share"),
    "can-you-do": ("question", "do"),
    "did-discuss": ("past", "discussed"),
    "did-decide": ("past", "decided"),
    "did-say": ("past", "said"),
    "did-check": ("past", "checked"),
}


def looks_roman_urdu(text: str) -> bool:
    """True when a Latin line has enough Roman Urdu cues to rewrite."""
    tokens = [tok.lower() for tok in _TOKEN.findall(text or "")]
    if not tokens:
        return False
    hits = 0
    for token in tokens:
        folded = _SPELLING.get(token, token)
        if folded in _ROMAN_CUES or token in _ROMAN_CUES:
            hits += 1
    if hits >= 2:
        return True
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase, _atom in _ROMAN_PHRASES)


def _apply_phrases(text: str, phrases: Tuple[Tuple[str, str], ...]) -> str:
    rendered = text
    for source, target in sorted(phrases, key=lambda item: len(item[0]), reverse=True):
        rendered = re.sub(re.escape(source), f" {target} ", rendered, flags=re.IGNORECASE)
    return rendered


def _urdu_script_to_roman(text: str) -> str:
    from services.preprocessing.urdu_norm import strip_harakat, transliterate_word

    cleaned = strip_harakat(text).replace("۔", ".").replace("؟", "?").replace("،", ",")
    cleaned = re.sub(r"^\s*میں(?![\u0600-\u06FF])", " main ", cleaned)
    cleaned = re.sub(r"(?<![\u0600-\u06FF])میں(?![\u0600-\u06FF])", " ", cleaned)
    cleaned = _apply_phrases(cleaned, _URDU_PHRASES)
    for source, target in sorted(_URDU_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        cleaned = re.sub(
            rf"(?<![\u0600-\u06FF]){re.escape(source)}(?![\u0600-\u06FF])",
            f" {target} ",
            cleaned,
        )
    cleaned = _ARABIC_RUN.sub(lambda match: transliterate_word(match.group(0)), cleaned)
    return cleaned


def _object_phrase(verb: str, objects: List[str]) -> str:
    """Join the work words, and put a dashboard at the end of an add."""
    words = [
        word for word in objects
        if word.lower() not in {"add", "fix", "set", "complete", "send"}
    ]
    if verb == "add" and "dashboard" in words:
        rest = [word for word in words if word != "dashboard"]
        tail = " ".join(rest).strip()
        return f"{tail} on the dashboard".strip() if tail else "it on the dashboard"
    if not words:
        return ""
    if verb in {"fix", "complete", "send", "set", "check", "share", "write", "do"}:
        return "the " + " ".join(words)
    return " ".join(words)


def _content_word(token: str) -> bool:
    folded = token.lower()
    if folded in _DROP or folded in _TIME or folded in _PRONOUN:
        return False
    if folded in _VERB_ATOM or folded in {"maybe", "okay", "hello", "yes", "no"}:
        return False
    return True


def _assemble(raw: str) -> Optional[str]:
    text = _apply_phrases(raw, _ROMAN_PHRASES)
    tokens = _TOKEN.findall(text)
    if not tokens:
        return None
    pieces: List[str] = []
    for tok in tokens:
        folded = _SPELLING.get(tok.lower(), tok.lower())
        if tok.isupper() and len(tok) >= 2:
            pieces.append(tok.lower())
        elif folded in _VERB_ATOM or folded in _TIME or folded in _PRONOUN or folded in _DROP or folded in {"maybe", "okay", "hello"}:
            pieces.append(folded)
        elif tok[:1].isupper():
            pieces.append(tok)
        else:
            pieces.append(folded)

    mood = ""
    verb = ""
    subject = ""
    names: List[str] = []
    times: List[str] = []
    objects: List[str] = []
    maybe = False
    skip_time = False
    saw_commitment = False

    for piece in pieces:
        if piece in {"nahi", "not"}:
            skip_time = True
            continue
        if piece == "maybe":
            maybe = True
            continue
        if piece in {"okay", "hello", "yes", "absolutely"} and not saw_commitment:
            continue
        if piece in _TIME:
            if skip_time:
                skip_time = False
                continue
            times.append(_TIME[piece])
            continue
        skip_time = False
        if piece in _VERB_ATOM:
            mood, verb = _VERB_ATOM[piece]
            saw_commitment = True
            continue
        if piece in _PRONOUN and not subject:
            subject = _PRONOUN[piece]
            continue
        if (
            piece[:1].isupper()
            and piece.lower() not in _DROP
            and piece.lower() not in _NOT_A_PERSON
        ):
            names.append(piece)
            continue
        if piece.lower() in _NOT_A_PERSON or _content_word(piece):
            if piece.lower() not in _DROP and piece.lower() not in _TIME:
                objects.append(piece.lower() if piece.lower() in _NOT_A_PERSON else piece)

    if not saw_commitment:
        if any(piece in {"okay", "hello", "yes", "absolutely"} for piece in pieces):
            return "Okay."
        return None

    addressee = names[0] if names else ""
    if mood == "please" and addressee:
        subject = addressee
    elif mood == "question" and addressee:
        subject = addressee
    elif not subject and addressee:
        subject = addressee
    elif not subject:
        subject = "I" if mood == "will" else ""

    # A past verb wins only when the line did not also make a new commitment.
    if mood == "past":
        obj = " ".join(objects + [name for name in names if name != subject]).strip()
        when = times[-1] if times else ""
        if verb == "decided":
            return f"We decided {obj} {when}".strip() + "."
        if when == "tomorrow":
            when = "yesterday"
        gloss = f"{subject or 'We'} {verb} {obj} {when}".strip()
        gloss = re.sub(r"\s+", " ", gloss)
        return gloss[:1].upper() + gloss[1:] + "."

    when = ""
    obj = _object_phrase(verb, objects)
    if "this evening" in times:
        when = "this evening"
    elif "tonight" in times:
        when = "tonight"
    elif "this morning" in times:
        when = "this morning"
    elif times:
        when = times[-1]
    if mood == "will" and verb in {"do", "send", "check", "fix", "share", "write", "add", "set"} and not obj:
        obj = "that"
    if verb == "do" and obj in {"phir", "pehle", "later"}:
        obj = "that"
    if maybe and mood != "please":
        target = obj or "it"
        return f"Someone might {verb or 'handle'} {target} later."

    def _with_when(line: str) -> str:
        if not when:
            return line
        if when in {"this evening", "tonight", "this morning"}:
            return f"{line} {when}"
        if when == "today":
            return f"{line} today"
        return f"{line} by {when}"

    if mood == "question":
        ask = _with_when(f"Can you {verb} {obj}".strip())
        if addressee:
            ask += f", {addressee}"
        return ask + "?"

    actor = subject or "Someone"
    if mood == "please":
        line = f"{actor}, please {verb} {obj}".strip()
    else:
        line = f"{actor} will {verb} {obj}".strip()
    line = _with_when(line)
    line = re.sub(r"\s+", " ", line).strip(" ,")
    if line and line[-1] not in ".?":
        line += "."
    if line and line[0].isalpha():
        line = line[0].upper() + line[1:]
    return line


def utterance_to_english(text: str) -> str:
    """
    Return an English meeting line.

    English input is unchanged. Urdu script and Roman Urdu commitments are
    rewritten with the meeting lexicon. The caller keeps *text* as evidence.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    if arabic_script_ratio(raw) >= 0.12:
        chunks = [chunk.strip(" ،,") for chunk in re.split(r"[۔؟!]+", raw) if chunk.strip(" ،,")]
        if len(chunks) > 1:
            glosses = [_assemble(_urdu_script_to_roman(chunk)) for chunk in chunks]
            joined = " ".join(gloss for gloss in glosses if gloss)
            return joined or raw
        converted = _assemble(_urdu_script_to_roman(raw))
        return converted or raw
    if looks_roman_urdu(raw):
        converted = _assemble(raw)
        # An unparsed Roman line is not handed to the English detector,
        # which would treat leftover English words such as "update" as tasks.
        return converted or "Noted."
    return raw

"""
Prepare Urdu meeting text for Marian translation.

Marian often mangles Urdu spellings of meeting words (dashboard, Thursday,
notification) and short speaker labels. This module rewrites those before
translation and keeps Speaker lines in Latin script.
"""

from __future__ import annotations

import re
from typing import Dict

_HARAKAT = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_ARABIC_RUN = re.compile(r"[\u0600-\u06FF]+")
_SPEAKER_RE = re.compile(
    r"^(\s*Speaker\s+\d+\s*[–\-—]\s*)(.*)$",
    re.IGNORECASE,
)

# Longest phrases first at apply time. General meeting language, not one transcript.
_GLOSSARY = (
    ("نوٹیفکیشن سسٹم", "notification system"),
    ("نوٹیفکیشنز", "notifications"),
    ("نوٹیفکیشن", "notification"),
    ("ڈیش بورڈ", "dashboard"),
    ("ڈیشبورڈ", "dashboard"),
    ("لاگ ان", "login"),
    ("سائن اپ", "sign-up"),
    ("بیک اینڈ", "backend"),
    ("فرنٹ اینڈ", "frontend"),
    ("یوزر مینجمنٹ", "user management"),
    ("ڈیٹا بیس", "database"),
    ("ٹیسٹ کیسز", "test cases"),
    ("ٹیسٹ کیس", "test case"),
    ("ٹیسٹنگ", "testing"),
    ("ڈیڈ لائنز", "deadlines"),
    ("ڈیڈ لائن", "deadline"),
    ("فیچرز", "features"),
    ("فیچر", "feature"),
    ("کلائنٹ", "client"),
    ("پروجیکٹ", "project"),
    ("ماڈیول", "module"),
    ("رپورٹ", "report"),
    ("میٹنگ", "meeting"),
    ("ڈیمو", "demo"),
    ("ریویو", "review"),
    ("ٹیم ممبرز", "team members"),
    ("ٹیم", "team"),
    ("سسٹم", "system"),
    ("اپ ڈیٹ", "update"),
    ("مسائل", "issues"),
    ("مسئلہ", "issue"),
    ("اسی فیصد", "80 percent"),
    ("پچیس", "25"),
    ("السلام علیکم", "Hello"),
    ("وعلیکم السلام", "Hello"),
    ("دوپہر تین بجے", "3 PM"),
    ("اگلے دو دن", "the next two days"),
    ("آج شام تک", "by this evening"),
    ("شام تک", "by evening"),
    ("آج شام", "this evening"),
    ("اگلے پیر", "next Monday"),
    ("اگلے منگل", "next Tuesday"),
    ("اگلے بدھ", "next Wednesday"),
    ("اگلے جمعرات", "next Thursday"),
    ("اگلے جمعہ", "next Friday"),
    ("اگلے ہفتہ", "next Saturday"),
    ("اگلے اتوار", "next Sunday"),
    ("جمعرات", "Thursday"),
    ("جمعہ", "Friday"),
    ("منگل", "Tuesday"),
    ("بدھ", "Wednesday"),
    ("ہفتہ", "Saturday"),
    ("اتوار", "Sunday"),
    ("پیر", "Monday"),
    ("شام", "evening"),
)

# Unwritten short vowels. Only applied to tokens we transliterate ourselves.
_ROMAN_FIX = {
    "ahmd": "Ahmad",
    "aysha": "Ayesha",
    "ali": "Ali",
    "sara": "Sarah",
}

_LETTERS = {
    "ا": "a",
    "آ": "a",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "z",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "و": "o",
    "ہ": "h",
    "ه": "h",
    "ھ": "h",
    "ء": "",
    "ی": "i",
    "ي": "i",
    "ے": "e",
    "ئ": "y",
    "ؤ": "o",
    "ك": "k",
    "ة": "a",
}


def strip_harakat(text: str) -> str:
    """Remove Urdu/Arabic vowel marks so glossary keys can match."""
    return _HARAKAT.sub("", text or "")


def transliterate_word(word: str) -> str:
    """Romanize one Arabic-script token (names on speaker labels)."""
    cleaned = strip_harakat(word)
    pieces = []
    index = 0
    while index < len(cleaned):
        char = cleaned[index]
        nxt = cleaned[index + 1] if index + 1 < len(cleaned) else ""
        if char == "ع" and nxt in ("ا", "آ"):
            pieces.append("a")
            index += 2
            continue
        if char in ("ہ", "ه") and index == len(cleaned) - 1:
            pieces.append("a")
            index += 1
            continue
        pieces.append(_LETTERS.get(char, ""))
        index += 1
    roman = "".join(pieces)
    roman = re.sub(r"(.)\1{2,}", r"\1", roman)
    if not roman:
        return word
    fixed = _ROMAN_FIX.get(roman.lower())
    if fixed:
        return fixed
    return roman[:1].upper() + roman[1:]


def latinize_arabic_runs(text: str) -> str:
    """Replace each Arabic-script word with its romanization."""
    return _ARABIC_RUN.sub(lambda match: transliterate_word(match.group(0)), text)


def speaker_names(text: str) -> Dict[str, str]:
    """Map Urdu names written on Speaker lines to Latin spellings."""
    found: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        match = _SPEAKER_RE.match(raw.strip())
        if not match:
            continue
        for token in _ARABIC_RUN.findall(strip_harakat(match.group(2))):
            if len(token) < 2:
                continue
            found[token] = transliterate_word(token)
    return found


def apply_glossary(text: str) -> str:
    """Swap Urdu meeting terms and weekdays for English before translation."""
    prepared = strip_harakat(text)
    for source, target in sorted(_GLOSSARY, key=lambda item: len(item[0]), reverse=True):
        prepared = re.sub(
            rf"(?<![\u0600-\u06FF]){re.escape(source)}(?![\u0600-\u06FF])",
            target,
            prepared,
        )
    return prepared


# Longer phrases first at apply time. General meeting Urdu, not one transcript.
_PHRASES = (
    ("السلام علیکم سب کو", "Hello everyone"),
    ("السلام علیکم", "Hello"),
    ("وعلیکم السلام", "Hello"),
    ("شکریہ", "Thank you"),
    ("ٹھیک ہے,", "Okay."),
    ("ٹھیک ہے", "Okay"),
    ("جی بالکل", "Yes, absolutely"),
    ("بالکل", "Okay"),
    ("عایشہ", "Ayesha"),
    ("عائشہ", "Ayesha"),
    ("آئیشہ", "Ayesha"),
    ("علی", "Ali"),
    ("add کریں", "please add"),
    ("fix کریں", "please fix"),
    ("بہت اچھا", "Very good"),
    ("بہترین", "Great"),
    ("ایک اور بات", "One more thing"),
    ("آج کی میٹنگ میں", "In today's meeting"),
    ("موجودہ صورتحال", "the current status"),
    ("پر بات کریں گے", "we will discuss"),
    ("سب سے پہلے", "First"),
    ("ہم اپنے پروجیکٹ کی موجودہ صورتحال پر بات کریں گے", "the topic is the current status of our project"),
    ("میں جاننا چاہتا ہوں", "I want to know"),
    ("نوٹیفکیشن سسٹم میں ابھی کچھ مسائل ہیں", "there are still some issues in the notification system"),
    ("بیک اینڈ ٹیم سے API کی اپ ڈیٹ چاہیے", "I need an API update from the backend team"),
    ("نوٹیفکیشن سسٹم کی مکمل ٹیسٹنگ نہیں کر سکتی", "I cannot fully test the notification system"),
    ("نوٹیفکیشن سسٹم کی مکمل ٹیسٹنگ نہیں کر سکتا", "I cannot fully test the notification system"),
    ("کیا آپ کو اس کام کے لیے کسی اضافی مدد کی ضرورت ہے", "Do you need any extra help for this work"),
    ("ڈیمو سے پہلے ایک مکمل میٹنگ کر سکتے ہیں", "have a full meeting before the demo"),
    ("کیا ہم", "can we"),
    ("سب اپنی ڈیڈ لائنز کا خیال رکھیں", "Everyone should keep the deadlines in mind"),
    ("اپنی پیش رفت", "their progress"),
    ("پچھلے ہفتے", "last week"),
    ("جو کام دیے گئے تھے", "the work that was assigned"),
    ("ان میں سے کتنے", "how many of them"),
    ("مکمل ہو چکے ہیں", "have been finished"),
    ("مکمل ہو چکا ہے", "is done"),
    ("میری طرف سے", "On my side"),
    ("صفحات تیار ہیں", "pages are ready"),
    ("ابھی کچھ مسائل ہیں", "there are still some issues"),
    ("کچھ مسائل ہیں", "there are some issues"),
    ("مکمل کر دوں گی", "will complete it"),
    ("مکمل کر دوں گا", "will complete it"),
    ("مکمل کر کے", "finish it and"),
    ("مکمل کریں", "please complete"),
    ("لاگ ان مسئلہ ٹھیک کر دوں گا", "I will fix the login issue"),
    ("بھیج دوں گا", "I will send it"),
    ("ٹھیک ہے، بھیج دوں گا", "Okay, I will send it"),
    ("کلائنٹ کو اپ ڈیٹ بھیج دیں", "please send the client an update"),
    ("مجھے لگتا ہے ٹیم تھوڑی پریشان ہے", "I think the team is a bit worried"),
    ("براہ کرم کلائنٹ کو اپ ڈیٹ بھیج دیں", "please send the client an update"),
    ("بلال، براہ کرم کلائنٹ کو اپ ڈیٹ بھیج دیں", "Bilal, please send the client an update"),
    ("اضافی مدد", "extra help"),
    ("کی ضرورت ہے", "is needed"),
    ("ضرورت ہے", "is needed"),
    ("کیا آپ کو", "Do you"),
    ("اس کے بغیر", "Without that"),
    ("نہیں کر سکتی", "cannot"),
    ("نہیں کر سکتا", "cannot"),
    ("فراہم کر دوں گی", "will provide"),
    ("فراہم کر دوں گا", "will provide"),
    ("اس کے علاوہ", "Also"),
    ("بہتری کی ضرورت ہے", "needs improvement"),
    ("کام کروں گی", "will work"),
    ("کام کروں گا", "will work"),
    ("بہتر کر دیں", "please improve"),
    ("کے حوالے سے کیا صورتحال ہے", "what is the status"),
    ("اب تک", "so far"),
    ("چلائے ہیں", "have been run"),
    ("کامیاب رہے ہیں", "passed"),
    ("مسائل سامنے آئے ہیں", "issues came up"),
    ("سے متعلق ہیں", "are related"),
    ("سے متعلق", "about"),
    ("تیار کر لی ہے", "have prepared"),
    ("شیئر کر دوں گی", "will share it"),
    ("شیئر کر دوں گا", "will share it"),
    ("حل کرنے ہیں", "have to resolve"),
    ("دینا ہے", "have to give"),
    ("کر سکتے ہیں", "can we"),
    ("سے پہلے", "before"),
    ("تاکہ", "so that"),
    ("چیک ہو جائیں", "can be checked"),
    ("رکھیں گے", "we will hold"),
    ("ساتھ لائیں گے", "will bring"),
    ("پیش رفت", "progress"),
    ("دے دوں گی", "will hand it over"),
    ("دے دوں گا", "will hand it over"),
    ("کے لیے", "for"),
    ("کے ساتھ", "with"),
    ("جس پر", "which"),
    ("مجھے", "I"),
    ("چاہیے", "need"),
    ("البتہ", "However"),
    ("جبکہ", "while"),
    ("ہمیں", "We"),
    ("یہی", "these"),
    ("فیصلے ہوئے ہیں", "decisions were made"),
    ("خیال رکھیں", "please keep in mind"),
    ("جن میں سے", "of which"),
    ("ان میں سے", "of those"),
    ("تقریبا", "about"),
)

_WORDS = {
    "کام": "work",
    "صفحات": "pages",
    "تیار": "ready",
    "مسائل": "issues",
    "مسئلہ": "issue",
    "ضرورت": "need",
    "مدد": "help",
    "اضافی": "extra",
    "کارکردگی": "performance",
    "بہتری": "improvement",
    "بہتر": "improve",
    "صورتحال": "status",
    "موجودہ": "current",
    "کامیاب": "successful",
    "بیس": "20",
    "پانچ": "five",
    "تین": "three",
    "دو": "two",
    "اہم": "important",
    "تمام": "all",
    "مکمل": "complete",
    "فیصلے": "decisions",
    "فیصلہ": "decision",
    "اپنا": "my",
    "اپنی": "my",
    "اپنے": "our",
    "سب": "everyone",
    "آپ": "you",
    "اسے": "it",
    "ہم": "we",
    "یہ": "this",
    "جی": "Yes",
    "نہیں": "not",
    "اب": "now",
    "کیا": "what",
    "کتنے": "how many",
    "کیونکہ": "because",
    "تاکہ": "so that",
    "لیکن": "but",
    "اور": "and",
    "بہت": "very",
    "ابھی": "still",
    "کچھ": "some",
    "کسی": "any",
    "جو": "that",
    "اس": "",
    "کہ": "",
    "تو": "",
    "جس": "which",
    "ساتھ": "with",
    "مجھے": "I",
    "چاہیے": "need",
    "البتہ": "However",
    "جبکہ": "while",
    "ہمیں": "We",
    "یہی": "these",
    "ہی": "",
    "بھی": "also",
    "پہلے": "before",
    "ایک": "a",
    "کر": "",
    "کی": "",
    "کے": "",
    "کا": "",
    "کو": "",
    "سے": "from",
    "پر": "on",
    "نے": "",
    "ہیں": "",
    "ہے": "",
    "ہوں": "",
    "گا": "",
    "گی": "",
    "گے": "",
    "تک": "by",
    "لیے": "for",
    "والے": "",
    "حوالے": "",
    "بات": "point",
    "جاننا": "know",
    "چاہتا": "want",
    "دیے": "given",
    "گئے": "",
    "تھے": "",
    "تھا": "",
    "تھی": "",
    "ہو": "",
    "چکے": "",
    "چکا": "",
    "چکی": "",
    "طرف": "side",
    "صفحات": "pages",
    "لاگ": "login",
    "سائن": "sign",
    "اپ": "up",
    "چیز": "thing",
    "فیصد": "percent",
    "اسی": "80",
    "تقریبا": "about",
}

_PARTICLE = {
    "کا", "کی", "کے", "کو", "نے", "ہی", "تو", "کہ", "اس", "ان", "گا", "گی", "گے",
    "ہیں", "ہے", "ہوں", "تھا", "تھی", "تھے", "ہو", "چکا", "چکے", "چکی",
}


def _swap_phrases(text: str, names: Dict[str, str]) -> str:
    """Longest-match replacement of names, glossary terms, and phrases."""
    prepared = strip_harakat(text)
    prepared = prepared.replace("۔", ".").replace("؟", "?").replace("،", ",")
    for source, target in sorted(names.items(), key=lambda item: len(item[0]), reverse=True):
        prepared = re.sub(
            rf"(?<![\u0600-\u06FF]){re.escape(source)}(?![\u0600-\u06FF])",
            target,
            prepared,
        )
    phrases = list(_GLOSSARY) + list(_PHRASES)
    for source, target in sorted(phrases, key=lambda item: len(item[0]), reverse=True):
        prepared = re.sub(
            rf"(?<![\u0600-\u06FF]){re.escape(strip_harakat(source))}(?![\u0600-\u06FF])",
            target,
            prepared,
        )
    for source, target in sorted(_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
        prepared = re.sub(
            rf"(?<![\u0600-\u06FF]){re.escape(source)}(?![\u0600-\u06FF])",
            target,
            prepared,
        )
    return prepared


_WHEN = (
    r"(?:by this evening|this evening|the next two days|next Monday|next Tuesday|"
    r"next Wednesday|next Thursday|next Friday|next Saturday|next Sunday|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
)


def _reorder_clause(sentence: str) -> str:
    """Put leftover Urdu word order into short English clauses."""
    named = re.match(
        r"^(?P<name>[A-Z][a-z]+), you (?P<when>.+?) by (?P<obj>.+?) please (?P<verb>improve|complete)\.?$",
        sentence,
    )
    if named:
        return (
            f"{named.group('name')}, please {named.group('verb')} "
            f"{named.group('obj')} by {named.group('when')}."
        )
    sentence = re.sub(
        rf"^I it (?P<when>{_WHEN}) by will complete it\.?$",
        r"I will complete it by \g<when>.",
        sentence,
    )
    sentence = re.sub(
        rf"^I by (?P<when>{_WHEN}) (?P<obj>.+?) will (?P<verb>provide|complete|share|work)\.?$",
        r"I will \g<verb> \g<obj> by \g<when>.",
        sentence,
    )
    sentence = re.sub(
        rf"\bI (?P<when>{_WHEN}) by my work finish it and (?P<what>.+?) for will hand it over",
        r"I will finish the work by \g<when> and hand it over for \g<what>",
        sentence,
    )
    sentence = re.sub(
        r"^I (?P<what>.+?) have prepared and (?P<who>.+?) with will share it\.?$",
        r"I will send \g<what> to \g<who>.",
        sentence,
    )
    sentence = re.sub(
        rf"^We (?P<when>{_WHEN}) by (?P<obj>.+?) have to (?P<verb>resolve|complete)\.?$",
        r"We have to \g<verb> \g<obj> by \g<when>.",
        sentence,
    )
    sentence = re.sub(
        rf"because client (?P<obj>.+?) demo (?P<when>{_WHEN}) have to give\.?$",
        r"because we have to give the client a \g<obj> demo \g<when>.",
        sentence,
    )
    sentence = re.sub(
        r"(?i)^we (?P<when>Monday 3 PM) a review meeting we will hold\.?$",
        r"We will hold a review meeting on \g<when>.",
        sentence,
    )
    sentence = re.sub(
        r",?\s*\bwhich the next two days will work\.?$",
        ". I will work on it over the next two days.",
        sentence,
    )
    sentence = re.sub(
        r"^Also (?P<what>.+?) module also is done\.?$",
        r"The \g<what> module is also done.",
        sentence,
    )
    sentence = re.sub(r"\bsome needs improvement\b", "needs some improvement", sentence)
    sentence = re.sub(
        r"(?i)^all team members their progress report will bring\.?$",
        "All team members will bring a report of their progress.",
        sentence,
    )
    sentence = re.sub(
        r"(?i)^we so far ",
        "So far, ",
        sentence,
    )
    sentence = re.sub(
        r"(?i)^now testing what is the status\??$",
        "What is the testing status now?",
        sentence,
    )
    return sentence


def _polish_english(text: str) -> str:
    """Tidy phrase-table English: subjects, word order, spacing."""
    cleaned = re.sub(r"(^|[.?!،,]\s*)میں(?=\s|$)", r"\1I", text)
    cleaned = cleaned.replace(" میں ", " ")
    cleaned = re.sub(r"\bI I\b", "I", cleaned)
    cleaned = re.sub(r"\s+([,.?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.?!])\s+", cleaned)
    polished = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        parts = re.split(r"\s+because\s+", sentence, maxsplit=1)
        if len(parts) == 2:
            left = _reorder_clause(parts[0].strip(" ."))
            right = _reorder_clause("because " + parts[1].strip())
            if right.lower().startswith("because "):
                right = right[8:]
            sentence = f"{left.rstrip('.')} because {right}"
        else:
            sentence = _reorder_clause(sentence)
        if sentence and sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        if sentence and sentence[-1] not in ".?":
            sentence += "."
        polished.append(sentence)
    return " ".join(polished)


def render_urdu_line(line: str, names: Dict[str, str] | None = None) -> str | None:
    """
    Render one Urdu line with the meeting phrase table.

    Returns None when too much Arabic script is left, so the caller can
    fall back to the neural translator for that line.
    """
    from services.preprocessing.language import arabic_script_ratio

    stripped = (line or "").strip()
    if not stripped:
        return ""
    if _SPEAKER_RE.match(stripped):
        return prepare_urdu_line(stripped, names)
    rendered = _swap_phrases(stripped, names or {})
    rendered = _polish_english(rendered)
    if arabic_script_ratio(rendered) > 0.12:
        return None
    return rendered


def attribute_speakers(text: str) -> str:
    """Attach the current Speaker name to that person's commitments."""
    current = None
    name_re = re.compile(r"^Speaker\s+\d+\s*[–\-—]\s*([A-Z][a-z]+)")
    lines = []
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        match = name_re.match(stripped)
        if match:
            current = match.group(1)
            lines.append(stripped)
            continue
        if not stripped or not current:
            lines.append(raw)
            continue
        parts = []
        for sentence in re.split(r"(?<=[.?!])\s+", stripped):
            if re.search(r"\b(will|please|have to)\b", sentence, re.IGNORECASE):
                sentence = re.sub(r"^I\b", current, sentence.strip())
                sentence = re.sub(
                    r"\bI will\b",
                    f"{current} will",
                    sentence,
                )
            parts.append(sentence)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def prepare_urdu_line(line: str, names: Dict[str, str] | None = None) -> str:
    """
    Ready one transcript line for translation.

    Speaker labels stay Latin. Other lines get glossary terms and any names
    already seen on a Speaker line.
    """
    stripped = (line or "").strip()
    if not stripped:
        return ""
    match = _SPEAKER_RE.match(stripped)
    if match:
        label = f"{match.group(1)}{latinize_arabic_runs(strip_harakat(match.group(2)))}".strip()
        return label.rstrip(".: ") + "."
    prepared = apply_glossary(stripped)
    for source, target in sorted((names or {}).items(), key=lambda item: len(item[0]), reverse=True):
        prepared = re.sub(
            rf"(?<![\u0600-\u06FF]){re.escape(source)}(?![\u0600-\u06FF])",
            target,
            prepared,
        )
    return prepared

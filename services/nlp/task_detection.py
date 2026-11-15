"""
services/nlp/task_detection.py
---------------------------
spaCy + Regex action-item / task detection engine.
Extracted from Clario Project notebook (cells 110-112, 114, 121).

Improvements over v1
--------------------
- Negation guard  : sentences containing "will not / won't / nobody should /
                    don't / doesn't / didn't / no one" are skipped.
- Hypothetical guard : sentences starting with "if / whether" or containing
                    "would / could / might / may" without an obligation cue
                    are skipped.
- Question guard  : interrogative sentences (ending '?' OR auxiliary-led
                    without an obligation cue) are filtered.
- Obligation cues : extended with "responsible for", "assigned to",
                    "can you", "could you", "let's", "have to", "make sure",
                    "follow up", "ensure", "take care of", "action item".
- Weak-verb guard : generic verbs (review, check, discuss, consider) only
                    trigger if an obligation cue is also present.
- Dep-parse check : sentences without a keyword cue are accepted when spaCy's
                    dependency parse finds a modal/future ROOT with a subject
                    that is not a pronoun (or is a first-person pronoun with a
                    commitment verb).
- Relative dates  : "tomorrow", "next Friday", "Monday" etc. are resolved to
                    ISO YYYY-MM-DD via dateparser when a meeting_date is
                    passed through detect_tasks; the original phrase is kept
                    in `due_date` and the ISO form in `due_date_iso`.
- participants    : keyword-only param from Google Meet integration — unchanged.
"""

from __future__ import annotations

import re
import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level NLP cache
# ---------------------------------------------------------------------------
_nlp = None
_nlp_unavailable = False


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Strong obligation/request cues — any one is sufficient to flag a sentence
STRONG_CUES: List[str] = [
    r"\bwill\b",
    r"\bneeds?\s+to\b",
    r"\bmust\b",
    r"\bhas\s+to\b",
    r"\bhave\s+to\b",
    r"\bplease\b",
    r"\bdeadline\b",
    r"\bresponsible\s+for\b",
    r"\bassigned\s+to\b",
    r"\bcan\s+you\b",
    r"\bcould\s+you\b",
    r"\bi\s+want\s+you\s+to\b",
    r"\bi\s+need\s+you\s+to\b",
    r"\bi(?:'d| would)\s+like\s+you\s+to\b",
    r"\blet['']?s\b",
    r"\bmake\s+sure\b",
    r"\baction\s+item\b",
    r"\bfollow[\s-]up\b",
    r"\btake\s+care\s+of\b",
    r"\bensure\b",
    r"\bpromised\s+to\b",
    r"\bit\s+is\s+important\s+that\b",   # "it is important that John submits"
    r"\bit['']?s\s+important\s+that\b",  # "it's important that"
    r"\bit\s+would\s+be\s+(?:great|nice|helpful|good|ideal|best|wonderful|appreciated)\s+if\b",
    r"\bi\s+was\s+hoping\s+(?:you|someone)\s+could\b",
    r"\bwould\s+(?:you|someone)\s+(?:mind|be\s+able\s+to)\b",
]

# Verbs that imply a concrete action even without a strong cue
STRONG_ACTION_VERBS: List[str] = [
    r"\bsubmit\b",
    r"\bcomplete\b",
    r"\bfinish\b",
    r"\bsend\b",
    r"\bfix\b",
    r"\bhandle\b",
    r"\bprepare\b",
    r"\bdeliver\b",
    r"\bcoordinate\b",
    r"\bschedule\b",
    r"\bupdate\b",
    r"\bdeploy\b",
    r"\bpublish\b",
    r"\brelease\b",
    r"\bbuild\b",
    r"\bcreate\b",
    r"\bwrite\b",
    r"\bdraft\b",
    r"\btest\b",
    r"\bimplement\b",
    r"\btrack\b",
    r"\bmonitor\b",
    r"\bfinaliz[es]\b",
    r"\brefactor\b",
    r"\bmigrate\b",
    r"\bprototype\b",
    r"\bdocument\b",
    r"\bset\s*up\b",
    r"\bresolve\b",
]

# Weak verbs that only trigger a task if an obligation cue is ALSO present
WEAK_VERBS: List[str] = [
    r"\breview\b",
    r"\bcheck\b",
    r"\bdiscuss\b",
    r"\bconsider\b",
    r"\blook\s+at\b",
    r"\bthink\s+about\b",
    r"\btalk\s+about\b",
    r"\bexplore\b",
]

# Combined TASK_KEYWORDS kept for public API compatibility
TASK_KEYWORDS: List[str] = STRONG_CUES + STRONG_ACTION_VERBS

PRIORITY_KEYWORDS = {
    "High":   [    r"\burgent\b", r"\basap\b", r"\bimmediately\b",
               r"\bcritical\b", r"\bmust\b", r"\bimportant\b",
               r"\bhigh\s+priority\b",
               r"\bit\s+is\s+important\b", r"\bit['']?s\s+important\b"],
    "Medium": [r"\bsoon\b", r"\bby this week\b", r"\bshould\b",
               r"\bneeds?\s+to\b", r"\bmedium\s+priority\b"],
}

# ---------------------------------------------------------------------------
# Negation / hypothetical / question filters
# ---------------------------------------------------------------------------

# If any of these match, skip the sentence (high-confidence non-tasks)
# A status remark ("it's recording", "we are live") is not an assignment.
_STATUS_RE = re.compile(
    r"\b(?:it(?:'s| is)|this is|that(?:'s| is)|we are|they are)\s+"
    r"(?:now\s+)?(?:recording|recorded|live|done|ready|working|happening|over)\b",
    re.IGNORECASE,
)


_NEGATION_RE = re.compile(
    r"\b(won'?t|will\s+not|not\s+be|nobody|no\s+one|n'?t\b|didn'?t|"
    r"doesn'?t|don'?t|never|cannot|can'?t)\b",
    re.IGNORECASE,
)

_HYPOTHETICAL_TRIGGERS = re.compile(
    r"^(if|whether)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_CUES = re.compile(
    r"\b(would|could|might|may|perhaps|possibly|probably)\b",
    re.IGNORECASE,
)

# WH-words that always make a sentence a question (even if it has obligation cues)
_WH_QUESTION_RE = re.compile(
    r"^\s*(what|who|where|when|why|how|which|whom|whose)\b",
    re.IGNORECASE,
)

# Question: ends with '?' or starts with a WH/aux word
_QUESTION_RE = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|will|would|should|"
    r"could|can|may|might|shall|what|who|where|when|why|how)\b",
    re.IGNORECASE,
)

# "yesterday" or "last [day]" signals a past-tense/completed event
_PAST_CONTEXT_RE = re.compile(
    r"\b(yesterday|last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)|"
    r"ago|earlier\s+today|this\s+morning\s+already)\b",
    re.IGNORECASE,
)

# "should" on its own is not an obligation unless there's a named subject
_WEAK_SHOULD_RE = re.compile(r"^\s*\bshould\b", re.IGNORECASE)

# Relative date expressions understood by dateparser
# Polite-indirect patterns that ARE tasks despite containing "would/could"
# e.g. "It would be great if Sara could handle the slides"
_POLITE_REQUEST_RE = re.compile(
    r"\bit\s+would\s+be\s+(?:great|nice|helpful|good|ideal|best|wonderful|appreciated)\s+if\b"
    r"|\bi\s+was\s+hoping\s+(?:you|someone)\s+could\b"
    r"|\bwould\s+(?:you|someone)\s+(?:mind|be\s+able\s+to)\b",
    re.IGNORECASE,
)

_ABSOLUTE_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june|july|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
    r"apr(?:il)?|may|june|july|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)

_RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|yesterday|next\s+\w+|last\s+\w+|this\s+\w+|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"end\s+of\s+(?:day|week|month)|by\s+(?:end\s+of\s+)?(?:day|week)|"
    r"tonight|this\s+morning|this\s+afternoon|this\s+evening)\b",
    re.IGNORECASE,
)

# "Name, can you please … and Other, please …" is more than one request.
_NAME_WORD = r"(?!(?i:please|can|could|will|would|you|i|need)\b)[A-Z][\w'.-]+"
_NAME_RE = rf"{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}}"
_POLITE_PREFIX_RE = (
    r"(?:can you please|could you please|would you please|will you please|"
    r"i need you to|i want you to|i would like you to|i'd like you to|"
    r"can you|could you|will you|would you|please|"
    r"complete|finish|submit|send|fix|prepare|update|review|check|handle|"
    r"deliver|create|write|draft|deploy|schedule)"
)
# A comma is required before a bare verb ("Sara, send"). Speech often drops the
# comma only before please / can you ("Sara can you send this").
_SPOKEN_REQUEST_RE = (
    r"please|can you|could you|will you|would you|i need you to|i want you to"
)
_VOCATIVE_RE = re.compile(
    rf"(?:^|(?<=\s))(?:(?i:and|also)\s+)?({_NAME_RE})"
    rf"(?:,\s*(?i:{_POLITE_PREFIX_RE})\b|(?:,\s*|\s+)(?i:{_SPOKEN_REQUEST_RE})\b)",
)
# "can" and "you" are the request, not part of the name. Without this, a
# greedy match turns "sara can you" into one fake person.
_LOWER_NAME = (
    r"(?!(?:please|can|could|will|would|you|i|need|and|also|the|to|for|of|"
    r"with|by|on|in|a|an|notes|report|recording|budget)\b)[a-z][\w'.-]{2,}"
)
_LEADING_ADDRESSEE_RE = re.compile(
    rf"\b({_LOWER_NAME}(?:\s+{_LOWER_NAME}){{0,2}})"
    rf"((?:,\s*|\s+)(?=(?:{_SPOKEN_REQUEST_RE})\b))",
)
_NEXT_ACTION_RE = re.compile(
    r"\s+and\s+(?:also\s+|then\s+)?(?=(?:please\s+|kindly\s+)?"
    r"(?:send|check|review|fix|update|prepare|write|share|complete|finish|"
    r"schedule|create|draft|test|deploy|publish|build|document|ensure|"
    r"submit|deliver|coordinate|handle|follow|make|look)\b)",
    re.IGNORECASE,
)
_GROUP_FOLLOWUP_RE = re.compile(
    r"^(?:please\s+)?(?:both of you|all of you|you both|the two of you|everyone|you all)"
    r"(?:,)?(?:\s+please)?(?:\s+(?:do|handle|complete|finish|take care of))?"
    r"(?:\s+(?:this|that|these|the))?(?:\s+tasks?)?$",
    re.IGNORECASE,
)
_TRAILING_FOLLOWUP_RE = re.compile(
    r"(?:[.]\s*|\s+)(?:please\s+)?(?:both of you|all of you|you both|the two of you)\b"
    r"(?:\s+please)?\s+(?:do|handle|complete|finish)\s+(?:this|that|the)\s+tasks?\b.*$",
    re.IGNORECASE,
)

_EVENT_NOUN_RE = re.compile(
    r"\b(?:meetings?|calls?|sessions?|standups?|syncs?|appointments?|workshops?)\b",
    re.IGNORECASE,
)
_SCHEDULE_COPULA_RE = re.compile(
    r"\b(?:will\s+be|is\s+on|is\s+at|are\s+on|scheduled\s+for)\b",
    re.IGNORECASE,
)
_ACTION_REQUEST_RE = re.compile(
    r"\b(?:please|must|needs?\s+to|have\s+to|has\s+to|complete|submit|finish|"
    r"send|fix|prepare|deliver|assigned|responsible|ensure|make\s+sure)\b",
    re.IGNORECASE,
)
_CLOSING_RE = re.compile(
    r"^(?:(?:okay|ok|alright|so|well|thanks|thank you|and|also)[, ]+)*(?:now[, ]+)?"
    r"(?:"
    r"(?:i am|i'm|we are|we're)\s+"
    r"(?:just\s+)?(?:ending|leaving|stopping|signing off|wrapping up|done|finished)"
    r"|(?:i will|i'll|we will|we'll)\s+"
    r"(?:just\s+)?(?:end|leave|stop|sign off|wrap up|finish)"
    r")"
    r"(?:\s+(?:the\s+)?(?:meeting|call|session|recording|now|here))?[.!\s]*$",
    re.IGNORECASE,
)


def _is_schedule_announcement(sentence: str) -> bool:
    """A statement of when an event happens, not a request that someone do work."""
    text = (sentence or "").strip()
    if not text or _ACTION_REQUEST_RE.search(text):
        return False
    if re.search(r"\b(?:can you|could you|will you|would you|please)\b", text, re.IGNORECASE):
        return False
    if re.match(
        r"^(?:please\s+)?(?:schedule|book|arrange|set up|organize|plan)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if not _EVENT_NOUN_RE.search(text):
        return False
    has_when = bool(
        _SCHEDULE_COPULA_RE.search(text)
        or _RELATIVE_DATE_RE.search(text)
        or _ABSOLUTE_DATE_RE.search(text)
    )
    if not has_when:
        return False
    if re.search(r"\b[A-Z][a-z]+\s+will\s+(?!be\b)\w+", text):
        return False
    return True


_MEETING_PHASE_RE = re.compile(
    r"\b(?:the\s+)?(?:meeting|call|session|recording)\s+"
    r"(?:has\s+|is\s+)?(?:started|ended|begun|finished|stopped|starting|over)\b",
    re.IGNORECASE,
)
_VAGUE_TELL_RE = re.compile(
    r"^(?:(?:can|could|will|would)\s+you\s+)?"
    r"(?:please\s+)?(?:tell|give|ask)\s+(?:me|us)\s+"
    r"(?:any|a|an|some|the)?\s*"
    r"(?:tasks?|work|anything|something|more|whatever)"
    r"[.!\s?]*$",
    re.IGNORECASE,
)
_RECORDING_RE = re.compile(
    r"\b(?:i am|i'm|we are|we're)\s+recording\b",
    re.IGNORECASE,
)
_OFFER_WORK_Q_RE = re.compile(
    r"\b(?:do you have|are there|is there|got)\s+any\s+"
    r"(?:tasks?|questions?|updates?|action items?|decisions?)\b",
    re.IGNORECASE,
)
_BARE_TASK_RE = re.compile(
    r"\b(?:do|finish|complete|handle|take)\s+(?:the\s+|a\s+|an\s+|any\s+|some\s+)?tasks?\b",
    re.IGNORECASE,
)
_TASK_DETAIL_RE = re.compile(
    r"\btasks?\s+(?:number\s+)?(?:\d+|[a-z]\b)|\b\d+\b",
    re.IGNORECASE,
)
_DATE_TAIL_RE = re.compile(
    r"\b(?:by|on|before)\s+(?:this\s+|next\s+)?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|week|month)\b",
    re.IGNORECASE,
)
_DO_IMPERATIVE_RE = re.compile(r"^do\s+(?!you\b)", re.IGNORECASE)
_SESSION_OPEN_RE = re.compile(
    r"^(?:i (?:have |'ve )?(?:just )?(?:turned|switched) on the "
    r"(?:meeting|call|session|recording)"
    r"(?: and (?:you can|you may) speak now)?"
    r"|(?:you can|you may) speak now)"
    r"[.!\s]*$",
    re.IGNORECASE,
)


def _only_says_task(sentence: str) -> bool:
    """'Do task by Friday' names no piece of work. 'Do task 2' does."""
    text = (sentence or "").strip()
    if not text or not _BARE_TASK_RE.search(text):
        return False
    if _TASK_DETAIL_RE.search(text):
        return False
    rest = _DATE_TAIL_RE.sub(" ", _BARE_TASK_RE.sub(" ", text))
    rest = re.sub(
        r"\b(?:can|could|will|would|you|please|i|want|need|to|and|also|just|"
        r"the|a|an|any|some|okay|ok|so|me|us)\b",
        " ",
        rest,
        flags=re.IGNORECASE,
    )
    return not re.findall(r"[a-z0-9]+", rest.lower())


def is_not_an_action(sentence: str) -> bool:
    """A status line or an empty 'tell me' is not work for someone."""
    text = (sentence or "").strip()
    if not text:
        return True
    text = re.sub(
        r"^(?:(?:okay|ok|um|uh|so|and|also|then|right|well|yes|yeah)[, ]+)+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,")
    if _is_closing_remark(text) or _SESSION_OPEN_RE.match(text):
        return True
    if _OFFER_WORK_Q_RE.search(text) and not re.search(
        r"\b(?:please|send|finish|complete|i want you to|i need you to)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if _RECORDING_RE.search(text) and not re.search(
        r"\b(?:please|send|finish|complete|i want you to|i need you to)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if _MEETING_PHASE_RE.search(text) and not re.search(
        r"\b(?:please|send|prepare|finish|complete|share|write|fix)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if _only_says_task(text):
        return True
    return bool(_VAGUE_TELL_RE.match(text))


def _is_closing_remark(sentence: str) -> bool:
    """Ending the conversation is not an assignment."""
    return bool(_CLOSING_RE.match((sentence or "").strip()))


_LEADING_FILLER_RE = re.compile(
    r"^(?:(?:okay|ok|um|uh|so|and|also|then|right|well|yes|yeah)[, ]+)+",
    re.IGNORECASE,
)
_TRAILING_CLOSE_RE = re.compile(
    r"(?:[,.]|\s)\s*(?:(?:okay|ok|so|and|also|well|right)[, ]+)*"
    r"(?:"
    r"(?:i am|i'm|we are|we're)\s+"
    r"(?:just\s+)?(?:ending|leaving|stopping|signing off|wrapping up)"
    r"|(?:i will|i'll|we will|we'll)\s+"
    r"(?:just\s+)?(?:end|leave|stop|sign off|wrap up|finish)"
    r")"
    r"(?:\s+the)?(?:\s+(?:meeting|call|session|recording))?"
    r"(?:\s+now)?"
    r"(?:\s+(?:bye|goodbye|by))?"
    r"[.!\s]*$",
    re.IGNORECASE,
)


def strip_spoken_asides(sentence: str) -> str:
    """Drop a leading okay/and and a trailing 'I am ending the meeting'.

    Recorded lines often glue the request and the goodbye into one sentence.
    The goodbye is not part of the task.
    """
    text = _TRAILING_CLOSE_RE.sub("", (sentence or "").strip()).strip(" ,")
    text = _LEADING_FILLER_RE.sub("", text).strip(" ,")
    parts = [
        part.strip(" ,")
        for part in re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
        if part.strip(" ,") and not _MEETING_PHASE_RE.search(part)
    ]
    if parts:
        text = " and ".join(parts)
    return text if text and not is_not_an_action(text) else ""


def leading_clauses(sentence: str) -> List[str]:
    """Clauses left after a goodbye, with a status remark removed.

    The parts stay separate so a request and an agreement are not one item.
    """
    text = _TRAILING_CLOSE_RE.sub("", (sentence or "").strip()).strip(" ,")
    text = _LEADING_FILLER_RE.sub("", text).strip(" ,")
    parts = [
        part.strip(" ,")
        for part in re.split(r"\s+and\s+", text, flags=re.IGNORECASE)
        if part.strip(" ,")
        and not _MEETING_PHASE_RE.search(part)
        and not _is_closing_remark(part)
        and not is_not_an_action(part)
    ]
    return parts


# "let's get started / move on / wrap up" are pleasantries, NOT tasks
_PLEASANTRY_LETS_RE = re.compile(
    r"\blet['']?s\s+(?:get\s+started|begin\b|start\b|move\s+on|wrap\s+up|"
    r"jump\s+in|dive\s+in|kick\s+off|go\s+ahead|proceed)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches_any(patterns: List[str], text_lower: str) -> bool:
    return any(re.search(p, text_lower) for p in patterns)


def _has_strong_cue(text_lower: str) -> bool:
    return _matches_any(STRONG_CUES, text_lower)


def _has_strong_action_verb(text_lower: str) -> bool:
    return _matches_any(STRONG_ACTION_VERBS, text_lower)


# A listed action word used as a noun ("test cases", "the release") is not a task.
_NOUN_CONTEXT = {
    r"\btest\b": re.compile(r"\btest\s+cases?\b", re.IGNORECASE),
    r"\bupdate\b": re.compile(r"\b(?:an|the|api|status)\s+update\b", re.IGNORECASE),
    r"\bbuild\b": re.compile(r"\b(?:the|a|this)\s+build\b", re.IGNORECASE),
    r"\brelease\b": re.compile(r"\b(?:the|a|this)\s+release\b", re.IGNORECASE),
    r"\bdocument\b": re.compile(r"\b(?:the|a|this)\s+document\b", re.IGNORECASE),
    r"\bdraft\b": re.compile(r"\b(?:the|a|this)\s+draft\b", re.IGNORECASE),
}


def _used_as_noun(pattern: str, sentence: str) -> bool:
    """True when *pattern* only appears inside a noun phrase in *sentence*."""
    guard = _NOUN_CONTEXT.get(pattern)
    return bool(guard and guard.search(sentence))


def _has_weak_verb(text_lower: str) -> bool:
    return _matches_any(WEAK_VERBS, text_lower)


_PERSON_SKIP = {
    "the", "this", "that", "please", "let", "team", "we", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "today",
    "tomorrow", "tonight", "next", "will", "make", "sure",
    "yes", "so", "okay", "ok", "sorry", "hello", "thanks", "thank", "great",
    "can", "could", "will", "would", "should", "please", "also", "how",
    "tell", "ask", "give", "say", "start", "stop", "let", "go", "do", "make",
    "task", "tasks", "yes",
    "and", "but", "or", "our", "their", "your", "other", "others",
    "participant", "participants", "ending", "leaving", "meeting",
    "invoice", "deployment", "notification", "dashboard", "checklist",
    "metrics", "release", "budget", "caching", "design", "client", "report",
    "also", "however", "first", "without", "well", "sure", "right", "hi",
    "hey", "good", "very", "still", "about", "maybe", "actually", "instead",
    "now", "then", "here", "there", "currently", "already", "just", "recording",
    "one", "what", "how", "absolutely", "noted", "fine", "yeah", "yep",
    "perfect", "cool", "good", "someone", "somebody", "duplicate",
    "records", "authentication", "filter", "priority", "updates",
}


def _person_token_ok(text: str, pronouns: set) -> bool:
    """Skip pronouns, agreement words, and Urdu tokens spaCy labels as people."""
    from services.preprocessing.language import arabic_script_ratio

    cleaned = (text or "").strip(" ،,.").lower()
    if not cleaned or cleaned in pronouns or cleaned in _PERSON_SKIP:
        return False
    if arabic_script_ratio(cleaned) >= 0.2:
        return False
    return True


def _guess_person(sentence: str) -> str:
    """Pick the first capitalised name when spaCy NER is unavailable."""
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", sentence):
        if match.group(0).lower() not in _PERSON_SKIP:
            return match.group(0)
    return "Team"


def _is_group_followup(sentence: str) -> bool:
    """True when a line only repeats work already given to named people."""
    text = re.sub(r"\s+", " ", (sentence or "").strip().rstrip(".!"))
    return bool(text and _GROUP_FOLLOWUP_RE.match(text))


def _acceptable_addressee(name: str) -> bool:
    cleaned = (name or "").casefold().replace("’", "'")
    if not cleaned or "'" in cleaned or cleaned in _PERSON_SKIP:
        return False
    return cleaned not in {
        "and", "also", "both", "please", "can", "could", "you",
        "i'll", "i'm", "we'll", "they'll", "i’ve",
    }


def _capitalize_addressees(text: str) -> str:
    """Restore name capitals that speech-to-text dropped before a request.

    "sara can you send the notes" is the same address as "Sara, can you".
    A weekday or a work noun in that position stays as it is.
    """
    def leading(match: re.Match) -> str:
        words = match.group(1).split()
        if any(not _acceptable_addressee(word) for word in words):
            return match.group(0)
        titled = " ".join(word[:1].upper() + word[1:] for word in words)
        return titled + match.group(2)

    return _LEADING_ADDRESSEE_RE.sub(leading, text or "")


def _polish_action(text: str) -> str:
    """Shorten a request without dropping a second requirement in the same action."""

    def meeting_noun(match: re.Match) -> str:
        return f"the meeting {match.group(1).lower()}"

    text = re.sub(
        r"\bthe ([A-Za-z]+) of (?:this|the) meeting\b",
        meeting_noun,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bcheck if\b", "check whether", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis meeting\b", "the meeting", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis$", "it", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,")


def concise_action(sentence: str) -> str:
    """Drop the name and the request wrapper, leaving the action."""
    text = re.sub(r"\s+", " ", (sentence or "").strip().rstrip(".!?"))
    text = _TRAILING_FOLLOWUP_RE.sub("", text).strip(" ,")
    text = re.sub(
        rf"^(?:and\s+|also\s+)?{_NAME_RE}(?:,\s*|\s+)(?=(?i:{_POLITE_PREFIX_RE})\b)",
        "",
        text,
        count=1,
    )
    text = re.sub(rf"^(?:and\s+|also\s+)?{_NAME_RE},\s*", "", text, count=1)
    text = re.sub(
        rf"^(?:{_POLITE_PREFIX_RE})\s+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:also|just|kindly)\s+", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\bsend\s+me\s+", "send ", text, count=1, flags=re.IGNORECASE)
    text = _polish_action(text)
    if not text:
        return (sentence or "").strip()
    return text[:1].upper() + text[1:]


def _expand_same_assignee(clauses: List[str]) -> List[str]:
    """Give a repeated action verb its own clause, keeping the same addressee."""
    expanded: List[str] = []
    for clause in clauses:
        match = _VOCATIVE_RE.match(clause.strip())
        assignee = match.group(1) if match and _acceptable_addressee(match.group(1)) else ""
        parts = [part.strip(" ,") for part in _NEXT_ACTION_RE.split(clause) if part.strip(" ,")]
        if len(parts) <= 1 or not assignee:
            expanded.append(clause.strip(" ,"))
            continue
        expanded.append(parts[0])
        for part in parts[1:]:
            if re.match(r"(?i)please\b", part):
                expanded.append(f"{assignee}, {part}")
            else:
                expanded.append(f"{assignee}, please {part}")
    return expanded


def split_coordinated_requests(sentence: str) -> List[str]:
    """
    Split one sentence into one clause per addressed action.

    A single request is returned unchanged. A reminder such as
    "both of you do this task" is removed instead of becoming a clause.
    """
    text = _TRAILING_FOLLOWUP_RE.sub("", (sentence or "").strip()).strip(" ,")
    if not text:
        return [sentence.strip()] if sentence and sentence.strip() else []
    matches = [
        match for match in _VOCATIVE_RE.finditer(text)
        if _acceptable_addressee(match.group(1))
    ]
    if len(matches) >= 2:
        clauses: List[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            clause = text[match.start():end]
            clause = re.sub(r"^(?:and|also)\s+", "", clause.strip(), count=1, flags=re.IGNORECASE)
            clause = clause.strip(" ,")
            if clause:
                clauses.append(clause)
        expanded = _expand_same_assignee(clauses)
        return expanded if len(expanded) > 1 else [sentence.strip()]
    if len(matches) == 1:
        expanded = _expand_same_assignee([text])
        return expanded if len(expanded) > 1 else [sentence.strip()]
    return [sentence.strip()] if sentence and sentence.strip() else []


def _action_verb_is_verbal(pattern: str, sent_doc) -> bool:
    """
    Return True only if the token matching *pattern* is used as a VERB or AUX
    in the parsed sentence (i.e. not as a noun/adjective).
    This disambiguates 'build the feature' (VERB) vs 'the build pipeline' (NOUN).

    Note: spaCy POS-tags all-caps text poorly (e.g. SUBMIT → PROPN).
    The caller should pass a normalised doc when needed; this function also
    checks for PROPN/NOUN tokens that ARE the root of the sentence (which
    indicates an all-caps imperative mis-tagged as PROPN).
    """
    for token in sent_doc:
        if re.search(pattern, token.text, re.IGNORECASE):
            if token.pos_ in ("VERB", "AUX"):
                return True
            # Fallback: all-caps imperatives are mis-tagged as PROPN/ROOT by spaCy
            if token.pos_ in ("PROPN", "NOUN") and token.dep_ == "ROOT":
                return True
    return False


def _is_negated(text_lower: str) -> bool:
    return bool(_NEGATION_RE.search(text_lower))


def _is_hypothetical(sentence: str, text_lower: str) -> bool:
    # Polite indirect requests like "It would be great if Sara could handle…"
    # are tasks — exempt them before the hypothetical check.
    if _POLITE_REQUEST_RE.search(sentence):
        return False
    if _HYPOTHETICAL_TRIGGERS.match(sentence):
        return True
    # "could/might/may/would" without any obligation cue → hypothetical
    if _HYPOTHETICAL_CUES.search(text_lower) and not _has_strong_cue(text_lower):
        return True
    return False


def _is_question(sentence: str) -> bool:
    if sentence.rstrip().endswith("?"):
        return True
    if _QUESTION_RE.match(sentence):
        return True
    return False


def _is_wh_question(sentence: str) -> bool:
    """WH-questions are ALWAYS filtered, even when they contain obligation cues."""
    if sentence.rstrip().endswith("?") and _WH_QUESTION_RE.match(sentence):
        return True
    return False


def _has_past_context(text_lower: str) -> bool:
    """Returns True if the sentence contains temporal markers signalling a past event."""
    return bool(_PAST_CONTEXT_RE.search(text_lower))


def _dep_parse_is_task(sent_doc) -> bool:
    """
    Fallback: use dependency parse to detect imperative or modal-future pattern.
    Returns True when:
      - ROOT is a non-past-tense verb with a named subject (PROPN/NOUN) or
        first-person pronoun plus a modal aux (will/must/shall).
      - OR sentence is imperative (ROOT verb is first real word).
    Past-tense ROOT verbs are excluded (they describe completed actions, not tasks).
    """
    root = None
    for token in sent_doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root = token
            break
    if root is None:
        return False

    # Exclude past-tense verbs (described past events, not tasks)
    if root.tag_ in ("VBD", "VBN"):   # past tense / past participle
        return False

    children = list(root.children)
    children_deps = {c.dep_ for c in children}
    has_subj = "nsubj" in children_deps or "nsubjpass" in children_deps

    # Require a named (non-pronoun) subject or a first-person commitment
    meaningful_subj = False
    for child in children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            if child.pos_ in ("PROPN", "NOUN"):
                meaningful_subj = True
            elif child.pos_ == "PRON" and child.text.lower() in ("i",):
                meaningful_subj = True
            break

    # Imperative: ROOT verb is the first alpha token (or follows please/now/also)
    if not has_subj and root.pos_ == "VERB":
        for tok in sent_doc:
            if tok.is_alpha:
                if tok.i == root.i or (
                    tok.i < root.i and tok.text.lower() in ("please", "also")
                ):
                    return True
                break

    # Modal (will/must/shall) with a named or first-person subject
    has_modal = any(
        c.dep_ == "aux" and c.text.lower() in ("will", "must", "shall")
        for c in children
    )
    if has_modal and meaningful_subj:
        return True

    # Named subject + present-tense action verb (no modal needed for proper names).
    # "I think the team is worried" is an opinion, not an assignment.
    if meaningful_subj and has_subj and root.tag_ in ("VB", "VBZ", "VBP", "VBG"):
        if root.lemma_.lower() in {"think", "believe", "feel", "seem", "know", "guess"} and not has_modal:
            return False
        return True

    return False


def _meeting_day(meeting_date: Any) -> Optional[datetime.date]:
    """The calendar day a relative deadline is counted from."""
    if meeting_date is None:
        return None
    if isinstance(meeting_date, datetime.datetime):
        return meeting_date.date()
    if isinstance(meeting_date, datetime.date):
        return meeting_date
    text = str(meeting_date).strip()
    if not text:
        return None
    try:
        return datetime.datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _resolve_date(phrase: str, meeting_date: Optional[datetime.date]) -> Optional[str]:
    """
    Resolve a relative date phrase to ISO YYYY-MM-DD using dateparser.
    Returns None if dateparser is not installed or resolution fails.
    """
    try:
        import dateparser
        settings: Dict[str, Any] = {"PREFER_DATES_FROM": "future"}
        day = _meeting_day(meeting_date)
        if day is not None:
            # Anchor relative resolution to the meeting date
            settings["RELATIVE_BASE"] = datetime.datetime.combine(
                day, datetime.time()
            )
        result = dateparser.parse(phrase, settings=settings)
        if result:
            return result.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nlp(model: str = "en_core_web_sm"):
    """
    Load (or return cached) spaCy English model.

    Parameters
    ----------
    model : spaCy model name.
            Download first with: python -m spacy download en_core_web_sm

    Returns
    -------
    spacy.language.Language
    """
    global _nlp, _nlp_unavailable
    if _nlp_unavailable:
        return None
    if _nlp is None:
        from services.native_compat import install
        install()
        try:
            import spacy
            print(f"Loading spaCy model '{model}' ...")
            _nlp = spacy.load(model)
            print("spaCy model loaded.")
        except Exception as exc:
            _nlp_unavailable = True
            _nlp = None
            print(f"spaCy is unavailable ({exc}). Using the regex task detector.")
            return None
    return _nlp


def detect_tasks(
    text: str,
    nlp=None,
    nlp_model=None,
    *,
    participants: Optional[List[Dict[str, Any]]] = None,
    meeting_date: Optional[datetime.date] = None,
) -> List[Dict[str, Any]]:
    """
    Detect action items / tasks in *text* using regex + spaCy NER + dep-parse.

    Algorithm
    ---------
    1. Split text into sentences via spaCy.
    2. For each sentence apply guard filters (negation, hypothetical, question).
    3. Check for strong obligation cues OR strong action verbs.
       Weak verbs only count when a strong cue is also present.
    4. If no keyword match, fall back to dependency-parse heuristic.
    5. Extract assignee (first PERSON NER entity, else "Team") and due date.
    6. Resolve relative date phrases to ISO YYYY-MM-DD (via dateparser).
    7. If participants provided, match assigned_to to participant roster.
    8. Return list of task dicts.

    Parameters
    ----------
    text          : Plain-text meeting transcript or any multi-sentence string.
    nlp           : Pre-loaded spaCy model (preferred kwarg name).
    nlp_model     : Alias for nlp (for backwards compatibility).
    participants  : Optional list of participant dicts for attendee matching.
    meeting_date  : Optional date to anchor relative date resolution.

    Returns
    -------
    List of dicts with keys:
      id, description, assigned_to, due_date, [due_date_iso,] keyword,
      status, priority, [assigned_email]
    """
    # Accept both `nlp` and `nlp_model` kwarg names
    loaded_nlp = nlp or nlp_model
    if loaded_nlp is None:
        loaded_nlp = load_nlp()

    tasks: List[Dict[str, Any]] = []
    if loaded_nlp is None:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]
    else:
        doc = loaded_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    pieces: List[tuple] = []
    for sentence in sentences:
        sentence = strip_spoken_asides(_capitalize_addressees(sentence))
        if not sentence:
            continue
        clauses = split_coordinated_requests(sentence)
        if len(clauses) > 1:
            pieces.extend((clause, True) for clause in clauses)
        else:
            pieces.append((sentence, False))

    for sentence, split_clause in pieces:
        sl = sentence.lower()
        from services.preprocessing.language import arabic_script_ratio
        if arabic_script_ratio(sentence) >= 0.12:
            continue

        if _is_group_followup(sentence):
            continue

        # --- Guard: negation ---
        if _is_negated(sl):
            continue

        if _STATUS_RE.search(sl) and not _has_strong_cue(sl):
            continue

        # --- Guard: past temporal context (yesterday, last week, etc.) ---
        if _has_past_context(sl):
            continue

        # --- Guard: hypothetical ---
        if _is_hypothetical(sentence, sl):
            continue

        # --- Guard: WH-questions (always filtered, even with cues) ---
        if _is_wh_question(sentence):
            continue

        # --- Guard: plain question (no strong cue) ---
        if (
            _is_question(sentence)
            and not _has_strong_cue(sl)
            and not _DO_IMPERATIVE_RE.match(sentence)
        ):
            continue

        # --- Guard: "let's get started / move on" pleasantries ---
        if _PLEASANTRY_LETS_RE.search(sentence):
            continue

        if (
            not sentence
            or is_not_an_action(sentence)
            or _is_closing_remark(sentence)
            or _is_schedule_announcement(sentence)
        ):
            continue

        # --- Keyword matching ---
        matched_keyword = ""
        is_task = False

        if _DO_IMPERATIVE_RE.match(sentence) or _has_strong_cue(sl):
            is_task = True
            # find the specific cue that matched for the keyword field
            for p in STRONG_CUES:
                if re.search(p, sl):
                    matched_keyword = p.replace(r"\b", "").replace("?", "").strip()
                    break

        elif _has_strong_action_verb(sl):
            # Verify the matched verb is actually used as a VERB (not a noun)
            # e.g. "the build pipeline" → build is NOUN → skip
            # Parse the sentence for POS checking. Without spaCy, keep the verb.
            if loaded_nlp is None:
                for p in STRONG_ACTION_VERBS:
                    if re.search(p, sl) and not _used_as_noun(p, sentence):
                        matched_keyword = p.replace(r"\b", "").replace("?", "").strip()
                        is_task = True
                        break
            else:
                sent_doc_for_pos = loaded_nlp(sentence)
                for p in STRONG_ACTION_VERBS:
                    if (
                        re.search(p, sl)
                        and not _used_as_noun(p, sentence)
                        and _action_verb_is_verbal(p, sent_doc_for_pos)
                    ):
                        matched_keyword = p.replace(r"\b", "").replace("?", "").strip()
                        is_task = True
                        break

        elif _has_weak_verb(sl) and _has_strong_cue(sl):
            # Weak verb is only valid when accompanied by a strong cue
            # (already covered above, this branch won't be reached; kept for clarity)
            is_task = True

        # --- Dep-parse fallback ---
        if not is_task and loaded_nlp is not None:
            sent_doc = loaded_nlp(sentence)
            if _dep_parse_is_task(sent_doc):
                is_task = True
                matched_keyword = "dep_parse"

        if not is_task:
            continue

        # Assignee: the addressed name, else the first PERSON entity, else "Team"
        assigned_to = "Team"
        _pronoun_set = {"i", "we", "they", "he", "she", "it", "you",
                        "i'll", "we'll", "they'll"}
        addressed = _VOCATIVE_RE.match(sentence.strip())
        if not (addressed and _acceptable_addressee(addressed.group(1))):
            addressed = next(
                (
                    match for match in _VOCATIVE_RE.finditer(sentence)
                    if _acceptable_addressee(match.group(1))
                ),
                None,
            )
        if (
            addressed
            and _person_token_ok(addressed.group(1), _pronoun_set)
            and _acceptable_addressee(addressed.group(1))
        ):
            assigned_to = addressed.group(1)
        sent_doc = loaded_nlp(sentence) if loaded_nlp is not None else None
        if assigned_to == "Team" and sent_doc is not None:
            for ent in sent_doc.ents:
                if ent.label_ == "PERSON" and _person_token_ok(ent.text, _pronoun_set):
                    assigned_to = ent.text
                    break
        if assigned_to == "Team":
            guessed = _guess_person(sentence)
            if guessed != "Team":
                assigned_to = guessed

        assigned_email = None
        if participants is not None:
            from services.nlp.participant_matching import match
            matched_p = match(assigned_to, participants)
            if matched_p:
                assigned_to = matched_p.get("name", assigned_to)
                assigned_email = matched_p.get("email")

        # Due date: first DATE or TIME entity
        due_date = "Not specified"
        due_date_iso: Optional[str] = None
        if sent_doc is not None:
            for ent in sent_doc.ents:
                if ent.label_ in ("DATE", "TIME"):
                    due_date = ent.text
                    # Try resolving relative date to ISO
                    due_date_iso = _resolve_date(ent.text, meeting_date)
                    break

        # If no NER date but a relative phrase exists, try to resolve it
        if due_date == "Not specified":
            m = _ABSOLUTE_DATE_RE.search(sentence) or _RELATIVE_DATE_RE.search(sentence)
            if m:
                due_date = m.group(0)
                due_date_iso = _resolve_date(due_date, meeting_date)
        if "@" in (due_date or "") or re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", due_date or ""):
            due_date = "Not specified"
            due_date_iso = None

        # --- Priority ---
        priority = "Low"
        for p_pattern in PRIORITY_KEYWORDS["High"]:
            if re.search(p_pattern, sl):
                priority = "High"
                break

        if priority != "High":
            due_lower = due_date.lower()
            if "tomorrow" in due_lower or "today" in due_lower or "tonight" in due_lower:
                priority = "High"
            elif "friday" in due_lower or "end of week" in due_lower:
                priority = "Low"
            else:
                days = ["monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday"]
                anchor = _meeting_day(meeting_date) or datetime.datetime.now().date()
                current_day_idx = anchor.weekday()
                for i, day_name in enumerate(days):
                    if day_name in due_lower:
                        diff = (i - current_day_idx) % 7
                        if diff == 1:
                            priority = "High"
                        elif diff == 2:
                            priority = "Medium"
                        elif diff >= 3:
                            priority = "Low"
                        break

        if priority == "Low":
            for p_pattern in PRIORITY_KEYWORDS["Medium"]:
                if re.search(p_pattern, sl):
                    priority = "Medium"
                    break

        description = concise_action(sentence) if split_clause else sentence
        task_data: Dict[str, Any] = {
            "id":          len(tasks) + 1,
            "description": description,
            "assigned_to": assigned_to,
            "due_date":    due_date,
            "keyword":     matched_keyword,
            "priority":    priority,
            "status":      "pending",
        }
        if due_date_iso:
            task_data["due_date_iso"] = due_date_iso
        if participants is not None:
            task_data["assigned_email"] = assigned_email

        tasks.append(task_data)

    return tasks


def get_task_summary(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics over a list of detected tasks.

    Returns
    -------
    dict with keys: total, by_person (dict name->count), unassigned_count
    """
    from collections import Counter

    people = [t["assigned_to"] for t in tasks]
    person_counts = Counter(people)
    unassigned = person_counts.pop("Unknown", 0)

    return {
        "total":            len(tasks),
        "by_person":        dict(person_counts),
        "unassigned_count": unassigned,
    }

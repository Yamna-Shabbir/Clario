"""
services/task_detection.py
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
    r"\blet['']?s\b",
    r"\bmake\s+sure\b",
    r"\baction\s+item\b",
    r"\bfollow[\s-]up\b",
    r"\btake\s+care\s+of\b",
    r"\bensure\b",
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
    "High":   [r"\burgent\b", r"\basap\b", r"\bimmediately\b",
               r"\bcritical\b", r"\bmust\b", r"\bimportant\b",
               r"\bit\s+is\s+important\b", r"\bit['']?s\s+important\b"],
    "Medium": [r"\bsoon\b", r"\bby this week\b", r"\bshould\b",
               r"\bneeds?\s+to\b"],
}

# ---------------------------------------------------------------------------
# Negation / hypothetical / question filters
# ---------------------------------------------------------------------------

# If any of these match, skip the sentence (high-confidence non-tasks)
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

_RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|yesterday|next\s+\w+|last\s+\w+|this\s+\w+|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"end\s+of\s+(?:day|week|month)|by\s+(?:end\s+of\s+)?(?:day|week)|"
    r"tonight|this\s+morning|this\s+afternoon|this\s+evening)\b",
    re.IGNORECASE,
)

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


def _has_weak_verb(text_lower: str) -> bool:
    return _matches_any(WEAK_VERBS, text_lower)


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
                    tok.i < root.i and tok.text.lower() in ("please", "now", "also")
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

    # Named subject + present-tense action verb (no modal needed for proper names)
    if meaningful_subj and has_subj and root.tag_ in ("VB", "VBZ", "VBP", "VBG"):
        return True

    return False


def _resolve_date(phrase: str, meeting_date: Optional[datetime.date]) -> Optional[str]:
    """
    Resolve a relative date phrase to ISO YYYY-MM-DD using dateparser.
    Returns None if dateparser is not installed or resolution fails.
    """
    try:
        import dateparser
        settings: Dict[str, Any] = {"PREFER_DATES_FROM": "future"}
        if meeting_date is not None:
            # Anchor relative resolution to the meeting date
            settings["RELATIVE_BASE"] = datetime.datetime.combine(
                meeting_date, datetime.time()
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
    global _nlp
    if _nlp is None:
        import spacy
        print(f"Loading spaCy model '{model}' ...")
        _nlp = spacy.load(model)
        print("spaCy model loaded.")
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
    doc = loaded_nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    for sentence in sentences:
        sl = sentence.lower()

        # --- Guard: negation ---
        if _is_negated(sl):
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
        if _is_question(sentence) and not _has_strong_cue(sl):
            continue

        # --- Guard: "let's get started / move on" pleasantries ---
        if _PLEASANTRY_LETS_RE.search(sentence):
            continue

        # --- Keyword matching ---
        matched_keyword = ""
        is_task = False

        if _has_strong_cue(sl):
            is_task = True
            # find the specific cue that matched for the keyword field
            for p in STRONG_CUES:
                if re.search(p, sl):
                    matched_keyword = p.replace(r"\b", "").replace("?", "").strip()
                    break

        elif _has_strong_action_verb(sl):
            # Verify the matched verb is actually used as a VERB (not a noun)
            # e.g. "the build pipeline" → build is NOUN → skip
            # Parse the sentence for POS checking
            sent_doc_for_pos = loaded_nlp(sentence)
            for p in STRONG_ACTION_VERBS:
                if re.search(p, sl) and _action_verb_is_verbal(p, sent_doc_for_pos):
                    matched_keyword = p.replace(r"\b", "").replace("?", "").strip()
                    is_task = True
                    break

        elif _has_weak_verb(sl) and _has_strong_cue(sl):
            # Weak verb is only valid when accompanied by a strong cue
            # (already covered above, this branch won't be reached; kept for clarity)
            is_task = True

        # --- Dep-parse fallback ---
        if not is_task:
            sent_doc = loaded_nlp(sentence)
            if _dep_parse_is_task(sent_doc):
                is_task = True
                matched_keyword = "dep_parse"

        if not is_task:
            continue

        # --- NER pass on the sentence ---
        sent_doc = loaded_nlp(sentence)

        # Assignee: first PERSON entity; fall back to "Team"
        assigned_to = "Team"
        _pronoun_set = {"i", "we", "they", "he", "she", "it", "you",
                        "i'll", "we'll", "they'll"}
        for ent in sent_doc.ents:
            if ent.label_ == "PERSON":
                if ent.text.lower() not in _pronoun_set:
                    assigned_to = ent.text
                    break

        assigned_email = None
        if participants is not None:
            from services.participant_matching import match
            matched_p = match(assigned_to, participants)
            if matched_p:
                assigned_to = matched_p.get("name", assigned_to)
                assigned_email = matched_p.get("email")

        # Due date: first DATE or TIME entity
        due_date = "Not specified"
        due_date_iso: Optional[str] = None
        for ent in sent_doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                due_date = ent.text
                # Try resolving relative date to ISO
                due_date_iso = _resolve_date(ent.text, meeting_date)
                break

        # If no NER date but a relative phrase exists, try to resolve it
        if due_date == "Not specified":
            m = _RELATIVE_DATE_RE.search(sentence)
            if m:
                due_date = m.group(0)
                due_date_iso = _resolve_date(due_date, meeting_date)

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
                current_day_idx = datetime.datetime.now().weekday()
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

        task_data: Dict[str, Any] = {
            "id":          len(tasks) + 1,
            "description": sentence,
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

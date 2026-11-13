"""
Review meeting tasks against the full transcript.

The sentence detector stays the source of action language. This module adds
speaker context, timestamps, duplicate and deadline changes, and a review
list for anything the evidence does not support.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional

from services.preprocessing.cleaning import clean_transcript

_TIME_RE = re.compile(r"[\[(](\d{1,2}:\d{2}(?::\d{2})?)[\])]")
_INLINE_SPEAKER_RE = re.compile(
    r"^(?:Speaker\s+\d+\s*[–\-—:]\s*)?"
    r"([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,2})"
    r"(?:\s*\([^)]*\))?\s*[:\-–—]\s+(.+)$"
)
_ACK_RE = re.compile(
    r"^(?:okay|ok|thanks|thank you|hello|hi|hey|yes|yeah|yep|sure|"
    r"sounds good|great|right|alright|cool|perfect|got it|no problem|"
    r"good morning|good afternoon|absolutely|noted|fine)\.?$",
    re.IGNORECASE,
)
_NOT_A_NAME = {
    "thanks", "thank", "okay", "ok", "yes", "hello", "hi", "hey", "sorry",
    "great", "sure", "right", "well", "um", "uh", "speaker", "absolutely",
    "noted", "fine", "yeah", "yep", "perfect", "cool", "good",
}
_URDU_NOT_SPEAKER = {"بالکل", "ٹھیک", "جی", "ہاں", "اچھا", "شکریہ", "ہے"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONTRACTIONS = (
    (re.compile(r"\bI'll\b", re.IGNORECASE), "I will"),
    (re.compile(r"\bwe'll\b", re.IGNORECASE), "we will"),
    (re.compile(r"\bthey'll\b", re.IGNORECASE), "they will"),
    (re.compile(r"\bI'm going to\b", re.IGNORECASE), "I am going to"),
    (re.compile(r"\blet me\b", re.IGNORECASE), "I will"),
)
_FIRST_PERSON_RE = re.compile(
    r"\b(I will|I can|I shall|I am going to|I promised to)\b",
    re.IGNORECASE,
)
_REQUEST_RE = re.compile(
    r"\b(can you|could you|will you|would you|please)\b",
    re.IGNORECASE,
)
_VAGUE_OBJECT_RE = re.compile(
    r"\b(?:do|handle|take|finish|complete|check|review|send|fix|share|update)\s+"
    r"(?:that|it)\b"
    r"|\b(?:do|handle|take|finish|complete|check|review|send|fix|share|update)\s+"
    r"this\b(?!\s+(?:evening|morning|afternoon|week|month))",
    re.IGNORECASE,
)
_LETS_DECISION_RE = re.compile(
    r"\blet['’]?s\s+(?:use|go with|keep|stick with|adopt)\b",
    re.IGNORECASE,
)
_GROUP_RE = re.compile(
    r"\b(all team|the team|we all|all of us)\b",
    re.IGNORECASE,
)
_DATE_WORD_RE = re.compile(
    r"\b(this evening|tonight|this morning|this afternoon|today|tomorrow|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|evening|"
    r"morning|afternoon)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "the", "to", "of", "for", "and", "or", "on", "in", "by", "with",
    "i", "we", "you", "will", "can", "my", "our", "that", "this", "it", "be",
    "is", "are", "do", "before", "after", "from", "at", "as",
}
_SOFT_COMMITMENT_RE = re.compile(
    r"\b(?:someone|somebody)\s+(?:should|ought to|needs to|might)\b",
    re.IGNORECASE,
)
_SPEAKER_ANY_RE = re.compile(
    r"^Speaker\s+\d+\s*[–\-—:]\s*(\S+?)(?:\s*\([^)]*\))?\s*[:.\-–—]?\s*$"
)
_URDU_INLINE_RE = re.compile(r"^(\S+)\s*[:：]\s+(\S.*)$")
_STANDALONE_SPEAKER_RE = re.compile(
    r"^(?:Speaker\s+\d+\s*[–\-—:]\s*)?"
    r"([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,2})"
    r"(?:\s*\([^)]*\))?\s*:\s*$"
)
_DIRECTED_REQUEST_RE = re.compile(
    r"^([A-Z][\w'.-]+),\s*(?:please|can you|could you|will you|would you)\b",
    re.IGNORECASE,
)
_DECISION_STATEMENT_RE = re.compile(
    r"\b(?:we\s+)?decided\b|\bdecision\s+is\b|\bgoing\s+with\b",
    re.IGNORECASE,
)
_BAD_ASSIGNEE = {
    "team", "unassigned", "yes", "so", "okay", "ok", "sorry", "hello", "thanks",
    "all", "we", "i", "the", "also", "however", "first", "without", "great",
    "very", "one", "well", "sure", "right", "unknown", "absolutely", "noted",
    "fine", "yeah", "yep", "perfect", "cool", "good", "hi", "hey", "someone",
    "somebody", "bilkul", "theek", "thik", "haan", "han", "ji", "acha", "achha",
}
_NOT_DIRECTED = {"someone", "somebody", "team", "okay", "ok", "yes", "absolutely"}


def segment_transcript(text: str) -> List[Dict[str, Any]]:
    """Split a transcript into speaker turns, keeping any timestamps."""
    segments: List[Dict[str, Any]] = []
    speaker = ""
    label = ""
    timestamp = ""
    line_id = 0
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        found_time = _TIME_RE.search(line)
        if found_time:
            timestamp = found_time.group(1)
            line = _TIME_RE.sub(" ", line).strip()
        if not line:
            continue
        any_header = _SPEAKER_ANY_RE.match(line)
        if any_header:
            speaker = _display_name(any_header.group(1))
            label = line.rstrip(".")
            continue
        inline = _INLINE_SPEAKER_RE.match(line)
        if inline and inline.group(1).lower() not in _NOT_A_NAME:
            speaker = inline.group(1)
            label = speaker
            line = inline.group(2).strip()
        urdu_inline = _URDU_INLINE_RE.match(line)
        if (
            urdu_inline
            and _has_arabic(urdu_inline.group(1))
            and urdu_inline.group(1) not in _URDU_NOT_SPEAKER
        ):
            speaker = _display_name(urdu_inline.group(1))
            label = speaker
            line = urdu_inline.group(2).strip()
        header_only = _STANDALONE_SPEAKER_RE.match(line)
        if header_only and header_only.group(1).lower() not in _NOT_A_NAME:
            speaker = header_only.group(1)
            label = speaker
            continue
        line = line.replace("۔", ". ").replace("؟", "? ")
        cleaned = clean_transcript(line)
        if not cleaned:
            continue
        line_id += 1
        for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
            sentence = sentence.strip()
            if not sentence:
                continue
            segments.append({
                "speaker": speaker,
                "label": label,
                "timestamp": timestamp,
                "text": sentence,
                "line_id": line_id,
            })
    if segments:
        return segments
    cleaned = clean_transcript(text or "")
    if not cleaned:
        return []
    return [
        {"speaker": "", "label": "", "timestamp": "", "text": sentence.strip()}
        for sentence in _SENTENCE_SPLIT_RE.split(cleaned)
        if sentence.strip()
    ]


def render_segments(segments: List[Dict[str, Any]]) -> str:
    """Rebuild a readable transcript. Timestamps stay on the task, not the prose."""
    if not segments:
        return ""
    if not any(segment.get("speaker") for segment in segments):
        return " ".join(segment["text"] for segment in segments if segment.get("text")).strip()
    lines: List[str] = []
    current = None
    for segment in segments:
        speaker = segment.get("speaker") or ""
        if speaker and speaker != current:
            lines.append(segment.get("label") or speaker)
            current = speaker
        if segment.get("text"):
            lines.append(segment["text"])
    return "\n".join(lines).strip()


def _expand(sentence: str) -> str:
    expanded = sentence
    for pattern, replacement in _CONTRACTIONS:
        expanded = pattern.sub(replacement, expanded)
    return expanded


def _for_detector(sentence: str, speaker: str) -> str:
    """Give the detector a named subject when the speaker said I will / I can."""
    expanded = _expand(sentence)
    if speaker and _FIRST_PERSON_RE.search(expanded):
        return _FIRST_PERSON_RE.sub(f"{speaker} will", expanded, count=1)
    return expanded


def _is_ack(sentence: str) -> bool:
    return bool(_ACK_RE.match(sentence.strip()))


def _content_tokens(text: str) -> set:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", (text or "").lower())
    return {word for word in words if word not in _STOPWORDS and not _DATE_WORD_RE.fullmatch(word)}


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _request_action(sentence: str) -> str:
    text = sentence.strip().rstrip("?.!")
    text = re.sub(r"^[A-Z][\w'.-]+,\s*", "", text)
    text = re.sub(
        r"^(?:can you|could you|will you|would you|please)\s+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:also|just|please)\s+", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def _is_request(sentence: str) -> bool:
    return bool(_REQUEST_RE.search(sentence)) and sentence.rstrip().endswith("?")


def _directed_to_other(sentence: str, speaker: str) -> Optional[str]:
    """Name on a please-line when someone else is speaking (assignment to them)."""
    match = _DIRECTED_REQUEST_RE.match(sentence.strip())
    if not match:
        return None
    addressee = match.group(1)
    if addressee.casefold() in _NOT_DIRECTED:
        return None
    if speaker and addressee.casefold() == speaker.casefold():
        return None
    return addressee


def _is_decision_statement(sentence: str) -> bool:
    if _DECISION_STATEMENT_RE.search(sentence) and not re.search(
        r"\b(?:will|please|must|has to|have to|needs to)\b",
        sentence,
        re.IGNORECASE,
    ):
        return True
    return bool(_looks_like_decision_only(sentence))


def _vague_acceptance(sentence: str, original: str) -> bool:
    expanded = _expand(sentence)
    if _VAGUE_OBJECT_RE.search(expanded):
        return True
    return bool(
        re.search(r"\b(?:kar dunga|bhej dunga|kar dungi|bhej dungi)\b", original, re.IGNORECASE)
        and re.search(r"\b(?:that|it|woh|yeh)\b", expanded, re.IGNORECASE)
    )


def _copy_task(task: Dict[str, Any], *, source: str, timestamp: str, context: str) -> Dict[str, Any]:
    copied = dict(task)
    copied["source_text"] = source
    copied["evidence_time"] = timestamp or ""
    copied["source_context"] = context or ""
    return copied


def _apply_participant(task: Dict[str, Any], participants: Optional[List[Dict[str, Any]]]) -> None:
    if participants is None:
        task.pop("assigned_email", None)
        return
    from services.nlp.participant_matching import match

    matched = match(task.get("assigned_to") or "", participants)
    if matched:
        task["assigned_to"] = matched.get("name") or task.get("assigned_to")
        task["assigned_email"] = matched.get("email")
    else:
        task["assigned_email"] = None


def _has_arabic(token: str) -> bool:
    from services.preprocessing.language import arabic_script_ratio
    return arabic_script_ratio(token) >= 0.2


def _unusable_assignee(name: str) -> bool:
    """Agreement words and leftover Urdu are not people."""
    cleaned = (name or "").strip(" ،,.")
    if not cleaned:
        return True
    if _has_arabic(cleaned):
        return True
    return cleaned.casefold() in _BAD_ASSIGNEE


def _display_name(token: str) -> str:
    if not _has_arabic(token):
        return token
    from services.preprocessing.urdu_norm import transliterate_word
    return transliterate_word(token)


def _known_latin_speakers(turns: List[Dict[str, Any]]) -> List[str]:
    found: List[str] = []
    for turn in turns:
        speaker = (turn.get("speaker") or "").strip()
        for part in re.findall(r"[A-Za-z][\w'.-]+", speaker):
            if part[:1].isupper() and part.casefold() not in {"speaker"} and part not in found:
                found.append(part)
    return found


def _align_known_names(text: str, names: List[str]) -> str:
    """Use the speaker-label spelling when a transliterated name is close."""
    if not text or not names:
        return text

    def replace(match: re.Match) -> str:
        word = match.group(0)
        best = ""
        best_ratio = 0.0
        for name in names:
            ratio = difflib.SequenceMatcher(None, word.casefold(), name.casefold()).ratio()
            if ratio > best_ratio:
                best, best_ratio = name, ratio
        if best and best_ratio >= 0.84 and word.casefold() != best.casefold():
            return best
        return word

    return re.sub(r"\b[A-Z][A-Za-z'.-]+\b", replace, text)


def _content_gloss_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text or "")
    skipped = {
        "i", "we", "you", "will", "please", "by", "the", "a", "an", "it", "that",
        "to", "and", "of", "for",
    }
    return [word.lower() for word in words if word.lower() not in skipped]


def _choose_urdu_english(phrase: Optional[str], lexicon: str) -> str:
    """Keep the gloss that preserves more of the spoken commitment."""
    from services.preprocessing.language import arabic_script_ratio

    options: List[str] = []
    for candidate in (phrase, lexicon):
        if not candidate:
            continue
        if candidate.strip() in {"Noted."}:
            continue
        if arabic_script_ratio(candidate) >= 0.02:
            continue
        options.append(candidate)
    if not options:
        return phrase or lexicon or ""

    def rank(text: str) -> tuple:
        # "please add" at the end is leftover Urdu word order.
        trailing = 1 if re.search(r"\bplease\s+[A-Za-z]+\s*\.?$", text.strip()) else 0
        ordered = 1 if re.search(
            r"\b(?:will|please)\s+(?:add|fix|send|complete|set|check|share|write|do|update)\b",
            text,
            re.IGNORECASE,
        ) else 0
        scrambled = 1 if re.search(r"\bI\s+[A-Za-z]+\s+by\b", text) else 0
        return (ordered - trailing, len(_content_gloss_words(text)), -scrambled, len(text))

    return max(options, key=rank)


def _explode_gloss(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One English sentence per turn after an Urdu line is rewritten."""
    exploded: List[Dict[str, Any]] = []
    for turn in turns:
        parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", turn.get("text") or "")
            if part.strip()
        ]
        if len(parts) <= 1:
            exploded.append(turn)
            continue
        for part in parts:
            piece = dict(turn)
            piece["text"] = part
            exploded.append(piece)
    return exploded


def _carry_addressee(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the person just named when the next please-line lost that name."""
    last = ""
    last_speaker = None
    for turn in turns:
        speaker = turn.get("speaker") or ""
        if speaker != last_speaker:
            last = ""
            last_speaker = speaker
        text = (turn.get("text") or "").strip()
        if _is_ack(text):
            continue
        match = re.match(r"^([A-Z][\w'.-]+),\s+please\b", text)
        if match and match.group(1).casefold() not in _NOT_DIRECTED:
            last = match.group(1)
            continue
        if last and re.match(r"^(?:Someone|Somebody),\s+please\b", text):
            turn["text"] = re.sub(r"^(?:Someone|Somebody)", last, text, count=1)
    return turns


def prepare_meeting_turns(transcript: str, *, translate: bool = True) -> List[Dict[str, Any]]:
    """Segment a transcript and rewrite Urdu lines without dropping evidence.

    *translate* controls the neural Urdu-to-English model only. The meeting
    lexicon still runs when it is false, which is the free-hosting path.
    """
    from services.preprocessing.language import arabic_script_ratio, ensure_english_transcript
    from services.preprocessing.roman_urdu import utterance_to_english
    from services.preprocessing.urdu_norm import render_urdu_line, speaker_names

    names = speaker_names(transcript)
    turns = segment_transcript(transcript)
    known_speakers = _known_latin_speakers(turns)
    for turn in turns:
        original = turn.get("text") or ""
        turn["original"] = original
        if arabic_script_ratio(original) >= 0.12:
            phrase = render_urdu_line(original, names)
            lexicon = utterance_to_english(original)
            english = _choose_urdu_english(phrase, lexicon)
        else:
            english = utterance_to_english(original)
        if translate and arabic_script_ratio(english or "") >= 0.12:
            english = ensure_english_transcript(english)
        turn["text"] = _align_known_names(english or original, known_speakers)
    return _carry_addressee(_explode_gloss(turns))


def _looks_like_decision_only(sentence: str) -> bool:
    return bool(_LETS_DECISION_RE.search(sentence))


def verify_meeting_tasks(
    transcript: str,
    tasks: Optional[List[Dict[str, Any]]] = None,
    *,
    nlp=None,
    participants: Optional[List[Dict[str, Any]]] = None,
    meeting_date=None,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Return confirmed tasks, decisions, and items that need a person to check.

    Parameters
    ----------
    transcript : English transcript. Used when *segments* is omitted.
    tasks : Optional tasks already found by the sentence detector.
    segments : Speaker turns from the original transcript, timestamps included.
    """
    from services.nlp.task_detection import detect_tasks
    from services.nlp.utterances import classify_utterances, decision_utterances

    turns = segments if segments is not None else segment_transcript(transcript)
    confirmed: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []
    decision_snippets: List[Dict[str, Any]] = []
    consumed_requests = set()
    last_request: Optional[Dict[str, Any]] = None

    for turn in turns:
        sentence = (turn.get("text") or "").strip()
        original = (turn.get("original") or sentence).strip()
        speaker = (turn.get("speaker") or "").strip()
        timestamp = turn.get("timestamp") or ""
        if not sentence or _is_ack(sentence):
            continue
        from services.preprocessing.language import arabic_script_ratio
        if arabic_script_ratio(sentence) >= 0.12:
            review.append({
                "kind": "unclear_meaning",
                "description": sentence,
                "assigned_to": "Unassigned",
                "due_date": "Not specified",
                "source_text": original,
                "source_context": "",
                "evidence_time": timestamp,
                "reason": "This line is still in Urdu, so it was not saved as an action item.",
            })
            continue
        sl = sentence.lower()
        from services.nlp.task_detection import _has_past_context
        if _has_past_context(sl) and not re.search(
            r"\b(?:will|please|kar dunga|kar dungi|bhej dunga|bhej dungi|"
            r"karunga|karungi|likh dunga|complete kar)\b",
            original,
            re.IGNORECASE,
        ):
            continue
        directed = _directed_to_other(sentence, speaker)
        if _is_request(sentence) or _REQUEST_RE.search(sentence) or directed:
            last_request = {
                "text": sentence,
                "original": original,
                "speaker": speaker,
                "timestamp": timestamp,
                "assignee": directed,
            }

        if _is_decision_statement(sentence):
            decision_snippets.append({
                "text": sentence,
                "confidence": 0.8,
                "source": original,
            })
            continue

        contextual = _for_detector(sentence, speaker)
        found = detect_tasks(
            contextual,
            nlp=nlp,
            participants=participants,
            meeting_date=meeting_date,
        )
        if directed and found:
            # A question waits for the reply. A named "please" is the assignment.
            if sentence.rstrip().endswith("?"):
                continue
            task = _copy_task(found[0], source=original, timestamp=timestamp, context="")
            task["description"] = sentence
            task["assigned_to"] = directed
            _apply_participant(task, participants)
            confirmed.append(task)
            continue
        if found and (
            _is_request(sentence)
            or (sentence.rstrip().endswith("?") and _REQUEST_RE.search(sentence))
        ):
            continue
        vague = _vague_acceptance(sentence, original)
        if vague and last_request and last_request["text"] != sentence and speaker:
            action = _request_action(last_request["text"])
            if not action or _VAGUE_OBJECT_RE.search(action):
                from services.preprocessing.roman_urdu import utterance_to_english
                action = _request_action(
                    utterance_to_english(last_request.get("original") or last_request["text"])
                )
            if action and not _VAGUE_OBJECT_RE.search(action):
                when_parts = _DATE_WORD_RE.findall(f"{last_request['text']} {sentence}")
                when = when_parts[0] if when_parts else ""
                resolved = f"{speaker} will {action}".strip()
                if when and when.lower() not in resolved.lower():
                    if when in {"this evening", "tonight", "this morning"}:
                        resolved = f"{resolved} {when}"
                    else:
                        resolved = f"{resolved} by {when}"
                due = ""
                dated = detect_tasks(
                    resolved,
                    nlp=nlp,
                    participants=participants,
                    meeting_date=meeting_date,
                )
                if dated:
                    due = dated[0].get("due_date") or ""
                assignee = speaker or ((last_request or {}).get("assignee") or "")
                task = {
                    "description": resolved,
                    "assigned_to": assignee,
                    "due_date": due or "Not specified",
                    "keyword": "response",
                    "priority": (dated[0].get("priority") if dated else "Low"),
                    "status": "pending",
                    "source_text": original,
                    "evidence_time": timestamp,
                    "source_context": last_request.get("original") or last_request["text"],
                }
                if dated and dated[0].get("due_date_iso"):
                    task["due_date_iso"] = dated[0]["due_date_iso"]
                _apply_participant(task, participants)
                confirmed.append(task)
                consumed_requests.add(last_request["text"])
                consumed_requests.add(last_request.get("original") or last_request["text"])
                last_request = None
                continue
            if not (action and not _VAGUE_OBJECT_RE.search(action)):
                review.append({
                    "kind": "unclear_meaning",
                    "description": sentence,
                    "assigned_to": speaker or "Unassigned",
                    "due_date": "Not specified",
                    "source_text": original,
                    "source_context": (last_request.get("original") or last_request["text"]) if last_request else "",
                    "evidence_time": timestamp,
                    "reason": "The reply points at an earlier line, but the action is not explicit.",
                })
                if not found:
                    continue

        if not found:
            if _SOFT_COMMITMENT_RE.search(sentence):
                review.append({
                    "kind": "possible_miss",
                    "description": sentence,
                    "assigned_to": "Unassigned",
                    "due_date": "Not specified",
                    "source_text": original,
                    "source_context": "",
                    "evidence_time": timestamp,
                    "reason": "This may be a commitment, but it does not name an owner.",
                })
                continue
            if _FIRST_PERSON_RE.search(_expand(sentence)) and speaker:
                review.append({
                    "kind": "possible_miss",
                    "description": sentence,
                    "assigned_to": speaker,
                    "due_date": "Not specified",
                    "source_text": original,
                    "source_context": "",
                    "evidence_time": timestamp,
                    "reason": "This sounds like a commitment, but the action is not clear enough to save.",
                })
            continue

        task = _copy_task(found[0], source=original, timestamp=timestamp, context="")
        task["description"] = sentence
        assignee = (task.get("assigned_to") or "").strip()
        first_person = bool(_FIRST_PERSON_RE.search(_expand(sentence)))
        names_the_group = bool(_GROUP_RE.search(sentence) or re.search(r"\b(we|let's|lets)\b", sentence, re.IGNORECASE))
        if names_the_group or assignee.lower() == "all":
            task["assigned_to"] = "Team"
        elif first_person and speaker:
            task["assigned_to"] = speaker
        elif _unusable_assignee(assignee) or assignee.lower() in {
            "invoice", "deployment", "notification", "dashboard", "checklist",
            "metrics", "design", "client", "report", "api", "release", "budget",
        } or not assignee:
            if speaker and (first_person or assignee.lower() in {
                "invoice", "deployment", "notification", "dashboard", "checklist",
                "metrics", "design", "client", "report", "api", "release", "budget",
            }):
                task["assigned_to"] = speaker
            else:
                review.append({
                    "kind": "unclear_assignee",
                    "description": sentence,
                    "assigned_to": "Unassigned",
                    "due_date": task.get("due_date") or "Not specified",
                    "source_text": original,
                    "source_context": "",
                    "evidence_time": timestamp,
                    "reason": "A commitment is here, but nobody is named. It was not assigned to a guessed person.",
                })
                continue
        else:
            task["assigned_to"] = assignee
        _apply_participant(task, participants)
        confirmed.append(task)

    confirmed = [
        task for task in confirmed
        if task.get("source_context") or task.get("source_text") not in consumed_requests
    ]
    review = [
        item for item in review
        if item.get("source_text") not in consumed_requests
    ]

    prior = tasks or []
    for earlier in prior:
        description = (earlier.get("description") or "").strip()
        if not description:
            continue
        tokens = _content_tokens(description)
        if any(_jaccard(tokens, _content_tokens(task.get("description") or "")) >= 0.6 for task in confirmed):
            continue
        if any(_jaccard(tokens, _content_tokens(item.get("description") or "")) >= 0.6 for item in review):
            continue
        carried = dict(earlier)
        carried.setdefault("source_text", description)
        carried.setdefault("evidence_time", "")
        carried.setdefault("source_context", "")
        confirmed.append(carried)

    confirmed, review = _collapse_repeats(confirmed, review)
    for index, task in enumerate(confirmed, start=1):
        task["id"] = index

    rendered = render_segments(turns) if turns else (transcript or "")
    utterances = classify_utterances(rendered, nlp=nlp)
    decisions = decision_utterances(utterances) + decision_snippets
    for turn in turns:
        sentence = (turn.get("text") or "").strip()
        if sentence and _looks_like_decision_only(sentence):
            decisions.append({
                "text": sentence,
                "confidence": 0.75,
                "source": turn.get("original") or sentence,
            })
    decisions = _unique_decisions(decisions)

    return {
        "tasks": confirmed,
        "review": review,
        "decisions": decisions,
        "transcript": rendered,
    }


def _collapse_repeats(
    tasks: List[Dict[str, Any]],
    review: List[Dict[str, Any]],
) -> tuple:
    """Keep the later commitment when the same person restates it."""
    kept: List[Dict[str, Any]] = []
    for task in tasks:
        tokens = _content_tokens(task.get("description") or "")
        assignee = (task.get("assigned_to") or "").casefold()
        match_at = None
        for index, previous in enumerate(kept):
            if (previous.get("assigned_to") or "").casefold() != assignee:
                continue
            if _jaccard(tokens, _content_tokens(previous.get("description") or "")) >= 0.6:
                match_at = index
                break
        if match_at is None:
            kept.append(task)
            continue
        previous = kept[match_at]
        later_due = (task.get("due_date") or "").strip()
        earlier_due = (previous.get("due_date") or "").strip()
        changed = later_due and earlier_due and later_due.casefold() != earlier_due.casefold()
        reason = (
            "This restates an earlier commitment with a different deadline. The later one was kept."
            if changed
            else "This repeats a commitment already saved."
        )
        review.append({
            "kind": "superseded" if changed else "duplicate",
            "description": previous.get("description") or "",
            "assigned_to": previous.get("assigned_to") or "",
            "due_date": earlier_due or "Not specified",
            "source_text": previous.get("source_text") or previous.get("description") or "",
            "source_context": "",
            "evidence_time": previous.get("evidence_time") or "",
            "reason": reason,
        })
        kept[match_at] = task
    return kept, review


def _unique_decisions(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for decision in decisions:
        text = (decision.get("text") or decision.get("content") or "").strip()
        key = re.sub(r"\s+", " ", text).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(decision)
    return unique

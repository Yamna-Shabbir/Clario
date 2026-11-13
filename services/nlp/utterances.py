"""
services/nlp/utterances.py
--------------------------
Lightweight utterance classifier for Phase 2.

Each sentence is labelled with one of the report categories:
  Decision, Task Assignment, General Discussion.

This is a rule classifier (the report allows a lightweight scikit-learn style
labeler). It does not load a transformer. Task sentences are the only ones
the pipeline should pass to detect_tasks().
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

LABEL_DECISION = "Decision"
LABEL_TASK = "Task Assignment"
LABEL_DISCUSSION = "General Discussion"

_DECISION_RE = re.compile(
    r"\b(?:we(?:'ll| will)|let(?:'s| us) (?:go with|use|switch|adopt)|"
    r"decided to|(?:the )?decision is|agreed to|going with|approved)\b",
    re.IGNORECASE,
)
_NAMED_WILL_RE = re.compile(r"\b[A-Z][a-z]{1,}\s+will\b")
_TASK_RE = re.compile(
    r"\b(?:needs? to|has to|have to|must|please|responsible for|assigned to|"
    r"can you|could you|make sure|follow[- ]up|action item|ensure|"
    r"submit|send|fix|deliver|prepare|complete|finish|design|deploy)\b",
    re.IGNORECASE,
)
# Newlines split speaker labels from the next sentence. Without that,
# "Sarah\nWe decided …" is one sentence when spaCy is unavailable.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str, nlp: Any) -> List[str]:
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def label_sentence(sentence: str) -> Dict[str, Any]:
    """Label one sentence. Named assignments beat collective decisions."""
    text = sentence.strip()
    named_match = _NAMED_WILL_RE.search(text)
    # "We will use X" is a decision, not a person being assigned work.
    named_assignment = named_match is not None and not named_match.group(0).lower().startswith("we ")
    task_cue = _TASK_RE.search(text) is not None or named_assignment
    decision_cue = _DECISION_RE.search(text) is not None

    if task_cue and (named_assignment or not decision_cue):
        label = LABEL_TASK
        confidence = 0.8 if named_assignment else 0.7
    elif decision_cue:
        label = LABEL_DECISION
        confidence = 0.8
    else:
        label = LABEL_DISCUSSION
        confidence = 0.6

    return {
        "text": text,
        "label": label,
        "confidence": confidence,
    }


def classify_utterances(text: str, nlp: Any = None) -> List[Dict[str, Any]]:
    """Split *text* into sentences and label each one.

    Parameters
    ----------
    text : Cleaned transcript.
    nlp  : Optional spaCy model. When omitted, sentences split on punctuation.
    """
    if not text or not text.strip():
        return []
    return [label_sentence(sentence) for sentence in _sentences(text, nlp)]


def task_assignment_text(utterances: List[Dict[str, Any]]) -> str:
    """Join sentences labelled Task Assignment, in order."""
    parts = [u["text"] for u in utterances if u.get("label") == LABEL_TASK and u.get("text")]
    return " ".join(parts)


def decision_utterances(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return utterances labelled Decision."""
    return [u for u in utterances if u.get("label") == LABEL_DECISION]

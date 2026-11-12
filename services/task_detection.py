"""
services/task_detection.py
---------------------------
spaCy + Regex action-item / task detection engine.
Extracted from Clario Project notebook (cells 110-112, 114, 121).
"""

import re
import datetime
from typing import List, Dict, Any

# Module-level NLP cache
_nlp = None

# Task trigger keywords (regex patterns)
TASK_KEYWORDS: List[str] = [
    r"\bwill\b",        # "John WILL complete ..."
    r"\bneeds? to\b",   # "Sarah NEEDS TO review ..."
    r"\bmust\b",        # "Report MUST be submitted ..."
    r"\bshould\b",      # "Team SHOULD review ..."
    r"\bhas to\b",      # "Mike HAS TO present ..."
    r"\bplease\b",      # "PLEASE review ..."
    r"\bdeadline\b",    # mentions a deadline
    r"\bsubmit\b",      # "submit the report"
    r"\bcomplete\b",    # "complete the setup"
    r"\bfinish\b",      # "finish the design"
    r"\bsend\b",        # "send feedback"
    r"\breview\b",      # "review the document"
    r"\bschedule\b",    # "schedule a meeting"
    r"\bfix\b",         # "fix the bug"
    r"\bhandle\b",      # "handle the presentation"
    r"\bprepare\b",     # "prepare the report"
    r"\bdeliver\b",     # "deliver the demo"
    r"\bensure\b",      # "ensure it works"
    r"\bcoordinate\b",  # "coordinate with the team"
]


PRIORITY_KEYWORDS = {
    "High": [r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\bcritical\b", r"\bmust\b", r"\bimportant\b"],
    "Medium": [r"\bsoon\b", r"\bby this week\b", r"\bshould\b", r"\bneeds? to\b"]
}


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


def detect_tasks(text: str,
                 nlp=None,
                 nlp_model=None,
                 *,
                 participants: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Detect action items / tasks in *text* using regex + spaCy NER.

    Algorithm
    ---------
    1. Split text into sentences via spaCy.
    2. For each sentence, test against TASK_KEYWORDS.
    3. If matched, extract:
       - assigned_to  : first PERSON entity (or "Unknown")
       - due_date     : first DATE / TIME entity (or "Not specified")
    4. If participants provided, match assigned_to to participant roster.
    5. Return list of task dicts.

    Parameters
    ----------
    text         : Plain-text meeting transcript or any multi-sentence string.
    nlp          : Pre-loaded spaCy model (preferred kwarg name).
    nlp_model    : Alias for nlp (for backwards compatibility).
    participants : Optional list of participant dicts for attendee matching.

    Returns
    -------
    List of dicts with keys: id, description, assigned_to, due_date, keyword, status, priority
    """
    # Accept both `nlp` and `nlp_model` kwarg names
    loaded_nlp = nlp or nlp_model
    if loaded_nlp is None:
        loaded_nlp = load_nlp()

    tasks: List[Dict[str, Any]] = []
    doc = loaded_nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_keyword = ""
        is_task = False

        for pattern in TASK_KEYWORDS:
            if re.search(pattern, sentence_lower):
                is_task = True
                matched_keyword = pattern.replace(r"\b", "").replace("?", "")
                break

        if not is_task:
            continue

        # NER pass on the sentence
        sent_doc = loaded_nlp(sentence)

        # Assignee: first PERSON entity
        assigned_to = "Team"
        for ent in sent_doc.ents:
            if ent.label_ == "PERSON":
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
        for ent in sent_doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                due_date = ent.text
                break

        # Priority extraction
        priority = "Low"
        for p_pattern in PRIORITY_KEYWORDS["High"]:
            if re.search(p_pattern, sentence_lower):
                priority = "High"
                break
        
        if priority != "High":
            # Dynamic proximity-based priority
            due_lower = due_date.lower()
            if "tomorrow" in due_lower or "today" in due_lower:
                priority = "High"
            elif "friday" in due_lower or "end of week" in due_lower:
                priority = "Low"
            else:
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                # Use current time to find distance
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

        # Fallback to keyword Medium if not already High
        if priority == "Low":
            for p_pattern in PRIORITY_KEYWORDS["Medium"]:
                if re.search(p_pattern, sentence_lower):
                    priority = "Medium"
                    break

        task_data = {
            "id":          len(tasks) + 1,
            "description": sentence,
            "assigned_to": assigned_to,
            "due_date":    due_date,
            "keyword":     matched_keyword,
            "priority":    priority,
            "status":      "pending",
        }
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

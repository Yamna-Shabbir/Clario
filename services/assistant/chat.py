"""Answer one assistant question from documentation and meeting records."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from services.assistant.knowledge import search_help
from services.assistant.provider import (
    AssistantProviderError,
    assistant_configured,
    complete,
    keeps_to_notes,
)
from services.assistant.retrieve import retrieve_meetings
from services.assistant.router import classify, is_mutation, question_tokens

HELP_EMPTY = "I don't have that in the Clario documentation."
MEETING_EMPTY = "I don't have that in your meetings."
READONLY = (
    "I can only look up information. I can't change, complete, or delete "
    "tasks, decisions, or meetings."
)
MODEL_NOTE = "The language model could not be reached, so this answer is taken directly from Clario."

CompleteFn = Callable[[Sequence[Dict[str, str]]], str]


def _focus(question: str) -> str:
    tokens = question_tokens(question)
    if tokens & {"task", "tasks", "assigned", "assignee", "pending", "deadline", "deadlines", "due", "priority"}:
        return "tasks"
    if tokens & {"decision", "decisions"}:
        return "decisions"
    if tokens & {"summary", "summaries"}:
        return "summary"
    if tokens & {"transcript", "said", "speaker", "evidence"}:
        return "transcript"
    if tokens & {"participant", "participants"}:
        return "participants"
    return "overview"


def _render_help(passages: Sequence[Dict[str, str]]) -> str:
    blocks = []
    for passage in passages:
        title = passage.get("title") or "Clario"
        text = (passage.get("text") or "").strip()
        source = passage.get("source") or ""
        blocks.append(f"{title}\n\n{text}\n\nSource: {source}")
    return "\n\n".join(blocks)


def _line_task(task: Dict[str, Any]) -> str:
    parts = [task.get("description") or "Task"]
    if task.get("assigned_to"):
        parts.append(f"assigned to {task['assigned_to']}")
    if task.get("due_date"):
        parts.append(f"due {task['due_date']}")
    if task.get("status"):
        parts.append(task["status"])
    if task.get("priority"):
        parts.append(f"{task['priority']} priority")
    return "- " + " — ".join(parts)


def _render_meetings(bundles: Sequence[Dict[str, Any]], focus: str) -> str:
    blocks = []
    for bundle in bundles:
        lines = [bundle["title"]]
        if bundle.get("meeting_date"):
            lines[0] += f" ({bundle['meeting_date']})"
        if not bundle.get("opened") and focus in {"summary", "transcript", "decisions", "participants"}:
            lines.append("The meeting report is not available to this account.")
        if focus in {"overview", "summary"} and bundle.get("summary"):
            lines.append("Summary")
            lines.append(bundle["summary"].strip())
        if focus in {"overview", "tasks"} and bundle.get("tasks"):
            lines.append("Tasks")
            lines.extend(_line_task(task) for task in bundle["tasks"][:8])
        if focus in {"overview", "decisions"} and bundle.get("decisions"):
            lines.append("Decisions")
            lines.extend(f"- {item['content']}" for item in bundle["decisions"][:6] if item.get("content"))
        if focus in {"overview", "participants"} and bundle.get("participants"):
            lines.append("Participants")
            lines.extend(
                "- " + (person.get("name") or person.get("email") or "Participant")
                for person in bundle["participants"][:12]
            )
        if focus in {"overview", "transcript"} and bundle.get("excerpt"):
            lines.append("Transcript")
            lines.append(bundle["excerpt"])
        if len(lines) == 1 and bundle.get("tasks"):
            lines.append("Tasks")
            lines.extend(_line_task(task) for task in bundle["tasks"][:8])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _evidence_for(bundles: Sequence[Dict[str, Any]], focus: str) -> List[Dict[str, Any]]:
    kinds = {"task", "decision"}
    if focus == "tasks":
        kinds = {"task"}
    elif focus == "decisions":
        kinds = {"decision"}
    chosen: List[Dict[str, Any]] = []
    for bundle in bundles:
        if not bundle.get("opened"):
            continue
        for item in bundle.get("evidence") or []:
            if item.get("kind") not in kinds:
                continue
            chosen.append(item)
            if len(chosen) >= 6:
                return chosen
    return chosen


def _phrase(notes: str, question: str, complete_fn: Optional[CompleteFn]) -> str:
    """Rephrase notes when a model is configured. Drop any wording the notes do not contain."""
    runner = complete_fn
    if runner is None and assistant_configured():
        runner = complete
    if runner is None:
        return notes
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only with words supported by the notes. "
                "Do not add features, people, dates, or steps that the notes do not contain. "
                "If the notes do not answer the question, say you do not have that information."
            ),
        },
        {"role": "user", "content": f"Notes:\n{notes}\n\nQuestion: {question}"},
    ]
    try:
        reply = runner(messages)
    except (AssistantProviderError, Exception):
        return f"{notes}\n\n{MODEL_NOTE}"
    if isinstance(reply, str) and keeps_to_notes(reply, notes, question):
        return reply.strip()
    return notes


def answer_question(
    question: str,
    *,
    user_id: int,
    email: str,
    db_path: str,
    previous: Optional[Dict[str, Any]] = None,
    complete_fn: Optional[CompleteFn] = None,
) -> Dict[str, Any]:
    """
    Answer from retrieved documentation and meetings.

    The reply does not invent a feature or a meeting fact. A missing note is
    stated as unavailable. This function does not write tasks or meetings.
    """
    cleaned = " ".join((question or "").split())
    if is_mutation(cleaned):
        return {
            "content": READONLY,
            "route": "meetings",
            "evidence": [],
            "sources": [],
            "context": {"route": "meetings", "meeting_ids": (previous or {}).get("meeting_ids") or [], "focus": "tasks"},
            "unavailable": False,
        }
    route, followup = classify(cleaned, previous)
    search_text = cleaned
    if followup and previous and previous.get("question"):
        search_text = f"{previous['question']} {cleaned}"
    meeting_ids = previous.get("meeting_ids") if followup and previous else None
    focus = _focus(search_text if followup else cleaned)
    help_passages: List[Dict[str, str]] = []
    bundles: List[Dict[str, Any]] = []
    if route in {"help", "both"}:
        help_passages = search_help(search_text)
    if route in {"meetings", "both"}:
        bundles = retrieve_meetings(
            search_text,
            user_id=user_id,
            email=email,
            db_path=db_path,
            meeting_ids=meeting_ids,
        )
    parts: List[str] = []
    unavailable = False
    if route in {"help", "both"}:
        if help_passages:
            parts.append(_render_help(help_passages))
        else:
            parts.append(HELP_EMPTY)
            unavailable = route == "help"
    if route in {"meetings", "both"}:
        rendered = _render_meetings(bundles, focus)
        if rendered:
            parts.append(rendered)
            unavailable = False
        else:
            parts.append(MEETING_EMPTY)
            unavailable = unavailable or route == "meetings"
    notes = "\n\n".join(parts).strip() or HELP_EMPTY
    if unavailable and route == "both":
        content = notes
    elif unavailable:
        content = notes
    else:
        content = _phrase(notes, cleaned, complete_fn)
    evidence = _evidence_for(bundles, focus) if route in {"meetings", "both"} else []
    return {
        "content": content,
        "route": route,
        "evidence": evidence,
        "sources": [
            {"title": passage.get("title") or "", "source": passage.get("source") or ""}
            for passage in help_passages
        ],
        "context": {
            "route": route,
            "meeting_ids": [bundle["id"] for bundle in bundles],
            "focus": focus,
            "question": cleaned,
        },
        "unavailable": unavailable and not bundles and not help_passages,
    }

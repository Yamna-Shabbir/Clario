"""
services/knowledge_graph/builder.py
-----------------------------------
Decision knowledge graph for Phase 3.

Node types: Person, Task, Decision, Deadline, Meeting.
Relations: ASSIGNED_TO, DECIDED_BY, DEPENDS_ON, RELATED_TO, DEADLINE, BELONGS_TO.

Each node keeps a confidence score and the source sentence when one exists.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_STOP = {
    "about", "after", "again", "being", "could", "decision", "design",
    "going", "meeting", "should", "their", "there", "these", "thing",
    "those", "using", "which", "would", "write",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "item"


def _tokens(text: str) -> set:
    return {tok for tok in re.findall(r"[a-z]{5,}", (text or "").lower()) if tok not in _STOP}


def _short(text: str, limit: int = 60) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_decision_graph(
    tasks: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    *,
    meeting_label: str = "Meeting",
) -> Dict[str, Any]:
    """Build a node/edge graph from classified decisions and detected tasks.

    Parameters
    ----------
    tasks          : Task dicts from detect_tasks (description, assigned_to, due_date).
    decisions      : Utterance dicts with text and confidence.
    meeting_label  : Display label for the Meeting node.

    Returns
    -------
    dict with keys ``nodes`` and ``edges``.
    """
    nodes: List[Dict[str, Any]] = [{
        "id": "meeting",
        "type": "Meeting",
        "label": meeting_label or "Meeting",
        "confidence": 1.0,
        "source": "",
    }]
    edges: List[Dict[str, str]] = []
    people: Dict[str, str] = {}

    def person_node(name: str) -> Optional[str]:
        cleaned = (name or "").strip()
        if not cleaned or cleaned.lower() in {"team", "unknown", "someone"}:
            return None
        key = cleaned.lower()
        if key not in people:
            node_id = f"person-{_slug(cleaned)}"
            people[key] = node_id
            nodes.append({
                "id": node_id,
                "type": "Person",
                "label": cleaned,
                "confidence": 0.75,
                "source": cleaned,
            })
        return people[key]

    decision_ids: List[tuple] = []
    for index, decision in enumerate(decisions, start=1):
        text = (decision.get("text") or decision.get("content") or "").strip()
        if not text:
            continue
        node_id = f"decision-{index}"
        confidence = float(decision.get("confidence") or 0.8)
        nodes.append({
            "id": node_id,
            "type": "Decision",
            "label": _short(text),
            "confidence": confidence,
            "source": text,
        })
        edges.append({"source": node_id, "target": "meeting", "relation": "BELONGS_TO"})
        decision_ids.append((node_id, text, _tokens(text)))

    for index, task in enumerate(tasks, start=1):
        description = (task.get("description") or "").strip()
        if not description:
            continue
        task_id = f"task-{index}"
        nodes.append({
            "id": task_id,
            "type": "Task",
            "label": _short(description),
            "confidence": 0.75,
            "source": description,
        })
        edges.append({"source": task_id, "target": "meeting", "relation": "BELONGS_TO"})

        assignee = person_node(task.get("assigned_to", ""))
        if assignee:
            edges.append({"source": assignee, "target": task_id, "relation": "ASSIGNED_TO"})

        due = (task.get("due_date") or "").strip()
        if due and due.lower() != "not specified":
            due_id = f"deadline-{index}"
            nodes.append({
                "id": due_id,
                "type": "Deadline",
                "label": due,
                "confidence": 0.7,
                "source": due,
            })
            edges.append({"source": task_id, "target": due_id, "relation": "DEADLINE"})

        task_tokens = _tokens(description)
        related = [item for item in decision_ids if task_tokens & item[2]]
        if len(related) == 1:
            edges.append({
                "source": task_id,
                "target": related[0][0],
                "relation": "RELATED_TO",
            })
        if re.search(r"\bdepends on\b", description, re.IGNORECASE) and len(decision_ids) == 1:
            edges.append({
                "source": task_id,
                "target": decision_ids[0][0],
                "relation": "DEPENDS_ON",
            })

    known_people = {node["label"].lower(): node["id"] for node in nodes if node["type"] == "Person"}
    for node_id, text, _tokens_unused in decision_ids:
        for name, person_id in known_people.items():
            if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                edges.append({"source": node_id, "target": person_id, "relation": "DECIDED_BY"})

    return {"nodes": nodes, "edges": edges}

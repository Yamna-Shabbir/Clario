#!/usr/bin/env python
"""
Score meeting-level action items against tests/fixtures/meeting_actions.json.

A confirmed task matches a gold task when the assignee matches and every
required word appears in the description or the stored evidence. Each gold
task and each prediction is used at most once.

Urdu meetings are rendered with the phrase table only. Lines the phrase
table cannot render are reported as misses and are not sent to a neural
translator.

    python scripts/eval_meeting_tasks.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp.task_verification import segment_transcript, verify_meeting_tasks
from services.preprocessing.language import arabic_script_ratio

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "meeting_actions.json"
MULTILINGUAL_PATH = PROJECT_ROOT / "tests" / "fixtures" / "multilingual_meetings.json"
EVAL_DOC = PROJECT_ROOT / "docs" / "meeting-task-eval.md"


def _render_urdu_meeting(transcript: str) -> tuple[str, list[str]]:
    """Return English text and any Urdu lines the phrase table left untouched."""
    from services.preprocessing.urdu_norm import render_urdu_line, speaker_names

    names = speaker_names(transcript)
    kept = []
    uncovered = []
    for raw in transcript.splitlines():
        if not raw.strip():
            kept.append("")
            continue
        rendered = render_urdu_line(raw, names)
        if rendered is None or arabic_script_ratio(rendered) > 0.12:
            uncovered.append(raw.strip())
            kept.append(raw)
        else:
            kept.append(rendered)
    return "\n".join(kept).strip(), uncovered


def _blob(task: dict) -> str:
    parts = [
        task.get("description") or "",
        task.get("source_text") or "",
        task.get("source_context") or "",
        task.get("due_date") or "",
        task.get("assigned_to") or "",
    ]
    return " ".join(parts).casefold()


def _match(gold: dict, task: dict) -> bool:
    assignee = (gold.get("assignee") or "").casefold()
    if assignee and assignee not in (task.get("assigned_to") or "").casefold():
        return False
    blob = _blob(task)
    if not all(word.casefold() in blob for word in gold.get("includes") or []):
        return False
    evidence = (gold.get("evidence") or "").casefold()
    if evidence and evidence not in (task.get("source_text") or "").casefold():
        return False
    stamp = gold.get("time") or ""
    if stamp and stamp not in (task.get("evidence_time") or ""):
        return False
    return True


def _score_meeting(meeting: dict, *, prepare: bool = False) -> dict:
    transcript = meeting["transcript"]
    uncovered: list[str] = []
    if prepare:
        from services.nlp.task_verification import prepare_meeting_turns
        turns = prepare_meeting_turns(transcript)
        result = verify_meeting_tasks(transcript, segments=turns)
    else:
        if meeting.get("language") == "urdu":
            transcript, uncovered = _render_urdu_meeting(transcript)
        turns = segment_transcript(transcript)
        result = verify_meeting_tasks(transcript, segments=turns)
    predictions = list(result["tasks"])
    used = set()
    matched = []
    missed = []
    for gold in meeting.get("tasks") or []:
        found = None
        for index, task in enumerate(predictions):
            if index in used:
                continue
            if _match(gold, task):
                found = index
                break
        if found is None:
            missed.append(gold)
        else:
            used.add(found)
            matched.append({"gold": gold, "task": predictions[found]})
    incorrect = [task for index, task in enumerate(predictions) if index not in used]
    decision_text = " ".join(
        (item.get("text") or item.get("content") or "") for item in result["decisions"]
    ).casefold()
    missed_decisions = [
        phrase for phrase in meeting.get("decisions") or []
        if phrase.casefold() not in decision_text
    ]
    forbidden_hits = []
    for phrase in meeting.get("not_tasks") or []:
        if any(phrase.casefold() in _blob(task) for task in predictions):
            forbidden_hits.append(phrase)
    tp = len(matched)
    fp = len(incorrect) + len(forbidden_hits)
    fn = len(missed) + len(missed_decisions)
    return {
        "id": meeting["id"],
        "language": meeting.get("language"),
        "split": meeting.get("split") or "core",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched": matched,
        "missed": missed,
        "incorrect": incorrect,
        "missed_decisions": missed_decisions,
        "forbidden_hits": forbidden_hits,
        "review": result["review"],
        "uncovered": uncovered,
        "tasks": predictions,
        "decisions": result["decisions"],
    }


def _rates(rows: list[dict]) -> dict:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _write(rows: list[dict], totals: dict, run_at: str) -> None:
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["language"], []).append(row)
    lines = [
        f"## Meeting eval {run_at}",
        "",
        f"Combined precision {totals['precision']:.3f}, "
        f"recall {totals['recall']:.3f}, F1 {totals['f1']:.3f} "
        f"(TP {totals['tp']}, FP {totals['fp']}, FN {totals['fn']}).",
        "",
        "Recall counts a gold task or gold decision only when the confirmed "
        "output contains it. Review items are listed and are not counted as "
        "correct tasks. This is not a claim of perfect extraction.",
        "",
        "### By language",
        "",
    ]
    for language in sorted(grouped):
        rates = _rates(grouped[language])
        lines.append(
            f"- **{language}**: P={rates['precision']:.3f} R={rates['recall']:.3f} "
            f"F1={rates['f1']:.3f} (TP {rates['tp']}, FP {rates['fp']}, FN {rates['fn']})"
        )
    lines.append("")
    for row in rows:
        lines.append(f"### {row['id']} ({row['language']})")
        lines.append("")
        lines.append(f"TP {row['tp']}, FP {row['fp']}, FN {row['fn']}.")
        lines.append("")
        if row["missed"]:
            lines.append("Missed tasks:")
            for gold in row["missed"]:
                lines.append(f"- {gold.get('assignee')}: {', '.join(gold.get('includes') or [])} — {gold.get('note', '')}")
        else:
            lines.append("Missed tasks: none")
        lines.append("")
        if row["incorrect"] or row["forbidden_hits"]:
            lines.append("Incorrect tasks:")
            for task in row["incorrect"]:
                lines.append(
                    f"- {task.get('assigned_to')}: {task.get('description')} "
                    f"(spoken: {task.get('source_text')})"
                )
            for phrase in row["forbidden_hits"]:
                lines.append(f"- Forbidden phrase still extracted: {phrase}")
        else:
            lines.append("Incorrect tasks: none")
        lines.append("")
        if row["missed_decisions"]:
            lines.append("Missed decisions: " + "; ".join(row["missed_decisions"]))
        else:
            lines.append("Missed decisions: none")
        lines.append("")
        if row["uncovered"]:
            lines.append("Urdu lines left untranslated:")
            for line in row["uncovered"]:
                lines.append(f"- {line}")
            lines.append("")
        if row["review"]:
            lines.append("Flagged for review:")
            for item in row["review"]:
                lines.append(f"- {item.get('reason')} | {item.get('source_text')}")
            lines.append("")
    EVAL_DOC.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_DOC.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n## {title}")
    for row in rows:
        print(f"\n=== {row['id']} ({row['language']}, {row['split']}) ===")
        print(f"  TP={row['tp']}  FP={row['fp']}  FN={row['fn']}")
        for gold in row["missed"]:
            print(f"  MISSED {gold.get('assignee')}: {gold.get('includes')} — {gold.get('note', '')}")
        for task in row["incorrect"]:
            print(f"  INCORRECT {task.get('assigned_to')}: {task.get('description')}")
            print(f"    spoken: {task.get('source_text')}")
        for phrase in row["forbidden_hits"]:
            print(f"  FORBIDDEN {phrase}")
        for phrase in row["missed_decisions"]:
            print(f"  MISSED DECISION {phrase}")
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["language"], []).append(row)
        grouped.setdefault(f"split:{row['split']}", []).append(row)
    for key in sorted(grouped):
        rates = _rates(grouped[key])
        print(
            f"  {key}: P={rates['precision']:.3f} R={rates['recall']:.3f} "
            f"F1={rates['f1']:.3f} TP={rates['tp']} FP={rates['fp']} FN={rates['fn']}"
        )


def main() -> int:
    core = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    multi = json.loads(MULTILINGUAL_PATH.read_text(encoding="utf-8"))
    core_rows = [_score_meeting(meeting, prepare=True) for meeting in core["meetings"]]
    multi_rows = [_score_meeting(meeting, prepare=True) for meeting in multi["meetings"]]
    _print_rows("Core meetings", core_rows)
    _print_rows("Urdu, Roman Urdu, and mixed meetings", multi_rows)
    totals = _rates(core_rows + multi_rows)
    print(
        f"\nCombined P={totals['precision']:.3f} "
        f"R={totals['recall']:.3f} F1={totals['f1']:.3f}"
    )
    run_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _write(core_rows + multi_rows, totals, run_at)
    print(f"Results appended to {EVAL_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

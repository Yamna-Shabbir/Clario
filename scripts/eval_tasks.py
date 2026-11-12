#!/usr/bin/env python
"""
scripts/eval_tasks.py
---------------------
Evaluate task-detection precision, recall and F1 on the labelled fixture
tests/fixtures/task_labels.json.

Usage
-----
    python scripts/eval_tasks.py               # eval current detector
    python scripts/eval_tasks.py --baseline    # same, but marks output as BASELINE

Saves results to docs/task-detection-eval.md (appends a dated section).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of where the script is called from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp.task_detection import detect_tasks, load_nlp

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "task_labels.json"
EVAL_DOC     = PROJECT_ROOT / "docs" / "task-detection-eval.md"


def _run_detector_on_sentence(nlp, sentence: str) -> bool:
    """Return True if detect_tasks classifies the sentence as a task."""
    tasks = detect_tasks(sentence, nlp=nlp)
    return len(tasks) > 0


def evaluate(split_name: str, examples: list, nlp) -> dict:
    tp = fp = fn = tn = 0
    false_positives = []
    false_negatives = []

    for ex in examples:
        predicted = _run_detector_on_sentence(nlp, ex["text"])
        gold = bool(ex["label"])

        if predicted and gold:
            tp += 1
        elif predicted and not gold:
            fp += 1
            false_positives.append(ex)
        elif not predicted and gold:
            fn += 1
            false_negatives.append(ex)
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    return {
        "split":    split_name,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _fmt_fp_fn(label: str, examples: list) -> str:
    if not examples:
        return f"**{label}:** none\n"
    lines = [f"**{label}:**"]
    for ex in examples:
        lines.append(f"- [{ex['id']}] *{ex['text']}* — {ex.get('note','')}")
    return "\n".join(lines) + "\n"


def _write_report(results: list[dict], is_baseline: bool, run_at: str) -> str:
    tag = " *(BASELINE)*" if is_baseline else ""
    lines = [
        f"## Eval run {run_at}{tag}\n",
    ]
    for r in results:
        lines.append(
            f"### {r['split']} split  "
            f"(P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1']:.3f})\n"
        )
        lines.append(
            f"| TP | FP | FN | TN |\n"
            f"|----|----|----|----|\n"
            f"| {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |\n"
        )
        lines.append(_fmt_fp_fn("False Positives", r["false_positives"]))
        lines.append(_fmt_fp_fn("False Negatives", r["false_negatives"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Mark this run as the baseline")
    args = parser.parse_args()

    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    dev_examples      = data["dev"]
    held_out_examples = data["held_out"]

    print("Loading spaCy model …")
    nlp = load_nlp()

    dev_result      = evaluate("dev (20)", dev_examples, nlp)
    held_out_result = evaluate("held-out (10)", held_out_examples, nlp)

    # Print to console
    for r in [dev_result, held_out_result]:
        print(f"\n=== {r['split']} ===")
        print(f"  Precision : {r['precision']:.3f}")
        print(f"  Recall    : {r['recall']:.3f}")
        print(f"  F1        : {r['f1']:.3f}")
        print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")
        if r["false_positives"]:
            print("  False Positives:")
            for ex in r["false_positives"]:
                print(f"    [{ex['id']}] {ex['text']}")
        if r["false_negatives"]:
            print("  False Negatives:")
            for ex in r["false_negatives"]:
                print(f"    [{ex['id']}] {ex['text']}")

    # Write to docs
    run_at  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    section = _write_report([dev_result, held_out_result], args.baseline, run_at)

    EVAL_DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DOC, "a", encoding="utf-8") as f:
        f.write("\n---\n")
        f.write(section)
        f.write("\n")

    print(f"\nResults appended to {EVAL_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


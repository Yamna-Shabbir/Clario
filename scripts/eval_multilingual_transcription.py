"""
Evaluate Whisper variants on Urdu / English / mixed fixtures.

Writes a dated section to docs/multilingual-transcription-eval.md and prints
a summary table (WER when jiwer is installed).

Usage:
  python scripts/generate_multilingual_samples.py
  python scripts/eval_multilingual_transcription.py
  python scripts/eval_multilingual_transcription.py --models base small --tasks translate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "multilingual_audio")
DOC_PATH = os.path.join(ROOT, "docs", "multilingual-transcription-eval.md")

try:
    import whisper
except ImportError:
    sys.exit("Install openai-whisper (requirements-ml.txt) before running this eval.")

try:
    import jiwer
except ImportError:
    jiwer = None


def _normalize_transform():
    if jiwer is None:
        return None
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])


def wer(ref: str, hyp: str):
    transform = _normalize_transform()
    if not transform or not ref.strip():
        return None
    out = jiwer.process_words(ref, hyp, reference_transform=transform, hypothesis_transform=transform)
    return round(out.wer * 100, 1)


def run_one(model, audio_path: str, task: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    kwargs = {"task": task, "fp16": False}
    if task == "translate":
        kwargs["initial_prompt"] = "Business meeting in English and Urdu."
    result = model.transcribe(audio_path, **kwargs)
    elapsed = time.perf_counter() - t0
    return (result.get("text") or "").strip(), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Multilingual Whisper eval on Clario fixtures.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny.en", "tiny", "base", "small"],
        help="Whisper checkpoints to compare",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["translate"],
        help="Whisper tasks (translate recommended for English output)",
    )
    args = parser.parse_args()

    manifest_path = os.path.join(FIXTURE_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        print("Run scripts/generate_multilingual_samples.py first.", file=sys.stderr)
        return 1

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    rows = []
    for sample in manifest["samples"]:
        audio_path = os.path.join(FIXTURE_DIR, sample["audio"])
        ref_path = os.path.join(FIXTURE_DIR, sample["reference"])
        with open(ref_path, encoding="utf-8") as fh:
            reference = fh.read().strip()
        audio = whisper.load_audio(audio_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE
        for model_name in args.models:
            if model_name.endswith(".en") and "translate" in args.tasks:
                tasks = ["transcribe"]
            else:
                tasks = args.tasks
            print(f"Loading {model_name} …")
            try:
                model = whisper.load_model(model_name)
            except Exception as exc:
                print(f"  skip {model_name}: {exc}")
                continue
            load_note = model_name
            for task in tasks:
                text, elapsed = run_one(model, audio_path, task)
                w = wer(reference, text)
                rows.append({
                    "sample": sample["id"],
                    "model": load_note,
                    "task": task,
                    "duration_s": round(duration, 1),
                    "rtf": round(elapsed / duration, 2) if duration else 0,
                    "wer_pct": w,
                    "hypothesis": text[:200],
                })
                label = f"{w}%" if w is not None else "n/a"
                print(f"  {sample['id']} | {model_name} | {task} | WER {label} | {elapsed:.1f}s")

    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    section_lines = [
        f"\n## Eval run {date.today().isoformat()}\n",
        "| Sample | Model | Task | WER % | RTF | Snippet |",
        "|--------|-------|------|-------|-----|---------|",
    ]
    for row in rows:
        wer_cell = row["wer_pct"] if row["wer_pct"] is not None else "—"
        snippet = row["hypothesis"].replace("|", "/")[:80]
        section_lines.append(
            f"| {row['sample']} | {row['model']} | {row['task']} | {wer_cell} | "
            f"{row['rtf']} | {snippet} |"
        )
    section = "\n".join(section_lines) + "\n"
    if os.path.isfile(DOC_PATH):
        with open(DOC_PATH, encoding="utf-8") as fh:
            existing = fh.read()
    else:
        existing = "# Multilingual transcription evaluation\n"
    if section.strip() not in existing:
        with open(DOC_PATH, "w", encoding="utf-8") as fh:
            fh.write(existing.rstrip() + "\n" + section)
    print(f"\nAppended results to {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

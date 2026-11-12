"""
scripts/eval_summary.py
-----------------------
Evaluate BART-based summarization quality, topic coverage, and performance.

Takes:
  1. Path to transcript (.txt)
  2. Path to expected topics JSON (.json):
     Format: {"topic_name": ["keyword1", "keyword2", ...], ...}

Computes and reports:
  - Model load time (separately)
  - Summarize time
  - Chunk count and reduce levels
  - Topic coverage for:
      (a) Summary alone
      (b) Summary after task smart-merge

CLI Usage:
  python scripts/eval_summary.py [transcript.txt] [topics.json]
                                 [--model facebook/bart-large-cnn]
                                 [--chunk-tokens 700]
                                 [--beams 4]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BART_MODEL
from services.summarization import load_summarizer, smart_merge, summarize_text
from services.task_detection import detect_tasks


def check_topic_coverage(
    text: str, topics: Dict[str, List[str]]
) -> Tuple[int, int, Dict[str, bool], Dict[str, Optional[str]]]:
    """
    Check which topics are covered in *text*.

    A topic is covered if any of its keywords appears in the text (case-insensitive).

    Returns:
        (covered_count, total_count, {topic: is_covered}, {topic: matched_keyword})
    """
    text_lower = text.lower()
    covered = {}
    matched_kws = {}
    for topic_name, keywords in topics.items():
        match = None
        for kw in keywords:
            if kw.lower() in text_lower:
                match = kw
                break
        covered[topic_name] = match is not None
        matched_kws[topic_name] = match
    return sum(covered.values()), len(topics), covered, matched_kws


def eval_summary(
    transcript_path: str,
    topics_path: str,
    model_name: str = BART_MODEL,
    chunk_tokens: int = 700,
    num_beams: int = 4,
    device: int = -1,
    verbose: bool = True,
    summariser=None,
) -> Dict[str, Any]:
    """
    Run summarizer evaluation on a transcript and topics file.
    """
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if not os.path.isfile(topics_path):
        raise FileNotFoundError(f"Topics JSON not found: {topics_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    with open(topics_path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if verbose:
        print("=" * 80)
        print("CLARIO SUMMARIZATION EVALUATION")
        print("=" * 80)
        print(f"Transcript:   {transcript_path} ({len(transcript)} chars, ~{len(transcript.split())} words)")
        print(f"Topics file:  {topics_path} ({len(topics)} expected topics)")
        print(f"Model:        {model_name}")
        print(f"Chunk tokens: {chunk_tokens}")
        print(f"Num beams:    {num_beams}")
        print("-" * 80)

    # 1. Model load time (separate)
    if summariser is None:
        if verbose:
            print(f"Loading model '{model_name}' ...")
        t_load = time.perf_counter()
        summariser = load_summarizer(model_name=model_name, device=device)
        load_time = time.perf_counter() - t_load
        if verbose:
            print(f"Model loaded in {load_time:.2f}s")
    else:
        load_time = 0.0

    # 2. Extract tasks from transcript for smart-merge
    if verbose:
        print("Extracting action items via spaCy / regex ...")
    try:
        tasks = detect_tasks(transcript)
    except Exception as exc:
        if verbose:
            print(f"  Notice: detect_tasks exception ({exc}), using empty tasks")
        tasks = []
    if verbose:
        print(f"Detected {len(tasks)} action items.")

    # 3. Run summarization (alone)
    info: Dict[str, Any] = {}
    if verbose:
        print("Running map-reduce summarization ...")
    t_sum = time.perf_counter()
    summary_alone = summarize_text(
        transcript,
        model_name=model_name,
        device=device,
        summariser=summariser,
        chunk_tokens=chunk_tokens,
        num_beams=num_beams,
        _info=info,
    )
    summarize_time = time.perf_counter() - t_sum
    if verbose:
        print(f"Summarization completed in {summarize_time:.2f}s")

    # 4. Smart-merge action items
    summary_merged = smart_merge(summary_alone, tasks)

    # 5. Check coverage
    total_chunks = sum(info.get("chunk_counts", [1]))
    levels = info.get("levels", 1)

    cov_alone_count, total_topics, cov_alone_map, match_alone = check_topic_coverage(
        summary_alone, topics
    )
    cov_merged_count, _, cov_merged_map, match_merged = check_topic_coverage(
        summary_merged, topics
    )

    pct_alone = (cov_alone_count / total_topics * 100) if total_topics else 0.0
    pct_merged = (cov_merged_count / total_topics * 100) if total_topics else 0.0

    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)
        print(f"Total Chunks:           {total_chunks} (by level: {info.get('chunk_counts', [])})")
        print(f"Reduce Levels:          {levels}")
        print(f"Model Load Time:        {load_time:.2f}s")
        print(f"Summarize Time:         {summarize_time:.2f}s")
        print(f"Coverage (a) Summary:   {cov_alone_count}/{total_topics} ({pct_alone:.1f}%)")
        print(f"Coverage (b) +Tasks:    {cov_merged_count}/{total_topics} ({pct_merged:.1f}%)")
        print("-" * 80)

        print(f"\n{'Topic':<32} | {'(a) Summary Alone':<18} | {'(b) + Smart-Merge':<18}")
        print("-" * 32 + "-+-" + "-" * 18 + "-+-" + "-" * 18)
        for topic_name in topics:
            res_a = f"YES ({match_alone[topic_name]})" if cov_alone_map[topic_name] else "NO"
            res_b = f"YES ({match_merged[topic_name]})" if cov_merged_map[topic_name] else "NO"
            print(f"{topic_name:<32} | {res_a:<18} | {res_b:<18}")

        print("\n" + "-" * 80)
        print("[SUMMARY ALONE]:")
        print(summary_alone)
        print("\n" + "-" * 80)
        print("[SUMMARY WITH SMART-MERGE]:")
        print(summary_merged)
        print("=" * 80)

    return {
        "model": model_name,
        "chunk_tokens": chunk_tokens,
        "beams": num_beams,
        "total_chunks": total_chunks,
        "levels": levels,
        "load_time": load_time,
        "summarize_time": summarize_time,
        "coverage_alone": f"{cov_alone_count}/{total_topics} ({pct_alone:.1f}%)",
        "coverage_merged": f"{cov_merged_count}/{total_topics} ({pct_merged:.1f}%)",
        "cov_alone_count": cov_alone_count,
        "cov_merged_count": cov_merged_count,
        "total_topics": total_topics,
        "summary_alone": summary_alone,
        "summary_merged": summary_merged,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Clario BART summarization coverage and runtime."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        default="tests/fixtures/long_meeting.txt",
        help="Path to transcript file",
    )
    parser.add_argument(
        "topics",
        nargs="?",
        default="tests/fixtures/long_meeting_topics.json",
        help="Path to expected topics JSON",
    )
    parser.add_argument(
        "--model",
        default=BART_MODEL,
        help="Summarizer model variant (default: facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=700,
        help="Chunk token limit (default: 700)",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device (-1 for CPU, 0 for CUDA)",
    )

    args = parser.parse_args()
    eval_summary(
        transcript_path=args.transcript,
        topics_path=args.topics,
        model_name=args.model,
        chunk_tokens=args.chunk_tokens,
        num_beams=args.beams,
        device=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()


"""
scripts/run_eval_matrix.py
--------------------------
Run evaluation across transcripts and configurations, and produce a consolidated table:
  - bart-large-cnn, 700 tokens, beams 4
  - bart-large-cnn, 700 tokens, beams 2
  - sshleifer/distilbart-cnn-12-6, 700 tokens, beams 4

Transcripts:
  1. Short (2.1k chars): tests/fixtures/extended_meeting.txt
  2. Long (25k chars): tests/fixtures/long_meeting.txt
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval_summary import eval_summary
from services.nlp.summarization import load_summarizer


def run_matrix():
    transcripts = [
        {
            "name": "Short (2.1k chars)",
            "text": "tests/fixtures/extended_meeting.txt",
            "topics": "tests/fixtures/extended_meeting_topics.json",
        },
        {
            "name": "Long (25k chars)",
            "text": "tests/fixtures/long_meeting.txt",
            "topics": "tests/fixtures/long_meeting_topics.json",
        },
    ]

    configs = [
        {"model": "facebook/bart-large-cnn", "beams": 4, "chunk_tokens": 700, "label": "bart-large-cnn (beams=4)"},
        {"model": "facebook/bart-large-cnn", "beams": 2, "chunk_tokens": 700, "label": "bart-large-cnn (beams=2)"},
        {"model": "sshleifer/distilbart-cnn-12-6", "beams": 4, "chunk_tokens": 700, "label": "distilbart-12-6 (beams=4)"},
    ]

    results = []

    # Cache models to avoid reloading
    cached_models = {}

    for cfg in configs:
        model_name = cfg["model"]
        if model_name not in cached_models:
            print("\n" + "=" * 80)
            print(f"LOADING MODEL FOR BENCHMARK: {model_name}")
            print("=" * 80)
            t0 = time.perf_counter()
            cached_models[model_name] = load_summarizer(model_name=model_name)
            load_time = time.perf_counter() - t0
            print(f"Loaded {model_name} in {load_time:.2f}s")
        else:
            load_time = 0.0

        for tr in transcripts:
            # Enforce: do not run long transcript with beams 4 on bart-large-cnn more than once
            print("\n" + "-" * 80)
            print(f"RUNNING: {tr['name']} | {cfg['label']}")
            print("-" * 80)

            res = eval_summary(
                transcript_path=tr["text"],
                topics_path=tr["topics"],
                model_name=model_name,
                chunk_tokens=cfg["chunk_tokens"],
                num_beams=cfg["beams"],
                verbose=False,
                summariser=cached_models[model_name],
            )
            res["transcript_name"] = tr["name"]
            res["config_label"] = cfg["label"]
            results.append(res)
            print(
                f"Done: {tr['name']} | Chunks: {res['total_chunks']} | Levels: {res['levels']} | "
                f"Time: {res['summarize_time']:.1f}s | "
                f"Cov (a): {res['coverage_alone']} | Cov (b): {res['coverage_merged']}"
            )

    # Print consolidated table
    print("\n" + "=" * 105)
    print("CONSOLIDATED BENCHMARK EVALUATION TABLE")
    print("=" * 105)
    header = (
        f"{'Transcript':<18} | {'Model Config':<24} | {'Chunks':<6} | {'Levels':<6} | "
        f"{'Time (s)':<9} | {'Coverage (a) Alone':<19} | {'Coverage (b) +Tasks':<19}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            f"{r['transcript_name']:<18} | {r['config_label']:<24} | {r['total_chunks']:<6} | "
            f"{r['levels']:<6} | {r['summarize_time']:<9.2f} | {r['coverage_alone']:<19} | {r['coverage_merged']:<19}"
        )
        print(row)
    print("=" * 105)


if __name__ == "__main__":
    run_matrix()


"""
scripts/compare_summarizer.py
-----------------------------
Compare real BART summarization:
  - Old: hard-truncated to first 1000 characters
  - New: token-aware map-reduce processing the complete transcript

Usage:
  python scripts/compare_summarizer.py [path_to_transcript.txt]
"""

import os
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.nlp.summarization import load_summarizer, summarize_text


def run_comparison(transcript_path: str, model_name: str = "facebook/bart-large-cnn"):
    if not os.path.isfile(transcript_path):
        sys.exit(f"Transcript not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    print("=" * 80)
    print(f"BART SUMMARIZER COMPARISON ON: {transcript_path}")
    print(f"Total transcript length: {len(full_text)} characters (~{len(full_text.split())} words)")
    print("=" * 80)

    print("\nLoading real BART model and tokenizer (cached if already downloaded) ...")
    print(f"Model: {model_name}")
    t_load = time.perf_counter()
    summariser = load_summarizer(model_name=model_name)
    load_duration = time.perf_counter() - t_load
    print(f"Model ready in {load_duration:.2f}s")

    # ── 1. Old approach: hard truncation at 1000 chars ────────────────────────
    print("\n" + "-" * 80)
    print("RUNNING OLD LOGIC (Hard-truncated at 1000 characters)")
    print("-" * 80)
    old_input = full_text[:1000]
    t0 = time.perf_counter()
    # Simulate old logic directly on truncated slice
    model, tokenizer = summariser
    inputs = tokenizer(old_input, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=50,
        num_beams=4,
        early_stopping=True,
    )
    old_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    old_duration = time.perf_counter() - t0

    print(f"Time taken: {old_duration:.2f}s")
    print(f"Chars processed: {len(old_input)} of {len(full_text)} ({len(old_input)/len(full_text)*100:.1f}%)")
    print(f"Chars discarded: {len(full_text) - len(old_input)}")
    print(f"\n[OLD SUMMARY]:\n{old_summary}")

    # ── 2. New approach: Token-aware Map-Reduce ──────────────────────────────
    print("\n" + "-" * 80)
    print("RUNNING NEW LOGIC (Token-Aware Map-Reduce on full transcript)")
    print("-" * 80)
    progress_log = []

    def on_progress(done, total):
        progress_log.append(f"Chunk {done}/{total}")

    t1 = time.perf_counter()
    # Use chunk_tokens=150 to clearly demonstrate map-reduce on this 330-word transcript
    new_summary = summarize_text(
        full_text,
        max_length=150,
        min_length=50,
        summariser=summariser,
        chunk_tokens=150,
        progress=on_progress,
    )
    new_duration = time.perf_counter() - t1

    print(f"Time taken: {new_duration:.2f}s")
    print(f"Chars processed: {len(full_text)} of {len(full_text)} (100.0%)")
    print(f"Progress steps: {' -> '.join(progress_log)}")
    print(f"\n[NEW SUMMARY]:\n{new_summary}")

    # ── 3. Content Coverage Comparison ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TOPIC COVERAGE COMPARISON")
    print("=" * 80)
    key_topics = [
        ("Financials & John's budget", "john"),
        ("Dave & database locking issues", "dave"),
        ("Sarah & frontend pull requests", "sarah"),
        ("Alice & SOC 2 compliance", "alice"),
        ("Bob & Acme architectural diagrams", "bob"),
        ("Emma & social media scheduling", "emma"),
        ("Liam & influencer partners", "liam"),
        ("Mark & Stripe billing webhooks", "mark"),
    ]

    print(f"{'Meeting Agenda Item':<38} | {'Old (1000 chars)':<16} | {'New (Map-Reduce)':<16}")
    print("-" * 38 + "-+-" + "-" * 16 + "-+-" + "-" * 16)
    for topic_label, kw in key_topics:
        in_old = "✅ Included" if kw in old_summary.lower() else "❌ DROPPED"
        in_new = "✅ Included" if kw in new_summary.lower() else "❌ Missed"
        print(f"{topic_label:<38} | {in_old:<16} | {in_new:<16}")
    print("=" * 80)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_audios/extended_meeting.txt"
    model = sys.argv[2] if len(sys.argv) > 2 else "facebook/bart-large-cnn"
    run_comparison(path, model_name=model)


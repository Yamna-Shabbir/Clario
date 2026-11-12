"""
scripts/benchmark_whisper.py
----------------------------
Benchmark OpenAI Whisper models on an audio file.

Measures:
  - Model load time
  - Transcription time
  - Audio duration
  - Real-time factor (RTF = transcribe_time / audio_duration)
  - Word Error Rate (WER via jiwer, with normalized text)
  - First 300 characters of transcription output
  - Optional word-level diffs (substitutions, insertions, deletions) via --show-diff

Usage:
  python scripts/benchmark_whisper.py sample_audios/engineering_standup.mp3
  python scripts/benchmark_whisper.py sample_audios/engineering_standup.mp3 --show-diff
  python scripts/benchmark_whisper.py audio.mp3 --reference path/to/transcript.txt
  python scripts/benchmark_whisper.py audio.mp3 --models tiny.en base.en --full
"""

import argparse
import os
import sys
import time

# Ensure ffmpeg binaries are found on Windows
_FFMPEG_PATHS = [
    r"C:\Users\Dell\AppData\Roaming\Python\Python314\Scripts",
    r"C:\Users\Dell\AppData\Roaming\Python\Python314\site-packages\imageio_ffmpeg\binaries",
]
for _p in _FFMPEG_PATHS:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

try:
    import whisper
except ImportError:
    sys.exit("Error: openai-whisper is not installed. Install via: pip install openai-whisper")

try:
    import jiwer
except ImportError:
    jiwer = None


def get_normalization_transform():
    """
    Construct a jiwer.Compose normalization transform.

    Transforms:
      1. Replace hyphens/dashes with spaces
      2. Lowercase text
      3. Strip punctuation
      4. Collapse whitespace
      5. Strip leading/trailing whitespace
      6. Reduce to list of list of words (required by jiwer 3+ for word-level ops)
    """
    if jiwer is None:
        return None
    return jiwer.Compose([
        jiwer.SubstituteRegexes({r"[-–—]": " "}),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])


def format_table(rows, headers):
    """Render a clean ASCII/Markdown table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    table_lines = [header_line, separator]

    for row in rows:
        line = " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        table_lines.append(line)

    return "\n".join(table_lines)


def print_word_level_diff(model_name: str, out):
    """Print detailed word-level substitutions, insertions, deletions and alignment."""
    print(f"\n{'=' * 75}")
    print(f"WORD-LEVEL DIFF FOR [{model_name}]")
    print(f"{'=' * 75}")
    print(
        f"Summary: Substitutions={out.substitutions}, "
        f"Insertions={out.insertions}, "
        f"Deletions={out.deletions}, "
        f"Hits={out.hits} | WER={out.wer * 100:.2f}%\n"
    )

    diff_items = []
    for sent_idx, alignment in enumerate(out.alignments):
        ref_words = out.references[sent_idx]
        hyp_words = out.hypotheses[sent_idx]
        for chunk in alignment:
            if chunk.type == "substitute":
                r_str = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                h_str = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                diff_items.append(f"  [SUBSTITUTION] '{r_str}' -> '{h_str}'")
            elif chunk.type == "insert":
                h_str = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                diff_items.append(f"  [INSERTION]    '{h_str}'")
            elif chunk.type == "delete":
                r_str = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                diff_items.append(f"  [DELETION]     '{r_str}'")

    if diff_items:
        print("Detailed Differences:")
        for item in diff_items:
            print(item)
    else:
        print("  (Perfect match - no word errors detected!)")

    print("\nAlignment Visualization:")
    print(jiwer.visualize_alignment(out))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper models on an audio file."
    )
    parser.add_argument("audio_path", help="Path to audio file (mp3, wav, m4a, etc.)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny.en", "base.en", "small.en"],
        help="Whisper model variants to benchmark (default: tiny.en base.en small.en)",
    )
    parser.add_argument(
        "--reference",
        "-r",
        default=None,
        help="Optional path to reference transcript .txt file (default: looks for <basename>.txt next to audio)",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Print word-level substitutions, insertions, and deletions per model",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Allow audio longer than 2 minutes (120 seconds)",
    )
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio_path)
    if not os.path.isfile(audio_path):
        sys.exit(f"Error: Audio file not found: {audio_path}")

    # Determine reference file path
    ref_path = None
    if args.reference:
        candidate = os.path.abspath(args.reference)
        if not os.path.isfile(candidate):
            sys.exit(f"Error: Specified reference file not found: {candidate}")
        ref_path = candidate
    else:
        candidate = os.path.splitext(audio_path)[0] + ".txt"
        if os.path.isfile(candidate):
            ref_path = candidate

    ref_text = None
    if ref_path and os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        print(f"Found reference transcript: {ref_path} ({len(ref_text)} chars)")
    else:
        print("No reference file provided or found next to audio. WER calculation skipped.")

    # Load audio to check duration
    print(f"\nLoading audio: {audio_path} ...")
    try:
        audio = whisper.load_audio(audio_path)
    except Exception as exc:
        sys.exit(f"Error decoding audio with Whisper/ffmpeg: {exc}")

    duration = len(audio) / whisper.audio.SAMPLE_RATE
    print(f"Audio duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

    # Enforce 2-minute safety limit unless --full is provided
    if duration > 120.0 and not args.full:
        sys.exit(
            f"\nRefusing to run on audio longer than 2 minutes ({duration:.1f}s).\n"
            f"Pass --full if you wish to benchmark this file."
        )

    # Flatten models list in case user passes comma-separated strings
    model_list = []
    for item in args.models:
        for sub in item.split(","):
            sub = sub.strip()
            if sub and sub not in model_list:
                model_list.append(sub)

    headers = [
        "Model",
        "Load Time",
        "Transcribe Time",
        "Duration",
        "RTF",
    ]
    if ref_text and jiwer:
        headers.append("WER")
    headers.append("First 300 Chars")

    transform = get_normalization_transform()

    results = []
    transcripts = {}
    word_outputs = {}

    print(f"\nBenchmarking models: {', '.join(model_list)}")
    print("=" * 70)

    for model_name in model_list:
        print(f"\n--> Benchmarking model: {model_name} ...")
        t0 = time.perf_counter()
        try:
            model = whisper.load_model(model_name)
        except Exception as exc:
            print(f"    Failed to load model '{model_name}': {exc}")
            continue
        load_time = time.perf_counter() - t0
        print(f"    Loaded in {load_time:.2f}s")

        t1 = time.perf_counter()
        res = model.transcribe(audio_path)
        transcribe_time = time.perf_counter() - t1
        text = res.get("text", "").strip()
        transcripts[model_name] = text
        print(f"    Transcribed in {transcribe_time:.2f}s")

        rtf = transcribe_time / duration if duration > 0 else 0.0

        wer_display = None
        if ref_text and jiwer and transform:
            out = jiwer.process_words(
                ref_text,
                text,
                reference_transform=transform,
                hypothesis_transform=transform,
            )
            word_outputs[model_name] = out
            wer_display = f"{out.wer * 100:.1f}%"
            print(f"    WER (normalized): {wer_display}")

        first_300 = text[:300].replace("\n", " ")

        row = [
            model_name,
            f"{load_time:.2f}s",
            f"{transcribe_time:.2f}s",
            f"{duration:.2f}s",
            f"{rtf:.3f}x",
        ]
        if ref_text and jiwer:
            row.append(wer_display)
        row.append(first_300)

        results.append(row)

    print("\n" + "=" * 80)
    print("WHISPER BENCHMARK RESULTS")
    print("=" * 80)
    print(format_table(results, headers))
    print("=" * 80)

    # Show detailed diffs if requested
    if args.show_diff and ref_text and jiwer:
        for model_name in model_list:
            if model_name in word_outputs:
                print_word_level_diff(model_name, word_outputs[model_name])

    # Print full transcripts comparison
    print("\n--- TRANSCRIPTS COMPARISON ---")
    if ref_path and ref_text:
        print(f"\n[REFERENCE ({ref_path})]:\n{ref_text}\n")
    for m_name in model_list:
        if m_name in transcripts:
            print(f"[{m_name}]:\n{transcripts[m_name]}\n")


if __name__ == "__main__":
    main()

"""
services/pipeline.py
---------------------
Master Clario pipeline:
  audio / text  →  transcript  →  summary  →  tasks  →  graph  →  DB

Public API
----------
    results = run_clario_pipeline(
        audio_path  = "/path/to/recording.mp3",   # or None
        text_input  = "Raw transcript text …",     # or None
        meeting_id  = 7,                           # pre-created DB row
        user_id     = 3,
        db_path     = "/path/to/clario.db",
        graph_dir   = "/path/to/graphs/",
    )
"""

import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import (
    DB_PATH, GRAPH_DIR,
    WHISPER_MODEL, BART_MODEL, SPACY_MODEL,
    SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH,
)
import pathlib


# ── Lazy model cache ───────────────────────────────────────────────────────────
_whisper_model    = None
_whisper_lock     = threading.Lock()
_summariser       = None
_summariser_lock  = threading.Lock()
_nlp              = None
_nlp_lock         = threading.Lock()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from services.transcription import load_whisper
                _whisper_model = load_whisper(WHISPER_MODEL)
    return _whisper_model


def _get_summariser():
    global _summariser
    if _summariser is None:
        with _summariser_lock:
            if _summariser is None:
                from services.summarization import load_summarizer
                _summariser = load_summarizer(BART_MODEL)
    return _summariser


def _get_nlp():
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:
                from services.task_detection import load_nlp
                _nlp = load_nlp(SPACY_MODEL)
    return _nlp


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_clario_pipeline(
    audio_path: Optional[str]  = None,
    text_input: Optional[str]  = None,
    meeting_id: Optional[int]  = None,
    user_id:    Optional[int]  = None,
    db_path:    str            = DB_PATH,
    graph_dir:  str            = GRAPH_DIR,
    *,
    participants: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the full Clario AI pipeline and persist results.

    Parameters
    ----------
    audio_path   : Path to an audio file (if provided, Whisper is used).
    text_input   : Raw transcript text (used if no audio_path or as fallback).
    meeting_id   : Pre-created meeting row id in the DB.
    user_id      : Authenticated user id (stored for graph naming).
    db_path      : SQLite database path.
    graph_dir    : Directory where the knowledge graph PNG will be saved.
    participants : Optional list of participant dicts for attendee matching.

    Returns
    -------
    dict with keys: transcript, summary, tasks, graph_path, stats
    None if no usable input was provided.
    """

    # Limit OpenBLAS/OpenMP threads to reduce memory pressure on constrained machines
    import os as _os
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")

    from models.database import (
        insert_transcript, insert_summary,
        insert_tasks,
    )
    from services.transcription import transcribe_audio
    from services.summarization import summarize_text
    from services.task_detection import detect_tasks

    # ── 1. Transcription ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("transcribing")
    if audio_path and os.path.exists(audio_path):
        print(" Transcribing audio …")
        try:
            transcript = transcribe_audio(audio_path, model=_get_whisper())
        except Exception as exc:
            if text_input and text_input.strip():
                # Text was also supplied — use it and warn
                print(f"⚠️  Whisper failed: {exc}. Using supplied text_input instead.")
                transcript = text_input
            else:
                # No fallback text — surface the error so the route can tell the user
                print(f"❌ Whisper failed and no text_input fallback: {exc}")
                raise RuntimeError(
                    f"Audio transcription failed: {exc}\n"
                    "Make sure FFmpeg is installed and on your PATH, or paste the "
                    "transcript text directly."
                ) from exc
    elif text_input:
        transcript = text_input
    else:
        print(" No audio file or text input provided.")
        return None

    if not transcript.strip():
        print(" Empty transcript — cannot proceed.")
        return None

    print(f" Transcript length: {len(transcript)} chars")

    # ── 2. Task detection ──────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("detecting_tasks")
    print(" Detecting tasks …")
    try:
        tasks = detect_tasks(transcript, nlp_model=_get_nlp(), participants=participants)
    except Exception as exc:
        print(f"️  Task detection failed: {exc}")
        tasks = []

    print(f" {len(tasks)} task(s) detected")

    # ── 3. Summarisation ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("summarizing")
    print(" Summarising …")
    try:
        from services.summarization import smart_merge
        raw_summary = summarize_text(
            transcript,
            summariser  = _get_summariser(),
            max_length  = SUMMARY_MAX_LENGTH,
            min_length  = SUMMARY_MIN_LENGTH,
            tasks       = tasks,
        )
        summary = smart_merge(raw_summary, tasks)
    except (Exception, MemoryError) as exc:
        print(f"⚠️  Summarisation failed: {exc}")
        try:
            from services.summarization import extractive_summarize
            nlp_for_summary = _get_nlp()
            summary = extractive_summarize(transcript, nlp=nlp_for_summary, tasks=tasks)
            print(f"   Extractive fallback summary: {len(summary)} chars")
        except Exception as fallback_exc:
            print(f"   Extractive fallback also failed: {fallback_exc}")
            summary = transcript[:1000]

    print(f" Summary length: {len(summary)} chars")

    # ── 4. Persist to DB ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("saving")
    graph_path = ""
    vibe = "Neutral"
    if meeting_id is not None:
        try:
            # Calculate Sentiment
            try:
                from textblob import TextBlob
                from models.database import update_meeting_sentiment
                polarity = TextBlob(transcript).sentiment.polarity
                if polarity > 0.15:
                    vibe = "Positive"
                elif polarity < -0.05:
                    vibe = "Tense"
                else:
                    vibe = "Neutral"
                update_meeting_sentiment(meeting_id, vibe, db_path)
            except Exception as e:
                print(f"  Sentiment analysis failed: {e}")

            insert_transcript(meeting_id, transcript, db_path)
            insert_summary(meeting_id,    summary,    db_path)
            if tasks:
                insert_tasks(meeting_id, tasks, db_path)
            print(f" Results saved to DB (meeting_id={meeting_id}) with vibe: {vibe}")
        except Exception as exc:
            print(f"️  DB save failed: {exc}")

        # ── 5. Analytics & Graphs ──────────────────────────────────────────────
        if progress_callback:
            progress_callback("generating_charts")
        try:
            from utils.visualization import (
                build_knowledge_graph,
                generate_assignee_bar_chart,
                generate_priority_donut_chart,
                generate_status_pie_chart
            )
            from models.database import set_meeting_graphs
            
            app_root = pathlib.Path(__file__).resolve().parent.parent
            static_graph_dir = app_root / "static" / "graphs"
            static_graph_dir.mkdir(parents=True, exist_ok=True)
            
            # Paths
            graph_filename = f"graph_m{meeting_id}.png"
            bar_filename   = f"bar_m{meeting_id}.png"
            donut_filename = f"donut_m{meeting_id}.png"
            status_filename = f"status_m{meeting_id}.png"
            
            graph_path_abs = static_graph_dir / graph_filename
            bar_path_abs   = static_graph_dir / bar_filename
            donut_path_abs = static_graph_dir / donut_filename
            status_path_abs = static_graph_dir / status_filename
            
            # Limit to top 25 tasks (High priority first) to avoid OOM in graph
            sorted_tasks = sorted(tasks, key=lambda t: (0 if t.get("priority") == "High" else 1 if t.get("priority") == "Medium" else 2))
            tasks_for_graph = [dict(t, id=i+1) for i, t in enumerate(sorted_tasks[:25])]
            
            # Generate
            build_knowledge_graph(tasks_for_graph, output_path=str(graph_path_abs))
            generate_assignee_bar_chart(tasks_for_graph, output_path=str(bar_path_abs))
            generate_priority_donut_chart(tasks_for_graph, output_path=str(donut_path_abs))
            generate_status_pie_chart(tasks_for_graph, output_path=str(status_path_abs))
            
            # Relative paths for DB
            graph_path = f"graphs/{graph_filename}"
            bar_path   = f"graphs/{bar_filename}"
            donut_path = f"graphs/{donut_filename}"
            status_path = f"graphs/{status_filename}"
            
            set_meeting_graphs(meeting_id, graph_path, bar_path, donut_path, status_path, db_path)
            print(f"  Analytics graphs generated for meeting {meeting_id}")
        except BaseException as exc:
            print(f"  Analytics generation failed: {exc}")

    return {
        "transcript": transcript,
        "summary":    summary,
        "tasks":      tasks,
        "graph_path": graph_path,
        "stats": {
            "meeting_id":   meeting_id,
            "task_count":   len(tasks),
            "summary_len":  len(summary),
            "transcript_len": len(transcript),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

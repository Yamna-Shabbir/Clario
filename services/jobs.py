"""
services/jobs.py
----------------
Background job executor and queue management for the Clario AI pipeline.

Features:
- ThreadPoolExecutor worker pool with configurable worker count (CLARIO_PIPELINE_WORKERS).
- Non-blocking job submission from HTTP routes (submit_pipeline_job).
- Automatic status transitions: queued -> processing -> completed / failed.
- Independent SQLite connection per worker thread (via models.database context managers).
- Post-completion hook registry for pluggable callbacks (e.g. email notifications)
  that never raise into the job.
- Optional synchronous execution mode (CLARIO_SYNC_JOBS=True) for deterministic tests.
"""

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from config import CLARIO_PIPELINE_WORKERS, CLARIO_SYNC_JOBS, DB_PATH, GRAPH_DIR

logger = logging.getLogger(__name__)

# Type for post-completion callbacks: hook(results_dict) -> None
PostCompletionHook = Callable[[Dict[str, Any]], None]

# ── Global worker pool & hooks ─────────────────────────────────────────────────
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

_post_completion_hooks: List[PostCompletionHook] = []
_hooks_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Return the singleton ThreadPoolExecutor, initializing if necessary."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                workers = max(1, CLARIO_PIPELINE_WORKERS)
                _executor = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="clario-pipeline-worker",
                )
                logger.info("Initialized ThreadPoolExecutor with %d worker(s).", workers)
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Shut down the executor (useful in test teardown)."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=wait)
            _executor = None


# ── Post-completion hook registry ──────────────────────────────────────────────

def register_post_completion_hook(hook: PostCompletionHook) -> None:
    """Register a callback to run after successful pipeline completion."""
    with _hooks_lock:
        if hook not in _post_completion_hooks:
            _post_completion_hooks.append(hook)


def clear_post_completion_hooks() -> None:
    """Clear all registered post-completion hooks (useful in tests)."""
    with _hooks_lock:
        _post_completion_hooks.clear()


def _run_post_completion_hooks(results: Dict[str, Any]) -> None:
    """Execute all registered post-completion hooks safely."""
    with _hooks_lock:
        hooks = list(_post_completion_hooks)
    for hook in hooks:
        try:
            hook(results)
        except Exception as exc:
            logger.exception("Post-completion hook %r failed: %s", hook, exc)


# ── Pipeline worker logic ──────────────────────────────────────────────────────

def _run_pipeline_job(
    meeting_id: int,
    user_id: int,
    audio_path: Optional[str] = None,
    text_input: Optional[str] = None,
    db_path: str = DB_PATH,
    graph_dir: str = GRAPH_DIR,
    *,
    participants: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Worker function executed in the background thread (or inline in sync mode).

    Handles status transitions:
      1. Move status from queued -> processing.
      2. Call run_clario_pipeline with progress_callback.
      3. Move status to completed on success and invoke post-completion hooks.
      4. Catch any Exception, record readable error_message, and set status to failed.
    """
    from models.database import update_meeting_status, update_meeting_progress
    from services.pipeline import run_clario_pipeline

    initial_stage = "transcribing" if (audio_path and os.path.exists(audio_path)) else "detecting_tasks"

    # Step 1: Move to processing
    try:
        update_meeting_status(
            meeting_id,
            status="processing",
            error_message=None,
            progress_stage=initial_stage,
            db_path=db_path,
        )
    except Exception as exc:
        logger.error("Failed to update meeting %s status to processing: %s", meeting_id, exc)

    def progress_callback(stage: str) -> None:
        try:
            update_meeting_progress(meeting_id, stage, db_path=db_path)
        except Exception as exc:
            logger.warning("Failed to update progress stage for meeting %s to %s: %s", meeting_id, stage, exc)

    # Step 2: Execute pipeline
    try:
        results = run_clario_pipeline(
            audio_path=audio_path,
            text_input=text_input,
            meeting_id=meeting_id,
            user_id=user_id,
            db_path=db_path,
            graph_dir=graph_dir,
            participants=participants,
            progress_callback=progress_callback,
        )

        if not results:
            raise RuntimeError("Pipeline returned no results. Check meeting input.")

        # Step 3: Record completion
        update_meeting_status(
            meeting_id,
            status="completed",
            error_message=None,
            progress_stage="completed",
            db_path=db_path,
        )

        # Step 4: Fire post-completion hooks
        _run_post_completion_hooks(results)

        return results

    except Exception as exc:
        error_msg = str(exc) or type(exc).__name__
        logger.error("Pipeline job failed for meeting %s: %s", meeting_id, error_msg, exc_info=True)
        try:
            update_meeting_status(
                meeting_id,
                status="failed",
                error_message=error_msg,
                progress_stage="failed",
                db_path=db_path,
            )
        except Exception as db_exc:
            logger.critical("Failed to record failure for meeting %s: %s", meeting_id, db_exc)
        return None


def submit_pipeline_job(
    meeting_id: int,
    user_id: int,
    audio_path: Optional[str] = None,
    text_input: Optional[str] = None,
    db_path: str = DB_PATH,
    graph_dir: str = GRAPH_DIR,
    *,
    participants: Optional[List[Dict[str, Any]]] = None,
    sync: Optional[bool] = None,
) -> Any:
    """
    Submit a meeting pipeline job.

    In async mode (default), queues the task into the ThreadPoolExecutor and
    returns a concurrent.futures.Future.

    In sync mode (CLARIO_SYNC_JOBS=True or sync=True), executes inline on the
    calling thread and returns the result dict (or None on failure).
    """
    is_sync = sync if sync is not None else CLARIO_SYNC_JOBS
    if is_sync:
        return _run_pipeline_job(
            meeting_id=meeting_id,
            user_id=user_id,
            audio_path=audio_path,
            text_input=text_input,
            db_path=db_path,
            graph_dir=graph_dir,
            participants=participants,
        )

    executor = get_executor()
    return executor.submit(
        _run_pipeline_job,
        meeting_id,
        user_id,
        audio_path,
        text_input,
        db_path,
        graph_dir,
        participants=participants,
    )


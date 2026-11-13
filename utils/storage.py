"""
Disk space, SQLite checkpoint, and safe temporary-file cleanup.

Nothing in this module deletes users, meetings, transcripts, summaries,
tasks, uploads, charts, outbox copies, or a finished model download.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
from typing import Iterable, List

logger = logging.getLogger("clario.storage")

FREE_TIER_AUDIO_MESSAGE = (
    "Audio transcription is not available on the free plan. "
    "Paste the meeting transcript instead. "
    "Clario will not download speech models on this plan."
)

_SKIP_DIRS = {
    "venv",
    ".venv",
    "env",
    "node_modules",
    ".git",
    "data",
    "uploads",
    "static",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
}


class ModelDownloadBlocked(RuntimeError):
    """A model download was refused before any file was fetched."""


def is_disk_full_error(exc: BaseException) -> bool:
    """True when SQLite or the OS reports that the filesystem is full."""
    if not isinstance(exc, (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)):
        return False
    text = str(exc).lower()
    return "disk is full" in text or "sqlite_full" in text or "no space left" in text


def free_bytes(path: str) -> int:
    """Free bytes on the filesystem that holds ``path``."""
    probe = path or "."
    if not os.path.exists(probe):
        probe = os.path.dirname(os.path.abspath(probe)) or "."
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            probe = "."
            break
        probe = parent
    return shutil.disk_usage(probe).free


def directory_bytes(path: str) -> int:
    """Size of a file or directory. Missing paths count as zero."""
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _mb(num: int) -> str:
    return f"{num / (1024 * 1024):.1f} MB"


def disk_report() -> str:
    """One-line inventory of free space and the folders that can grow."""
    from config import (
        DATA_DIR,
        DB_PATH,
        EMAIL_DIR,
        HF_MODELS_DIR,
        STATIC_GRAPH_DIR,
        UPLOAD_DIR,
    )

    probe = DB_PATH or DATA_DIR or "."
    try:
        usage = shutil.disk_usage(probe if os.path.exists(probe) else os.path.dirname(os.path.abspath(probe)) or ".")
        free_text = f"free {_mb(usage.free)} of {_mb(usage.total)}"
    except OSError as exc:
        free_text = f"free space unreadable ({exc})"
    parts = [
        free_text,
        f"db {_mb(directory_bytes(DB_PATH))}",
        f"wal {_mb(directory_bytes(DB_PATH + '-wal'))}",
        f"shm {_mb(directory_bytes(DB_PATH + '-shm'))}",
        f"uploads {_mb(directory_bytes(UPLOAD_DIR))}",
        f"charts {_mb(directory_bytes(STATIC_GRAPH_DIR))}",
        f"outbox {_mb(directory_bytes(EMAIL_DIR))}",
        f"models {_mb(directory_bytes(HF_MODELS_DIR))}",
    ]
    return "; ".join(parts)


def log_disk_usage(reason: str) -> None:
    """Write the disk inventory to the process log."""
    logger.info("Disk usage (%s): %s", reason, disk_report())


def upload_block_reason(directory: str, incoming_bytes: int = 0) -> str:
    """
    Return a user-facing message when this upload would drop free space
    below the configured reserve. An empty string means the upload may proceed.
    """
    from config import DISK_RESERVE_MB

    reserve = max(0, DISK_RESERVE_MB) * 1024 * 1024
    free = free_bytes(directory)
    incoming = max(0, int(incoming_bytes or 0))
    if free < reserve or free - incoming < reserve:
        return (
            f"Not enough free disk space to store this upload "
            f"({free // (1024 * 1024)} MB free, "
            f"{DISK_RESERVE_MB} MB must stay available). "
            "Existing meetings were not deleted."
        )
    return ""


def assert_model_download_allowed(purpose: str) -> None:
    """
    Refuse a download on the free plan, and on a paid plan whose disk cannot
    hold a model without eating the safety reserve.

    Call this only when the weights are not already on disk.
    """
    from config import (
        DATA_DIR,
        DISK_RESERVE_MB,
        FREE_TIER,
        HF_MODELS_DIR,
        MODEL_DOWNLOAD_MIN_FREE_MB,
    )

    if FREE_TIER:
        raise ModelDownloadBlocked(FREE_TIER_AUDIO_MESSAGE)
    probe = HF_MODELS_DIR or DATA_DIR or "."
    free = free_bytes(probe)
    need = (max(0, MODEL_DOWNLOAD_MIN_FREE_MB) + max(0, DISK_RESERVE_MB)) * 1024 * 1024
    if free < need:
        log_disk_usage(f"blocked download of {purpose}")
        raise ModelDownloadBlocked(
            f"Not enough free disk space to download {purpose}. "
            f"{free // (1024 * 1024)} MB is free and "
            f"{MODEL_DOWNLOAD_MIN_FREE_MB} MB is required above the "
            f"{DISK_RESERVE_MB} MB safety reserve. "
            "Existing meetings were not deleted."
        )


def friendly_storage_error(exc: BaseException) -> str:
    """User-facing text for a full disk or a refused download. Empty otherwise."""
    if isinstance(exc, ModelDownloadBlocked):
        log_disk_usage("model download blocked")
        return str(exc)
    if is_disk_full_error(exc):
        log_disk_usage("disk full")
        return (
            "The server disk is full, so this meeting could not be saved. "
            "Existing meetings, transcripts, and tasks were not deleted."
        )
    return ""


def _remove_incomplete(roots: Iterable[str]) -> List[str]:
    removed: List[str] = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for current, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".incomplete"):
                    continue
                path = os.path.join(current, name)
                try:
                    os.remove(path)
                except OSError:
                    continue
                removed.append(path)
    return removed


def cleanup_safe_files(
    *,
    project_root: str,
    models_dir: str,
    extra_roots: Iterable[str] | None = None,
) -> List[str]:
    """
    Delete incomplete model downloads, project ``__pycache__`` folders, and
    Clario partial files in the temp directory.

    ``data/``, ``uploads/``, and ``static/`` are not walked, so the database,
    outbox, audio, and charts stay where they are. The home Hugging Face
    cache is included only when the caller passes it in ``extra_roots``.
    """
    cache_roots = [models_dir, *(extra_roots or ())]
    removed = _remove_incomplete(cache_roots)

    partials = os.path.join(tempfile.gettempdir(), "clario-partials")
    if os.path.isdir(partials):
        for name in os.listdir(partials):
            if not (name.endswith(".incomplete") or name.endswith(".tmp")):
                continue
            path = os.path.join(partials, name)
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
            except OSError:
                continue
            removed.append(path)

    if project_root and os.path.isdir(project_root):
        for current, dirs, _files in os.walk(project_root, topdown=True):
            kept = []
            for name in dirs:
                if name in _SKIP_DIRS:
                    continue
                if name == "__pycache__":
                    path = os.path.join(current, name)
                    shutil.rmtree(path, ignore_errors=True)
                    removed.append(path)
                    continue
                kept.append(name)
            dirs[:] = kept
    return removed


def checkpoint_database(db_path: str) -> None:
    """
    Fold the WAL back into the database file.

    ``wal_checkpoint(TRUNCATE)`` does not remove rows. It only shrinks the
    ``-wal`` sidecar after committed pages are copied into the database.
    """
    if not db_path or not os.path.isfile(db_path):
        return
    from models.database import get_connection

    with get_connection(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def maintain_storage(db_path: str) -> None:
    """Startup maintenance: safe cleanup, WAL checkpoint, then a disk log."""
    from config import HF_MODELS_DIR, ROOT_DIR

    try:
        removed = cleanup_safe_files(
            project_root=ROOT_DIR,
            models_dir=HF_MODELS_DIR,
            extra_roots=(
                os.environ.get("HF_HOME", ""),
                os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            ),
        )
        if removed:
            logger.info("Removed %d temporary storage files.", len(removed))
    except Exception:
        logger.exception("Temporary-file cleanup failed")
    try:
        checkpoint_database(db_path)
    except sqlite3.OperationalError as exc:
        if is_disk_full_error(exc):
            log_disk_usage("checkpoint")
        logger.warning("WAL checkpoint skipped: %s", exc)
    except Exception:
        logger.exception("WAL checkpoint failed")
    log_disk_usage("startup")

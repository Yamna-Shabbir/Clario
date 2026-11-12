"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Clario test suite.

Model patching strategy (CLAUDE.md rule 3, updated):
  - Whisper  → always mocked (returns {"text": FAKE_TRANSCRIPT})
  - BART     → always mocked (returns a short deterministic string)
  - spaCy    → loaded for real (en_core_web_sm is small; session-scoped)
"""

import os
import pathlib
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants shared by tests
# ---------------------------------------------------------------------------

FAKE_TRANSCRIPT = (
    "Alice will prepare the project proposal by Friday. "
    "Bob must review the design documents immediately. "
    "The team should schedule a follow-up meeting next week."
)
FAKE_SUMMARY = "Alice and Bob discussed project milestones and assigned tasks."


# ---------------------------------------------------------------------------
# Fake ML models
# ---------------------------------------------------------------------------

class _FakeWhisperModel:
    """Minimal stand-in for a whisper.Whisper model."""

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        return {"text": FAKE_TRANSCRIPT}


def _make_fake_summariser():
    """
    Return a (model, tokenizer) tuple whose generate/decode chain returns
    FAKE_SUMMARY, matching the interface expected by services/summarization.py.
    """
    tokenizer = MagicMock()
    # tokenizer(text, ...) → dict with "input_ids"
    tokenizer.return_value = {"input_ids": MagicMock()}

    model = MagicMock()
    # model.generate(input_ids, ...) → list of one id-tensor
    model.generate.return_value = [MagicMock()]
    # tokenizer.decode(ids, skip_special_tokens=True) → FAKE_SUMMARY
    tokenizer.decode.return_value = FAKE_SUMMARY

    return model, tokenizer


# ---------------------------------------------------------------------------
# Session-scoped spaCy fixture (loaded once for the whole test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_nlp():
    """Load the real en_core_web_sm once per test session."""
    import spacy  # noqa: PLC0415
    return spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Autouse patches — replace heavy models in every test automatically
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_whisper():
    """Replace pipeline._get_whisper with a lightweight fake for every test."""
    fake = _FakeWhisperModel()
    with patch("services.pipeline._get_whisper", return_value=fake):
        yield fake


@pytest.fixture(autouse=True)
def patch_summariser():
    """Replace pipeline._get_summariser with a lightweight fake for every test."""
    fake_pair = _make_fake_summariser()
    with patch("services.pipeline._get_summariser", return_value=fake_pair):
        yield fake_pair


@pytest.fixture(autouse=True)
def patch_nlp(real_nlp):
    """
    Wire the real session-scoped spaCy model into pipeline._get_nlp.
    spaCy en_core_web_sm is small enough to load for real in tests.
    """
    with patch("services.pipeline._get_nlp", return_value=real_nlp):
        yield real_nlp


# ---------------------------------------------------------------------------
# Also patch visualization so tests don't write PNGs to static/graphs/
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_visualization(request, tmp_path):
    """
    Redirect all chart/graph generation to a per-test temp directory.
    Patches the pathlib.Path resolution inside services/pipeline.py so
    generated PNGs go to tmp_path instead of static/graphs/.
    Skipped for test_visualization which tests real chart generation.
    """
    if "test_visualization" in getattr(request.module, "__name__", ""):
        yield
        return
    # pipeline.py uses: pathlib.Path(__file__).resolve().parent.parent / "static" / "graphs"
    # We replace that resolved path with tmp_path.
    fake_static_graph = tmp_path / "graphs"
    fake_static_graph.mkdir(parents=True, exist_ok=True)

    _original_path_cls = pathlib.Path

    class _PatchedPath(type(pathlib.Path())):
        """Subclass that returns tmp_path for the specific static/graphs chain."""

    # Simpler: just stub build_knowledge_graph and the chart functions
    # so they write to tmp_path / "graphs" and return a valid path string.
    def _noop_graph(tasks, output_path=None, show=False):
        if output_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(output_path).write_bytes(b"PNG")
        return output_path or ""

    with (
        patch("utils.visualization.build_knowledge_graph",       side_effect=_noop_graph),
        patch("utils.visualization.generate_assignee_bar_chart",  side_effect=_noop_graph),
        patch("utils.visualization.generate_priority_donut_chart", side_effect=_noop_graph),
        patch("utils.visualization.generate_status_pie_chart",    side_effect=_noop_graph),
    ):
        # Also make pipeline's pathlib.Path(__file__).resolve().parent.parent
        # point at tmp_path so static_graph_dir is writable.
        with patch("services.pipeline.pathlib") as mock_pl:
            mock_pl.Path.return_value.resolve.return_value.parent.parent = tmp_path
            yield


# ---------------------------------------------------------------------------
# Temp-dir fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dirs(tmp_path) -> dict:
    """
    Return a dict of temp paths for db, uploads, and graphs.
    Uses pytest's built-in ``tmp_path`` for automatic cleanup.
    """
    db_path    = str(tmp_path / "test.db")
    upload_dir = str(tmp_path / "uploads")
    graph_dir  = str(tmp_path / "graphs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(graph_dir,  exist_ok=True)
    return {
        "db_path":    db_path,
        "upload_dir": upload_dir,
        "graph_dir":  graph_dir,
    }


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_dirs):
    """
    Create a fresh Flask app wired to temp directories for each test.
    DB is initialised; uploads and graphs go into throwaway temp dirs.
    """
    from app import create_app as _create_app  # noqa: PLC0415

    flask_app = _create_app(
        test_config={
            "db_path":    tmp_dirs["db_path"],
            "upload_dir": tmp_dirs["upload_dir"],
            "graph_dir":  tmp_dirs["graph_dir"],
            "TESTING":    True,
        }
    )
    flask_app.config["WTF_CSRF_ENABLED"] = False
    yield flask_app


@pytest.fixture()
def client(app):
    """Unauthenticated test client."""
    return app.test_client()


@pytest.fixture()
def logged_in_client(client):
    """
    Test client that has completed register + login.

    Yields ``(client, creds_dict)`` where ``creds_dict`` has keys:
    ``username``, ``email``, ``password``.
    """
    creds = {
        "username": "testuser",
        "email":    "test@example.com",
        "password": "TestPass123!",
    }

    # Register (auto-logs in)
    resp = client.post(
        "/register",
        data={
            "username": creds["username"],
            "email":    creds["email"],
            "password": creds["password"],
            "confirm":  creds["password"],
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200, f"Register failed: {resp.status_code}"

    yield client, creds

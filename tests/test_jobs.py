"""
tests/test_jobs.py
------------------
Tests for background jobs, asynchronous pipeline executor, and status polling:
- Status transitions: queued -> processing -> completed
- Failure path: queued -> processing -> failed with error_message stored
- Status endpoint GET /meeting/<id>/status: ownership enforcement and 404 checks
- Startup recovery: fail_interrupted_meetings marks queued/processing as failed
- Participant pass-through: verify participants reach run_clario_pipeline
- Post-completion hook registry: callback execution and error isolation
- Async submission via ThreadPoolExecutor
- Result page rendering across states (queued/processing, failed, completed)
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from models.database import (
    init_db,
    insert_user,
    insert_meeting,
    get_user_by_email,
    get_meeting_detail,
    get_meeting_status,
    fail_interrupted_meetings,
)
from services.jobs import (
    submit_pipeline_job,
    register_post_completion_hook,
    clear_post_completion_hooks,
)


@pytest.fixture(autouse=True)
def clean_hooks():
    """Ensure hook registry is clean before and after each test."""
    clear_post_completion_hooks()
    yield
    clear_post_completion_hooks()


class TestStatusTransitionsAndExecution:

    def test_sync_job_status_transition_to_completed(self, tmp_dirs):
        """Test happy path: queued -> processing -> completed."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("alice", "alice@example.com", "hash", db_path)
        meeting_id = insert_meeting(
            user_id=user_id,
            title="Sprint Planning",
            status="queued",
            progress_stage="queued",
            db_path=db_path,
        )

        # Mock run_clario_pipeline to simulate work and invoke progress callback
        fake_result = {
            "transcript": "Alice will document the API.",
            "summary": "Meeting summary",
            "tasks": [{"description": "Alice will document the API", "assigned_to": "Alice"}],
            "graph_path": "",
            "stats": {"task_count": 1},
        }

        stages_recorded = []

        def fake_pipeline(*args, **kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                for stage in ("detecting_tasks", "summarizing", "saving", "generating_charts"):
                    cb(stage)
                    stages_recorded.append(stage)
            return fake_result

        with patch("services.pipeline.run_clario_pipeline", side_effect=fake_pipeline):
            result = submit_pipeline_job(
                meeting_id=meeting_id,
                user_id=user_id,
                text_input="Alice will document the API.",
                db_path=db_path,
                graph_dir=tmp_dirs["graph_dir"],
                sync=True,
            )

        assert result == fake_result
        assert "detecting_tasks" in stages_recorded
        assert "summarizing" in stages_recorded

        # Verify DB status
        status_info = get_meeting_status(meeting_id, user_id, db_path)
        assert status_info["status"] == "completed"
        assert status_info["stage"] == "completed"
        assert status_info["error"] is None

    def test_job_failure_path_records_error(self, tmp_dirs):
        """Test error path: pipeline exception moves status to failed with readable error_message."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("bob", "bob@example.com", "hash", db_path)
        meeting_id = insert_meeting(
            user_id=user_id,
            title="Corrupted Recording",
            status="queued",
            progress_stage="queued",
            db_path=db_path,
        )

        error_text = "Whisper audio decoder crashed: invalid header"

        with patch("services.pipeline.run_clario_pipeline", side_effect=RuntimeError(error_text)):
            result = submit_pipeline_job(
                meeting_id=meeting_id,
                user_id=user_id,
                audio_path="/fake/path.mp3",
                db_path=db_path,
                graph_dir=tmp_dirs["graph_dir"],
                sync=True,
            )

        assert result is None

        # Verify DB records failure
        status_info = get_meeting_status(meeting_id, user_id, db_path)
        assert status_info["status"] == "failed"
        assert status_info["stage"] == "failed"
        assert error_text in status_info["error"]

    def test_participants_passed_through_to_pipeline(self, tmp_dirs):
        """Verify participants kwarg passed to submit_pipeline_job reaches run_clario_pipeline."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("charlie", "charlie@example.com", "hash", db_path)
        meeting_id = insert_meeting(user_id=user_id, status="queued", db_path=db_path)

        participants_list = [
            {"name": "Sara Chen", "email": "sara@company.com"},
            {"name": "David Kim", "email": "david@company.com"},
        ]

        captured_kwargs = {}

        def fake_pipeline(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return {"transcript": "text", "summary": "sum", "tasks": []}

        with patch("services.pipeline.run_clario_pipeline", side_effect=fake_pipeline):
            submit_pipeline_job(
                meeting_id=meeting_id,
                user_id=user_id,
                text_input="Sara will review the PR.",
                db_path=db_path,
                graph_dir=tmp_dirs["graph_dir"],
                participants=participants_list,
                sync=True,
            )

        assert captured_kwargs.get("participants") == participants_list

    def test_async_submission_with_thread_pool(self, tmp_dirs):
        """Verify async submission (sync=False) returns a Future and completes in background thread."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("dana", "dana@example.com", "hash", db_path)
        meeting_id = insert_meeting(user_id=user_id, status="queued", db_path=db_path)

        def slow_fake_pipeline(*args, **kwargs):
            time.sleep(0.05)
            return {"transcript": "done", "summary": "done", "tasks": []}

        with patch("services.pipeline.run_clario_pipeline", side_effect=slow_fake_pipeline):
            future = submit_pipeline_job(
                meeting_id=meeting_id,
                user_id=user_id,
                text_input="Dana will check the logs.",
                db_path=db_path,
                graph_dir=tmp_dirs["graph_dir"],
                sync=False,
            )
            assert hasattr(future, "result")
            result = future.result(timeout=5)

        assert result is not None
        status_info = get_meeting_status(meeting_id, user_id, db_path)
        assert status_info["status"] == "completed"


class TestPostCompletionHooks:

    def test_post_completion_hook_called_on_success(self, tmp_dirs):
        """Registered hooks are called with results dictionary when pipeline succeeds."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("eve", "eve@example.com", "hash", db_path)
        meeting_id = insert_meeting(user_id=user_id, status="queued", db_path=db_path)

        hook_called_with = []

        def my_hook(results):
            hook_called_with.append(results)

        register_post_completion_hook(my_hook)

        fake_res = {"transcript": "abc", "summary": "xyz", "tasks": []}
        with patch("services.pipeline.run_clario_pipeline", return_value=fake_res):
            submit_pipeline_job(
                meeting_id=meeting_id,
                user_id=user_id,
                text_input="Transcript text",
                db_path=db_path,
                graph_dir=tmp_dirs["graph_dir"],
                sync=True,
            )

        assert len(hook_called_with) == 1
        assert hook_called_with[0] == fake_res

    def test_hook_exception_does_not_fail_job(self, tmp_dirs):
        """If a post-completion hook raises an exception, the job remains completed."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("frank", "frank@example.com", "hash", db_path)
        meeting_id = insert_meeting(user_id=user_id, status="queued", db_path=db_path)

        def crashing_hook(results):
            raise ValueError("Email notification service offline")

        register_post_completion_hook(crashing_hook)

        fake_res = {"transcript": "abc", "summary": "xyz", "tasks": []}
        with patch("services.pipeline.run_clario_pipeline", return_value=fake_res):
            submit_pipeline_job(
                meeting_id=meeting_id,
                user_id=user_id,
                text_input="Transcript text",
                db_path=db_path,
                graph_dir=tmp_dirs["graph_dir"],
                sync=True,
            )

        # Meeting status should still be completed!
        status_info = get_meeting_status(meeting_id, user_id, db_path)
        assert status_info["status"] == "completed"


class TestStartupRecovery:

    def test_fail_interrupted_meetings(self, tmp_dirs):
        """Any meetings in 'queued' or 'processing' are marked as failed with recovery error."""
        db_path = tmp_dirs["db_path"]
        init_db(db_path)
        user_id = insert_user("george", "george@example.com", "hash", db_path)

        m1 = insert_meeting(user_id=user_id, title="M1", status="queued", db_path=db_path)
        m2 = insert_meeting(user_id=user_id, title="M2", status="processing", db_path=db_path)
        m3 = insert_meeting(user_id=user_id, title="M3", status="completed", db_path=db_path)
        m4 = insert_meeting(user_id=user_id, title="M4", status="failed", error_message="Previous error", db_path=db_path)

        count = fail_interrupted_meetings(db_path)
        assert count == 2

        s1 = get_meeting_status(m1, user_id, db_path)
        assert s1["status"] == "failed"
        assert s1["error"] == "Interrupted by server restart"

        s2 = get_meeting_status(m2, user_id, db_path)
        assert s2["status"] == "failed"
        assert s2["error"] == "Interrupted by server restart"

        # Completed and already failed meetings must not be modified
        s3 = get_meeting_status(m3, user_id, db_path)
        assert s3["status"] == "completed"

        s4 = get_meeting_status(m4, user_id, db_path)
        assert s4["status"] == "failed"
        assert s4["error"] == "Previous error"


class TestStatusEndpointAndOwnership:

    def test_get_status_success(self, logged_in_client, tmp_dirs):
        """Owner can get meeting status JSON {status, stage, error}."""
        c, creds = logged_in_client
        db_path = tmp_dirs["db_path"]
        user_id = get_user_by_email(creds["email"], db_path)["id"]

        meeting_id = insert_meeting(
            user_id=user_id,
            title="Status Test",
            status="processing",
            progress_stage="summarizing",
            db_path=db_path,
        )

        resp = c.get(f"/meeting/{meeting_id}/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "processing"
        assert data["stage"] == "summarizing"
        assert data["error"] is None

    def test_get_status_404_for_non_owner(self, app, tmp_dirs):
        """User B querying User A's meeting status receives 404."""
        db_path = tmp_dirs["db_path"]
        user_a = insert_user("usera", "usera@example.com", "hash", db_path)
        meeting_a = insert_meeting(user_id=user_a, title="User A Meeting", db_path=db_path)

        # Login as user B
        c_b = app.test_client()
        c_b.post(
            "/register",
            data=dict(username="userb", email="userb@example.com", password="Password1!", confirm="Password1!"),
        )

        resp = c_b.get(f"/meeting/{meeting_a}/status")
        assert resp.status_code == 404

    def test_get_status_404_for_non_existent(self, logged_in_client):
        """Querying a non-existent meeting id returns 404."""
        c, _ = logged_in_client
        resp = c.get("/meeting/999999/status")
        assert resp.status_code == 404


class TestResultPageStatusRendering:

    def test_result_page_shows_loading_ui_when_queued(self, logged_in_client, tmp_dirs):
        """When meeting is queued/processing, result page renders processing container."""
        c, creds = logged_in_client
        db_path = tmp_dirs["db_path"]
        user_id = get_user_by_email(creds["email"], db_path)["id"]
        meeting_id = insert_meeting(
            user_id=user_id,
            title="In Flight Meeting",
            status="processing",
            progress_stage="transcribing",
            db_path=db_path,
        )

        resp = c.get(f"/result/{meeting_id}")
        assert resp.status_code == 200
        assert b"processing-container" in resp.data
        assert b"Transcribing audio" in resp.data

    def test_result_page_shows_failed_ui_when_failed(self, logged_in_client, tmp_dirs):
        """When meeting failed, result page displays error message and link to upload again."""
        c, creds = logged_in_client
        db_path = tmp_dirs["db_path"]
        user_id = get_user_by_email(creds["email"], db_path)["id"]
        meeting_id = insert_meeting(
            user_id=user_id,
            title="Failed Meeting",
            status="failed",
            error_message="Whisper out of memory",
            db_path=db_path,
        )

        resp = c.get(f"/result/{meeting_id}")
        assert resp.status_code == 200
        assert b"Meeting processing failed" in resp.data
        assert b"Whisper out of memory" in resp.data
        assert b"Upload again" in resp.data


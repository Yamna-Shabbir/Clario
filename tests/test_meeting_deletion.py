"""Meeting deletion tests against an isolated temporary database."""

import gc
import os
import tempfile
import unittest
from pathlib import Path

from flask import Flask
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash

import routes.dashboard as dashboard_routes
import routes.integrations as integrations_routes
import routes.meeting as meeting_routes
import routes.search as search_routes
from models.database import (
    get_connection,
    get_meetings_by_user,
    get_people_stats,
    init_db,
    insert_meeting,
    insert_summary,
    insert_tasks,
    insert_transcript,
    insert_user,
    set_meeting_graphs,
)
from routes.auth import auth_bp
from routes.dashboard import dashboard_bp
from routes.integrations import integrations_bp
from routes.meeting import meeting_bp
from routes.search import search_bp

ROOT = Path(__file__).resolve().parents[1]


class MeetingDeletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "deletion.db")
        self.upload_dir = os.path.join(self.tmp.name, "uploads")
        self.graph_dir = os.path.join(self.tmp.name, "graphs")
        self.outside = os.path.join(self.tmp.name, "outside")
        os.makedirs(self.upload_dir)
        os.makedirs(self.graph_dir)
        os.makedirs(self.outside)

        self._saved = {
            "meeting_db": meeting_routes.DB_PATH,
            "meeting_upload": meeting_routes.UPLOAD_DIR,
            "meeting_graphs": meeting_routes.STATIC_GRAPHS_DIR,
            "dashboard_db": dashboard_routes.DB_PATH,
            "integrations_db": integrations_routes.DB_PATH,
            "search_db": search_routes.DB_PATH,
        }
        meeting_routes.DB_PATH = self.db_path
        meeting_routes.UPLOAD_DIR = self.upload_dir
        meeting_routes.STATIC_GRAPHS_DIR = self.graph_dir
        dashboard_routes.DB_PATH = self.db_path
        integrations_routes.DB_PATH = self.db_path
        search_routes.DB_PATH = self.db_path

        init_db(self.db_path)
        self.owner_id = insert_user(
            "Owner", "owner@example.com", generate_password_hash("password1"), self.db_path
        )
        self.other_id = insert_user(
            "Other", "other@example.com", generate_password_hash("password2"), self.db_path
        )

        self.app = Flask(
            "deletion-tests",
            template_folder=str(ROOT / "templates"),
            static_folder=str(ROOT / "static"),
        )
        self.app.secret_key = "deletion-test-secret"
        self.app.config["WTF_CSRF_ENABLED"] = False
        CSRFProtect(self.app)
        from routes.live import live_bp
        from routes.tasks import tasks_bp
        self.app.register_blueprint(auth_bp)
        self.app.register_blueprint(dashboard_bp)
        self.app.register_blueprint(integrations_bp)
        self.app.register_blueprint(meeting_bp)
        self.app.register_blueprint(search_bp)
        self.app.register_blueprint(live_bp)
        self.app.register_blueprint(tasks_bp)
        self.client = self.app.test_client()

    def tearDown(self):
        meeting_routes.DB_PATH = self._saved["meeting_db"]
        meeting_routes.UPLOAD_DIR = self._saved["meeting_upload"]
        meeting_routes.STATIC_GRAPHS_DIR = self._saved["meeting_graphs"]
        dashboard_routes.DB_PATH = self._saved["dashboard_db"]
        integrations_routes.DB_PATH = self._saved["integrations_db"]
        search_routes.DB_PATH = self._saved["search_db"]
        gc.collect()
        self.tmp.cleanup()

    def _login(self, user_id, username):
        with self.client.session_transaction() as sess:
            sess["user_id"] = user_id
            sess["username"] = username
            sess["email"] = f"{username.lower()}@example.com"

    def _seed_meeting(self, user_id, summary, audio_name=None, graphs=True):
        meeting_id = insert_meeting(
            user_id=user_id,
            title="Planning",
            audio_filename=audio_name,
            db_path=self.db_path,
        )
        insert_transcript(meeting_id, f"Transcript for {summary}", self.db_path)
        insert_summary(meeting_id, summary, self.db_path)
        insert_tasks(
            meeting_id,
            [{
                "description": f"Task for {summary}",
                "assigned_to": "Alex",
                "due_date": "Friday",
                "status": "pending",
                "keyword": "will",
                "priority": "High",
            }],
            self.db_path,
        )
        if audio_name and "/" not in audio_name and ".." not in audio_name:
            with open(os.path.join(self.upload_dir, audio_name), "w", encoding="utf-8") as fh:
                fh.write("audio")
        if graphs:
            names = {
                "graph": f"graph_m{meeting_id}.png",
                "bar": f"bar_m{meeting_id}.png",
                "donut": f"donut_m{meeting_id}.png",
                "status": f"status_m{meeting_id}.png",
            }
            for filename in names.values():
                with open(os.path.join(self.graph_dir, filename), "w", encoding="utf-8") as fh:
                    fh.write("png")
            set_meeting_graphs(
                meeting_id,
                f"graphs/{names['graph']}",
                f"graphs/{names['bar']}",
                f"graphs/{names['donut']}",
                f"graphs/{names['status']}",
                self.db_path,
            )
        return meeting_id

    def _counts(self, meeting_id):
        conn = get_connection(self.db_path)
        try:
            meeting = conn.execute(
                "SELECT COUNT(*) AS c FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()["c"]
            transcript = conn.execute(
                "SELECT COUNT(*) AS c FROM transcripts WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()["c"]
            summary = conn.execute(
                "SELECT COUNT(*) AS c FROM summaries WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()["c"]
            tasks = conn.execute(
                "SELECT COUNT(*) AS c FROM tasks WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()["c"]
        finally:
            conn.close()
        return meeting, transcript, summary, tasks

    def test_dashboard_shows_delete_control(self):
        meeting_id = self._seed_meeting(self.owner_id, "Owner summary stays listed")
        self._login(self.owner_id, "Owner")
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn(f"/meeting/{meeting_id}/delete", html)
        self.assertIn("This cannot be undone.", html)
        self.assertIn("Owner summary stays listed", html)

    def test_owner_can_delete_meeting_and_associated_records(self):
        audio_name = f"u{self.owner_id}_notes.mp3"
        meeting_id = self._seed_meeting(
            self.owner_id, "Unique owner summary", audio_name=audio_name
        )
        kept_id = self._seed_meeting(
            self.owner_id, "Meeting that must remain", audio_name=f"u{self.owner_id}_keep.mp3"
        )
        self.assertEqual(self._counts(meeting_id), (1, 1, 1, 1))

        self._login(self.owner_id, "Owner")
        response = self.client.post(
            f"/meeting/{meeting_id}/delete", follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Meeting deleted.", html)
        self.assertNotIn("Unique owner summary", html)
        self.assertIn("Meeting that must remain", html)

        self.assertEqual(self._counts(meeting_id), (0, 0, 0, 0))
        self.assertEqual(self._counts(kept_id)[0], 1)
        self.assertFalse(os.path.exists(os.path.join(self.upload_dir, audio_name)))
        for prefix in ("graph", "bar", "donut", "status"):
            self.assertFalse(
                os.path.exists(os.path.join(self.graph_dir, f"{prefix}_m{meeting_id}.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(self.graph_dir, f"{prefix}_m{kept_id}.png"))
            )
        self.assertTrue(
            os.path.exists(os.path.join(self.upload_dir, f"u{self.owner_id}_keep.mp3"))
        )
        self.assertEqual(
            [row["id"] for row in get_meetings_by_user(self.owner_id, self.db_path)],
            [kept_id],
        )
        self.assertEqual(get_people_stats(self.owner_id, self.db_path), [("Alex", 1)])

        search = self.client.get("/search?q=Unique+owner+summary")
        search_html = search.get_data(as_text=True)
        self.assertIn("Found 0 results", search_html)
        self.assertIn("No results found", search_html)
        detail = self.client.get(f"/result/{meeting_id}")
        self.assertEqual(detail.status_code, 404)

    def test_other_user_cannot_delete_meeting(self):
        meeting_id = self._seed_meeting(self.owner_id, "Protected owner summary")
        self._login(self.other_id, "Other")
        response = self.client.post(
            f"/meeting/{meeting_id}/delete", follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access denied", response.get_data(as_text=True))
        self.assertEqual(self._counts(meeting_id), (1, 1, 1, 1))
        self.assertTrue(
            os.path.exists(os.path.join(self.graph_dir, f"graph_m{meeting_id}.png"))
        )

    def test_unauthenticated_delete_is_rejected(self):
        meeting_id = self._seed_meeting(self.owner_id, "Still stored")
        response = self.client.post(f"/meeting/{meeting_id}/delete")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])
        self.assertEqual(self._counts(meeting_id), (1, 1, 1, 1))

    def test_missing_meeting_does_not_delete_anything(self):
        meeting_id = self._seed_meeting(self.owner_id, "Existing meeting")
        self._login(self.owner_id, "Owner")
        response = self.client.post("/meeting/99999/delete", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("access denied", response.get_data(as_text=True))
        self.assertEqual(self._counts(meeting_id), (1, 1, 1, 1))

    def test_unsafe_paths_are_not_removed(self):
        outside_file = os.path.join(self.outside, "secret.txt")
        with open(outside_file, "w", encoding="utf-8") as fh:
            fh.write("keep")
        meeting_id = insert_meeting(
            user_id=self.owner_id,
            title="Unsafe paths",
            audio_filename="../outside/secret.txt",
            db_path=self.db_path,
        )
        insert_transcript(meeting_id, "unsafe transcript", self.db_path)
        insert_summary(meeting_id, "unsafe summary", self.db_path)
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """UPDATE meetings
                   SET graph_path = ?, bar_chart_path = ?
                   WHERE id = ?""",
                ("graphs/../../outside/secret.txt", "/etc/passwd", meeting_id),
            )
            conn.commit()
        finally:
            conn.close()

        self._login(self.owner_id, "Owner")
        response = self.client.post(
            f"/meeting/{meeting_id}/delete", follow_redirects=True
        )
        self.assertIn("Meeting deleted.", response.get_data(as_text=True))
        self.assertEqual(self._counts(meeting_id), (0, 0, 0, 0))
        self.assertTrue(os.path.exists(outside_file))
        self.assertFalse(any(os.scandir(self.upload_dir)))
        self.assertFalse(any(os.scandir(self.graph_dir)))


if __name__ == "__main__":
    unittest.main()

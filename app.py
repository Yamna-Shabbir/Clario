"""
app.py
------
Clario – AI Meeting Assistant
Flask application factory and entry point.
"""

import logging
import os
import sys

# ── Force UTF-8 output on Windows (prevents charmap codec errors in logs) ──────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from services.native_compat import install as _install_native_compat

_install_native_compat()

from datetime import timedelta
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, send_from_directory
from flask_wtf.csrf import CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix
from config import (
    SECRET_KEY, DEBUG, PORT, DB_PATH, DATA_DIR, UPLOAD_DIR, GRAPH_DIR,
    HF_MODELS_DIR, STATIC_GRAPH_DIR, MAX_CONTENT_LENGTH, MAX_UPLOAD_MB,
)


def create_app(*, test_config: Optional[Dict] = None) -> Flask:
    """
    Application factory — create, configure, and return the Flask app.

    Parameters
    ----------
    test_config : Optional dict of overrides injected by the test suite.
        Recognised keys (all optional):

        ``db_path``
            SQLite file path (replaces the module-level DB_PATH used by all
            blueprints).  The factory patches each blueprint module's local
            ``DB_PATH`` binding so routes see the override immediately.
        ``upload_dir``
            Directory for uploaded audio files (patches ``routes.meeting.UPLOAD_DIR``).
        ``graph_dir``
            Directory where chart PNGs are written (patches both
            ``routes.meeting.GRAPH_DIR`` and ``config.GRAPH_DIR``).
        ``secret_key``
            Flask session secret.  Defaults to a random hex string in tests.
        ``MAX_CONTENT_LENGTH``
            Maximum request payload size in bytes.
        ``WTF_CSRF_ENABLED``
            Enable/disable CSRF protection in tests.
        ``TESTING``
            Passed directly to ``app.config`` (disables Flask error catching).

    All keys default to production values so this parameter is safe to omit.
    """
    app = Flask(__name__)

    # Determine effective values before registering blueprints
    _db_path    = DB_PATH
    _secret_key = SECRET_KEY

    if test_config:
        import secrets as _secrets
        app.config["TESTING"] = test_config.get("TESTING", True)
        _secret_key = test_config.get("secret_key", _secrets.token_hex(16))
        if "db_path" in test_config:
            _db_path = test_config["db_path"]

    app.secret_key = _secret_key

    # ── Upload limits & security settings ──────────────────────────────────────
    app.config["MAX_CONTENT_LENGTH"] = (
        test_config.get("MAX_CONTENT_LENGTH", MAX_CONTENT_LENGTH)
        if test_config and "MAX_CONTENT_LENGTH" in test_config
        else MAX_CONTENT_LENGTH
    )

    # ── Secure session cookies ────────────────────────────────────────────────
    # SESSION_COOKIE_SECURE follows DEBUG so production (DEBUG=False) requires HTTPS.
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"]   = not DEBUG

    if test_config:
        for k in ("WTF_CSRF_ENABLED", "SESSION_COOKIE_SECURE", "SESSION_COOKIE_HTTPONLY", "SESSION_COOKIE_SAMESITE"):
            if k in test_config:
                app.config[k] = test_config[k]

    # Behind Render / other reverse proxies, trust X-Forwarded-* so HTTPS and
    # host headers are correct for cookies and OAuth redirects.
    if not DEBUG and not (test_config or {}).get("TESTING"):
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            )

    # ── CSRF protection ────────────────────────────────────────────────────────
    csrf = CSRFProtect()
    csrf.init_app(app)

    # Sessions expire after 7 days of inactivity
    app.permanent_session_lifetime = timedelta(days=7)

    # ── Runtime directories (persistent disk on Render when paths point there) ─
    if not (test_config or {}).get("TESTING"):
        for path in (
            DATA_DIR,
            os.path.dirname(os.path.abspath(_db_path)),
            UPLOAD_DIR,
            GRAPH_DIR,
            HF_MODELS_DIR,
            STATIC_GRAPH_DIR,
            os.path.join(app.root_path, "static", "graphs"),
        ):
            if path:
                os.makedirs(path, exist_ok=True)

    # ── Initialise database & recover interrupted jobs ─────────────────────────
    from models.database import init_db, fail_interrupted_meetings
    init_db(_db_path)
    fail_interrupted_meetings(_db_path)

    # ── Register blueprints ────────────────────────────────────────────────────
    from routes.auth         import auth_bp
    from routes.dashboard    import dashboard_bp
    from routes.meeting      import meeting_bp
    from routes.search       import search_bp
    from routes.integrations import integrations_bp
    from routes.live         import live_bp
    from routes.tasks        import tasks_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(meeting_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(integrations_bp)
    app.register_blueprint(live_bp)
    app.register_blueprint(tasks_bp)

    from services.jobs import register_post_completion_hook
    from services.live import link_task_assigners
    from services.notifications import send_meeting_followups
    register_post_completion_hook(link_task_assigners)
    register_post_completion_hook(send_meeting_followups)

    # ── Patch blueprint-local bindings AFTER blueprints are imported ───────────
    # Routes use `from config import DB_PATH` which binds the name in each
    # module's own namespace.  We must patch those names, not config.DB_PATH.
    if test_config:
        import routes.auth         as _r_auth
        import routes.dashboard    as _r_dash
        import routes.meeting      as _r_meet
        import routes.search       as _r_search
        import routes.integrations as _r_integ
        import routes.live         as _r_live
        import routes.tasks        as _r_tasks

        for _mod in (_r_auth, _r_dash, _r_meet, _r_search, _r_integ, _r_live, _r_tasks):
            if hasattr(_mod, "DB_PATH"):
                setattr(_mod, "DB_PATH", _db_path)

        if "upload_dir" in test_config:
            _r_meet.UPLOAD_DIR = test_config["upload_dir"]
            _r_live.UPLOAD_DIR = test_config["upload_dir"]

        if "graph_dir" in test_config:
            _r_meet.GRAPH_DIR = test_config["graph_dir"]

    # ── Health check (Render / load balancers; does not load ML models) ───────
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    # Charts may live on a persistent disk outside package static/.
    @app.get("/static/graphs/<path:filename>")
    def serve_meeting_graphs(filename: str):
        return send_from_directory(STATIC_GRAPH_DIR, filename)

    # ── Error handlers ─────────────────────────────────────────────────────────
    @app.errorhandler(413)
    def request_entity_too_large(e):
        return render_template(
            "error.html",
            code    = 413,
            title   = "File too large",
            message = f"The uploaded file exceeds the maximum allowed size of {MAX_UPLOAD_MB} MB.",
        ), 413

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template(
            "error.html",
            code    = 404,
            title   = "Page not found",
            message = "The page you're looking for doesn't exist or has been moved.",
        ), 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.exception("Unhandled server error")
        return render_template(
            "error.html",
            code    = 500,
            title   = "Internal server error",
            message = "Something went wrong. Please try again in a moment.",
        ), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=DEBUG, port=PORT, use_reloader=False)

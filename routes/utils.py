"""
routes/utils.py
---------------
Shared helpers used across all route blueprints.
"""

from functools import wraps
from flask import redirect, request, session, url_for


def login_required(f):
    """
    Decorator that redirects unauthenticated requests to /login.

    Usage::

        @dashboard_bp.route("/dashboard")
        @login_required
        def index():
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.method == "GET":
                return redirect(url_for("auth.login", next=request.path))
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


def current_user_id() -> int:
    """Return the logged-in user's id (assumes login_required already ran)."""
    return session["user_id"]


def current_user() -> dict:
    """Return the lightweight user dict stored in the session."""
    return {
        "id":       session.get("user_id"),
        "username": session.get("username"),
        "email":    session.get("email"),
    }

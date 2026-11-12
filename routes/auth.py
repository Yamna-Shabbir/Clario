"""
routes/auth.py
--------------
Authentication routes — register, login, logout, landing.
Password hashing via Werkzeug. Session stores user_id + username.
"""

from flask import (
    Blueprint, flash, redirect, render_template,
    request, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import DB_PATH
from models.database import (
    email_exists, get_user_by_email, insert_user,
    init_db, username_exists,
)

auth_bp = Blueprint("auth", __name__)


# ── Landing ────────────────────────────────────────────────────────────────────
@auth_bp.route("/")
def landing():
    if "user_id" in session:
        return redirect(url_for("dashboard.index"))
    return render_template("landing.html")


# ── Register ───────────────────────────────────────────────────────────────────
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm",  "").strip()

        # ── Validation ─────────────────────────────────────────────────────────
        errors = []
        if not username or not email or not password:
            errors.append("All fields are required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if email_exists(email, DB_PATH):
            errors.append("An account with that email already exists.")
        if username_exists(username, DB_PATH):
            errors.append("That username is already taken.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("auth/register.html",
                                   username=username, email=email)

        # ── Create user ────────────────────────────────────────────────────────
        pw_hash = generate_password_hash(password)
        try:
            user_id = insert_user(username, email, pw_hash, DB_PATH)
        except Exception as exc:
            flash(f"Could not create account: {exc}", "error")
            return render_template("auth/register.html")

        # Auto-login after registration
        session.permanent = True
        session["user_id"]  = user_id
        session["username"] = username
        session["email"]    = email

        flash(f"Welcome to Clario, {username}! 🎉", "success")
        return redirect(url_for("dashboard.index"))

    return render_template("auth/register.html")


# ── Login ──────────────────────────────────────────────────────────────────────
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        email    = request.form.get("email",    "").strip().lower()
        password = request.form.get("password", "").strip()

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("auth/login.html", email=email)

        user = get_user_by_email(email, DB_PATH)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("auth/login.html", email=email)

        session.permanent = True
        session["user_id"]  = user["id"]
        session["username"] = user["username"]
        session["email"]    = user["email"]

        flash(f"Welcome back, {user['username']}! 👋", "success")
        from urllib.parse import urlparse
        raw_next = request.args.get("next", "").strip()
        target_url = None
        if raw_next and raw_next.startswith("/") and not raw_next.startswith("//"):
            parsed = urlparse(raw_next)
            if not parsed.scheme and not parsed.netloc:
                target_url = raw_next

        return redirect(target_url or url_for("dashboard.index"))

    return render_template("auth/login.html")


# ── Logout ─────────────────────────────────────────────────────────────────────
@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.landing"))

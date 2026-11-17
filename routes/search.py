import re

from flask import Blueprint, render_template, request, session
from config import DB_PATH
from models.database import get_connection, init_db
from routes.utils import login_required, current_user

search_bp = Blueprint("search", __name__)

_WORDS = re.compile(r"[a-z0-9]+")


def _words(value: str) -> list:
    return _WORDS.findall((value or "").lower())


def query_matches_meeting(title: str, people: list, query: str) -> bool:
    """True when the query is the title, or an attendee's name or email.

    A short fragment inside the transcript, a summary, or part of a longer
    name does not count. ``no`` does not match ``Noona``.
    """
    needle = " ".join((query or "").split()).lower()
    if not needle:
        return False
    title_text = " ".join((title or "").split()).lower()
    if needle == title_text:
        return True
    query_words = _words(needle)
    title_words = _words(title_text)
    if len(needle) >= 3 and query_words and all(word in title_words for word in query_words):
        return True
    for person in people or []:
        name = " ".join((person.get("name") or "").split()).lower()
        email = (person.get("email") or "").strip().lower()
        if needle == name or (email and needle == email):
            return True
        if len(needle) < 3:
            continue
        if query_words and all(word in _words(name) for word in query_words):
            return True
        local = email.split("@", 1)[0]
        if local and needle == local:
            return True
    return False


def search_owned_meetings(user_id: int, query: str, db_path: str) -> list:
    """Meetings this account owns whose title or attendee matches the query."""
    needle = " ".join((query or "").split())
    if not needle:
        return []
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.title, m.meeting_date, m.sentiment, m.created_at,
                   s.content AS summary,
                   p.name AS attendee_name, p.email AS attendee_email
            FROM meetings m
            LEFT JOIN summaries s ON s.meeting_id = m.id
            LEFT JOIN meeting_participants p ON p.meeting_id = m.id
            WHERE m.user_id = ?
            ORDER BY m.created_at DESC, m.id DESC
            """,
            (user_id,),
        ).fetchall()
    grouped = {}
    for row in rows:
        item = grouped.get(row["id"])
        if item is None:
            item = {
                "id": row["id"],
                "title": row["title"],
                "meeting_date": row["meeting_date"],
                "sentiment": row["sentiment"],
                "created_at": row["created_at"],
                "summary": row["summary"],
                "people": [],
            }
            grouped[row["id"]] = item
        if row["attendee_name"] or row["attendee_email"]:
            item["people"].append({
                "name": row["attendee_name"] or "",
                "email": row["attendee_email"] or "",
            })
    matched = [
        item for item in grouped.values()
        if query_matches_meeting(item["title"], item["people"], needle)
    ]
    return matched[:50]


@search_bp.route("/search")
@login_required
def search():
    query = request.args.get("q", "").strip()
    user_id = session["user_id"]
    init_db(DB_PATH)
    results = search_owned_meetings(user_id, query, DB_PATH) if query else []
    return render_template(
        "dashboard/search.html",
        user=current_user(),
        query=query,
        results=results,
    )

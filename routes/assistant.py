"""In-app assistant. Read-only over meetings and documentation."""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, redirect, render_template, request, session, url_for

from config import DB_PATH
from models.database import init_db
from routes.utils import current_user, login_required
from services.assistant.chat import answer_question
from services.assistant.store import (
    add_message,
    create_conversation,
    get_conversation,
    latest_context,
    list_conversations,
    list_messages,
)

assistant_bp = Blueprint("assistant", __name__)

SUGGESTED = (
    "How do I start a meeting?",
    "Where are tasks and recordings?",
    "What should I do if transcription fails?",
    "What tasks are still pending?",
    "What did we decide in my latest meeting?",
)

_MAX_QUESTION = 2000


def _links(evidence: list) -> list:
    linked = []
    for item in evidence or []:
        copy = dict(item)
        meeting_id = item.get("meeting_id")
        if meeting_id:
            copy["meeting_url"] = url_for("meeting.result", meeting_id=meeting_id)
        recording_id = item.get("recording_id")
        if meeting_id and recording_id is not None:
            copy["audio_url"] = url_for(
                "meeting.recording_file",
                meeting_id=meeting_id,
                recording_id=recording_id,
            )
        linked.append(copy)
    return linked


def _public_message(message: dict) -> dict:
    context = message.get("context") or {}
    return {
        "id": message.get("id"),
        "role": message.get("role"),
        "content": message.get("content") or "",
        "route": message.get("route") or context.get("route") or "",
        "sources": context.get("sources") or [],
        "evidence": _links(context.get("evidence") or []),
        "created_at": message.get("created_at"),
    }


@assistant_bp.route("/assistant")
@login_required
def index():
    """Chat page with history and suggested questions."""
    init_db(DB_PATH)
    user_id = session["user_id"]
    conversations = list_conversations(user_id, DB_PATH)
    active_id = request.args.get("c", type=int)
    messages = []
    if active_id:
        if get_conversation(active_id, user_id, DB_PATH) is None:
            abort(404)
        messages = [_public_message(message) for message in list_messages(active_id, user_id, DB_PATH)]
    return render_template(
        "assistant/index.html",
        user=current_user(),
        conversations=conversations,
        messages=messages,
        active_id=active_id,
        suggestions=SUGGESTED,
    )


@assistant_bp.route("/assistant/conversations", methods=["POST"])
@login_required
def create():
    """Start an empty conversation."""
    init_db(DB_PATH)
    conversation_id = create_conversation(session["user_id"], DB_PATH)
    if request.is_json or request.accept_mimetypes.best == "application/json":
        return jsonify({"id": conversation_id, "title": "New conversation"})
    return redirect(url_for("assistant.index", c=conversation_id))


@assistant_bp.route("/assistant/conversations/<int:conversation_id>", methods=["GET"])
@login_required
def conversation(conversation_id: int):
    """Messages for a conversation owned by the signed-in user."""
    init_db(DB_PATH)
    user_id = session["user_id"]
    row = get_conversation(conversation_id, user_id, DB_PATH)
    if row is None:
        abort(404)
    return jsonify({
        "id": row["id"],
        "title": row["title"],
        "messages": [_public_message(message) for message in list_messages(conversation_id, user_id, DB_PATH)],
    })


@assistant_bp.route("/assistant/conversations/<int:conversation_id>/messages", methods=["POST"])
@login_required
def messages(conversation_id: int):
    """Append a question and a read-only answer."""
    init_db(DB_PATH)
    user_id = session["user_id"]
    if get_conversation(conversation_id, user_id, DB_PATH) is None:
        abort(404)
    payload = request.get_json(silent=True) or {}
    question = payload.get("message") if isinstance(payload, dict) else None
    if question is None:
        question = request.form.get("message", "")
    question = " ".join(str(question or "").split())
    if not question:
        return jsonify({"error": "Enter a question."}), 400
    if len(question) > _MAX_QUESTION:
        return jsonify({"error": "That question is too long."}), 400
    try:
        reply = answer_question(
            question,
            user_id=user_id,
            email=session.get("email") or "",
            db_path=DB_PATH,
            previous=latest_context(conversation_id, user_id, DB_PATH) or None,
        )
    except Exception:
        return jsonify({"error": "The assistant could not read that information."}), 500
    context = dict(reply.get("context") or {})
    context["evidence"] = reply.get("evidence") or []
    context["sources"] = reply.get("sources") or []
    add_message(
        conversation_id,
        user_id,
        DB_PATH,
        role="user",
        content=question,
        route=reply.get("route") or "",
    )
    message_id = add_message(
        conversation_id,
        user_id,
        DB_PATH,
        role="assistant",
        content=reply.get("content") or "",
        route=reply.get("route") or "",
        context=context,
    )
    body = _public_message({
        "id": message_id,
        "role": "assistant",
        "content": reply.get("content") or "",
        "route": reply.get("route") or "",
        "context": context,
    })
    return jsonify({"conversation_id": conversation_id, "message": body, "question": question})

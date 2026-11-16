"""Conversation history for the assistant. Scoped to the signed-in user."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from models.database import get_connection


def create_conversation(user_id: int, db_path: str, *, title: str = "New conversation") -> int:
    """Insert an empty conversation and return its id."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO assistant_conversations (user_id, title) VALUES (?, ?)",
            (user_id, (title or "New conversation")[:80]),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_conversations(user_id: int, db_path: str) -> List[Dict[str, Any]]:
    """Newest conversations for this user."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, title, created_at, updated_at
               FROM assistant_conversations
               WHERE user_id = ?
               ORDER BY updated_at DESC, id DESC
               LIMIT 40""",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_conversation(conversation_id: int, user_id: int, db_path: str) -> Optional[Dict[str, Any]]:
    """Return one conversation owned by this user."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT id, user_id, title, created_at, updated_at
               FROM assistant_conversations
               WHERE id = ? AND user_id = ?""",
            (conversation_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def list_messages(conversation_id: int, user_id: int, db_path: str) -> List[Dict[str, Any]]:
    """Messages for a conversation this user owns. Empty when they do not."""
    if get_conversation(conversation_id, user_id, db_path) is None:
        return []
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, role, content, route, context_json, created_at
               FROM assistant_messages
               WHERE conversation_id = ?
               ORDER BY id ASC""",
            (conversation_id,),
        ).fetchall()
    messages = []
    for row in rows:
        item = dict(row)
        try:
            item["context"] = json.loads(item.pop("context_json") or "{}")
        except ValueError:
            item["context"] = {}
        messages.append(item)
    return messages


def add_message(
    conversation_id: int,
    user_id: int,
    db_path: str,
    *,
    role: str,
    content: str,
    route: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Append a message when the conversation belongs to this user."""
    if get_conversation(conversation_id, user_id, db_path) is None:
        return None
    payload = json.dumps(context or {}, ensure_ascii=False)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO assistant_messages
               (conversation_id, role, content, route, context_json)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, role, content, route, payload),
        )
        if role == "user":
            title = " ".join(content.split())[:80] or "New conversation"
            conn.execute(
                """UPDATE assistant_conversations
                   SET title = CASE WHEN title = 'New conversation' THEN ? ELSE title END,
                       updated_at = datetime('now')
                   WHERE id = ? AND user_id = ?""",
                (title, conversation_id, user_id),
            )
        else:
            conn.execute(
                """UPDATE assistant_conversations
                   SET updated_at = datetime('now')
                   WHERE id = ? AND user_id = ?""",
                (conversation_id, user_id),
            )
        conn.commit()
        return int(cur.lastrowid)


def latest_context(conversation_id: int, user_id: int, db_path: str) -> Dict[str, Any]:
    """Context saved with the newest assistant reply, or an empty dict."""
    messages = list_messages(conversation_id, user_id, db_path)
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("context"):
            return message["context"]
    return {}

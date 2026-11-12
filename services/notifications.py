"""
services/notifications.py
-------------------------
Post-meeting follow-up mail.

After the pipeline finishes, each person with an email address receives a
message. Assignees see their own tasks in the body. Everyone receives a PDF
of the summary, decisions, and the full task list.

Delivery prefers an HTTPS provider (Brevo or Resend). Those work on Render's
free plan, which blocks SMTP ports 25, 465, and 587. SMTP is used when no
HTTPS key is set and CLARIO_SMTP_HOST plus CLARIO_SMTP_FROM are set.
Otherwise a copy is written under CLARIO_EMAIL_DIR, or data/outbox in debug.
"""

from __future__ import annotations

import base64
import json
import os
import re
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

from config import (
    BREVO_API_KEY,
    DATA_DIR,
    DB_PATH,
    DEBUG,
    EMAIL_DIR,
    RESEND_API_KEY,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_TLS,
    SMTP_USER,
)

_HTTPS_SETUP = (
    "This server cannot send mail yet. Render's free plan blocks Gmail SMTP "
    "(ports 587 and 465). Add CLARIO_BREVO_API_KEY in the Render environment "
    "(a verified sender in Brevo), then invite again. Until then, copy the "
    "link from the meeting page."
)

_PAGE_LINES = 46
_LINE_WIDTH = 90


def send_meeting_followups(results: Dict[str, Any]) -> None:
    """Send or save follow-up mail for one finished meeting.

    Safe to register as a post-completion hook. Returns without sending when
    the result payload has no meeting id.
    """
    stats = results.get("stats") or {}
    meeting_id = stats.get("meeting_id")
    user_id = stats.get("user_id")
    if not meeting_id or not user_id:
        return

    db_path = stats.get("db_path") or DB_PATH
    from models.database import (
        get_decisions_by_meeting,
        get_meeting_detail,
        get_meeting_participants,
        get_tasks_by_meeting,
        get_user_by_id,
        replace_meeting_emails,
    )
    from services.nlp.participant_matching import match, rematch_meeting_tasks

    meeting = get_meeting_detail(int(meeting_id), int(user_id), db_path)
    owner = get_user_by_id(int(user_id), db_path)
    if not meeting or not owner:
        return

    participants = get_meeting_participants(int(meeting_id), db_path)
    if participants:
        rematch_meeting_tasks(int(meeting_id), int(user_id), db_path)
    tasks = get_tasks_by_meeting(int(meeting_id), db_path)
    decisions = get_decisions_by_meeting(int(meeting_id), db_path)

    from services.nlp.participant_matching import pending_name_collisions
    from models.database import replace_pending_confirmations

    collisions = pending_name_collisions(tasks, participants)
    replace_pending_confirmations(int(meeting_id), collisions, db_path)
    if collisions:
        replace_meeting_emails(int(meeting_id), [{
            "recipient": owner.get("email") or "",
            "subject": f"Confirm assignees for {meeting.get('title') or 'Meeting'}",
            "status": "waiting",
            "detail": "Two or more people share a name. The person who assigned the task needs to choose an email before mail is sent.",
        }], db_path)
        return

    title = meeting.get("title") or "Meeting"
    summary = results.get("summary") or meeting.get("summary") or ""
    pdf = build_summary_pdf(title, summary, decisions, tasks)
    recipients = _recipients(owner, participants, tasks, match)

    log: List[Dict[str, str]] = []
    if not recipients:
        log.append({
            "recipient": "",
            "subject": "",
            "status": "skipped",
            "detail": "No email addresses were available.",
        })
    for person in recipients:
        subject, body = _message_copy(title, person)
        status, detail = _deliver(person["email"], subject, body, pdf, int(meeting_id))
        log.append({
            "recipient": person["email"],
            "subject": subject,
            "status": status,
            "detail": detail,
        })
    replace_meeting_emails(int(meeting_id), log, db_path)


def build_summary_pdf(
    title: str,
    summary: str,
    decisions: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> bytes:
    """Build a one-or-more page PDF brief of the meeting."""
    lines = [title or "Meeting", ""]
    lines.extend(_wrap("Summary"))
    lines.extend(_wrap(summary or "No summary was generated."))
    lines.append("")
    lines.append("Decisions")
    if decisions:
        for decision in decisions:
            text = decision.get("content") or decision.get("text") or ""
            lines.extend(_wrap(f"- {text}"))
    else:
        lines.append("- None recorded")
    lines.append("")
    lines.append("Tasks")
    if tasks:
        for task in tasks:
            who = task.get("assigned_to") or "Team"
            due = task.get("due_date") or "Not specified"
            lines.extend(_wrap(f"- {who}: {task.get('description', '')} (due {due})"))
    else:
        lines.append("- None recorded")
    return _lines_to_pdf(lines)


def _recipients(owner: Dict[str, Any], participants: List[Dict[str, Any]],
                tasks: List[Dict[str, Any]], match_fn) -> List[Dict[str, Any]]:
    """One entry per email. Owner and attendees get the brief; assignees get tasks."""
    people: Dict[str, Dict[str, Any]] = {}

    def add(email: str, name: str) -> Optional[Dict[str, Any]]:
        cleaned = (email or "").strip().lower()
        if not cleaned or "@" not in cleaned:
            return None
        person = people.setdefault(cleaned, {"email": cleaned, "name": name or cleaned, "tasks": []})
        if name and person["name"] == cleaned:
            person["name"] = name
        return person

    add(owner.get("email", ""), owner.get("username") or "there")
    for participant in participants:
        add(participant.get("email") or "", participant.get("name") or "")

    for task in tasks:
        email = task.get("assigned_email") or ""
        if not email:
            matched = match_fn(task.get("assigned_to") or "", participants)
            if matched:
                email = matched.get("email") or ""
        person = add(email, task.get("assigned_to") or "")
        if person is not None:
            person["tasks"].append(task)
    return list(people.values())


def _message_copy(title: str, person: Dict[str, Any]) -> Tuple[str, str]:
    name = person.get("name") or "there"
    own = person.get("tasks") or []
    if own:
        subject = f"Your tasks from {title}"
        lines = [f"Hello {name},", "", "These tasks were assigned to you:", ""]
        for task in own:
            due = task.get("due_date") or "Not specified"
            lines.append(f"- {task.get('description', '')} (due {due})")
        lines.extend(["", "The attached PDF includes the meeting summary and the other tasks."])
    else:
        subject = f"Summary of {title}"
        lines = [
            f"Hello {name},",
            "",
            "The meeting summary, decisions, and full task list are in the attached PDF.",
        ]
    return subject, "\n".join(lines)


def mail_is_configured() -> bool:
    """True when an HTTPS API key or SMTP host is available."""
    return bool(BREVO_API_KEY or RESEND_API_KEY or (SMTP_HOST and SMTP_FROM))


def _deliver(recipient: str, subject: str, body: str, pdf: bytes, meeting_id: int) -> Tuple[str, str]:
    saved = _save_copy(recipient, subject, body, pdf, meeting_id)
    if not mail_is_configured():
        if saved:
            return "saved", f"{_HTTPS_SETUP} A copy was saved at {saved}."
        return "skipped", _HTTPS_SETUP
    try:
        message = _build_message(recipient, subject, body, pdf)
        _dispatch(message)
    except Exception as exc:
        detail = _explain_send_error(exc)
        if saved:
            detail = f"{detail} A copy was saved at {saved}."
        return "failed", detail
    if saved:
        return "sent", f"Sent. A copy was saved at {saved}."
    return "sent", "Sent."


def _save_copy(recipient: str, subject: str, body: str, pdf: bytes, meeting_id: int) -> str:
    folder = EMAIL_DIR or (os.path.join(DATA_DIR, "outbox") if DEBUG else "")
    if not folder:
        return ""
    meeting_dir = os.path.join(folder, f"meeting_{meeting_id}")
    os.makedirs(meeting_dir, exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "-", recipient.lower()).strip("-") or "recipient"
    pdf_path = os.path.join(meeting_dir, f"{stem}.pdf")
    eml_path = os.path.join(meeting_dir, f"{stem}.eml")
    message = _build_message(recipient, subject, body, pdf)
    with open(pdf_path, "wb") as handle:
        handle.write(pdf)
    with open(eml_path, "wb") as handle:
        handle.write(message.as_bytes())
    return pdf_path


def send_invite_emails(recipients: List[str], join_url: str, title: str,
                       host_name: str) -> List[Dict[str, str]]:
    """Email a copyable join link. Saves a local copy when SMTP is not set."""
    subject = f"Join {title} on Clario"
    body = (
        f"{host_name} invited you to a Clario meeting: {title}\n\n"
        f"Open this link and sign in with your email:\n{join_url}\n\n"
        "After you join, record your microphone. Clario labels that audio "
        "with your account name and email."
    )
    sent: List[Dict[str, str]] = []
    seen = set()
    for raw in recipients:
        email = (raw or "").strip().lower()
        if not email or "@" not in email or email in seen:
            continue
        seen.add(email)
        status, detail = _deliver_text(email, subject, body, "invite")
        sent.append({"recipient": email, "status": status, "detail": detail})
    return sent


def _sender_address() -> str:
    return SMTP_FROM or SMTP_USER or "clario@localhost"


def _dispatch(message: EmailMessage) -> None:
    """Send through HTTPS when a key is set, otherwise SMTP."""
    if BREVO_API_KEY or RESEND_API_KEY:
        _send_https(message)
        return
    _smtp_send(message)


def _send_https(message: EmailMessage) -> None:
    recipient = str(message["To"] or "")
    subject = str(message["Subject"] or "")
    plain = message.get_body(preferencelist=("plain",))
    text = plain.get_content() if plain is not None else ""
    attachment = None
    filename = "meeting-summary.pdf"
    for part in message.iter_attachments():
        attachment = part.get_payload(decode=True)
        filename = part.get_filename() or filename
        break
    if BREVO_API_KEY:
        _post_brevo(recipient, subject, text, attachment, filename)
        return
    _post_resend(recipient, subject, text, attachment, filename)


def _post_json(url: str, payload: dict, headers: Dict[str, str]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(raw or f"HTTP {exc.code}") from exc


def _post_brevo(recipient: str, subject: str, text: str,
                attachment: Optional[bytes], filename: str) -> None:
    payload: Dict[str, Any] = {
        "sender": {"name": "Clario", "email": _sender_address()},
        "to": [{"email": recipient}],
        "subject": subject,
        "textContent": text,
    }
    if attachment:
        payload["attachment"] = [{
            "name": filename,
            "content": base64.b64encode(attachment).decode("ascii"),
        }]
    _post_json(
        "https://api.brevo.com/v3/smtp/email",
        payload,
        {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": BREVO_API_KEY,
        },
    )


def _post_resend(recipient: str, subject: str, text: str,
                 attachment: Optional[bytes], filename: str) -> None:
    payload: Dict[str, Any] = {
        "from": f"Clario <{_sender_address()}>",
        "to": [recipient],
        "subject": subject,
        "text": text,
    }
    if attachment:
        payload["attachments"] = [{
            "filename": filename,
            "content": base64.b64encode(attachment).decode("ascii"),
        }]
    _post_json(
        "https://api.resend.com/emails",
        payload,
        {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {RESEND_API_KEY}",
        },
    )


def _explain_send_error(exc: Exception) -> str:
    detail = str(exc) or type(exc).__name__
    if "Authentication" in detail or "Username and Password" in detail:
        return (
            "Gmail rejected the SMTP login. Use a Google App Password "
            "(not the normal account password) in CLARIO_SMTP_PASSWORD."
        )
    lowered = detail.lower()
    if any(token in lowered for token in ("timed out", "unreachable", "errno", "connection refused", "network")):
        return (
            "SMTP could not be reached. Render's free plan blocks ports 587 and 465. "
            "Set CLARIO_BREVO_API_KEY to send over HTTPS."
        )
    return detail


def _smtp_send(message: EmailMessage) -> None:
    """Send one message, trying STARTTLS then SSL when the first path fails."""
    last_error: Optional[Exception] = None
    attempts = []
    if SMTP_PORT == 465:
        attempts.append("ssl")
    else:
        attempts.extend(["starttls", "ssl"])
    for mode in attempts:
        try:
            if mode == "ssl":
                with smtplib.SMTP_SSL(SMTP_HOST, 465 if SMTP_PORT == 587 else SMTP_PORT, timeout=12) as client:
                    if SMTP_USER:
                        client.login(SMTP_USER, SMTP_PASSWORD)
                    client.send_message(message)
            else:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as client:
                    if SMTP_TLS:
                        client.starttls()
                    if SMTP_USER:
                        client.login(SMTP_USER, SMTP_PASSWORD)
                    client.send_message(message)
            return
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("SMTP send failed.")


def _deliver_text(recipient: str, subject: str, body: str, stem: str) -> Tuple[str, str]:
    saved = _save_text_copy(recipient, subject, body, stem)
    if not mail_is_configured():
        if saved:
            return "saved", f"{_HTTPS_SETUP} A copy was saved at {saved}."
        return "skipped", _HTTPS_SETUP
    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = _sender_address()
        message["To"] = recipient
        message.set_content(body)
        _dispatch(message)
    except Exception as exc:
        detail = _explain_send_error(exc)
        if saved:
            detail = f"{detail} A copy was saved at {saved}."
        return "failed", detail
    if saved:
        return "sent", f"Sent. A copy was saved at {saved}."
    return "sent", "Sent."


def _save_text_copy(recipient: str, subject: str, body: str, stem: str) -> str:
    folder = EMAIL_DIR or (os.path.join(DATA_DIR, "outbox") if DEBUG else "")
    if not folder:
        return ""
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^a-z0-9]+", "-", recipient.lower()).strip("-") or "recipient"
    path = os.path.join(folder, f"{stem}-{safe}.eml")
    message = EmailMessage()
    message["Subject"] = subject
    message["To"] = recipient
    message["From"] = SMTP_FROM or "clario@localhost"
    message.set_content(body)
    with open(path, "wb") as handle:
        handle.write(message.as_bytes())
    return path


def _send_smtp(recipient: str, subject: str, body: str, pdf: bytes) -> None:
    """Backward-compatible SMTP helper. Prefer :func:`_dispatch`."""
    _smtp_send(_build_message(recipient, subject, body, pdf))


def _build_message(recipient: str, subject: str, body: str, pdf: bytes) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_FROM or "clario@localhost"
    message["To"] = recipient
    message.set_content(body)
    message.add_attachment(
        pdf,
        maintype="application",
        subtype="pdf",
        filename="meeting-summary.pdf",
    )
    return message


def _wrap(text: str) -> List[str]:
    words = (text or "").replace("\r", "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) > _LINE_WIDTH:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}"
    lines.append(current)
    return lines


def _lines_to_pdf(lines: List[str]) -> bytes:
    pages = [lines[index:index + _PAGE_LINES] for index in range(0, len(lines), _PAGE_LINES)] or [[]]
    objects: List[bytes] = []

    def add(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    page_ids: List[int] = []
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_lines in pages:
        content = _page_stream(page_lines)
        content_id = add(
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
        )
        page_ids.append(content_id)

    page_object_ids: List[int] = []
    for content_id in page_ids:
        page_object_ids.append(add(
            b"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] "
            + f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode("ascii")
        ))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    pages_id = add(f"<< /Type /Pages /Count {len(page_object_ids)} /Kids [{kids}] >>".encode("ascii"))
    catalog_id = add(b"<< /Type /Catalog /Pages " + f"{pages_id} 0 R >>".encode("ascii"))

    # Parent references were written as "0 0 R" before the pages object existed.
    fixed: List[bytes] = []
    for obj in objects:
        fixed.append(obj.replace(b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii")))

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(fixed, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(fixed) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(fixed) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii")
    )
    return bytes(output)


def _page_stream(lines: List[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        safe = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .encode("latin-1", "replace")
            .decode("latin-1")
        )
        commands.append(f"({safe}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", "replace")

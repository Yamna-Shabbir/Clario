"""
services/notifications.py
-------------------------
Post-meeting follow-up mail.

After the pipeline finishes, each person with an email address receives a
message. Assignees see their own tasks in the body. Everyone receives a PDF
of the summary, decisions, and the full task list.

The From address is always the meeting host's own account. Clario does not
send every user's mail from one shared address.

Delivery prefers that host's Gmail, using the Google account they connected
(gmail.send over HTTPS, so Render free can deliver). Brevo or Resend is used
only when that same host address is already authorized on the provider.
SMTP is used when no HTTPS path is available. Gmail's SMTP server is used
only when its login is that same host. Otherwise a copy is written under
CLARIO_EMAIL_DIR, or data/outbox in debug.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
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
    "Clario sends from the address you logged in with. Connect that Google "
    "account under Integrations and allow Gmail send. A different mailbox is "
    "not used as From. Render's free plan blocks Gmail SMTP (ports 587 and 465). "
    "Optional: CLARIO_BREVO_API_KEY delivers only after this same address is an "
    "active Brevo sender. Until then, copy the link from the meeting page."
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

    from models.database import get_meeting_emails
    if any(row.get("status") == "sent" for row in get_meeting_emails(int(meeting_id), db_path)):
        return

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
            "detail": "A name matches more than one address, or none of them. Choose who should receive each task. Mail is sent after every choice is saved.",
        }], db_path)
        return

    if results.get("require_approval") and tasks and any(not int(task.get("approved") or 0) for task in tasks):
        replace_meeting_emails(int(meeting_id), [{
            "recipient": owner.get("email") or "",
            "subject": f"Confirm tasks for {meeting.get('title') or 'Meeting'}",
            "status": "waiting",
            "detail": "Confirm every task on the meeting page before follow-up mail is sent. You can edit a task or the summary first.",
        }], db_path)
        return

    title = meeting.get("title") or "Meeting"
    summary = results.get("summary") or meeting.get("summary") or ""
    pdf = build_summary_pdf(title, summary, decisions, tasks)
    recipients = _recipients(
        owner, participants, tasks, match, db_path=db_path,
    )

    log: List[Dict[str, str]] = []
    if not recipients:
        log.append({
            "recipient": "",
            "subject": "",
            "status": "skipped",
            "detail": "No email addresses were available.",
        })
    host_email = _clean_email(owner.get("email") or "")
    host_name = (owner.get("username") or "the host").strip() or "the host"
    reply_to = _reply_header(host_name, host_email)
    for person in recipients:
        subject, body, rich = _message_copy(
            title, person, meeting=meeting, meeting_id=int(meeting_id),
            all_tasks=tasks, decisions=decisions, summary=summary,
        )
        if host_email:
            body = (
                f"{body}\n\nThis meeting was hosted by {host_name} ({host_email}). "
                "Reply to this message to reach them."
            )
        status, detail = _deliver(
            person["email"], subject, body, pdf, int(meeting_id),
            reply_to=reply_to, sender_name=host_name, sender_email=host_email,
            host_user_id=int(user_id), db_path=db_path, html=rich,
        )
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
    lines.append("Key points")
    lines.extend(_key_point_lines(decisions, tasks))
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
    lines.append("Assignee | Deadline | Priority | Task")
    if tasks:
        for task in tasks:
            who = task.get("assigned_to") or "Team"
            due = task.get("due_date") or "Not specified"
            level = task.get("priority") or "Low"
            lines.extend(_wrap(
                f"{who} | {due} | {level} | {task.get('description', '')}"
            ))
    else:
        lines.append("- None recorded")
    return _lines_to_pdf(lines)


def _key_point_lines(decisions: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> List[str]:
    """Decisions and tasks, listed together as the meeting's key points."""
    points: List[str] = []
    for decision in decisions:
        text = decision.get("content") or decision.get("text") or ""
        if text:
            points.extend(_wrap(f"- Decision: {text}"))
    for task in tasks:
        who = task.get("assigned_to") or "Team"
        due = task.get("due_date") or "Not specified"
        points.extend(_wrap(
            f"- Task: {who} — {task.get('description', '')} (due {due})"
        ))
    return points or ["- None recorded"]


def _recipients(owner: Dict[str, Any], participants: List[Dict[str, Any]],
                tasks: List[Dict[str, Any]], match_fn, *,
                db_path: str = "") -> List[Dict[str, Any]]:
    """One entry per known email. A name or a spoken sentence is never an address."""
    people: Dict[str, Dict[str, Any]] = {}

    def add(email: str, name: str) -> Optional[Dict[str, Any]]:
        cleaned = _clean_email(email)
        if not cleaned:
            return None
        person = people.setdefault(cleaned.lower(), {
            "email": cleaned.lower(), "name": name or cleaned.lower(), "tasks": [],
        })
        if name and person["name"] == cleaned.lower():
            person["name"] = name
        return person

    add(owner.get("email", ""), owner.get("username") or "there")
    for participant in participants:
        add(participant.get("email") or "", participant.get("name") or "")

    for task in tasks:
        who = (task.get("assigned_to") or "").strip()
        if who.casefold() in {"", "team", "unassigned", "unknown"}:
            continue
        email = _task_recipient(task, participants, owner, match_fn, db_path)
        person = add(email, who)
        if person is not None:
            person["tasks"].append(task)
    return list(people.values())


def _task_recipient(task: Dict[str, Any], participants: List[Dict[str, Any]],
                    owner: Dict[str, Any], match_fn, db_path: str) -> str:
    """Email of the assignee, only when a participant or user record has it."""
    candidate = _clean_email(task.get("assigned_email") or "")
    if not candidate:
        candidate = _clean_email(task.get("assigned_to") or "")
    if not candidate:
        matched = match_fn(task.get("assigned_to") or "", participants)
        candidate = _clean_email((matched or {}).get("email") or "")
    if not candidate or not _known_address(candidate, participants, owner, db_path):
        return ""
    return candidate.lower()


def _known_address(email: str, participants: List[Dict[str, Any]],
                   owner: Dict[str, Any], db_path: str) -> bool:
    wanted = _clean_email(email).lower()
    if not wanted:
        return False
    known = {_clean_email(owner.get("email") or "").lower()}
    known.update(_clean_email(person.get("email") or "").lower() for person in participants)
    known.discard("")
    if wanted in known:
        return True
    if not db_path:
        return False
    from models.database import get_user_by_email
    return get_user_by_email(wanted, db_path) is not None


def _meeting_link(meeting_id: int) -> str:
    from config import OAUTH_REDIRECT_BASE
    base = (OAUTH_REDIRECT_BASE or "http://127.0.0.1:5000").rstrip("/")
    return f"{base}/result/{meeting_id}"


def _message_copy(title: str, person: Dict[str, Any], *,
                  meeting: Optional[Dict[str, Any]] = None,
                  meeting_id: int = 0,
                  all_tasks: Optional[List[Dict[str, Any]]] = None,
                  decisions: Optional[List[Dict[str, Any]]] = None,
                  summary: str = "") -> Tuple[str, str, str]:
    name = person.get("name") or "there"
    own = person.get("tasks") or []
    meeting = meeting or {}
    when = meeting.get("meeting_date") or (meeting.get("created_at") or "")[:10]
    shown = own if own else (all_tasks or [])
    if own:
        subject = f"Your tasks from {title}"
        lines = [f"Hello {name},", "", f"Meeting: {title}", f"Date: {when or 'Not specified'}", ""]
        for task in own:
            due = task.get("due_date") or "Not specified"
            spoken = (task.get("evidence_time") or "").strip()
            if spoken and spoken not in due:
                due = f"{due} (mentioned at {spoken})"
            lines.extend([
                f"Task: {task.get('description', '')}",
                f"Assigned to: {name}",
                f"Assigned by: {task.get('assigned_by') or 'the host'}",
                f"Due: {due}",
                f"Priority: {task.get('priority') or 'Low'}",
                f"Status: {task.get('status') or 'pending'}",
                "",
            ])
        lines.extend(_plain_task_table(shown))
        if meeting_id:
            lines.append(f"Open the meeting in Clario: {_meeting_link(meeting_id)}")
        lines.append("The attached PDF includes the meeting summary and the other tasks.")
    else:
        subject = f"Summary of {title}"
        lines = [
            f"Hello {name},",
            "",
            "The meeting summary, decisions, and full task list are in the attached PDF.",
            "",
        ]
        lines.extend(_plain_task_table(shown))
    rich = _html_message(
        title, name, when or "Not specified", shown, decisions or [], summary,
        meeting_id=meeting_id, personal=bool(own),
    )
    return subject, "\n".join(lines), rich


def _plain_task_table(tasks: List[Dict[str, Any]]) -> List[str]:
    lines = ["Task | Assignee | Deadline | Priority"]
    if not tasks:
        lines.append("None recorded")
        return lines
    for task in tasks:
        lines.append(" | ".join([
            (task.get("description") or "").replace("|", "/"),
            task.get("assigned_to") or "Team",
            task.get("due_date") or "Not specified",
            task.get("priority") or "Low",
        ]))
    lines.append("")
    return lines


def _html_message(title: str, name: str, when: str, tasks: List[Dict[str, Any]],
                  decisions: List[Dict[str, Any]], summary: str, *,
                  meeting_id: int = 0, personal: bool = False) -> str:
    """Organised HTML brief: key points, then a task table with deadlines."""
    def cell(value: object) -> str:
        return html.escape(str(value or ""))

    point_items = []
    for decision in decisions:
        text = decision.get("content") or decision.get("text") or ""
        if text:
            point_items.append(f"<li><strong>Decision.</strong> {cell(text)}</li>")
    for task in tasks:
        point_items.append(
            "<li><strong>Task.</strong> "
            f"{cell(task.get('assigned_to') or 'Team')} — {cell(task.get('description'))} "
            f"(due {cell(task.get('due_date') or 'Not specified')})</li>"
        )
    points = "".join(point_items) or "<li>None recorded.</li>"
    rows = []
    for task in tasks:
        rows.append(
            "<tr>"
            f"<td style=\"padding:8px 10px;border-bottom:1px solid #E4D4D7;\">{cell(task.get('description'))}</td>"
            f"<td style=\"padding:8px 10px;border-bottom:1px solid #E4D4D7;\">{cell(task.get('assigned_to') or 'Team')}</td>"
            f"<td style=\"padding:8px 10px;border-bottom:1px solid #E4D4D7;\">{cell(task.get('due_date') or 'Not specified')}</td>"
            f"<td style=\"padding:8px 10px;border-bottom:1px solid #E4D4D7;\">{cell(task.get('priority') or 'Low')}</td>"
            "</tr>"
        )
    body_rows = "".join(rows) or (
        "<tr><td colspan=\"4\" style=\"padding:8px 10px;\">None recorded.</td></tr>"
    )
    intro = (
        "These are your tasks from the meeting."
        if personal else
        "The full task list for this meeting is below."
    )
    link = ""
    if meeting_id:
        url = html.escape(_meeting_link(meeting_id))
        link = f"<p><a href=\"{url}\">Open this meeting in Clario</a></p>"
    summary_block = ""
    if summary:
        summary_block = f"<p>{cell(summary)}</p>"
    return (
        "<div style=\"font-family:Arial,sans-serif;color:#2E1A1F;line-height:1.45;\">"
        f"<p>Hello {cell(name)},</p>"
        f"<p><strong>{cell(title)}</strong><br>Date: {cell(when)}</p>"
        f"<p>{intro}</p>"
        f"{summary_block}"
        "<h2 style=\"font-size:16px;color:#7A2E40;\">Key points</h2>"
        f"<ul>{points}</ul>"
        "<h2 style=\"font-size:16px;color:#7A2E40;\">Tasks</h2>"
        "<table style=\"border-collapse:collapse;width:100%;font-size:14px;\">"
        "<thead><tr>"
        "<th style=\"text-align:left;padding:8px 10px;border-bottom:2px solid #7A2E40;\">Task</th>"
        "<th style=\"text-align:left;padding:8px 10px;border-bottom:2px solid #7A2E40;\">Assignee</th>"
        "<th style=\"text-align:left;padding:8px 10px;border-bottom:2px solid #7A2E40;\">Deadline</th>"
        "<th style=\"text-align:left;padding:8px 10px;border-bottom:2px solid #7A2E40;\">Priority</th>"
        "</tr></thead>"
        f"<tbody>{body_rows}</tbody></table>"
        f"{link}"
        "<p>The attached PDF has the summary, key points, decisions, and every task.</p>"
        "</div>"
    )


def sends_as_self(user_id: Optional[int] = None, db_path: Optional[str] = None) -> bool:
    """True when this login can deliver from their own mailbox.

    A shared Brevo key does not count. The host needs their own Gmail
    connection, or a Gmail SMTP login that is the same address.
    """
    if not user_id:
        return False
    path = db_path or DB_PATH
    if _host_gmail_can_send(int(user_id), path):
        return True
    return _is_gmail_smtp() and _smtp_login_is(_account_email(int(user_id), path))


def mail_is_configured(user_id: Optional[int] = None, db_path: Optional[str] = None) -> bool:
    """True when this host can send as themselves, or a shared HTTPS key exists.

    Gmail SMTP counts only when its login is this user's address. A server
    mailbox that belongs to someone else is not treated as ready.
    """
    path = db_path or DB_PATH
    if user_id and _host_gmail_can_send(int(user_id), path):
        return True
    if BREVO_API_KEY or RESEND_API_KEY:
        return True
    if not (SMTP_HOST and (SMTP_FROM or SMTP_USER)):
        return False
    if user_id and _is_gmail_smtp():
        return _smtp_login_is(_account_email(int(user_id), path))
    return True


def _account_email(user_id: int, db_path: str) -> str:
    try:
        from models.database import get_user_by_id
        owner = get_user_by_id(int(user_id), db_path) or {}
    except Exception:
        return ""
    return _clean_email(owner.get("email") or "")


def _is_gmail_smtp() -> bool:
    host = (SMTP_HOST or "").lower()
    return "gmail.com" in host or "googlemail.com" in host


def _smtp_login_is(email: str) -> bool:
    login = _clean_email(SMTP_USER or SMTP_FROM or "")
    host = _clean_email(email)
    return bool(login) and bool(host) and login.lower() == host.lower()


def _host_gmail_can_send(user_id: int, db_path: str) -> bool:
    """True when this user connected Google and granted gmail.send."""
    try:
        from models.database import get_oauth_connection
        row = get_oauth_connection(int(user_id), "google", db_path=db_path)
    except Exception:
        return False
    if not row or not row.get("access_token_enc"):
        return False
    return "gmail.send" in (row.get("scope") or "").lower()


def _deliver(recipient: str, subject: str, body: str, pdf: bytes, meeting_id: int,
            *, reply_to: str = "", sender_name: str = "", sender_email: str = "",
            host_user_id: Optional[int] = None, db_path: Optional[str] = None,
            html: str = "") -> Tuple[str, str]:
    saved = _save_copy(
        recipient, subject, body, pdf, meeting_id,
        reply_to=reply_to, sender_name=sender_name, sender_email=sender_email,
        html=html,
    )
    if not mail_is_configured(host_user_id, db_path):
        if saved:
            return "saved", f"{_HTTPS_SETUP} A copy was saved at {saved}."
        return "skipped", _HTTPS_SETUP
    try:
        message = _build_message(
            recipient, subject, body, pdf, reply_to=reply_to,
            sender_name=sender_name, sender_email=sender_email, html=html,
        )
        _dispatch(message, host_user_id=host_user_id, db_path=db_path)
    except Exception as exc:
        detail = _explain_send_error(exc)
        if saved:
            detail = f"{detail} A copy was saved at {saved}."
        return "failed", detail
    if saved:
        return "sent", f"Sent. A copy was saved at {saved}."
    return "sent", "Sent."


def _save_copy(recipient: str, subject: str, body: str, pdf: bytes, meeting_id: int,
               *, reply_to: str = "", sender_name: str = "", sender_email: str = "",
               html: str = "") -> str:
    folder = EMAIL_DIR or (os.path.join(DATA_DIR, "outbox") if DEBUG else "")
    if not folder:
        return ""
    meeting_dir = os.path.join(folder, f"meeting_{meeting_id}")
    os.makedirs(meeting_dir, exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "-", recipient.lower()).strip("-") or "recipient"
    pdf_path = os.path.join(meeting_dir, f"{stem}.pdf")
    eml_path = os.path.join(meeting_dir, f"{stem}.eml")
    message = _build_message(
        recipient, subject, body, pdf, reply_to=reply_to,
        sender_name=sender_name, sender_email=sender_email, html=html,
    )
    with open(pdf_path, "wb") as handle:
        handle.write(pdf)
    with open(eml_path, "wb") as handle:
        handle.write(message.as_bytes())
    return pdf_path


def send_invite_emails(recipients: List[str], join_url: str, title: str,
                       host_name: str, *, host_email: str = "",
                       host_user_id: Optional[int] = None,
                       db_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Email a copyable join link. Saves a local copy when mail cannot be sent.

    The logged-in host is the From address and the Reply-To address.
    When ``host_user_id`` is set and that person connected Gmail, the message
    is sent through their mailbox. A shared From address is not substituted.
    """
    host = (host_name or "A teammate").strip() or "A teammate"
    address = _clean_email(host_email)
    reply_to = _reply_header(host, address)
    subject = f"{host} invited you to {title}"
    who = f"{host} ({address})" if address else host
    body = (
        f"{who} invited you to a Clario meeting: {title}.\n\n"
        + (
            f"Reply to this message to reach {host} at {address}.\n\n"
            if address else ""
        )
        + f"Open this link and sign in with your email:\n{join_url}\n\n"
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
        status, detail = _deliver_text(
            email, subject, body, "invite",
            reply_to=reply_to, sender_name=host, sender_email=address,
            host_user_id=host_user_id, db_path=db_path,
        )
        sent.append({"recipient": email, "status": status, "detail": detail})
    return sent


_SENDER_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _clean_email(value: str) -> str:
    found = _SENDER_RE.search((value or "").strip().strip('"').strip("'"))
    return found.group(0) if found else ""


def _reply_header(name: str, email: str) -> str:
    """Reply-To value. The address is the host; the name is display only."""
    address = _clean_email(email)
    if not address:
        return ""
    display = " ".join((name or "").replace("\r", " ").replace("\n", " ").split())
    return formataddr((display, address)) if display else address


def _reply_parts(header: str) -> Tuple[str, str]:
    display, address = parseaddr(header or "")
    address = _clean_email(address or header)
    display = " ".join((display or "").split())
    return display, address


def _sender_address() -> str:
    """Parse an optional env address. Meeting mail does not use this as From."""
    raw = (
        os.environ.get("CLARIO_SMTP_FROM", "")
        or os.environ.get("CLARIO_SMTP_USER", "")
        or SMTP_FROM
        or SMTP_USER
        or ""
    ).strip().strip('"').strip("'")
    return _clean_email(raw)


def _from_parts(message: EmailMessage) -> Tuple[str, str]:
    display, address = parseaddr(str(message.get("From") or ""))
    address = _clean_email(address or str(message.get("From") or ""))
    display = " ".join((display or "").split())
    return display, address


def _brevo_active_senders() -> set:
    """Email addresses Brevo currently allows in the From line.

    ``GET /v3/senders`` lists each sender. ``active`` is true only after
    that address has been verified. An unverified host is absent.
    """
    request = urllib.request.Request(
        "https://api.brevo.com/v3/senders",
        headers={"accept": "application/json", "api-key": BREVO_API_KEY},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(raw or f"HTTP {exc.code}") from exc
    allowed = set()
    for item in payload.get("senders") or []:
        if item.get("active") is not True:
            continue
        email = _clean_email(str(item.get("email") or ""))
        if email:
            allowed.add(email.lower())
    return allowed


def _require_brevo_sender(email: str) -> None:
    """Refuse the send when this host address is not an active Brevo sender."""
    try:
        allowed = _brevo_active_senders()
    except Exception as exc:
        raise RuntimeError(
            f"Clario could not check whether {email} is an authorized Brevo sender ({exc}). "
            "The message was not sent, and no other From address was used."
        ) from exc
    if email.lower() not in allowed:
        raise RuntimeError(
            f"{email} is not an active sender in this Brevo account. "
            "Brevo only accepts a From address that has been verified under "
            "Brevo → Senders. Add and verify this host address, then send again. "
            "Clario did not substitute another sender."
        )


def _dispatch(message: EmailMessage, *, host_user_id: Optional[int] = None,
              db_path: Optional[str] = None) -> None:
    """Send as the meeting host. Never replace From with a shared address.

    The host's connected Gmail is used first. Brevo or Resend runs only for
    that same address. Gmail SMTP runs only when its login is that address.
    """
    _name, email = _from_parts(message)
    if not email:
        raise RuntimeError(
            "This message has no meeting-host address to send from. "
            "Clario did not substitute a fixed sender."
        )
    path = db_path or DB_PATH
    if host_user_id and _host_gmail_can_send(int(host_user_id), path):
        _send_via_gmail(message, int(host_user_id), path)
        return
    if BREVO_API_KEY or RESEND_API_KEY:
        if BREVO_API_KEY:
            _require_brevo_sender(email)
        _send_https(message)
        return
    if _is_gmail_smtp() and not _smtp_login_is(email):
        raise RuntimeError(
            f"Mail has to leave from {email}. This server's Gmail login is a "
            "different mailbox, so it was not used. Connect Google under "
            "Integrations with the address you logged in with. "
            "Clario did not substitute another sender."
        )
    _smtp_send(message)


def _google_account_email(token: str) -> str:
    """Address of the Google account that owns this access token."""
    request = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"accept": "application/json", "authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(
            "Clario could not confirm which Google account is connected "
            f"({raw or exc.code}). Reconnect under Integrations. "
            "The message was not sent, and no other From address was used."
        ) from exc
    found = _clean_email(str(payload.get("email") or ""))
    if not found:
        raise RuntimeError(
            "The connected Google account did not share its email address. "
            "Reconnect under Integrations and allow the email permission. "
            "Clario did not substitute another sender."
        )
    return found


def _send_via_gmail(message: EmailMessage, user_id: int, db_path: str) -> None:
    """Deliver one RFC822 message through the host's Gmail account."""
    from services.integrations.base import NotConnected, TokenRevoked
    from services.integrations.google_meet import get_valid_access_token

    _name, sender = _from_parts(message)
    try:
        token = get_valid_access_token(int(user_id), db_path)
    except NotConnected as exc:
        raise RuntimeError(
            f"Mail has to leave from {sender}. Connect that Google account "
            "under Integrations. Clario did not substitute another sender."
        ) from exc
    except TokenRevoked as exc:
        raise RuntimeError(
            "Google disconnected this account. Reconnect under Integrations "
            "with the address you logged in with. Clario did not substitute "
            "another sender."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Clario could not open this host's Google mailbox ({exc}). "
            "The message was not sent, and no other From address was used."
        ) from exc

    google_email = _google_account_email(token)
    if google_email.lower() != sender.lower():
        raise RuntimeError(
            f"The connected Google account is {google_email}, but this meeting "
            f"is hosted by {sender}. Reconnect Google with the address you "
            "logged in with. Clario did not substitute another sender."
        )
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
    _post_json(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        {"raw": raw},
        {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
        },
    )


def _send_https(message: EmailMessage) -> None:
    recipient = str(message["To"] or "")
    subject = str(message["Subject"] or "")
    plain = message.get_body(preferencelist=("plain",))
    text = plain.get_content() if plain is not None else ""
    html_part = message.get_body(preferencelist=("html",))
    rich = html_part.get_content() if html_part is not None else ""
    attachment = None
    filename = "meeting-summary.pdf"
    for part in message.iter_attachments():
        attachment = part.get_payload(decode=True)
        filename = part.get_filename() or filename
        break
    sender_name, sender_email = _from_parts(message)
    reply_name, reply_email = _reply_parts(str(message.get("Reply-To") or ""))
    if not reply_email:
        reply_email, reply_name = sender_email, sender_name
    if BREVO_API_KEY:
        _post_brevo(
            recipient, subject, text, attachment, filename,
            sender_email=sender_email, sender_name=sender_name or sender_email,
            reply_to=reply_email, reply_name=reply_name or sender_name,
            html=rich,
        )
        return
    _post_resend(
        recipient, subject, text, attachment, filename,
        sender_email=sender_email, sender_name=sender_name or sender_email,
        reply_to=reply_email, reply_name=reply_name or sender_name,
        html=rich,
    )


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
                attachment: Optional[bytes], filename: str, *,
                sender_email: str, sender_name: str,
                reply_to: str = "", reply_name: str = "",
                html: str = "") -> None:
    payload: Dict[str, Any] = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": recipient}],
        "subject": subject,
        "textContent": text,
    }
    if html:
        payload["htmlContent"] = html
    if reply_to:
        payload["replyTo"] = {"email": reply_to, "name": reply_name or reply_to}
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
                 attachment: Optional[bytes], filename: str, *,
                 sender_email: str, sender_name: str,
                 reply_to: str = "", reply_name: str = "",
                 html: str = "") -> None:
    payload: Dict[str, Any] = {
        "from": formataddr((sender_name, sender_email)),
        "to": [recipient],
        "subject": subject,
        "text": text,
    }
    if html:
        payload["html"] = html
    if reply_to:
        payload["reply_to"] = formataddr((reply_name, reply_to)) if reply_name else reply_to
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
    if "not an active sender" in lowered or "could not check whether" in lowered:
        return detail
    if "valid sender email" in lowered or "sender email" in lowered:
        return (
            f"{detail} Brevo rejected this host address and Clario did not "
            "replace it with a fixed sender. Verify the host under "
            "Brevo → Senders, then send again."
        )
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


def _deliver_text(recipient: str, subject: str, body: str, stem: str,
                 *, reply_to: str = "", sender_name: str = "", sender_email: str = "",
                 host_user_id: Optional[int] = None, db_path: Optional[str] = None) -> Tuple[str, str]:
    saved = _save_text_copy(
        recipient, subject, body, stem, reply_to=reply_to,
        sender_name=sender_name, sender_email=sender_email,
    )
    if not mail_is_configured(host_user_id, db_path):
        if saved:
            return "saved", f"{_HTTPS_SETUP} A copy was saved at {saved}."
        return "skipped", _HTTPS_SETUP
    try:
        message = EmailMessage()
        message["Subject"] = subject
        message["To"] = recipient
        email = _clean_email(sender_email)
        name = " ".join((sender_name or "").replace("\r", " ").replace("\n", " ").split())
        if email:
            message["From"] = formataddr((name, email)) if name else email
            message["Reply-To"] = reply_to or message["From"]
        elif reply_to:
            message["Reply-To"] = reply_to
        message.set_content(body)
        _dispatch(message, host_user_id=host_user_id, db_path=db_path)
    except Exception as exc:
        detail = _explain_send_error(exc)
        if saved:
            detail = f"{detail} A copy was saved at {saved}."
        return "failed", detail
    if saved:
        return "sent", f"Sent. A copy was saved at {saved}."
    return "sent", "Sent."


def _save_text_copy(recipient: str, subject: str, body: str, stem: str,
                   *, reply_to: str = "", sender_name: str = "", sender_email: str = "") -> str:
    folder = EMAIL_DIR or (os.path.join(DATA_DIR, "outbox") if DEBUG else "")
    if not folder:
        return ""
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^a-z0-9]+", "-", recipient.lower()).strip("-") or "recipient"
    path = os.path.join(folder, f"{stem}-{safe}.eml")
    message = EmailMessage()
    message["Subject"] = subject
    message["To"] = recipient
    email = _clean_email(sender_email)
    name = " ".join((sender_name or "").replace("\r", " ").replace("\n", " ").split())
    if email:
        message["From"] = formataddr((name, email)) if name else email
        message["Reply-To"] = reply_to or message["From"]
    elif reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    with open(path, "wb") as handle:
        handle.write(message.as_bytes())
    return path


def _send_smtp(recipient: str, subject: str, body: str, pdf: bytes) -> None:
    """Backward-compatible SMTP helper. Prefer :func:`_dispatch`."""
    _smtp_send(_build_message(recipient, subject, body, pdf))


def _build_message(recipient: str, subject: str, body: str, pdf: bytes,
                   *, reply_to: str = "", sender_name: str = "",
                   sender_email: str = "", html: str = "") -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["To"] = recipient
    email = _clean_email(sender_email)
    name = " ".join((sender_name or "").replace("\r", " ").replace("\n", " ").split())
    if email:
        message["From"] = formataddr((name, email)) if name else email
        message["Reply-To"] = reply_to or message["From"]
    elif reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    if html:
        message.add_alternative(html, subtype="html")
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

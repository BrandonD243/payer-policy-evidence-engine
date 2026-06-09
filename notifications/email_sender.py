from __future__ import annotations

from datetime import datetime, UTC
from email.message import EmailMessage
from pathlib import Path
import os
import re
import smtplib
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
OUTBOX_DIR = BASE_DIR / "demo_outbox"


def send_email_with_attachment(
    *,
    to_email: str,
    subject: str,
    body: str,
    attachment_bytes: bytes,
    attachment_filename: str,
    attachment_mime_type: str = "application/pdf",
) -> dict:
    return send_email_with_attachments(
        to_email=to_email,
        subject=subject,
        body=body,
        attachments=[
            {
                "filename": attachment_filename,
                "content": attachment_bytes,
                "mime_type": attachment_mime_type,
            }
        ],
    )


def send_email_with_attachments(
    *,
    to_email: str,
    subject: str,
    body: str,
    attachments: Optional[List[Dict[str, object]]] = None,
) -> dict:
    from_email = os.getenv("SMTP_FROM_EMAIL") or os.getenv("SMTP_USERNAME") or "demo@example.test"
    message = EmailMessage()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    attachment_names = []
    for attachment in attachments or []:
        filename = str(attachment.get("filename") or "attachment")
        mime_type = str(attachment.get("mime_type") or "application/octet-stream")
        content = attachment.get("content") or b""
        if isinstance(content, str):
            content = content.encode("utf-8")
        maintype, subtype = mime_type.split("/", 1)
        message.add_attachment(
            content,
            maintype=maintype,
            subtype=subtype,
            filename=filename,
        )
        attachment_names.append(filename)

    smtp_host = os.getenv("SMTP_HOST")
    if smtp_host:
        deliver_via_smtp(message, smtp_host)
        return {
            "delivery_status": "sent",
            "delivery_mode": "smtp",
            "to_email": to_email,
            "attachment_filename": attachment_names[0] if attachment_names else "",
            "attachment_filenames": attachment_names,
        }

    outbox_path = write_to_demo_outbox(message, attachment_names[0] if attachment_names else "prior-auth-submission")
    return {
        "delivery_status": "queued",
        "delivery_mode": "demo_outbox",
        "to_email": to_email,
        "attachment_filename": attachment_names[0] if attachment_names else "",
        "attachment_filenames": attachment_names,
        "outbox_path": str(outbox_path),
    }


def deliver_via_smtp(message: EmailMessage, smtp_host: str) -> None:
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        if use_tls:
            server.starttls()
        if smtp_username and smtp_password:
            server.login(smtp_username, smtp_password)
        server.send_message(message)


def write_to_demo_outbox(message: EmailMessage, attachment_filename: str) -> Path:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    safe_name = sanitize_filename(attachment_filename).replace(".pdf", "")
    path = OUTBOX_DIR / f"{timestamp}-{safe_name}.eml"
    path.write_bytes(bytes(message))
    return path


def sanitize_filename(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "prior-auth-form.pdf").strip("-")
    return cleaned or "prior-auth-form.pdf"

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .generator import generate_linkedin_draft_content
from .models import ContentBrief, GeneratedPost
from .store import list_generated_posts, save_generated_post, update_generated_post_status


@dataclass(frozen=True)
class ContentServiceError(Exception):
    message: str
    status_code: int = 400


def create_linkedin_draft(brief: ContentBrief) -> GeneratedPost:
    generated = generate_linkedin_draft_content(brief)
    return save_generated_post(generated)


def list_linkedin_drafts() -> List[GeneratedPost]:
    return list_generated_posts()


def update_linkedin_draft_status(draft_id: str, approval_status: str) -> GeneratedPost:
    updated = update_generated_post_status(draft_id, approval_status)
    if not updated:
        raise ContentServiceError(f"Generated draft '{draft_id}' was not found.", status_code=404)
    return updated

from __future__ import annotations

from datetime import datetime, UTC
from typing import Dict, List, Optional
import uuid

from .models import GeneratedPost


_DRAFTS: Dict[str, GeneratedPost] = {}


def save_generated_post(post_data: dict) -> GeneratedPost:
    now = datetime.now(UTC)
    post = GeneratedPost(
        id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        **post_data,
    )
    _DRAFTS[post.id] = post
    return post


def list_generated_posts() -> List[GeneratedPost]:
    return sorted(_DRAFTS.values(), key=lambda post: post.created_at, reverse=True)


def get_generated_post(post_id: str) -> Optional[GeneratedPost]:
    return _DRAFTS.get(post_id)


def update_generated_post_status(post_id: str, approval_status: str) -> Optional[GeneratedPost]:
    existing = _DRAFTS.get(post_id)
    if not existing:
        return None

    updated = existing.model_copy(
        update={
            "approval_status": approval_status,
            "updated_at": datetime.now(UTC),
        }
    )
    _DRAFTS[post_id] = updated
    return updated


def clear_generated_posts() -> None:
    _DRAFTS.clear()

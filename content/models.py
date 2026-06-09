from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, field_validator


APPROVAL_STATUSES = {
    "draft",
    "needs_review",
    "approved",
    "scheduled",
    "published",
    "rejected",
}


class ContentBrief(BaseModel):
    topic: str = Field(..., min_length=3, max_length=240)
    audience: str = Field(..., min_length=3, max_length=160)
    content_pillar: str = Field(..., min_length=2, max_length=120)
    post_type: str = Field(..., min_length=2, max_length=80)
    tone: str = Field(..., min_length=2, max_length=80)
    cta: str = Field(..., min_length=2, max_length=160)
    visual_type: str = Field(..., min_length=2, max_length=80)

    @field_validator("*")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = " ".join(str(value).split())
        if not cleaned:
            raise ValueError("Field cannot be blank.")
        return cleaned


class GeneratedPost(BaseModel):
    id: str
    caption: str
    visual_title: str
    visual_sections: List[str]
    alt_text: str
    hashtags: List[str]
    compliance_notes: List[str]
    approval_status: str
    created_at: datetime
    updated_at: datetime


class UpdateApprovalStatusRequest(BaseModel):
    approval_status: str

    @field_validator("approval_status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in APPROVAL_STATUSES:
            allowed = ", ".join(sorted(APPROVAL_STATUSES))
            raise ValueError(f"approval_status must be one of: {allowed}")
        return normalized

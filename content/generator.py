from __future__ import annotations

import re
from typing import List

from .models import ContentBrief


def generate_linkedin_draft_content(brief: ContentBrief) -> dict:
    caption = build_caption(brief)
    visual_title = build_visual_title(brief)
    visual_sections = build_visual_sections(brief)
    alt_text = build_alt_text(brief, visual_title, visual_sections)
    hashtags = build_hashtags(brief)
    compliance_notes = build_compliance_notes(brief, caption)

    return {
        "caption": caption,
        "visual_title": visual_title,
        "visual_sections": visual_sections,
        "alt_text": alt_text,
        "hashtags": hashtags,
        "compliance_notes": compliance_notes,
        "approval_status": "needs_review" if compliance_notes else "draft",
    }


def build_caption(brief: ContentBrief) -> str:
    return (
        f"{brief.topic} is top of mind for {brief.audience}.\n\n"
        f"In our work around {brief.content_pillar}, we keep coming back to one practical question: "
        f"how can teams make this easier to understand, easier to act on, and easier to trust?\n\n"
        f"For a {brief.post_type}, the opportunity is to keep the message {brief.tone.lower()} "
        f"while giving people a clear next step.\n\n"
        f"{brief.cta}"
    )


def build_visual_title(brief: ContentBrief) -> str:
    return f"{brief.topic}: What {brief.audience} Should Know"


def build_visual_sections(brief: ContentBrief) -> List[str]:
    return [
        f"Why it matters: {brief.content_pillar}",
        f"Who it helps: {brief.audience}",
        f"Format: {brief.visual_type}",
        f"Next step: {brief.cta}",
    ]


def build_alt_text(brief: ContentBrief, visual_title: str, visual_sections: List[str]) -> str:
    sections = "; ".join(visual_sections)
    return (
        f"{brief.visual_type} graphic titled '{visual_title}' with sections: {sections}."
    )


def build_hashtags(brief: ContentBrief) -> List[str]:
    seeds = [
        brief.content_pillar,
        brief.topic,
        brief.audience,
        "healthcare innovation",
        "prior authorization",
    ]
    hashtags = []
    for seed in seeds:
        tag = re.sub(r"[^A-Za-z0-9]+", "", seed.title())
        if tag and f"#{tag}" not in hashtags:
            hashtags.append(f"#{tag}")
    return hashtags[:5]


def build_compliance_notes(brief: ContentBrief, caption: str) -> List[str]:
    notes = [
        "Human review required before publishing. This tool only creates an internal draft.",
        "Confirm the post does not imply clinical, legal, or payer-specific guarantees.",
    ]

    sensitive_terms = ["patient", "phi", "diagnosis", "claim", "denial", "approval"]
    lower_text = f"{brief.topic} {brief.audience} {caption}".lower()
    if any(term in lower_text for term in sensitive_terms):
        notes.append("Healthcare-sensitive language detected; verify no real PHI or case-specific details are included.")

    return notes

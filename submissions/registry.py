from __future__ import annotations

from typing import Dict

from .adapters import (
    APIPayloadAdapter,
    EmailSubmissionAdapter,
    PDFFormAdapter,
    PortalFieldMappingAdapter,
    SubmissionAdapter,
)


_ADAPTERS: Dict[str, SubmissionAdapter] = {
    APIPayloadAdapter.method: APIPayloadAdapter(),
    EmailSubmissionAdapter.method: EmailSubmissionAdapter(),
    PDFFormAdapter.method: PDFFormAdapter(),
    PortalFieldMappingAdapter.method: PortalFieldMappingAdapter(),
}


def get_submission_adapter(method: str) -> SubmissionAdapter:
    normalized = (method or "").strip().lower()
    if normalized not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unknown submission method '{method}'. Available methods: {available}")
    return _ADAPTERS[normalized]

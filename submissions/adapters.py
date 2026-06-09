from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from notifications.email_sender import send_email_with_attachments

from .models import PriorAuthArtifact, PriorAuthCase, SubmissionResult


class SubmissionAdapter(Protocol):
    method: str

    def prepare(
        self,
        case: PriorAuthCase,
        artifacts: Optional[List[PriorAuthArtifact]] = None,
        **options,
    ) -> SubmissionResult:
        ...

    def submit(self, prepared: SubmissionResult) -> SubmissionResult:
        ...


@dataclass(frozen=True)
class EmailSubmissionConfig:
    default_to_email: Optional[str] = None
    smtp_configured: bool = False

    @classmethod
    def from_environment(cls) -> "EmailSubmissionConfig":
        return cls(
            default_to_email=os.getenv("PRIOR_AUTH_TEST_EMAIL"),
            smtp_configured=bool(os.getenv("SMTP_HOST")),
        )


class EmailSubmissionAdapter:
    method = "email"

    def __init__(
        self,
        sender: Optional[Callable[..., dict]] = None,
        config: Optional[EmailSubmissionConfig] = None,
    ):
        self.sender = sender or send_email_with_attachments
        self.config = config or EmailSubmissionConfig.from_environment()

    def prepare(
        self,
        case: PriorAuthCase,
        artifacts: Optional[List[PriorAuthArtifact]] = None,
        **options,
    ) -> SubmissionResult:
        artifacts = artifacts or []
        to_email = options.get("to_email") or self.config.default_to_email
        if not to_email and self.config.smtp_configured:
            raise ValueError("Set PRIOR_AUTH_TEST_EMAIL or provide to_email before sending via SMTP.")
        to_email = to_email or "demo-test@example.com"

        subject = options.get("subject") or (
            f"Prior Authorization Submission - {case.patient_name} - CPT {case.cpt_code}"
        )
        body = options.get("body") or build_email_body(case, artifacts)

        return SubmissionResult(
            method=self.method,
            status="prepared",
            case=case,
            message="Email submission prepared.",
            artifacts=artifacts,
            prepared_payload={
                "to_email": to_email,
                "subject": subject,
                "body": body,
                "attachment_filenames": [artifact.filename for artifact in artifacts],
            },
        )

    def submit(self, prepared: SubmissionResult) -> SubmissionResult:
        to_email = prepared.prepared_payload.get("to_email")
        if not to_email:
            raise ValueError("Email submission requires a recipient.")

        delivery = self.sender(
            to_email=to_email,
            subject=prepared.prepared_payload.get("subject", ""),
            body=prepared.prepared_payload.get("body", ""),
            attachments=[
                {
                    "filename": artifact.filename,
                    "content": artifact.content,
                    "mime_type": artifact.mime_type,
                }
                for artifact in prepared.artifacts
            ],
        )
        prepared.status = delivery.get("delivery_status", "submitted")
        prepared.message = "Email submission completed."
        prepared.metadata["delivery"] = delivery
        return prepared


def build_email_body(case: PriorAuthCase, artifacts: List[PriorAuthArtifact]) -> str:
    artifact_text = "No generated artifacts are attached."
    if artifacts:
        artifact_text = "Attached artifacts:\n" + "\n".join(f"- {artifact.filename}" for artifact in artifacts)

    return (
        "A prior authorization submission has been prepared in the demo software.\n\n"
        f"Case ID: {case.case_id}\n"
        f"Patient: {case.patient_name}\n"
        f"Payer: {case.payer}\n"
        f"CPT Code: {case.cpt_code}\n\n"
        f"{artifact_text}"
    )


class PDFFormAdapter:
    method = "pdf_form"

    def prepare(
        self,
        case: PriorAuthCase,
        artifacts: Optional[List[PriorAuthArtifact]] = None,
        **options,
    ) -> SubmissionResult:
        artifacts = artifacts or []
        return SubmissionResult(
            method=self.method,
            status="prepared",
            case=case,
            message="Completed prior authorization PDF prepared.",
            artifacts=artifacts,
            prepared_payload={
                "artifact_filenames": [artifact.filename for artifact in artifacts],
                "download_hint": "Use the existing template preview/download endpoint for the rendered PDF.",
            },
        )

    def submit(self, prepared: SubmissionResult) -> SubmissionResult:
        prepared.status = "prepared"
        prepared.metadata["artifact_filenames"] = [
            artifact.filename for artifact in prepared.artifacts
        ]
        return prepared


class PortalFieldMappingAdapter:
    method = "portal_fields"

    def prepare(
        self,
        case: PriorAuthCase,
        artifacts: Optional[List[PriorAuthArtifact]] = None,
        **options,
    ) -> SubmissionResult:
        fields = build_portal_fields(case)
        return SubmissionResult(
            method=self.method,
            status="prepared",
            case=case,
            message="Portal field mapping prepared for staff review or browser automation.",
            artifacts=artifacts or [],
            prepared_payload={"fields": fields},
        )

    def submit(self, prepared: SubmissionResult) -> SubmissionResult:
        prepared.status = "prepared"
        return prepared


class APIPayloadAdapter:
    method = "api_payload"

    def prepare(
        self,
        case: PriorAuthCase,
        artifacts: Optional[List[PriorAuthArtifact]] = None,
        **options,
    ) -> SubmissionResult:
        payload = build_api_payload(case, artifacts or [])
        return SubmissionResult(
            method=self.method,
            status="prepared",
            case=case,
            message="API payload prepared. No real payer API was called.",
            artifacts=artifacts or [],
            prepared_payload={"payload": payload},
        )

    def submit(self, prepared: SubmissionResult) -> SubmissionResult:
        prepared.status = "prepared"
        prepared.metadata["simulated"] = True
        return prepared


def build_portal_fields(case: PriorAuthCase) -> dict:
    return {
        "caseId": case.case_id,
        "memberName": case.patient_name,
        "payer": case.payer,
        "cptCode": case.cpt_code,
        "providerName": case.provider_name or "",
        "placeOfService": case.metadata.get("place_of_service", "outpatient"),
        "requestType": "prior_authorization",
    }


def build_api_payload(case: PriorAuthCase, artifacts: List[PriorAuthArtifact]) -> dict:
    return {
        "case_id": case.case_id,
        "request_type": "prior_authorization",
        "patient": {
            "name": case.patient_name,
            "member_id": case.metadata.get("member_id", ""),
        },
        "payer": {
            "name": case.payer,
        },
        "authorization_request": {
            "cpt_code": case.cpt_code,
            "provider_name": case.provider_name or "",
            "place_of_service": case.metadata.get("place_of_service", "outpatient"),
        },
        "artifacts": [
            {
                "filename": artifact.filename,
                "mime_type": artifact.mime_type,
                "size_bytes": len(artifact.content),
            }
            for artifact in artifacts
        ],
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from forms.prior_auth_form import generate_prior_auth_form_pdf
from forms.template_filler import fill_prior_auth_template
from notifications.email_sender import sanitize_filename

from .models import PriorAuthArtifact, PriorAuthCase, SubmissionResult
from .registry import get_submission_adapter


@dataclass(frozen=True)
class SubmissionServiceError(Exception):
    message: str
    status_code: int = 502


def submit_prior_auth_case(
    *,
    case: Dict[str, Any],
    submission_method: str,
    artifacts: Optional[List[PriorAuthArtifact]] = None,
    to_email: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
) -> SubmissionResult:
    try:
        adapter = get_submission_adapter(submission_method)
        prepared = adapter.prepare(
            PriorAuthCase.from_mapping(case),
            artifacts=artifacts or [],
            to_email=to_email,
            subject=subject,
            body=body,
        )
        return adapter.submit(prepared)
    except ValueError as exc:
        raise SubmissionServiceError(str(exc), status_code=400) from exc
    except Exception as exc:
        raise SubmissionServiceError(f"Submission failed: {exc}", status_code=502) from exc


def build_case_artifacts(
    *,
    case: Dict[str, Any],
    documents: Iterable[Dict[str, Any]] = (),
    include_completed_template: bool = True,
    include_supporting_documents: bool = False,
) -> List[PriorAuthArtifact]:
    artifacts: List[PriorAuthArtifact] = []

    if include_completed_template:
        artifacts.append(build_completed_template_artifact(case))

    if include_supporting_documents:
        artifacts.extend(build_supporting_document_artifacts(documents))

    return artifacts


def build_completed_template_artifact(case: Dict[str, Any]) -> PriorAuthArtifact:
    try:
        template_pdf = fill_prior_auth_template(case)
    except Exception as exc:
        raise SubmissionServiceError(f"Template fill failed: {exc}", status_code=500) from exc

    return PriorAuthArtifact(
        filename=sanitize_filename(f"filled-prior-auth-template-{case['case_id']}.pdf"),
        content=template_pdf,
        mime_type="application/pdf",
    )


def build_supporting_document_artifacts(documents: Iterable[Dict[str, Any]]) -> List[PriorAuthArtifact]:
    artifacts: List[PriorAuthArtifact] = []
    for document in documents:
        filename = sanitize_filename(document.get("filename") or "supporting-document.txt")
        if not filename.lower().endswith(".txt"):
            filename = f"{filename}.txt"
        artifacts.append(
            PriorAuthArtifact(
                filename=filename,
                content=(document.get("text") or "").encode("utf-8"),
                mime_type="text/plain",
            )
        )
    return artifacts


def send_filled_template_for_case(case: Dict[str, Any], to_email: Optional[str] = None) -> Dict[str, Any]:
    result = submit_prior_auth_case(
        case=case,
        submission_method="email",
        to_email=to_email,
        artifacts=[build_completed_template_artifact(case)],
    )
    return delivery_from_result(result)


def send_standard_form_email(form_payload: Dict[str, Any], to_email: Optional[str] = None) -> Dict[str, Any]:
    attachment_bytes = generate_prior_auth_form_pdf(form_payload)
    filename_base = form_payload.get("case_id") or form_payload.get("patient_name") or "prior-auth-form"
    attachment_filename = sanitize_filename(f"prior-auth-form-{filename_base}.pdf")
    subject = f"Prior Authorization Form - {form_payload.get('patient_name')} - CPT {form_payload.get('cpt_code')}"
    body = (
        "A standardized prior authorization form has been completed in the demo software.\n\n"
        f"Patient: {form_payload.get('patient_name')}\n"
        f"Payer: {form_payload.get('payer')}\n"
        f"CPT Code: {form_payload.get('cpt_code')}\n\n"
        "The completed form is attached as a PDF."
    )
    case = {
        "case_id": form_payload.get("case_id") or filename_base,
        "patient_name": form_payload.get("patient_name") or "Unknown",
        "payer": form_payload.get("payer") or "unknown",
        "cpt_code": form_payload.get("cpt_code") or "",
        "provider_name": form_payload.get("provider_name"),
    }

    result = submit_prior_auth_case(
        case=case,
        submission_method="email",
        to_email=to_email,
        subject=subject,
        body=body,
        artifacts=[
            PriorAuthArtifact(
                filename=attachment_filename,
                content=attachment_bytes,
                mime_type="application/pdf",
            )
        ],
    )
    return delivery_from_result(result)


def delivery_from_result(result: SubmissionResult) -> Dict[str, Any]:
    return result.metadata.get("delivery", {})

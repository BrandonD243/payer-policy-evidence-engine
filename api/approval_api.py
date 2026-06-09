from __future__ import annotations

import os
import json
import yaml
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field

from content import (
    ContentBrief,
    ContentServiceError,
    GeneratedPost,
    UpdateApprovalStatusRequest,
    create_linkedin_draft,
    list_linkedin_drafts,
    update_linkedin_draft_status,
)
from extractors.openai_concept_extractor import extract_concepts_from_text
from extractors.section_extractor import extract_concepts_from_sections

from evaluators.clause_evaluator import evaluate_clauses
from evaluators.pa_required_evaluator import check_pa_required
from evaluators.coverage_evaluator import check_coverage
from evaluators.evidence_category_evaluator import group_clauses_by_evidence_category

# ✅ NEW IMPORTS
from resolvers.concept_resolver import infer_derived_concepts
from fhir.client import DEFAULT_MOCK_HEALTH_BASE_URL, FHIRClient, FHIRClientError
from fhir.mapper import (
    coverage_payer_display,
    patient_demographics,
    patient_display_name,
    patient_to_document_text,
    resources_to_document_text,
)

from storage.case_store import (
    create_case,
    update_case_status,
    get_all_cases,
    get_case
)

from storage.document_store import (
    save_document,
    get_documents_for_case,
    get_document
)
from forms.template_filler import fill_prior_auth_template
from reports.generator import generate_prior_auth_packet_pdf, build_report_context, render_report_html
from submissions import (
    SubmissionServiceError,
    build_case_artifacts,
    delivery_from_result,
    send_filled_template_for_case,
    send_standard_form_email,
    submit_prior_auth_case as service_submit_prior_auth_case,
)

# ===============================
# Base Directories
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCEPT_PATH = os.path.join(BASE_DIR, "concepts", "concept_registry.yaml")

with open(CONCEPT_PATH, "r") as f:
    concept_yaml = yaml.safe_load(f)

concept_registry = concept_yaml["concepts"]
openai_client = OpenAI()

RELEVANT_CONCEPTS = [
    {"concept_id": cid, **data}
    for cid, data in concept_registry.items()
    if data.get("type") == "atomic"
]

# ===============================
# Load Clauses
# ===============================
def load_clauses(payer_name: str, cpt_code: str):
    base = os.path.join(BASE_DIR, "payers", payer_name, cpt_code)

    approval_path = os.path.join(base, "approval_clauses.yaml")
    exclusion_path = os.path.join(base, "exclusion_clauses.yaml")

    if not os.path.exists(approval_path) or not os.path.exists(exclusion_path):
        raise FileNotFoundError(
            f"Clauses not found for payer={payer_name}, cpt={cpt_code}"
        )

    with open(approval_path, "r") as f:
        approval_clauses = yaml.safe_load(f).get("approval_clauses", [])

    with open(exclusion_path, "r") as f:
        exclusion_clauses = yaml.safe_load(f).get("exclusion_clauses", [])

    for c in approval_clauses:
        c["clause_type"] = "approval"

    for c in exclusion_clauses:
        c["clause_type"] = "exclusion"

    return approval_clauses + exclusion_clauses


def load_policy(payer_name: str, cpt_code: str):
    base = os.path.join(BASE_DIR, "payers", payer_name, cpt_code)
    policy_path = os.path.join(base, "policy.yaml")

    if not os.path.exists(policy_path):
        # Default logic if no policy.yaml
        return {
            "decision_logic": {
                "approvals": "ANY",
                "exclusions_override": True
            }
        }

    with open(policy_path, "r") as f:
        policy = yaml.safe_load(f)

    return policy


# ===============================
# Helpers
# ===============================
def extract_evidence_from_file(text: str):
    mentions = extract_concepts_from_sections(
        text,
        relevant_concepts=RELEVANT_CONCEPTS,
        extractor_fn=extract_concepts_from_text
    )

    for m in mentions:
        if "text_span" in m:
            m["evidence_text"] = m.pop("text_span")

        if "concept_id" not in m:
            m["concept_id"] = "unknown"

    return mentions


def compute_missing_concepts(clause_def, matched):
    required = clause_def.get("required_concepts", [])
    matched_ids = {
        m.get("concept_id")
        for m in matched
        if m.get("concept_id")
    }
    return [c for c in required if c and c not in matched_ids]


def normalize_confidence(raw_confidence):
    if isinstance(raw_confidence, (int, float)):
        return float(raw_confidence)
    if isinstance(raw_confidence, str):
        key = raw_confidence.strip().lower()
        if key == "strong":
            return 0.9
        if key == "moderate":
            return 0.7
        if key == "weak":
            return 0.5
        try:
            return float(key)
        except ValueError:
            return 0.0
    return 0.0


def wrap_matches(evidence_list):
    return [
        ConceptMatch(
            concept_id=ev.get("concept_id", ""),
            confidence=normalize_confidence(ev.get("confidence", 0.0)),
            certainty_level=ev.get("certainty_level"),
            evidence_text=ev.get("evidence_text", ""),
            section=ev.get("section", "")
        )
        for ev in evidence_list
    ]


def humanize_concept_id(concept_id: str) -> str:
    if not concept_id:
        return ""

    definition = None
    concept_data = concept_registry.get(concept_id)
    if isinstance(concept_data, dict):
        definition = concept_data.get("definition")

    if definition:
        return definition.rstrip(".")

    return concept_id.replace("_", " ").strip()


def summarize_partial_clause(clause: ClauseEvaluation) -> str:
    if clause.status != "partial" or not clause.missing_concepts:
        return ""

    missing_phrases = [humanize_concept_id(cid) for cid in clause.missing_concepts]
    missing_phrases = [p for p in missing_phrases if p]
    if not missing_phrases:
        return ""

    if len(missing_phrases) == 1:
        missing_text = missing_phrases[0]
    else:
        missing_text = ", ".join(missing_phrases[:-1]) + " and " + missing_phrases[-1]

    return f"{clause.policy_text} is partially supported but missing {missing_text}."


def join_clause_texts(clause_texts: List[str]) -> str:
    if not clause_texts:
        return ""
    if len(clause_texts) == 1:
        return clause_texts[0]
    if len(clause_texts) == 2:
        return f"{clause_texts[0]} and {clause_texts[1]}"
    return ", ".join(clause_texts[:-1]) + ", and " + clause_texts[-1]


def describe_required_clause_phrase(decision_logic: Dict[str, Any], total_approvals: int) -> str:
    approvals = (decision_logic or {}).get("approvals", "ANY")
    if isinstance(approvals, str):
        approvals_upper = approvals.upper()
        if approvals_upper == "ANY":
            return "only 1 approved clause is required"
        if approvals_upper == "ALL":
            return f"all {total_approvals} required clauses are satisfied"
        if approvals_upper.isdigit():
            return f"only {int(approvals_upper)} approved clauses are required"
    if isinstance(approvals, int):
        if approvals == 1:
            return "only 1 approved clause is required"
        return f"only {approvals} approved clauses are required"
    return "only 1 approved clause is required"


def normalize_clause_fragment(policy_text: str) -> str:
    return (policy_text or "").strip().rstrip(".;: ,")


def build_approved_summary(
    decision_logic: Dict[str, Any],
    approval_results: List["ClauseEvaluation"],
    approved_clause_texts: List[str],
) -> str:
    cleaned_texts = []
    for text in approved_clause_texts:
        cleaned = normalize_clause_fragment(text)
        if cleaned:
            cleaned_texts.append(cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned.lower())
    approved_text = join_clause_texts(cleaned_texts)

    approvals = str((decision_logic or {}).get("approvals", "ANY")).upper()
    if approvals == "ALL":
        if approved_text:
            return (
                "The prior authorization request has been approved because the submitted "
                f"documentation supports all required findings, including {approved_text}."
            )
        return "The prior authorization request has been approved because the submitted documentation supports all required findings."

    if approved_text:
        return (
            "The prior authorization request has been approved because the submitted "
            f"documentation supports {approved_text}."
        )
    return "The prior authorization request has been approved based on the submitted documentation."


def build_review_summary(
    approved_clause_texts: List[str],
    partial_exclusion_texts: List[str],
) -> str:
    approved_text = join_clause_texts([
        cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned.lower()
        for text in approved_clause_texts
        for cleaned in [normalize_clause_fragment(text)]
        if cleaned
    ])
    exclusion_text = join_clause_texts([
        normalize_clause_fragment(text)
        for text in partial_exclusion_texts
        if normalize_clause_fragment(text)
    ])

    if approved_text and exclusion_text:
        return (
            "The prior authorization request needs review because although the submitted "
            f"documentation supports {approved_text}, an exclusion clause regarding {exclusion_text} "
            "was partially triggered."
        )
    if exclusion_text:
        return (
            "The prior authorization request needs review because an exclusion clause regarding "
            f"{exclusion_text} was partially triggered."
        )
    return "The prior authorization request needs review based on the submitted documentation."


def summarize_clause_counts(
    approval_results: List["ClauseEvaluation"],
    exclusion_results: List["ClauseEvaluation"],
    decision_logic: Dict[str, Any] | None,
    approval_satisfied: bool,
    exclusions_triggered: bool,
) -> Dict[str, Optional[int] | str]:
    approval_logic = str((decision_logic or {}).get("approvals", "ANY")).upper()
    include_not_met = approval_logic == "ALL" or exclusions_triggered or not approval_satisfied

    counted_clauses = list(approval_results)
    if include_not_met:
        counted_clauses.extend(exclusion_results)

    return {
        "clauses_met": sum(1 for clause in counted_clauses if clause.status == "satisfied"),
        "clauses_need_attention": (
            sum(1 for clause in counted_clauses if clause.status == "partial")
            + sum(1 for clause in exclusion_results if clause.status == "partial" and clause not in counted_clauses)
        ),
        "clauses_not_met": (
            sum(1 for clause in counted_clauses if clause.status == "unsatisfied")
            if include_not_met
            else None
        ),
        "requirements_checked": f"{len(approval_results)}/{len(approval_results)}",
    }


def generate_case_summary(
    case_id: str,
    payer_name: str,
    cpt_code: str,
    pa_required: bool,
    coverage_status: str,
    approval_results: List[ClauseEvaluation],
    exclusion_results: List[ClauseEvaluation],
    evidence_review: List[Dict[str, Any]],
    decision_logic: Dict[str, Any],
    approval_satisfied: bool,
    evaluation_result_summary: str
) -> Tuple[str, Optional[str]]:
    approval_statuses = [
        f"{c.policy_text}: {c.status}" for c in approval_results
    ]
    exclusion_statuses = [
        f"{c.policy_text}: {c.status}" for c in exclusion_results
    ]
    categories = [
        f"{cat['evidence_category']}: {cat['status']}" for cat in evidence_review
    ]

    partial_clause_summaries = [
        summarize_partial_clause(c)
        for c in approval_results
        if c.status == "partial"
    ]
    partial_support_text = " ".join([s for s in partial_clause_summaries if s])

    decision_status = "APPROVED" if coverage_status == "covered" and approval_satisfied else "DENIED"
    exclusions_triggered = any(c.status == "satisfied" for c in exclusion_results) and decision_logic.get("exclusions_override", True)
    partial_exclusion_texts = [c.policy_text for c in exclusion_results if c.status == "partial"]

    approved_clause_texts = [c.policy_text for c in approval_results if c.status == "satisfied"]
    if coverage_status == "covered" and approval_satisfied and partial_exclusion_texts:
        return (
            build_review_summary(approved_clause_texts, partial_exclusion_texts),
            None,
        )
    if coverage_status == "covered" and approval_satisfied and approved_clause_texts:
        return (
            build_approved_summary(decision_logic, approval_results, approved_clause_texts),
            None,
        )

    prompt = f"""
You are a clinical review assistant. Your task is to summarize a prior authorization decision.

IMPORTANT: The final decision has already been determined by deterministic rules. Your summary MUST align with this decision.

FINAL DECISION: {decision_status}

Deterministic Rules:
- Coverage Status: {coverage_status}
- Prior Authorization Required: {pa_required}
- Decision Logic: Approval requires {decision_logic.get('approvals', 'ANY')} clauses to be satisfied
- Exclusions Override: {decision_logic.get('exclusions_override', True)}

Approval Clauses: {', '.join(approval_statuses) if approval_statuses else 'none evaluated'}
Exclusion Clauses: {', '.join(exclusion_statuses) if exclusion_statuses else 'none triggered'}
Evidence Categories: {', '.join(categories) if categories else 'none'}

{'Partial support details: ' + partial_support_text if partial_support_text else ''}

If the case is not approved, explain what evidence is missing for full support using the policy clause content and concept names in plain English.

Your task:
1. Return a JSON object with two keys: "summary_why" and "summary_fixes".
2. "summary_why" must always be present and explain the decision in plain clinical language.
3. "summary_fixes" should only be included if the case is not approved.
4. Use policy_text, not clause IDs.
5. Use the word "clauses" instead of "qualifying criterion" or similar.
6. Do not mention medical necessity.
7. If approved, say why using satisfied approval clause text and tie it to the evaluation result summary.
8. Keep the summary short and tie it into this evaluation result: {evaluation_result_summary}
9. Keep the JSON object as the only output.

Example: {{"summary_why": "...", "summary_fixes": "..."}}

Summary JSON:
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write concise clinical decision summaries as JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        # Strip Markdown fences
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].strip().startswith("```"):
                for i in range(1, len(lines)):
                    if lines[i].strip().startswith("```"):
                        content = "\n".join(lines[1:i]).strip()
                        break

        # Extract the first JSON object if additional text is present
        if "{" in content and not content.strip().startswith("{"):
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last != -1 and last > first:
                content = content[first:last+1]

        try:
            data = json.loads(content)
            return (
                data.get("summary_why", "").strip() if isinstance(data.get("summary_why"), str) else "",
                data.get("summary_fixes").strip() if isinstance(data.get("summary_fixes"), str) else None,
            )
        except Exception:
            # If JSON parsing fails, return raw content as summary_why
            return content, None
    except Exception:
        # Fallback logic using coverage_status (the source of truth from deterministic rules)
        satisfied_approvals = sum(1 for c in approval_results if c.status == "satisfied")
        total_approvals = len(approval_results)

        triggered_exclusions = sum(1 for c in exclusion_results if c.status == "satisfied")
        partial_clauses = [c for c in approval_results if c.status == "partial"]

        if coverage_status == "covered":
            required_phrase = describe_required_clause_phrase(decision_logic, total_approvals)
            if required_phrase.startswith("all "):
                return (
                    f"The case is approved because {required_phrase}.",
                    None,
                )
            return (
                f"The case is approved because {required_phrase} and {format_evaluation_result_summary(decision_logic, satisfied_approvals, total_approvals)}.",
                None,
            )

        if not pa_required:
            return (
                "The case is not covered because prior authorization is not required for this service.",
                None,
            )

        if triggered_exclusions > 0 and decision_logic.get("exclusions_override", True):
            return (
                f"The case is denied due to {triggered_exclusions} contraindication(s) that override approval.",
                "Provide documentation showing that the contraindication(s) do not apply or that the exclusion does not trigger.",
            )

        if partial_clauses:
            partial_details = " ".join(
                summarize_partial_clause(c) for c in partial_clauses if summarize_partial_clause(c)
            )
            missing_phrases = []
            for c in partial_clauses:
                for mid in c.missing_concepts:
                    p = humanize_concept_id(mid)
                    if p and p not in missing_phrases:
                        missing_phrases.append(p)
            if missing_phrases:
                if len(missing_phrases) == 1:
                    fixes = f"Provide documentation supporting {missing_phrases[0]}."
                else:
                    fixes = f"Provide documentation supporting {', '.join(missing_phrases[:-1])} and {missing_phrases[-1]}."
            else:
                fixes = f"Provide additional documentation to meet the remaining approval criteria ({satisfied_approvals}/{total_approvals})."

            if partial_details:
                return (
                    f"The case is not approved. {partial_details}",
                    fixes,
                )
            return (
                f"The case is not approved. Insufficient approval criteria satisfied ({satisfied_approvals} of {total_approvals}).",
                fixes,
            )

        return (
            f"The case is not approved. Insufficient approval criteria satisfied ({satisfied_approvals} of {total_approvals}).",
            None,
        )


# 🔥 NEW: Map extracted concepts to clause-required concepts
def map_to_clause_concepts(concepts: list) -> list:

    mapping = {
        "conservative_therapy_6_weeks": "symptoms_over_6_weeks",
        "conservative_therapy_4_weeks": "symptoms_over_6_weeks",
        "failed_conservative_therapy": "symptoms_over_6_weeks",
    }

    new_concepts = list(concepts)

    for c in concepts:

        if c["concept_id"] in mapping:

            mapped = c.copy()
            mapped["concept_id"] = mapping[c["concept_id"]]

            new_concepts.append(mapped)

    return new_concepts

def deduplicate_concepts(concepts: list) -> list:

    seen = set()
    unique = []

    for c in concepts:
        key = (
            c.get("concept_id"),
            c.get("evidence_text"),
            c.get("section")
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(c)

    return unique



#for evidence review, group satisfied/unsatisfied clauses by their evidence category (if defined) to provide more structured feedback on which types of evidence are sufficient vs insufficient. This is a more flexible approach than hardcoding specific categories in the clause definitions, while still allowing for categorization when available.    


# ===============================
# FastAPI App
# ===============================
app = FastAPI(
    title="Payer Policy Evidence Engine",
    version="2.4",
    description="Semantic clause evaluation for payer policies"
)

# ===============================
# API Models
# ===============================
class ConceptMatch(BaseModel):
    concept_id: str
    confidence: float
    certainty_level: str | None
    evidence_text: str
    section: str


class ClauseEvaluation(BaseModel):
    clause_id: str
    clause_type: str
    status: str
    policy_text: str
    missing_concepts: List[str]
    matched_concepts: List[ConceptMatch]
    evidence_category: str

class EvidenceClause(BaseModel):
    clause_id: str
    policy_text: str
    status: str

class EvidenceCategoryReview(BaseModel):
    evidence_category: str
    status: str
    clauses: List[EvidenceClause]


class EvaluationRule(BaseModel):
    decision_logic: str
    display: str


class ApprovalResponse(BaseModel):
    case_id: str
    evidence_review: List[EvidenceCategoryReview]
    approval_clauses: List[ClauseEvaluation]
    exclusion_clauses: Optional[List[ClauseEvaluation]] = None
    pa_required: bool
    coverage_status: str = "covered"
    evaluation_rule: EvaluationRule
    evaluation_result_summary: str
    num_documents: int
    clauses_met: Optional[int] = None
    clauses_need_attention: Optional[int] = None
    clauses_not_met: Optional[int] = None
    requirements_checked: str
    summary_why: str
    summary_fixes: Optional[str] = None
    review: str


class PriorAuthPacketRequest(BaseModel):
    case_id: str


class StandardizedPriorAuthForm(BaseModel):
    patient_name: str
    patient_dob: str
    member_id: str
    payer: str
    provider_name: str
    provider_npi: str
    provider_phone: str
    provider_fax: Optional[str] = None
    cpt_code: str
    diagnosis_codes: List[str] = []
    requested_service: str
    place_of_service: Optional[str] = "outpatient"
    urgency: Optional[str] = "standard"
    coverage_status: Optional[str] = None
    review_status: Optional[str] = None
    evidence_review: Optional[List[Dict[str, Any]]] = None
    clinical_summary: str
    supporting_notes: Optional[str] = None
    case_id: Optional[str] = None


class PriorAuthFormEmailRequest(BaseModel):
    form: StandardizedPriorAuthForm
    to_email: Optional[str] = None


class PriorAuthTemplateEmailRequest(BaseModel):
    case_id: str
    to_email: Optional[str] = None


class PriorAuthSubmissionRequest(BaseModel):
    case_id: str
    submission_method: str = "email"
    to_email: Optional[str] = None
    include_completed_template: bool = True
    include_supporting_documents: bool = False


class PriorAuthSubmissionResponse(BaseModel):
    case_id: str
    submission_method: str
    status: str
    message: Optional[str] = None
    prepared_payload: Dict[str, Any] = Field(default_factory=dict)
    artifact_filenames: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    delivery_status: Optional[str] = None
    delivery_mode: Optional[str] = None
    to_email: Optional[str] = None
    attachment_filename: Optional[str] = None
    attachment_filenames: List[str] = Field(default_factory=list)
    outbox_path: Optional[str] = None


class MockHealthPatientSearchResponse(BaseModel):
    fhir_base_url: str
    patients: List[Dict[str, Any]]


class MockHealthImportRequest(BaseModel):
    patient_id: str
    cpt_code: str = "64640"
    payer: Optional[str] = None
    fhir_base_url: Optional[str] = None
    bearer_token: Optional[str] = None


class MockHealthImportResponse(BaseModel):
    case_id: str
    patient_name: str
    payer: str
    cpt_code: str
    fhir_base_url: str
    documents_imported: int
    resources_imported: Dict[str, int]
    message: str


class MockHealthTemplateEmailRequest(MockHealthImportRequest):
    to_email: Optional[str] = None


class MockHealthTemplateEmailResponse(BaseModel):
    case_id: str
    patient_name: str
    payer: str
    cpt_code: str
    delivery_status: str
    delivery_mode: str
    to_email: str
    attachment_filename: str
    outbox_path: Optional[str] = None
    message: str


class PriorAuthFormEmailResponse(BaseModel):
    delivery_status: str
    delivery_mode: str
    to_email: str
    attachment_filename: str
    outbox_path: Optional[str] = None


class CaseSummary(BaseModel):
    case_id: str
    patient_name: str
    payer: str
    cpt_code: str
    status: str
    created_at: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    created_at: str


class CaseDetailResponse(BaseModel):
    case: CaseSummary
    documents: List[DocumentSummary]


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    case_id: str
    created_at: str
    text: str


class PARequiredResponse(BaseModel):
    payer: str
    cpt_code: str
    pa_required: bool
    message: str



# ===============================
# Core Evaluation Logic
# ===============================

def final_decision(coverage_covered: bool, pa_required: bool, exclusions_triggered: bool, approval_satisfied: bool) -> str:
    """Single truth source for coverage status determination."""
    if not coverage_covered:
        return "not covered"
    if not pa_required:
        return "covered"
    if exclusions_triggered:
        return "not covered"
    if approval_satisfied:
        return "covered"
    return "not covered"


def determine_review_status(
    coverage_status: str,
    approval_results: List[ClauseEvaluation],
    exclusion_results: List[ClauseEvaluation],
    approval_satisfied: bool,
    exclusions_triggered: bool
) -> str:
    has_partial_exclusion = any(c.status == "partial" for c in exclusion_results)

    if coverage_status == "covered" and approval_satisfied and not exclusions_triggered and not has_partial_exclusion:
        return "Ready to submit"

    if exclusions_triggered or coverage_status != "covered":
        return "Do not submit"

    if has_partial_exclusion or any(c.status == "partial" for c in approval_results):
        return "Review before submitting"

    return "Do not submit"


def format_evaluation_rule(decision_logic: Dict[str, Any] | None) -> Dict[str, str]:
    approvals = (decision_logic or {}).get("approvals", "ANY").upper()
    display = (
        "All clauses must be satisfied for approval."
        if approvals == "ALL"
        else "At least one clause must be satisfied."
    )
    return {
        "decision_logic": approvals,
        "display": display,
    }


def format_evaluation_result_summary(
    decision_logic: Dict[str, Any] | None,
    satisfied_approvals: int,
    total_approvals: int,
) -> str:
    approvals = (decision_logic or {}).get("approvals", "ANY").upper()
    if approvals == "ALL":
        return f"{satisfied_approvals} of {total_approvals} required clauses satisfied"
    return (
        f"{satisfied_approvals} clauses satisfied"
        if satisfied_approvals > 0
        else "no clauses satisfied"
    )


def evaluate_case(
    case_id: str,
    combined_text: str,
    payer_name: str,
    cpt_code: str,
    context: Dict[str, Any]
):

    # Collect all decision inputs
    coverage = check_coverage(
        payer_name,
        cpt_code,
        context=context
    )
    coverage_covered = coverage.get("coverage_status", "covered").lower() == "covered"

    pa_required = check_pa_required(
        payer_name,
        cpt_code,
        context=context
    )

    # If not covered or no PA required, still need to evaluate clauses for completeness
    all_clauses = load_clauses(payer_name, cpt_code)
    policy = load_policy(payer_name, cpt_code)
    decision_logic = policy.get("decision_logic", {"approvals": "ANY", "exclusions_override": True})

    extracted_evidence = extract_evidence_from_file(combined_text)

    # ✅ Derived concept inference
    extracted_evidence = infer_derived_concepts(
        extracted_evidence,
        concept_registry
    )

    extracted_evidence = deduplicate_concepts(extracted_evidence)

    # 🔥 Map concepts to clause-required IDs
    extracted_evidence = map_to_clause_concepts(extracted_evidence)

    extracted_evidence = infer_derived_concepts(
        extracted_evidence,
        concept_registry
    )

    def filter_duration_concepts(evidence_list):

        duration_terms = [
            "week",
            "weeks",
            "month",
            "months",
            "chronic",
            "persistent",
            "history",
            ">"
        ]

        therapy_terms = [
            "therapy",
            "physical therapy",
            "pt",
            "sessions",
            "treatment",
            "exercise",
            "program"
        ]

        filtered = []

        for ev in evidence_list:

            if ev.get("concept_id") == "symptoms_over_6_weeks":

                text = ev.get("evidence_text", "").lower()

                # Must contain duration indicator
                if not any(term in text for term in duration_terms):
                    continue

                # Exclude therapy duration references
                if any(term in text for term in therapy_terms):
                    continue

            filtered.append(ev)

        return filtered
    #NEW
    extracted_evidence = filter_duration_concepts(extracted_evidence)

    # Evaluate clauses
    clause_results = evaluate_clauses(
        extracted_evidence,
        all_clauses,
        concept_registry
    )

    approval_results = []
    exclusion_results = []

    clause_lookup = {c.get("id") or c.get("clause_id"): c for c in all_clauses}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type", "approval")
        evidence_category = clause_def.get("evidence_category", "uncategorized")  # 🔥 read from YAML

        matched = result.get("evidence", [])
        missing = compute_missing_concepts(clause_def, matched)

        clause_status = result.get("status", "unsatisfied")
        matched_concepts = wrap_matches(matched)

        # Keep exclusion clause status as the actual trigger state.
        # This ensures partial exclusions stay partial and only truly triggered
        # exclusions are marked satisfied.
        display_status = clause_status

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            status=display_status,
            policy_text=(clause_def.get("policy_text") or "").strip(),
            missing_concepts=missing,
            matched_concepts=matched_concepts,
            evidence_category=evidence_category,
            logic=(clause_def.get("logic") or "AND").upper()
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    satisfied_approvals = [c for c in approval_results if c.status == "satisfied"]
    triggered_exclusions = [c for c in exclusion_results if c.status == "satisfied"]

    # Determine approval and exclusion status
    if decision_logic.get("approvals", "ANY") == "ALL":
        approval_satisfied = len(satisfied_approvals) == len(approval_results) and len(approval_results) > 0
    else:  # ANY
        approval_satisfied = len(satisfied_approvals) > 0

    exclusions_triggered = decision_logic.get("exclusions_override", True) and bool(triggered_exclusions)

    # 🧱 SINGLE DECISION FUNCTION
    coverage_status = final_decision(
        coverage_covered=coverage_covered,
        pa_required=pa_required,
        exclusions_triggered=exclusions_triggered,
        approval_satisfied=approval_satisfied
    )

    # Update case status based on final decision
    if coverage_status == "covered":
        update_case_status(case_id, "approved")
    else:
        update_case_status(case_id, f"denied:{coverage_status}")

    return approval_results, exclusion_results, pa_required, coverage_status, decision_logic, approval_satisfied

def build_evidence_review(all_clauses: List[ClauseEvaluation], decision_logic: Dict = None):
    """
    Build evidence review for display.
    
    For ANY logic CPT codes:
    - If any approval clauses are satisfied: only show categories with satisfied approval clauses,
      and within those categories, only show satisfied approval clauses.
    - If no approval clauses are satisfied: show all categories with all approval and exclusion clauses.
    
    For ALL logic CPT codes: show all categories with all approval clauses.
    
    Always show exclusion clauses in their categories.
    """
    if decision_logic is None:
        decision_logic = {}

    approval_logic = decision_logic.get("approvals", "ANY").upper()
    
    # Check if any approval clauses are satisfied across all
    any_satisfied_approvals = any(
        c.status == "satisfied" and c.clause_type == "approval" 
        for c in all_clauses
    )
    
    categories = {}

    for clause in all_clauses:

        category = clause.evidence_category or "uncategorized"

        if category not in categories:
            categories[category] = {
                "evidence_category": category,
                "status": "Insufficient",
                "clauses": [],
                "has_approval": False,
                "has_partial": False,
                "raw_clauses": []  # Store for filtering
            }

        # Store raw clause
        categories[category]["raw_clauses"].append(clause)

        # Track clause results for status determination
        if clause.status == "satisfied":
            categories[category]["has_approval"] = True
        elif clause.status == "partial":
            categories[category]["has_partial"] = True

    # Determine final category status and filter clauses
    results = []

    for category_data in categories.values():

        if category_data["has_approval"]:
            category_data["status"] = "Sufficient"

        elif category_data["has_partial"]:
            category_data["status"] = "Partially sufficient"

        else:
            category_data["status"] = "Insufficient"

        # Filter clauses based on approval_logic and any_satisfied_approvals
        raw_clauses = category_data.pop("raw_clauses")
        filtered_clauses = []

        # Separate approval and exclusion clauses
        approval_clauses = [c for c in raw_clauses if c.clause_type == "approval"]
        exclusion_clauses = [c for c in raw_clauses if c.clause_type == "exclusion"]

        if approval_logic == "ANY":
            if any_satisfied_approvals:
                # Only show satisfied approval clauses in categories that have them
                satisfied = [c for c in approval_clauses if c.status == "satisfied"]
                if satisfied:
                    filtered_clauses.extend([{
                        "clause_id": c.clause_id,
                        "policy_text": c.policy_text,
                        "status": c.status
                    } for c in satisfied])
            else:
                # No satisfied approvals anywhere: show all approval clauses
                filtered_clauses.extend([{
                    "clause_id": c.clause_id,
                    "policy_text": c.policy_text,
                    "status": c.status
                } for c in approval_clauses])
        else:  # ALL logic
            # Always show all approval clauses
            filtered_clauses.extend([{
                "clause_id": c.clause_id,
                "policy_text": c.policy_text,
                "status": c.status
            } for c in approval_clauses])

        triggered_exclusions = [c for c in exclusion_clauses if c.status == "satisfied"]
        partial_exclusions = [c for c in exclusion_clauses if c.status == "partial"]

        if triggered_exclusions:
            category_data["status"] = "Insufficient"
            filtered_clauses.extend([{
                "clause_id": c.clause_id,
                "policy_text": c.policy_text,
                "status": c.status
            } for c in triggered_exclusions])

        if partial_exclusions:
            filtered_clauses.extend([{
                "clause_id": c.clause_id,
                "policy_text": c.policy_text,
                "status": c.status
            } for c in partial_exclusions])

        # Only include category if it has clauses to show
        if filtered_clauses:
            category_data["clauses"] = filtered_clauses

            # Remove internal tracking flags
            category_data.pop("has_approval")
            category_data.pop("has_partial")

            results.append(category_data)

    return results


def build_case_evaluation_response(case_id: str) -> ApprovalResponse:
    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    documents = get_documents_for_case(case_id)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents attached")

    payer_name = case["payer"].lower()

    context = {
        "state": "NY",
        "plan_type": case.get("plan_type", "commercial"),
        "place_of_service": case.get("place_of_service", "outpatient"),
    }

    try:
        update_case_status(case_id, "processing")

        combined_text = "\n\n".join(
            doc.get("text", "") for doc in documents
        )

        approval_results, exclusion_results, pa_required, coverage_status, decision_logic, approval_satisfied = evaluate_case(
            case_id,
            combined_text,
            payer_name,
            case["cpt_code"],
            context
        )

        # Filter exclusions passed into evidence review: include partial and triggered exclusions.
        filtered_exclusion_results = [
            c for c in exclusion_results if c.status in {"satisfied", "partial"}
        ]

        all_results = approval_results + filtered_exclusion_results
        evidence_review = build_evidence_review(all_results, decision_logic)

    except Exception:
        update_case_status(case_id, "error")
        raise

    evaluation_rule = format_evaluation_rule(decision_logic)
    evaluation_result_summary = format_evaluation_result_summary(
        decision_logic,
        satisfied_approvals=len([c for c in approval_results if c.status == "satisfied"]),
        total_approvals=len(approval_results),
    )

    summary_why, summary_fixes = generate_case_summary(
        case_id=case_id,
        payer_name=payer_name,
        cpt_code=case["cpt_code"],
        pa_required=pa_required,
        coverage_status=coverage_status,
        approval_results=approval_results,
        exclusion_results=exclusion_results,
        evidence_review=evidence_review,
        decision_logic=decision_logic,
        approval_satisfied=approval_satisfied,
        evaluation_result_summary=evaluation_result_summary
    )

    exclusions_triggered = decision_logic.get("exclusions_override", True) and any(
        c.status == "satisfied" for c in exclusion_results
    )
    review_status = determine_review_status(
        coverage_status=coverage_status,
        approval_results=approval_results,
        exclusion_results=exclusion_results,
        approval_satisfied=approval_satisfied,
        exclusions_triggered=exclusions_triggered,
    )
    clause_counts = summarize_clause_counts(
        approval_results=approval_results,
        exclusion_results=exclusion_results,
        decision_logic=decision_logic,
        approval_satisfied=approval_satisfied,
        exclusions_triggered=exclusions_triggered,
    )

    response_data = {
        "case_id": case_id,
        "evidence_review": evidence_review,
        "approval_clauses": approval_results,
        "pa_required": pa_required,
        "coverage_status": coverage_status,
        "evaluation_rule": evaluation_rule,
        "evaluation_result_summary": evaluation_result_summary,
        "num_documents": len(documents),
        "clauses_met": clause_counts["clauses_met"],
        "clauses_need_attention": clause_counts["clauses_need_attention"],
        "clauses_not_met": clause_counts["clauses_not_met"],
        "requirements_checked": clause_counts["requirements_checked"],
        "summary_why": summary_why,
        "review": review_status
    }
    if summary_fixes:
        response_data["summary_fixes"] = summary_fixes

    # Always include the full set of exclusion clauses in the response (show all possible exclusions)
    response_data["exclusion_clauses"] = exclusion_results

    return ApprovalResponse(**response_data)


FHIR_CASE_RESOURCE_TYPES = [
    "Condition",
    "Procedure",
    "Observation",
    "DiagnosticReport",
    "DocumentReference",
    "MedicationRequest",
]


def clean_optional_api_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"string", "null", "none"}:
        return None
    return cleaned


def create_case_from_fhir(
    *,
    patient_id: str,
    cpt_code: str,
    payer: Optional[str],
    fhir_base_url: Optional[str],
    bearer_token: Optional[str],
) -> MockHealthImportResponse:
    cleaned_patient_id = clean_optional_api_value(patient_id)
    if not cleaned_patient_id:
        raise HTTPException(
            status_code=400,
            detail="patient_id is required. Use an id returned by /fhir/mock-health/patients.",
        )

    cleaned_cpt_code = clean_optional_api_value(cpt_code) or "64640"
    cleaned_payer = clean_optional_api_value(payer)
    cleaned_fhir_base_url = clean_optional_api_value(fhir_base_url)
    cleaned_bearer_token = clean_optional_api_value(bearer_token)

    client = FHIRClient(
        base_url=cleaned_fhir_base_url or DEFAULT_MOCK_HEALTH_BASE_URL,
        bearer_token=cleaned_bearer_token,
    )

    try:
        patient = client.get_resource("Patient", cleaned_patient_id)
        coverages = client.search_entries("Coverage", {"beneficiary": f"Patient/{cleaned_patient_id}", "_count": 10})
        resources_by_type = {
            resource_type: client.search_entries(resource_type, {"patient": cleaned_patient_id, "_count": 20})
            for resource_type in FHIR_CASE_RESOURCE_TYPES
        }
    except FHIRClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    patient_name = patient_display_name(patient)
    resolved_payer = cleaned_payer or coverage_payer_display(coverages) or "unknown"
    case = create_case(
        patient_name=patient_name,
        payer=resolved_payer.lower(),
        cpt_code=cleaned_cpt_code,
    )
    case["fhir_patient"] = patient_demographics(patient, cleaned_patient_id)
    case["fhir_base_url"] = client.base_url

    save_document(
        case_id=case["case_id"],
        filename="fhir_patient_summary.txt",
        text=patient_to_document_text(patient),
    )

    documents_imported = 1
    resources_imported = {
        "Patient": 1,
        "Coverage": len(coverages),
    }
    for resource_type, resources in resources_by_type.items():
        resources_imported[resource_type] = len(resources)
        if not resources:
            continue
        save_document(
            case_id=case["case_id"],
            filename=f"fhir_{resource_type.lower()}_summary.txt",
            text=resources_to_document_text(resource_type, resources),
        )
        documents_imported += 1

    return MockHealthImportResponse(
        case_id=case["case_id"],
        patient_name=patient_name,
        payer=case["payer"],
        cpt_code=case["cpt_code"],
        fhir_base_url=client.base_url,
        documents_imported=documents_imported,
        resources_imported=resources_imported,
        message="FHIR resources imported. Run /run_case/{case_id} to analyze.",
    )


# ===============================
# Endpoints
# ===============================
@app.post("/upload_files")
async def upload_files(
    files: List[UploadFile] = File(...),
    payer: str = "humana",
    cpt_code: str = "64640",
    patient_name: str = "Unknown"
):
    payer_name = payer.lower()

    case = create_case(
        patient_name=patient_name,
        payer=payer_name,
        cpt_code=cpt_code,
    )

    for file in files:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")

        save_document(
            case_id=case["case_id"],
            filename=file.filename,
            text=text
        )

    return {
        "case_id": case["case_id"],
        "message": "Files uploaded successfully. Run /run_case to analyze."
    }


@app.post("/demo/preload-64640-case-1")
async def preload_64640_case_1():
    example_dir = os.path.join(BASE_DIR, "examples", "64640", "case_01")
    if not os.path.isdir(example_dir):
        raise HTTPException(status_code=404, detail="Demo case files not found")

    case = create_case(
        patient_name="Jay Doe",
        payer="humana",
        cpt_code="64640",
    )

    loaded_files = []
    for filename in sorted(os.listdir(example_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(example_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        save_document(
            case_id=case["case_id"],
            filename=filename,
            text=text,
        )
        loaded_files.append(filename)

    if not loaded_files:
        raise HTTPException(status_code=404, detail="No demo text files found")

    return {
        "case_id": case["case_id"],
        "payer": case["payer"],
        "cpt_code": case["cpt_code"],
        "patient_name": case["patient_name"],
        "documents_loaded": loaded_files,
        "message": "Demo 64640 case 1 loaded. Run /run_case/{case_id} to analyze.",
    }


@app.get("/fhir/mock-health/patients", response_model=MockHealthPatientSearchResponse)
async def list_mock_health_patients(count: int = 5, bearer_token: Optional[str] = None):
    client = FHIRClient(base_url=DEFAULT_MOCK_HEALTH_BASE_URL, bearer_token=bearer_token)
    try:
        patients = client.search_entries("Patient", {"_count": max(1, min(count, 20))})
    except FHIRClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return MockHealthPatientSearchResponse(
        fhir_base_url=client.base_url,
        patients=[
            {
                "id": patient.get("id"),
                "name": patient_display_name(patient),
                "birthDate": patient.get("birthDate"),
                "gender": patient.get("gender"),
            }
            for patient in patients
        ],
    )


@app.post("/fhir/mock-health/import-case", response_model=MockHealthImportResponse)
async def import_mock_health_case(request: MockHealthImportRequest):
    return create_case_from_fhir(
        patient_id=request.patient_id,
        cpt_code=request.cpt_code,
        payer=request.payer,
        fhir_base_url=request.fhir_base_url,
        bearer_token=request.bearer_token,
    )


@app.post("/fhir/mock-health/send-template-email", response_model=MockHealthTemplateEmailResponse)
async def send_mock_health_template_email(request: MockHealthTemplateEmailRequest):
    imported = create_case_from_fhir(
        patient_id=request.patient_id,
        cpt_code=request.cpt_code,
        payer=request.payer,
        fhir_base_url=request.fhir_base_url,
        bearer_token=request.bearer_token,
    )
    case = get_case(imported.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Imported case was not found")

    try:
        delivery = send_filled_template_for_case(case, request.to_email)
    except SubmissionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return MockHealthTemplateEmailResponse(
        case_id=case["case_id"],
        patient_name=case.get("patient_name", "Unknown"),
        payer=case.get("payer", "unknown"),
        cpt_code=case.get("cpt_code", ""),
        message="FHIR patient data was imported, inserted into the PDF template, and sent by email.",
        **delivery,
    )


@app.post("/run_case/{case_id}", response_model=ApprovalResponse, response_model_exclude_none=True)
async def run_case(case_id: str):
    return build_case_evaluation_response(case_id)

    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    documents = get_documents_for_case(case_id)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents attached")

    payer_name = case["payer"].lower()

    context = {
        "state": "NY",
        "plan_type": case.get("plan_type", "commercial"),
        "place_of_service": case.get("place_of_service", "outpatient"),
    }

    try:
        update_case_status(case_id, "processing")

        combined_text = "\n\n".join(
            doc.get("text", "") for doc in documents
        )

        approval_results, exclusion_results, pa_required, coverage_status, decision_logic, approval_satisfied = evaluate_case(
            case_id,
            combined_text,
            payer_name,
            case["cpt_code"],
            context
        )

        # 🔥 Filter exclusions passed into evidence review: include partial and triggered exclusions.
        filtered_exclusion_results = [
            c for c in exclusion_results if c.status in {"satisfied", "partial"}
        ]

        all_results = approval_results + filtered_exclusion_results
        evidence_review = build_evidence_review(all_results, decision_logic)

    except Exception:
        update_case_status(case_id, "error")
        raise

    evaluation_rule = format_evaluation_rule(decision_logic)
    evaluation_result_summary = format_evaluation_result_summary(
        decision_logic,
        satisfied_approvals=len([c for c in approval_results if c.status == "satisfied"]),
        total_approvals=len(approval_results),
    )

    summary_why, summary_fixes = generate_case_summary(
        case_id=case_id,
        payer_name=payer_name,
        cpt_code=case["cpt_code"],
        pa_required=pa_required,
        coverage_status=coverage_status,
        approval_results=approval_results,
        exclusion_results=exclusion_results,
        evidence_review=evidence_review,
        decision_logic=decision_logic,
        approval_satisfied=approval_satisfied,
        evaluation_result_summary=evaluation_result_summary
    )

    exclusions_triggered = decision_logic.get("exclusions_override", True) and any(
        c.status == "satisfied" for c in exclusion_results
    )
    review_status = determine_review_status(
        coverage_status=coverage_status,
        approval_results=approval_results,
        exclusion_results=exclusion_results,
        approval_satisfied=approval_satisfied,
        exclusions_triggered=exclusions_triggered,
    )
    clause_counts = summarize_clause_counts(
        approval_results=approval_results,
        exclusion_results=exclusion_results,
        decision_logic=decision_logic,
        approval_satisfied=approval_satisfied,
        exclusions_triggered=exclusions_triggered,
    )

    response_data = {
        "case_id": case_id,
        "evidence_review": evidence_review,
        "approval_clauses": approval_results,
        "pa_required": pa_required,
        "coverage_status": coverage_status,
        "evaluation_rule": evaluation_rule,
        "evaluation_result_summary": evaluation_result_summary,
        "num_documents": len(documents),
        "clauses_met": clause_counts["clauses_met"],
        "clauses_need_attention": clause_counts["clauses_need_attention"],
        "clauses_not_met": clause_counts["clauses_not_met"],
        "requirements_checked": clause_counts["requirements_checked"],
        "summary_why": summary_why,
        "review": review_status
    }
    if summary_fixes:
        response_data["summary_fixes"] = summary_fixes

    # Always include the full set of exclusion clauses in the response (show all possible exclusions)
    response_data["exclusion_clauses"] = exclusion_results

    return ApprovalResponse(**response_data)


@app.post("/prior-auth-forms/send-email", response_model=PriorAuthFormEmailResponse)
async def send_prior_auth_form_email(request: PriorAuthFormEmailRequest):
    form_payload = (
        request.form.model_dump(exclude_none=True)
        if hasattr(request.form, "model_dump")
        else request.form.dict(exclude_none=True)
    )
    try:
        result = send_standard_form_email(form_payload, request.to_email)
    except SubmissionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return PriorAuthFormEmailResponse(**result)


@app.post("/prior-auth-submissions", response_model=PriorAuthSubmissionResponse)
async def submit_prior_auth_case(request: PriorAuthSubmissionRequest):
    case = get_case(request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        artifacts = build_case_artifacts(
            case=case,
            documents=get_documents_for_case(request.case_id),
            include_completed_template=request.include_completed_template,
            include_supporting_documents=request.include_supporting_documents,
        )
        result = service_submit_prior_auth_case(
            case=case,
            submission_method=request.submission_method,
            artifacts=artifacts,
            to_email=request.to_email,
        )
    except SubmissionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    delivery = delivery_from_result(result)
    return PriorAuthSubmissionResponse(
        case_id=case["case_id"],
        submission_method=result.method,
        status=result.status,
        message=result.message,
        prepared_payload=result.prepared_payload,
        artifact_filenames=[artifact.filename for artifact in result.artifacts],
        metadata=result.metadata,
        delivery_status=delivery.get("delivery_status"),
        delivery_mode=delivery.get("delivery_mode"),
        to_email=delivery.get("to_email"),
        attachment_filename=delivery.get("attachment_filename"),
        attachment_filenames=delivery.get("attachment_filenames", []),
        outbox_path=delivery.get("outbox_path"),
    )


@app.post("/prior-auth-forms/send-template-email", response_model=PriorAuthFormEmailResponse)
async def send_prior_auth_template_email(request: PriorAuthTemplateEmailRequest):
    case = get_case(request.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        result = send_filled_template_for_case(case, request.to_email)
    except SubmissionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return PriorAuthFormEmailResponse(**result)


@app.post("/content/linkedin/drafts/generate", response_model=GeneratedPost)
async def generate_linkedin_content_draft(brief: ContentBrief):
    return create_linkedin_draft(brief)


@app.get("/content/linkedin/drafts", response_model=List[GeneratedPost])
async def get_linkedin_content_drafts():
    return list_linkedin_drafts()


@app.patch("/content/linkedin/drafts/{draft_id}/approval-status", response_model=GeneratedPost)
async def update_linkedin_content_draft_status(
    draft_id: str,
    request: UpdateApprovalStatusRequest,
):
    try:
        return update_linkedin_draft_status(draft_id, request.approval_status)
    except ContentServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@app.get("/prior-auth-forms/template-preview/{case_id}")
async def preview_filled_prior_auth_template(case_id: str):
    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        pdf_bytes = fill_prior_auth_template(case)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Template fill failed: {exc}") from exc

    filename = f"filled-prior-auth-template-{case_id}.pdf"
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.post("/reports/prior-auth-packet")
async def generate_prior_auth_packet(request: PriorAuthPacketRequest):
    evaluation = build_case_evaluation_response(request.case_id)
    case = get_case(request.case_id)
    documents = get_documents_for_case(request.case_id)

    evaluation_payload = (
        evaluation.model_dump(exclude_none=True)
        if hasattr(evaluation, "model_dump")
        else evaluation.dict(exclude_none=True)
    )
    pdf_bytes = generate_prior_auth_packet_pdf(
        case=case,
        documents=documents,
        evaluation=evaluation_payload,
    )

    filename = f"prior-auth-packet-{request.case_id}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.get("/reports/prior-auth-packet/{case_id}/preview", response_class=HTMLResponse)
async def preview_prior_auth_packet(case_id: str):
    evaluation = build_case_evaluation_response(case_id)
    case = get_case(case_id)
    documents = get_documents_for_case(case_id)
    evaluation_payload = (
        evaluation.model_dump(exclude_none=True)
        if hasattr(evaluation, "model_dump")
        else evaluation.dict(exclude_none=True)
    )
    context = build_report_context(
        case=case,
        documents=documents,
        evaluation=evaluation_payload,
    )
    return HTMLResponse(content=render_report_html(context))


@app.get("/cases", response_model=List[CaseSummary])
async def list_cases():
    return get_all_cases()


@app.get("/cases/{case_id}", response_model=CaseDetailResponse)
async def get_case_detail(case_id: str):
    return CaseDetailResponse(
        case=get_case(case_id),
        documents=get_documents_for_case(case_id)
    )


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_endpoint(document_id: str):
    return get_document(document_id)


@app.get("/prior-auth-required", response_model=PARequiredResponse)
def prior_auth_required(payer: str, cpt_code: str):

    payer_name = payer.lower()

    context = {
        "state": "NY",
        "plan_type": "commercial",
        "place_of_service": "outpatient",
    }

    coverage = check_coverage(
        payer_name,
        cpt_code,
        context=context
    )

    coverage_status = coverage.get("coverage_status", "covered").lower()

    if coverage_status != "covered":
        return {
            "payer": payer,
            "cpt_code": cpt_code,
            "pa_required": False,
            "message": f"Service is {coverage_status.replace('_', ' ')}"
        }

    pa_required = check_pa_required(
        payer_name,
        cpt_code,
        context=context
    )

    return {
        "payer": payer,
        "cpt_code": cpt_code,
        "pa_required": pa_required,
        "message": (
            "Prior authorization required"
            if pa_required
            else "No prior authorization required"
        )
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}

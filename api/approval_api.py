from __future__ import annotations

import os
import yaml
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from openai import OpenAI
from pydantic import BaseModel

from extractors.openai_concept_extractor import extract_concepts_from_text
from extractors.section_extractor import extract_concepts_from_sections

from evaluators.clause_evaluator import evaluate_clauses
from evaluators.pa_required_evaluator import check_pa_required
from evaluators.coverage_evaluator import check_coverage
from evaluators.evidence_category_evaluator import group_clauses_by_evidence_category

# ✅ NEW IMPORTS
from resolvers.concept_resolver import infer_derived_concepts

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


def wrap_matches(evidence_list):
    return [
        ConceptMatch(
            concept_id=ev.get("concept_id", ""),
            confidence=float(ev.get("confidence", 0.0)),
            certainty_level=ev.get("certainty_level"),
            evidence_text=ev.get("evidence_text", ""),
            section=ev.get("section", "")
        )
        for ev in evidence_list
    ]


def generate_case_summary(
    case_id: str,
    payer_name: str,
    cpt_code: str,
    pa_required: bool,
    coverage_status: str,
    approval_results: List[ClauseEvaluation],
    exclusion_results: List[ClauseEvaluation],
    evidence_review: List[Dict[str, Any]]
) -> str:
    clause_statuses = [
        f"{c.clause_id}: {c.status}" for c in approval_results + exclusion_results
    ]
    categories = [
        f"{cat['evidence_category']}: {cat['status']}" for cat in evidence_review
    ]

    prompt = f"""
You are a clinical review assistant.

Summarize the overall approval decision for a prior authorization case using the policy clause evaluation results.

Respond in one or two short sentences. Mention whether the case appears approved, denied, or partially supported, and cite the main evidence category status.

Case ID: {case_id}
Payer: {payer_name}
CPT code: {cpt_code}
Coverage status: {coverage_status}
Prior authorization required: {pa_required}
Approval clause results: {', '.join(clause_statuses) if clause_statuses else 'none'}
Evidence categories: {', '.join(categories) if categories else 'none'}

Summary:
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write concise clinical decision summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        if any(c.status == "satisfied" for c in approval_results):
            return "The case has sufficient evidence for approval based on satisfied approval clauses."
        if any(c.status == "partial" for c in approval_results):
            return "The case has partial evidence and may require additional support before approval."
        return "The case is not supported for approval based on the current evidence."


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


class ApprovalResponse(BaseModel):
    case_id: str
    evidence_review: List[EvidenceCategoryReview]
    approval_clauses: List[ClauseEvaluation]
    exclusion_clauses: Optional[List[ClauseEvaluation]] = None
    pa_required: bool
    coverage_status: str = "covered"
    evaluation_summary: str


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
def evaluate_case(
    case_id: str,
    combined_text: str,
    payer_name: str,
    cpt_code: str,
    context: Dict[str, Any]
):

    # STEP 1 — Coverage
    coverage = check_coverage(
        payer_name,
        cpt_code,
        context=context
    )

    coverage_status = coverage.get("coverage_status", "covered").lower()

    if coverage_status != "covered":
        update_case_status(case_id, f"not_covered:{coverage_status}")
        return [], [], False, coverage_status

    # STEP 2 — PA requirement
    pa_required = check_pa_required(
        payer_name,
        cpt_code,
        context=context
    )

    if not pa_required:
        update_case_status(case_id, "no_pa_required")
        return [], [], False, "covered"

    # STEP 3 — Medical necessity
    all_clauses = load_clauses(payer_name, cpt_code)

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

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            status=clause_status,
            policy_text=(clause_def.get("policy_text") or "").strip(),
            missing_concepts=missing,
            matched_concepts=matched_concepts,
            evidence_category=evidence_category
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    satisfied = any(c.status == "satisfied" for c in approval_results)
    update_case_status(case_id, "approved" if satisfied else "denied")

    return approval_results, exclusion_results, pa_required, "covered"

def build_evidence_review(all_clauses: List[ClauseEvaluation]):

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
                "has_exclusion": False
            }

        categories[category]["clauses"].append({
            "clause_id": clause.clause_id,
            "policy_text": clause.policy_text,
            "status": clause.status
        })

        # Track clause results
        if clause.clause_type == "approval":
            if clause.status == "satisfied":
                categories[category]["has_approval"] = True
            elif clause.status == "partial":
                categories[category]["has_partial"] = True

        if clause.clause_type == "exclusion" and clause.status == "satisfied":
            categories[category]["has_exclusion"] = True


    # Determine final category status
    results = []

    for category_data in categories.values():

        if category_data["has_exclusion"]:
            category_data["status"] = "Insufficient"

        elif category_data["has_approval"]:
            category_data["status"] = "Sufficient"

        elif category_data["has_partial"]:
            category_data["status"] = "Partially sufficient"

        else:
            category_data["status"] = "Insufficient"

        # Remove internal flags
        category_data.pop("has_approval")
        category_data.pop("has_partial")
        category_data.pop("has_exclusion")

        results.append(category_data)

    return results


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


@app.post("/run_case/{case_id}")
async def run_case(case_id: str):

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

        approval_results, exclusion_results, pa_required, coverage_status = evaluate_case(
            case_id,
            combined_text,
            payer_name,
            case["cpt_code"],
            context
        )

        # 🔥 Filter exclusions: only include if any are triggered
        filtered_exclusion_results = [
            c for c in exclusion_results if c.status or c.matched_concepts
        ]

        all_results = approval_results + filtered_exclusion_results
        evidence_review = build_evidence_review(all_results)

    except Exception:
        update_case_status(case_id, "error")
        raise

    case_summary = generate_case_summary(
        case_id=case_id,
        payer_name=payer_name,
        cpt_code=case["cpt_code"],
        pa_required=pa_required,
        coverage_status=coverage_status,
        approval_results=approval_results,
        exclusion_results=filtered_exclusion_results,
        evidence_review=evidence_review
    )

    # 🔥 Only include `exclusion_clauses` if any remain
    response_data = {
        "case_id": case_id,
        "evidence_review": evidence_review,
        "approval_clauses": approval_results,
        "pa_required": pa_required,
        "coverage_status": coverage_status,
        "evaluation_summary": case_summary
    }
    # Only include exclusions if any evidence exists    
    if filtered_exclusion_results:
        response_data["exclusion_clauses"] = filtered_exclusion_results

    return ApprovalResponse(**response_data)


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
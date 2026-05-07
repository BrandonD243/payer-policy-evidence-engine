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


def generate_case_summary(
    case_id: str,
    payer_name: str,
    cpt_code: str,
    pa_required: bool,
    coverage_status: str,
    approval_results: List[ClauseEvaluation],
    exclusion_results: List[ClauseEvaluation],
    evidence_review: List[Dict[str, Any]],
    decision_logic: Dict[str, Any]
) -> str:
    approval_statuses = [
        f"{c.clause_id}: {c.status}" for c in approval_results
    ]
    exclusion_statuses = [
        f"{c.clause_id}: {c.status}" for c in exclusion_results
    ]
    categories = [
        f"{cat['evidence_category']}: {cat['status']}" for cat in evidence_review
    ]

    # Determine if exclusions caused denial
    exclusions_triggered = any(c.status == "satisfied" for c in exclusion_results) and decision_logic.get("exclusions_override", True)

    prompt = f"""
You are a clinical review assistant.

Summarize the overall approval decision for a prior authorization case using the policy clause evaluation results.

Provide a concise summary in 1-2 sentences. Clearly state if the case is approved, denied, or partially supported. If denied, specify the reason (e.g., due to exclusion clauses, insufficient approval clauses, or coverage issues). Mention key satisfied clauses and evidence categories that influenced the decision.

Decision Logic: Approval requires {decision_logic.get('approvals', 'ANY')} approval clauses to be satisfied. Exclusions override approval if {decision_logic.get('exclusions_override', True)}.

Case ID: {case_id}
Payer: {payer_name}
CPT code: {cpt_code}
Coverage status: {coverage_status}
Prior authorization required: {pa_required}
Approval clause results: {', '.join(approval_statuses) if approval_statuses else 'none'}
Exclusion clause results: {', '.join(exclusion_statuses) if exclusion_statuses else 'none'}
Evidence categories: {', '.join(categories) if categories else 'none'}
Exclusions triggered denial: {exclusions_triggered}

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
        # Fallback logic using decision_logic
        satisfied_approvals = sum(1 for c in approval_results if c.status == "satisfied")
        total_approvals = len(approval_results)
        satisfied_exclusions = sum(1 for c in exclusion_results if c.status == "satisfied")

        if decision_logic.get("exclusions_override", True) and satisfied_exclusions > 0:
            return "The case is denied due to satisfied exclusion criteria."
        
        if decision_logic.get("approvals", "ANY") == "ALL":
            if satisfied_approvals == total_approvals and total_approvals > 0:
                return "The case is approved as all required approval clauses are satisfied."
            else:
                return f"The case is not approved as {total_approvals - satisfied_approvals} of {total_approvals} approval clauses remain unsatisfied."
        else:  # ANY
            if satisfied_approvals > 0:
                return f"The case is approved as {satisfied_approvals} of {total_approvals} approval clauses are satisfied."
            else:
                return "The case is not approved as no approval clauses are satisfied."


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


def format_evaluation_rule(decision_logic: Dict[str, Any] | None) -> Dict[str, str]:
    approvals = (decision_logic or {}).get("approvals", "ANY").upper()
    display = (
        "All policy criteria must be satisfied for approval."
        if approvals == "ALL"
        else "At least one qualifying medical necessity criterion must be satisfied."
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
        return f"{satisfied_approvals} of {total_approvals} required criteria satisfied"
    return (
        f"{satisfied_approvals} qualifying criterion satisfied"
        if satisfied_approvals > 0
        else "no qualifying criterion satisfied"
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

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            status=clause_status,
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
    satisfied_exclusions = [c for c in exclusion_results if c.status == "satisfied"]

    # Determine approval and exclusion status
    if decision_logic.get("approvals", "ANY") == "ALL":
        approval_satisfied = len(satisfied_approvals) == len(approval_results) and len(approval_results) > 0
    else:  # ANY
        approval_satisfied = len(satisfied_approvals) > 0

    exclusions_triggered = decision_logic.get("exclusions_override", True) and bool(satisfied_exclusions)

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

    return approval_results, exclusion_results, pa_required, coverage_status, decision_logic

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
                "has_exclusion": False,
                "has_partial_exclusion": False,
                "raw_clauses": []  # Store for filtering
            }

        # Store raw clause
        categories[category]["raw_clauses"].append(clause)

        # Track clause results for status determination
        if clause.clause_type == "approval":
            if clause.status == "satisfied":
                categories[category]["has_approval"] = True
            elif clause.status == "partial":
                categories[category]["has_partial"] = True

        if clause.clause_type == "exclusion":
            if clause.status == "satisfied":
                categories[category]["has_exclusion"] = True
            elif clause.status == "partial":
                categories[category]["has_partial_exclusion"] = True

    any_satisfied_exclusions = any(
        c.status == "satisfied" and c.clause_type == "exclusion" 
        for c in all_clauses
    )

    # Determine final category status and filter clauses
    results = []

    for category_data in categories.values():

        if category_data["has_exclusion"]:
            category_data["status"] = "Insufficient"

        elif category_data["has_approval"]:
            category_data["status"] = "Sufficient"

        elif category_data["has_partial"] or category_data.get("has_partial_exclusion", False):
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
                # Only show satisfied clauses in categories that have them
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

        # Always show exclusion clauses
        filtered_clauses.extend([{
            "clause_id": c.clause_id,
            "policy_text": c.policy_text,
            "status": c.status
        } for c in exclusion_clauses])

        # Only include category if it has clauses to show
        if filtered_clauses:
            category_data["clauses"] = filtered_clauses

            # Remove internal tracking flags
            category_data.pop("has_approval")
            category_data.pop("has_partial")
            category_data.pop("has_exclusion")
            category_data.pop("has_partial_exclusion")

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

        approval_results, exclusion_results, pa_required, coverage_status, decision_logic = evaluate_case(
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
        evidence_review = build_evidence_review(all_results, decision_logic)

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
        evidence_review=evidence_review,
        decision_logic=decision_logic
    )

    # 🔥 Only include `exclusion_clauses` if any remain
    evaluation_rule = format_evaluation_rule(decision_logic)
    evaluation_result_summary = format_evaluation_result_summary(
        decision_logic,
        satisfied_approvals=len([c for c in approval_results if c.status == "satisfied"]),
        total_approvals=len(approval_results),
    )

    response_data = {
        "case_id": case_id,
        "evidence_review": evidence_review,
        "approval_clauses": approval_results,
        "pa_required": pa_required,
        "coverage_status": coverage_status,
        "evaluation_rule": evaluation_rule,
        "evaluation_result_summary": evaluation_result_summary,
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

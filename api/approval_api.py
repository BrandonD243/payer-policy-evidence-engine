import os
import yaml
import random
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from extractors.openai_concept_extractor import extract_concepts_from_text
from extractors.section_extractor import extract_concepts_from_sections
from extractors.confidence_normalizer import normalize_confidence

from evaluators.clause_evaluator import evaluate_clauses
from resolvers.concept_resolver import infer_derived_concepts
from examples.case_loader import load_case_documents



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
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

PRELOADED_CASES = {
    "patient_01_approved_stenosis": os.path.join(
        EXAMPLES_DIR,
        "patient_01_approved_stenosis"
    ),
    "patient_03_denied_bonemri": os.path.join(
        EXAMPLES_DIR,
        "patient_03_denied_bonemri"
    ),
}


CONCEPT_PATH = os.path.join(BASE_DIR, "concepts", "concept_registry.yaml")

# Load concept registry
with open(CONCEPT_PATH, "r") as f:
    concept_registry = yaml.safe_load(f).get("concepts", {})

# Build relevant concepts list for extraction
RELEVANT_CONCEPTS = [
    {"concept_id": cid, **data}
    for cid, data in concept_registry.items()
]

# ===============================
# Load Clauses
# ===============================
def load_clauses(payer_name: str):
    approval_path = os.path.join(BASE_DIR, "payers", payer_name, "72148", "approval_clauses.yaml")
    exclusion_path = os.path.join(BASE_DIR, "payers", payer_name, "72148", "exclusion_clauses.yaml")
    recommendation_path = os.path.join(BASE_DIR, "payers", payer_name, "72148", "recommendation_registry.yaml")

    if not os.path.exists(approval_path) or not os.path.exists(exclusion_path):
        raise FileNotFoundError(f"Clauses not found for payer {payer_name}")

    with open(approval_path, "r") as f:
        approval_clauses = yaml.safe_load(f).get("approval_clauses", [])
    with open(exclusion_path, "r") as f:
        exclusion_clauses = yaml.safe_load(f).get("exclusion_clauses", [])
    with open(recommendation_path, "r") as f:
        rec_yaml = yaml.safe_load(f)

    # Tag clause types
    for c in approval_clauses:
        c["clause_type"] = "approval"
    for c in exclusion_clauses:
        c["clause_type"] = "exclusion"

    all_clauses = approval_clauses + exclusion_clauses

    recommendation_bank = [
        {
            "id": rec_id,
            "title": rec_id.replace("_", " ").title(),
            "description": rec.get("description", "").strip(),
            "related_clause": rec.get("linked_clause", ""),
            "category": rec.get("category", "")
        }
        for rec_id, rec in rec_yaml.get("recommendations", {}).items()
    ]

    return all_clauses, approval_clauses, exclusion_clauses, recommendation_bank

# ===============================
# Helpers
# ===============================
def extract_evidence_from_file(text: str, relevant_concepts: List[dict]):
    mentions = extract_concepts_from_sections(
        text,
        relevant_concepts=relevant_concepts,
        extractor_fn=extract_concepts_from_text
    )

    # normalize confidence AFTER extraction
    mentions = normalize_confidence(mentions)

    # normalize keys
    for m in mentions:
        if "text_span" in m:
            m["evidence_text"] = m.pop("text_span")

        if "concept_id" not in m:
            m["concept_id"] = "unknown"

    return mentions


# ===============================
# FastAPI App
# ===============================
app = FastAPI(
    title="Payer Policy Evidence Engine",
    version="1.0",
    description="Semantic clause evaluation for CPT 72148"
)



# ===============================
# API Models
# ===============================
class Evidence(BaseModel):
    concept_id: str
    confidence: str
    certainty_level: str | None
    evidence_text: str
    section: str

class ClauseEvaluation(BaseModel):
    clause_id: str
    clause_type: str
    satisfied: bool
    evidence: List[Evidence]

class Recommendation(BaseModel):
    id: str
    title: str
    description: str
    related_clause: str
    category: str

class ApprovalResponse(BaseModel):
    case_id: str
    approval_clauses: List[ClauseEvaluation]
    exclusion_clauses: List[ClauseEvaluation]
    recommendations: List[Recommendation] = []

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

# ===============================
# Analyze Files Endpoint
# ===============================

@app.on_event("startup")
def preload_example_cases():
    print("Preloading example cases...")

    for case_name, case_path in PRELOADED_CASES.items():
        if not os.path.exists(case_path):
            print(f"Skipping missing example folder: {case_path}")
            continue

        case = create_case(
            patient_name=case_name.replace("_", " ").title(),
            payer="aetna",
            cpt_code="72148"
        )

        combined_text = load_case_documents(case_path)

        save_document(
            case_id=case["case_id"],
            filename=f"{case_name}.txt",
            text=combined_text
        )

        print(f"Loaded example case: {case_name}")

    print("Example preload complete.")


@app.post("/analyze_files")
async def analyze_files(files: List[UploadFile] = File(...), policy_name: str = "aetna"):
    payer_name = policy_name.lower()
    try:
        all_clauses, approval_clauses, exclusion_clauses, recommendation_bank = load_clauses(payer_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Create a new case
    case = create_case(
        patient_name="Unknown",
        payer=payer_name,
        cpt_code="72148"
    )

    # Extract evidence
    extracted_evidence = []
    for file in files:
        content = await file.read()
        evidence = extract_evidence_from_file(
            content.decode("utf-8"),
            relevant_concepts=RELEVANT_CONCEPTS
        )
        extracted_evidence.extend(evidence)

    # Evaluate clauses
    clause_results = evaluate_clauses(
        extracted_evidence,
        all_clauses,
        concept_registry
    )

    def wrap_evidence(evidence_list):
        return [
            Evidence(
                concept_id=ev.get("concept_id", ""),
                confidence=ev.get("confidence", "weak"),
                certainty_level=ev.get("certainty_level"),
                evidence_text=ev.get("evidence_text", ""),
                section=ev.get("section", "")
            )
            for ev in evidence_list
        ]

    approval_results, exclusion_results = [], []
    clause_lookup = {c["id"]: c for c in all_clauses}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type", "approval")

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            satisfied=result.get("satisfied", False),
            evidence=wrap_evidence(result.get("evidence", []))
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    satisfied_approvals = [c for c in approval_results if c.satisfied]

    recommendations: List[Recommendation] = []
    if not satisfied_approvals and recommendation_bank:
        sampled = random.sample(recommendation_bank, k=min(5, len(recommendation_bank)))
        recommendations = [Recommendation(**rec) for rec in sampled]

    update_case_status(
        case["case_id"],
        "approved" if satisfied_approvals else "denied"
    )

    return ApprovalResponse(
        case_id=case["case_id"],
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results,
        recommendations=recommendations
    )

@app.post("/run_case/{case_id}")
async def run_case(case_id: str, policy_name: str = "aetna"):
    payer_name = policy_name.lower()

    try:
        all_clauses, approval_clauses, exclusion_clauses, recommendation_bank = load_clauses(payer_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    documents = get_documents_for_case(case_id)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents attached to case")

    # Combine stored document text
    combined_text = "\n\n".join([doc["text"] for doc in documents])

    extracted_evidence = extract_evidence_from_file(
    combined_text,
    relevant_concepts=RELEVANT_CONCEPTS
)

    print("---- Extracted concept IDs ----")
    print([e["concept_id"] for e in extracted_evidence])
    print("--------------------------------")


    clause_results = evaluate_clauses(
        extracted_evidence,
        all_clauses,
        concept_registry
    )

    def wrap_evidence(evidence_list):
        return [
            Evidence(
                concept_id=ev.get("concept_id", ""),
                confidence=ev.get("confidence", "weak"),
                certainty_level=ev.get("certainty_level"),
                evidence_text=ev.get("evidence_text", ""),
                section=ev.get("section", "")
            )
            for ev in evidence_list
        ]

    approval_results, exclusion_results = [], []
    clause_lookup = {c["id"]: c for c in all_clauses}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type", "approval")

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            satisfied=result.get("satisfied", False),
            evidence=wrap_evidence(result.get("evidence", []))
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    satisfied_approvals = [c for c in approval_results if c.satisfied]

    recommendations: List[Recommendation] = []
    if not satisfied_approvals and recommendation_bank:
        sampled = random.sample(recommendation_bank, k=min(5, len(recommendation_bank)))
        recommendations = [Recommendation(**rec) for rec in sampled]

    update_case_status(
        case_id,
        "approved" if satisfied_approvals else "denied"
    )

    return ApprovalResponse(
        case_id=case_id,
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results,
        recommendations=recommendations
    )

@app.get("/payers")
async def list_payers():
    payers_dir = os.path.join(BASE_DIR, "payers")

    if not os.path.exists(payers_dir):
        return []

    payers = []

    for name in os.listdir(payers_dir):
        full_path = os.path.join(payers_dir, name)
        if os.path.isdir(full_path):
            payers.append({
                "id": name,
                "display_name": name.replace("_", " ").title()
            })

    return sorted(payers, key=lambda x: x["display_name"])


# ===============================
# Case Endpoints
# ===============================
@app.get("/cases", response_model=List[CaseSummary])
async def list_cases():
    return get_all_cases()

@app.get("/cases/{case_id}", response_model=CaseDetailResponse)
async def get_case_detail(case_id: str):
    case = get_case(case_id)
    documents = get_documents_for_case(case_id)
    return CaseDetailResponse(case=case, documents=documents)

@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_endpoint(document_id: str):
    return get_document(document_id)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

import os
import yaml
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from extractors.openai_concept_extractor import extract_concepts_from_text
from extractors.section_extractor import extract_concepts_from_sections

from evaluators.clause_evaluator import evaluate_clauses
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
        EXAMPLES_DIR, "patient_01_approved_stenosis"
    ),
    "patient_03_denied_bonemri": os.path.join(
        EXAMPLES_DIR, "patient_03_denied_bonemri"
    ),
}

CONCEPT_PATH = os.path.join(BASE_DIR, "concepts", "concept_registry.yaml")

with open(CONCEPT_PATH, "r") as f:
    concept_yaml = yaml.safe_load(f)

# Canonical registry used by evaluators
concept_registry = concept_yaml["concepts"]

# Extractor-ready list
RELEVANT_CONCEPTS = [
    {"concept_id": cid, **data}
    for cid, data in concept_registry.items()
]



# ===============================
# Load Clauses
# ===============================
def load_clauses(payer_name: str):
    base = os.path.join(BASE_DIR, "payers", payer_name, "72148")

    approval_path = os.path.join(base, "approval_clauses.yaml")
    exclusion_path = os.path.join(base, "exclusion_clauses.yaml")

    if not os.path.exists(approval_path) or not os.path.exists(exclusion_path):
        raise FileNotFoundError(f"Clauses not found for payer {payer_name}")

    with open(approval_path, "r") as f:
        approval_clauses = yaml.safe_load(f).get("approval_clauses", [])

    with open(exclusion_path, "r") as f:
        exclusion_clauses = yaml.safe_load(f).get("exclusion_clauses", [])

    for c in approval_clauses:
        c["clause_type"] = "approval"

    for c in exclusion_clauses:
        c["clause_type"] = "exclusion"

    all_clauses = approval_clauses + exclusion_clauses

    return all_clauses


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
    matched_ids = {m.get("concept_id") for m in matched}
    return [c for c in required if c not in matched_ids]


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


# ===============================
# FastAPI App
# ===============================
app = FastAPI(
    title="Payer Policy Evidence Engine",
    version="2.0",
    description="Semantic clause evaluation for CPT 72148"
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
    status: bool
    policy_text: str
    missing_concepts: List[str]
    matched_concepts: List[ConceptMatch]

class ApprovalResponse(BaseModel):
    case_id: str
    approval_clauses: List[ClauseEvaluation]
    exclusion_clauses: List[ClauseEvaluation]


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
# Startup: Preload Examples
# ===============================
@app.on_event("startup")
def preload_example_cases():
    print("Preloading example cases...")

    for case_name, case_path in PRELOADED_CASES.items():
        if not os.path.exists(case_path):
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


# ===============================
# Core Evaluation Logic
# ===============================
def evaluate_case(case_id: str, combined_text: str, payer_name: str):

    all_clauses = load_clauses(payer_name)

    extracted_evidence = extract_evidence_from_file(combined_text)

    clause_results = evaluate_clauses(
        extracted_evidence,
        all_clauses,
        concept_registry
    )

    approval_results = []
    exclusion_results = []

    clause_lookup = {c["id"]: c for c in all_clauses}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type", "approval")

        matched = result.get("evidence", [])
        missing = compute_missing_concepts(clause_def, matched)

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            status=result.get("satisfied", False),
            policy_text=(clause_def.get("policy_text") or "").strip(),
            missing_concepts=missing,
            matched_concepts=wrap_matches(matched)
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    satisfied = any(c.status for c in approval_results)
    update_case_status(case_id, "approved" if satisfied else "denied")

    return approval_results, exclusion_results



# ===============================
# Endpoints
# ===============================
@app.post("/analyze_files")
async def analyze_files(files: List[UploadFile] = File(...), policy_name: str = "aetna"):

    payer_name = policy_name.lower()

    case = create_case(
        patient_name="Unknown",
        payer=payer_name,
        cpt_code="72148"
    )

    extracted_texts = []

    for file in files:
        content = await file.read()
        extracted_texts.append(content.decode("utf-8"))

    combined_text = "\n\n".join(extracted_texts)

    approval_results, exclusion_results= evaluate_case(
        case["case_id"],
        combined_text,
        payer_name
    )

    return ApprovalResponse(
        case_id=case["case_id"],
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results,
    )


@app.post("/run_case/{case_id}")
async def run_case(case_id: str, policy_name: str = "aetna"):

    payer_name = policy_name.lower()

    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    documents = get_documents_for_case(case_id)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents attached")

    combined_text = "\n\n".join([doc["text"] for doc in documents])

    approval_results, exclusion_results = evaluate_case(
        case_id,
        combined_text,
        payer_name
    )

    return ApprovalResponse(
        case_id=case_id,
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results,
    )


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


@app.get("/health")
async def health_check():
    return {"status": "ok"}

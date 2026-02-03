# approval_api.py
import os
import yaml
import random
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from extractors.openai_concept_extractor import extract_concepts_from_text
from extractors.section_extractor import extract_concepts_from_sections
from evaluators.clause_evaluator import evaluate_clauses
from resolvers.concept_resolver import infer_composite_concepts

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
# Load Registries
# ===============================
CLAUSE_PATH = os.path.join("payers", "aetna", "72148", "clause_registry.yaml")
CONCEPT_PATH = os.path.join("concepts", "concept_registry.yaml")
RECOMMENDATION_PATH = os.path.join(
    "payers", "aetna", "72148", "recommendation_registry.yaml"
)

with open(CLAUSE_PATH, "r") as f:
    clause_registry_raw = yaml.safe_load(f)

approval_clauses = clause_registry_raw.get("approval_clauses", [])
exclusion_clauses = clause_registry_raw.get("exclusion_clauses", [])

for c in approval_clauses:
    c["clause_type"] = "approval"
for c in exclusion_clauses:
    c["clause_type"] = "exclusion"

ALL_CLAUSES = approval_clauses + exclusion_clauses

with open(CONCEPT_PATH, "r") as f:
    concept_registry = yaml.safe_load(f).get("concepts", {})

with open(RECOMMENDATION_PATH, "r") as f:
    rec_yaml = yaml.safe_load(f)

recommendation_bank = [
    {
        "id": rec_id,
        "title": rec_id.replace("_", " ").title(),
        "description": rec.get("description", ""),
        "related_clause": rec.get("linked_clause", ""),
        "category": rec.get("category", "")
    }
    for rec_id, rec in rec_yaml.get("recommendations", {}).items()
]

RELEVANT_CONCEPTS = [
    {"concept_id": cid, **data}
    for cid, data in concept_registry.items()
]

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
# Analyze Multiple Files → One Case
# ===============================
@app.post("/analyze_files", response_model=ApprovalResponse)
async def analyze_files(files: List[UploadFile] = File(...)):
    """
    Analyze multiple clinical documents as ONE case.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # 1️⃣ Create case
    case = create_case(
        patient_name="Unknown",
        payer="Aetna",
        cpt_code="72148"
    )

    all_text_chunks: List[str] = []

    # 2️⃣ Persist documents + aggregate text
    for file in files:
        try:
            text = (await file.read()).decode("utf-8")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read {file.filename}: {e}"
            )

        if not text.strip():
            continue

        save_document(
            filename=file.filename,
            text=text,
            case_id=case["case_id"]
        )

        all_text_chunks.append(
            f"\n\n=== DOCUMENT: {file.filename} ===\n\n{text}"
        )

    if not all_text_chunks:
        raise HTTPException(status_code=400, detail="All uploaded documents were empty")

    combined_text = "\n".join(all_text_chunks)

    # 3️⃣ Concept extraction (case-level)
    concept_mentions = extract_concepts_from_sections(
        document_text=combined_text,
        relevant_concepts=RELEVANT_CONCEPTS,
        extractor_fn=extract_concepts_from_text
    )

    derived_concepts = infer_composite_concepts(concept_mentions)
    concept_mentions.extend(derived_concepts)

    # 4️⃣ Clause evaluation
    clause_results = evaluate_clauses(
        concept_mentions=concept_mentions,
        clause_registry=ALL_CLAUSES
    )

    # 5️⃣ Wrap results
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

    approval_results = []
    exclusion_results = []
    clause_lookup = {c["id"]: c for c in ALL_CLAUSES}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type")

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

    # 6️⃣ Recommendation logic (ONLY if zero approvals satisfied)
    satisfied_approvals = [c for c in approval_results if c.satisfied]
    recommendations: List[Recommendation] = []

    if not satisfied_approvals and recommendation_bank:
        sampled = random.sample(
            recommendation_bank,
            k=min(5, len(recommendation_bank))
        )
        recommendations = [Recommendation(**rec) for rec in sampled]

    # 7️⃣ Update case status
    update_case_status(
        case["case_id"],
        "approved" if satisfied_approvals else "denied"
    )

    # 8️⃣ Return response
    return ApprovalResponse(
        case_id=case["case_id"],
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results,
        recommendations=recommendations
    )

# ===============================
# Case Endpoints
# ===============================
@app.get("/cases", response_model=List[CaseSummary])
async def list_cases():
    return get_all_cases()

@app.get("/cases/{case_id}", response_model=CaseDetailResponse)
async def get_case_detail(case_id: str):
    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    documents = get_documents_for_case(case_id)

    return CaseDetailResponse(
        case=case,
        documents=[
            DocumentSummary(
                document_id=d["document_id"],
                filename=d["filename"],
                created_at=d["created_at"]
            )
            for d in documents
        ]
    )

# ===============================
# Document Endpoint
# ===============================
@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_endpoint(document_id: str):
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailResponse(**doc)

# ===============================
# Health Check
# ===============================
@app.get("/health")
async def health_check():
    return {"status": "ok"}

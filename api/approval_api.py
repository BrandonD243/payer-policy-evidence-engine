import os
import yaml
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from extractors.openai_concept_extractor import extract_concepts_from_text
from evaluators.clause_evaluator import evaluate_clauses


# ===============================
# Load Registries
# ===============================

CLAUSE_PATH = os.path.join("payers", "aetna", "72148", "clause_registry.yaml")
CONCEPT_PATH = os.path.join("concepts", "concept_registry.yaml")

try:
    with open(CLAUSE_PATH, "r") as f:
        clause_registry_raw = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError(f"Clause registry not found at {CLAUSE_PATH}")

approval_clauses = clause_registry_raw.get("approval_clauses", [])
exclusion_clauses = clause_registry_raw.get("exclusion_clauses", [])

ALL_CLAUSES = approval_clauses + exclusion_clauses

try:
    with open(CONCEPT_PATH, "r") as f:
        concept_registry = yaml.safe_load(f).get("concepts", {})
except FileNotFoundError:
    raise RuntimeError(f"Concept registry not found at {CONCEPT_PATH}")

# Convert concept registry to extractor-friendly format
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
    description="Semantic clause evaluation for CPT 72148 (MRI Lumbar Spine)"
)


# ===============================
# API Models
# ===============================

class DocumentRequest(BaseModel):
    patient_id: str
    text: str


class ClauseEvaluation(BaseModel):
    clause_id: str
    clause_type: str
    satisfied: bool
    matched_concepts: List[str]
    confidence_levels: List[str]


class ApprovalResponse(BaseModel):
    patient_id: str
    approval_clauses: List[ClauseEvaluation]
    exclusion_clauses: List[ClauseEvaluation]


# ===============================
# API Endpoint
# ===============================

@app.post("/analyze_document", response_model=ApprovalResponse)
async def analyze_document(request: DocumentRequest):
    """
    Analyze a clinical document and determine which payer policy
    approval or exclusion clauses are satisfied.
    """

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text is empty")

    # 1️⃣ Extract semantic concepts using OpenAI
    try:
        concept_mentions = extract_concepts_from_text(
            document_text=request.text,
            relevant_concepts=RELEVANT_CONCEPTS
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Concept extraction failed: {str(e)}"
        )

    # 2️⃣ Evaluate all clauses
    clause_results = evaluate_clauses(
        concept_mentions=concept_mentions,
        clause_registry=ALL_CLAUSES
    )

    # 3️⃣ Split approval vs exclusion
    approval_results = []
    exclusion_results = []

    clause_lookup = {c["id"]: c for c in ALL_CLAUSES}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type", "exclusion")

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            satisfied=result["satisfied"],
            matched_concepts=result["matched_concepts"],
            confidence_levels=result["confidence_levels"]
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    return ApprovalResponse(
        patient_id=request.patient_id,
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results
    )


# ===============================
# Health Check
# ===============================

@app.get("/health")
async def health_check():
    return {"status": "ok"}

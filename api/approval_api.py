# approval_api.py
import os
import yaml
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import random

from extractors.openai_concept_extractor import extract_concepts_from_text
from extractors.section_extractor import extract_concepts_from_sections
from evaluators.clause_evaluator import evaluate_clauses
from resolvers.concept_resolver import infer_composite_concepts

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

# 🔒 Stamp clause_type explicitly
for c in approval_clauses:
    c["clause_type"] = "approval"

for c in exclusion_clauses:
    c["clause_type"] = "exclusion"

ALL_CLAUSES = approval_clauses + exclusion_clauses

try:
    with open(CONCEPT_PATH, "r") as f:
        concept_registry = yaml.safe_load(f).get("concepts", {})
except FileNotFoundError:
    raise RuntimeError(f"Concept registry not found at {CONCEPT_PATH}")

RECOMMENDATION_PATH = os.path.join(
    "payers", "aetna", "72148", "recommendation_registry.yaml"
)

with open(RECOMMENDATION_PATH, "r") as f:
    rec_yaml = yaml.safe_load(f)

recommendation_bank = []

for rec_id, rec in rec_yaml.get("recommendations", {}).items():
    recommendation_bank.append({
        "id": rec_id,
        "title": rec_id.replace("_", " ").title(),
        "description": rec.get("description", ""),
        "related_clause": rec.get("linked_clause", ""),
        "category": rec.get("category", "")
    })


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
    patient_id: str
    approval_clauses: List[ClauseEvaluation]
    exclusion_clauses: List[ClauseEvaluation]
    recommendations: List[Recommendation] = []

# ===============================
# API Endpoint
# ===============================
@app.post("/analyze_file", response_model=ApprovalResponse)
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded clinical document (.txt) and determine
    which payer policy approval or exclusion clauses are satisfied.
    """

    # 1️⃣ Read the uploaded file as text
    try:
        text = (await file.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded document is empty")

    # 2️⃣ Extract concepts using OpenAI + Section-aware extraction
    try:
        concept_mentions = extract_concepts_from_sections(
            document_text=text,
            relevant_concepts=RELEVANT_CONCEPTS,
            extractor_fn=extract_concepts_from_text
        )

        # 🔁 Derive higher-level composite concepts
        derived_concepts = infer_composite_concepts(concept_mentions)
        concept_mentions.extend(derived_concepts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Concept extraction failed: {e}")

    # 🔍 DEBUG: Inspect extracted concepts
    print("\n=== CONCEPT MENTIONS ===")
    for m in concept_mentions:
        print(
            m["concept_id"],
            m.get("confidence"),
            m.get("certainty_level"),
            "|",
            m.get("evidence_text"),
            "| section:",
            m.get("section")
        )
    print("=======================\n")

    # 3️⃣ Evaluate all clauses
    try:
        clause_results = evaluate_clauses(
            concept_mentions=concept_mentions,
            clause_registry=ALL_CLAUSES
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clause evaluation failed: {e}")

    # 4️⃣ Convert evidence to Pydantic Evidence objects
    def wrap_evidence(evidence_list):
        wrapped = []
        for ev in evidence_list:
            wrapped.append(
                Evidence(
                    concept_id=ev.get("concept_id", ""),
                    confidence=ev.get("confidence", "weak"),
                    certainty_level=ev.get("certainty_level"),
                    evidence_text=ev.get("evidence_text", ""),
                    section=ev.get("section", "")
                )
            )
        return wrapped

    # 5️⃣ Split results into approval vs exclusion clauses
    approval_results = []
    exclusion_results = []
    clause_lookup = {c["id"]: c for c in ALL_CLAUSES}

    for result in clause_results:
        clause_def = clause_lookup.get(result["clause_id"], {})
        clause_type = clause_def.get("clause_type")


        evidence_wrapped = wrap_evidence(result.get("evidence", []))

        enriched = ClauseEvaluation(
            clause_id=result["clause_id"],
            clause_type=clause_type,
            satisfied=result.get("satisfied", False),
            evidence=evidence_wrapped
        )

        if clause_type == "exclusion":
            exclusion_results.append(enriched)
        else:
            approval_results.append(enriched)

    # 6️⃣ Recommendation logic
    satisfied_approvals = [
        c for c in approval_results if c.satisfied
    ]

    recommendations_out = []

    if len(satisfied_approvals) == 0 and recommendation_bank:
        sampled = random.sample(
            recommendation_bank,
            k=min(5, len(recommendation_bank))
        )

        for rec in sampled:
            recommendations_out.append(
                Recommendation(
                    id=rec["id"],
                    title=rec["title"],
                    description=rec["description"],
                    related_clause=rec.get("related_clause"),
                    category=rec.get("category")
                )
            )
    

    # 6️⃣ Return structured response
    return ApprovalResponse(
        patient_id=file.filename,
        approval_clauses=approval_results,
        exclusion_clauses=exclusion_results,
        recommendations=recommendations_out
    )


# ===============================
# Health Check
# ===============================
@app.get("/health")
async def health_check():
    return {"status": "ok"}

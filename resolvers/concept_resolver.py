from typing import List, Dict

# Composite concept definitions
COMPOSITE_CONCEPTS = [
    {
        "concept_id": "objective_neurologic_deficit",
        "required_atomic": {"motor_weakness", "decreased_reflexes", "sensory_deficit"},
        "confidence": "strong",
        "certainty_level": "confirmed",
        "section_hint": "PHYSICAL EXAM",
        "evidence_text_template": "Objective neurologic deficit inferred from exam findings"
    },
    # future derived concepts go here
]

def infer_composite_concepts(concept_mentions: List[Dict]) -> List[Dict]:
    """
    Converts atomic concept mentions into higher-level composite concepts
    based on rules defined in COMPOSITE_CONCEPTS.
    """
    inferred = []
    extracted_ids = {m["concept_id"] for m in concept_mentions}

    for composite in COMPOSITE_CONCEPTS:
        if extracted_ids & composite["required_atomic"]:
            inferred.append({
                "concept_id": composite["concept_id"],
                "confidence": composite["confidence"],
                "certainty_level": composite["certainty_level"],
                "evidence_text": composite["evidence_text_template"],
                "section": composite["section_hint"]
            })

    return inferred

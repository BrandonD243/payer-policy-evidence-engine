from typing import List, Dict


DERIVED_CONCEPTS = [
    {
        "concept_id": "objective_neurologic_deficit",
        "derived_from": [
            "motor_weakness",
            "decreased_reflexes",
            "sensory_deficit",
        ],
        "logic": "any_of",
        "confidence": "strong",
        "certainty_level": "confirmed",
        "section_hint": "PHYSICAL EXAM",
        "evidence_text_template": "Objective neurologic deficit inferred from exam findings",
    },
]


def load_derived_rules(concept_registry: Dict) -> List[Dict]:
    """
    Load derived concept rules directly from registry.
    """
    rules = []

    for cid, concept in concept_registry.items():
        if concept.get("type") == "derived" and "derived_from" in concept:
            rules.append({
                "concept_id": cid,
                "derived_from": concept["derived_from"],
                "logic": concept.get("logic", "any_of"),
                "confidence": "strong",
                "certainty_level": "confirmed",
                "section_hint": "INFERRED"
            })

    return rules


def infer_derived_concepts(
    concept_mentions: List[Dict],
    concept_registry: Dict
) -> List[Dict]:
    """
    Infers derived concepts from registry-defined rules.
    """

    derived_rules = load_derived_rules(concept_registry)

    inferred = []
    extracted_ids = {m["concept_id"] for m in concept_mentions}

    for rule in derived_rules:
        if rule["concept_id"] in extracted_ids:
            continue

        required = set(rule["derived_from"])
        triggered = False

        if rule["logic"] == "any_of":
            triggered = bool(extracted_ids & required)
        elif rule["logic"] == "all_of":
            triggered = required.issubset(extracted_ids)

        if triggered:
            inferred.append({
                "concept_id": rule["concept_id"],
                "confidence": rule["confidence"],
                "certainty_level": rule["certainty_level"],
                "evidence_text": "Derived from clinical findings",
                "section": rule["section_hint"],
            })

    return concept_mentions + inferred
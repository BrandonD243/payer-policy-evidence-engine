from typing import List, Dict


def normalize_confidence(confidence):
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if isinstance(confidence, str):
        key = confidence.strip().lower()
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


def load_derived_rules(concept_registry: Dict) -> List[Dict]:
    """
    Load derived concept rules from the concept registry.
    """

    rules = []

    for cid, concept in concept_registry.items():

        if concept.get("type") != "derived":
            continue

        derived_from = concept.get("derived_from")
        if not derived_from:
            continue

        rules.append({
            "concept_id": cid,
            "derived_from": derived_from,
            "logic": concept.get("logic", "all_of"),
            "confidence": normalize_confidence(concept.get("confidence", "strong")),
            "certainty_level": concept.get("certainty_level", "confirmed"),
            "section_hint": "INFERRED"
        })

    return rules


def infer_derived_concepts(
    concept_mentions: List[Dict],
    concept_registry: Dict
) -> List[Dict]:
    """
    Infers derived concepts using iterative rule evaluation
    until no new concepts can be inferred.
    """

    rules = load_derived_rules(concept_registry)

    # Current known concepts
    extracted_ids = {m["concept_id"] for m in concept_mentions}

    inferred_mentions = []

    changed = True

    while changed:
        changed = False

        for rule in rules:

            concept_id = rule["concept_id"]

            # Skip if already present
            if concept_id in extracted_ids:
                continue

            required = set(rule["derived_from"])

            triggered = False

            if rule["logic"] == "any_of":
                triggered = bool(extracted_ids & required)

            elif rule["logic"] == "all_of":
                triggered = required.issubset(extracted_ids)

            if triggered:

                new_concept = {
                    "concept_id": concept_id,
                    "confidence": rule["confidence"],
                    "certainty_level": rule["certainty_level"],
                    "evidence_text": f"Derived from: {', '.join(required)}",
                    "section": rule["section_hint"],
                }

                inferred_mentions.append(new_concept)
                extracted_ids.add(concept_id)

                changed = True

    return concept_mentions + inferred_mentions
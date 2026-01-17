from typing import List, Dict


CONFIDENCE_ORDER = {
    "weak": 1,
    "moderate": 2,
    "strong": 3
}


def evaluate_clauses(
    concept_mentions: List[Dict],
    clause_registry: List[Dict]
) -> List[Dict]:
    """
    Evaluates approval and exclusion clauses against extracted concept mentions.

    Args:
        concept_mentions: Output from OpenAI extractor
        clause_registry: All clauses (approval + exclusion)

    Returns:
        [
            {
                "clause_id": str,
                "satisfied": bool,
                "matched_concepts": List[str],
                "confidence_levels": List[str]
            }
        ]
    """

    results = []

    # Index mentions by concept_id
    mention_lookup = {
        m["concept_id"]: m for m in concept_mentions
    }

    for clause in clause_registry:
        clause_id = clause.get("id")

        required_concepts = clause.get("required_concepts", [])
        exclusion_concepts = clause.get("exclusion_concepts", [])

        min_confidence = clause.get("minimum_confidence", "weak")
        expected_certainty = clause.get("expected_certainty")

        min_conf_val = CONFIDENCE_ORDER.get(min_confidence, 1)

        matched_concepts = []
        confidence_levels = []

        # ===============================
        # EXCLUSION CLAUSE LOGIC
        # ===============================
        if exclusion_concepts:
            for concept_id in exclusion_concepts:
                mention = mention_lookup.get(concept_id)
                if mention:
                    matched_concepts.append(concept_id)
                    confidence_levels.append(mention.get("confidence", "weak"))

            satisfied = len(matched_concepts) > 0

            results.append({
                "clause_id": clause_id,
                "satisfied": satisfied,
                "matched_concepts": matched_concepts,
                "confidence_levels": confidence_levels
            })
            continue

        # ===============================
        # APPROVAL CLAUSE LOGIC
        # ===============================
        for concept_id in required_concepts:
            mention = mention_lookup.get(concept_id)
            if not mention:
                continue

            # Confidence check
            conf = mention.get("confidence", "weak")
            if CONFIDENCE_ORDER.get(conf, 1) < min_conf_val:
                continue

            # Certainty check (if defined)
            if expected_certainty:
                if mention.get("certainty_level") != expected_certainty:
                    continue

            matched_concepts.append(concept_id)
            confidence_levels.append(conf)

        satisfied = len(matched_concepts) == len(required_concepts)

        results.append({
            "clause_id": clause_id,
            "satisfied": satisfied,
            "matched_concepts": matched_concepts,
            "confidence_levels": confidence_levels
        })

    return results

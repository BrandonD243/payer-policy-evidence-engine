from typing import List, Dict
import re

# Confidence hierarchy
CONFIDENCE_ORDER = {
    "weak": 1,
    "moderate": 2,
    "strong": 3
}

# Example: therapy concepts that require duration matching
THERAPY_CONCEPTS = {
    "conservative_therapy_6_weeks": {
        "therapy_indicators": [
            "physical therapy",
            "NSAIDs",
            "NSAID therapy",
            "analgesics",
            "muscle relaxants"
        ],
        "duration_indicators": [
            "six weeks",
            "6 weeks",
            "more than one month"
        ]
    },
    "conservative_therapy_4_weeks": {
        "therapy_indicators": [
            "physical therapy",
            "NSAIDs",
            "NSAID therapy",
            "analgesics",
            "muscle relaxants"
        ],
        "duration_indicators": [
            "four weeks",
            "4 weeks",
            "one month"
        ]
    }
}

def find_therapy_concepts(concept_mentions: List[Dict]) -> List[Dict]:
    """
    Scans existing concept mentions for therapy-duration indicators and
    adds corresponding therapy concepts if matched.
    """
    # Collect new concept mentions
    new_mentions = []

    for mention in concept_mentions:
        text = mention.get("evidence_text", "").lower()

        for concept_id, data in THERAPY_CONCEPTS.items():
            # Skip if already present
            if any(m["concept_id"] == concept_id for m in concept_mentions + new_mentions):
                continue

            # Check if any therapy indicator matches (partial, case-insensitive)
            therapy_match = any(re.search(re.escape(indicator.lower()), text) for indicator in data["therapy_indicators"])
            duration_match = any(re.search(re.escape(dur.lower()), text) for dur in data["duration_indicators"])

            if therapy_match and duration_match:
                # Create a new concept mention for this therapy concept
                new_mentions.append({
                    "concept_id": concept_id,
                    "confidence": "strong",
                    "certainty_level": "confirmed",
                    "evidence_text": mention["evidence_text"],
                    "section": mention.get("section", "GENERAL")
                })

    return concept_mentions + new_mentions


def evaluate_clauses(
    concept_mentions: List[Dict],
    clause_registry: List[Dict]
) -> List[Dict]:
    """
    Evaluates which payer policy clauses are satisfied given extracted concept mentions.
    Supports both approval clauses and exclusion clauses.
    Approval clauses can specify logic: AND (default) or OR.
    """

    # Step 0: enrich concept_mentions with therapy-duration concepts
    concept_mentions = find_therapy_concepts(concept_mentions)

    results = []

    # Build a lookup of concept mentions by concept_id
    mention_lookup = {m["concept_id"]: m for m in concept_mentions}

    for clause in clause_registry:
        clause_id = clause.get("id")
        required_concepts = clause.get("required_concepts", [])
        exclusion_concepts = clause.get("exclusion_concepts", [])
        min_confidence = clause.get("minimum_confidence", "weak")
        expected_certainty = clause.get("expected_certainty")
        min_conf_val = CONFIDENCE_ORDER[min_confidence]

        # ===============================
        # EXCLUSION CLAUSES
        # ===============================
        if exclusion_concepts:
            matched_evidence = []

            for cid in exclusion_concepts:
                mention = mention_lookup.get(cid)
                if mention:
                    matched_evidence.append({
                        "concept_id": cid,
                        "confidence": mention["confidence"],
                        "certainty_level": mention.get("certainty_level"),
                        "evidence_text": mention.get("evidence_text"),
                        "section": mention.get("section")
                    })

            results.append({
                "clause_id": clause_id,
                "satisfied": len(matched_evidence) > 0,
                "evidence": matched_evidence
            })
            continue

        # ===============================
        # APPROVAL CLAUSES
        # ===============================
        matched_evidence = []
        logic = clause.get("logic", "AND").upper()  # default AND if not specified

        for cid in required_concepts:
            mention = mention_lookup.get(cid)
            if not mention:
                continue  # keep going; OR logic might still succeed

            if CONFIDENCE_ORDER[mention["confidence"]] < min_conf_val:
                continue

            if expected_certainty and mention.get("certainty_level") != expected_certainty:
                continue

            matched_evidence.append({
                "concept_id": cid,
                "confidence": mention["confidence"],
                "certainty_level": mention.get("certainty_level"),
                "evidence_text": mention.get("evidence_text"),
                "section": mention.get("section")
            })

        # Determine satisfaction based on logic
        if logic == "OR":
            satisfied = len(matched_evidence) > 0
        else:  # AND
            satisfied = len(matched_evidence) == len(required_concepts)

        results.append({
            "clause_id": clause_id,
            "satisfied": satisfied,
            "evidence": matched_evidence
        })

    return results

from collections import defaultdict
from typing import List, Dict
import re

CONFIDENCE_ORDER = {
    "weak": 1,
    "moderate": 2,
    "strong": 3
}

# ---------------------------
# Therapy concept derivation
# ---------------------------
def load_therapy_concepts(concept_registry: Dict) -> Dict:
    """Load concepts that have therapy/duration indicators."""
    return {
        cid: c
        for cid, c in concept_registry.items()
        if "therapy_indicators" in c and "duration_indicators" in c
    }

def find_therapy_concepts(
    concept_mentions: List[Dict],
    concept_registry: Dict
) -> List[Dict]:
    """Add derived therapy-duration concepts if both therapy & duration indicators are present."""
    therapy_concepts = load_therapy_concepts(concept_registry)
    new_mentions = []

    for mention in concept_mentions:
        text = mention.get("evidence_text", "").lower()
        for concept_id, data in therapy_concepts.items():
            # Skip if already added
            if any(m.get("concept_id") == concept_id for m in concept_mentions + new_mentions):
                continue

            therapy_match = any(
                re.search(re.escape(t.lower()), text)
                for t in data["therapy_indicators"]
            )
            duration_match = any(
                re.search(re.escape(d.lower()), text)
                for d in data["duration_indicators"]
            )

            if therapy_match and duration_match:
                print("\n==== DERIVED THERAPY CONCEPT ====")
                print("Derived concept_id:", concept_id)
                print("From evidence_text:", mention.get("evidence_text", ""))
                print("Section:", mention.get("section", "GENERAL"))
                print("therapy_match:", therapy_match)
                print("duration_match:", duration_match)
                print("=================================\n")

                new_mentions.append({
                    "concept_id": concept_id,
                    "confidence": "strong",
                    "certainty_level": "confirmed",
                    "evidence_text": mention.get("evidence_text", ""),
                    "section": mention.get("section", "GENERAL")
                })


    return concept_mentions + new_mentions

# ---------------------------
# Clause evaluation
# ---------------------------
def evaluate_clauses(
    concept_mentions: List[Dict],
    clause_registry: List[Dict],
    concept_registry: Dict
) -> List[Dict]:
    """Evaluates each clause against concept mentions."""

    # Step 1: derive therapy-duration concepts
    concept_mentions = find_therapy_concepts(concept_mentions, concept_registry)

    # Step 2: build lookup by concept_id
    mention_lookup = defaultdict(list)
    for m in concept_mentions:
        if "concept_id" in m:
            mention_lookup[m["concept_id"]].append(m)

    results = []

    # Step 3: evaluate each clause
    for clause in clause_registry:
        clause_id = clause.get("id", "unknown_clause")
        required_concepts = clause.get("required_concepts", [])
        exclusion_concepts = clause.get("exclusion_concepts", [])
        min_confidence = clause.get("minimum_confidence", "weak")
        expected_certainty = clause.get("expected_certainty")
        logic = clause.get("logic", "AND").upper()
        min_conf_val = CONFIDENCE_ORDER.get(min_confidence, 1)

        # Handle exclusions first
        if exclusion_concepts:
            matched = []
            for cid in exclusion_concepts:
                for m in mention_lookup.get(cid, []):
                    if CONFIDENCE_ORDER.get(m.get("confidence", "weak"), 1) >= min_conf_val:
                        if not expected_certainty or m.get("certainty_level") == expected_certainty:
                            matched.append(m)
            results.append({
                "clause_id": clause_id,
                "satisfied": len(matched) > 0,
                "evidence": matched
            })
            continue

        # -------------------------
        # Approvals / required concepts
        # -------------------------
        clause_matched = []

        # Evaluate each required concept individually
        concept_satisfaction = []
        for cid in required_concepts:
            valid_mentions = [
                m for m in mention_lookup.get(cid, [])
                if CONFIDENCE_ORDER.get(m.get("confidence", "weak"), 1) >= min_conf_val
                and (not expected_certainty or m.get("certainty_level") == expected_certainty)
            ]
            clause_matched.extend(valid_mentions)
            concept_satisfaction.append(len(valid_mentions) > 0)

        # Determine clause satisfaction based on logic
        if logic == "AND":
            satisfied = all(concept_satisfaction)
        else:  # OR
            satisfied = any(concept_satisfaction)

        results.append({
            "clause_id": clause_id,
            "satisfied": satisfied,
            "evidence": clause_matched
        })

    return results

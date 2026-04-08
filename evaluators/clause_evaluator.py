from collections import defaultdict
from typing import List, Dict
import re
import string

# ---------------------------
# Helper
# ---------------------------
def as_float(value, default=0.0):
    """Safely convert confidence values to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

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

            text_norm = text.translate(str.maketrans("", "", string.punctuation))
            therapy_match = any(
                re.search(re.escape(t.lower().translate(str.maketrans("", "", string.punctuation))), text_norm)
                for t in data["therapy_indicators"]
            )
            duration_match = any(
                re.search(re.escape(d.lower().translate(str.maketrans("", "", string.punctuation))), text_norm)
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
                    "confidence": as_float(mention.get("confidence", 0.0)),  # enforce float
                    "certainty_level": "confirmed",
                    "evidence_text": mention.get("evidence_text", ""),
                    "section": mention.get("section", "GENERAL")
                })

    return concept_mentions + new_mentions

# ---------------------------
# Clause evaluation
# ---------------------------
def evaluate_clauses(concept_mentions, clause_registry, concept_registry):
    # derive therapy concepts
    concept_mentions = find_therapy_concepts(concept_mentions, concept_registry)

    # build lookup
    mention_lookup = defaultdict(list)
    for m in concept_mentions:
        if "concept_id" in m:
            mention_lookup[m["concept_id"]].append(m)

    results = []

    for clause in clause_registry:
        clause_id = clause.get("clause_id") or clause.get("id") or "unknown_clause"
        required_concepts = clause.get("required_concepts", [])
        exclusion_concepts = clause.get("exclusion_concepts", [])
        min_confidence = as_float(clause.get("minimum_confidence", 0.0))  # ensure float
        logic = clause.get("logic", "AND").upper()

        # -------------------
        # Handle exclusions
        # -------------------
        if exclusion_concepts:
            matched = []
            for cid in exclusion_concepts:
                for m in mention_lookup.get(cid, []):
                    conf = as_float(m.get("confidence", 0.0))
                    if conf >= min_confidence:
                        matched.append(m)

            results.append({
                "clause_id": clause_id,
                "satisfied": len(matched) > 0,
                "evidence": matched
            })
            continue

        # -------------------
        # Handle approvals
        # -------------------
        clause_matched = []
        concept_satisfaction = []

        for cid in required_concepts:
            valid_mentions = []
            for m in mention_lookup.get(cid, []):
                conf = as_float(m.get("confidence", 0.0))
                if conf >= min_confidence:
                    valid_mentions.append(m)

            clause_matched.extend(valid_mentions)
            concept_satisfaction.append(len(valid_mentions) > 0)

        satisfied = all(concept_satisfaction) if logic == "AND" else any(concept_satisfaction)

        results.append({
            "clause_id": clause_id,
            "satisfied": satisfied,
            "evidence": clause_matched
        })

    return results

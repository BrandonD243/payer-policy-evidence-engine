import re
from typing import List, Dict

STRONG_PATTERNS = [
    r"\bconfirmed\b",
    r"\bdiagnosed\b",
    r"\bpositive\b",
    r"\bcompleted\b",
    r"\bobjective\b",
    r"\bmri shows\b",
]

MODERATE_PATTERNS = [
    r"\bconsistent with\b",
    r"\bsuggests\b",
    r"\blikely\b",
]

WEAK_PATTERNS = [
    r"\bpossible\b",
    r"\bmay\b",
    r"\bconsider\b",
    r"\bplanned\b",
    r"\brequest(ed)?\b",
]

def detect_confidence(text: str) -> str:
    text = text.lower()

    for pattern in STRONG_PATTERNS:
        if re.search(pattern, text):
            return "strong"

    for pattern in MODERATE_PATTERNS:
        if re.search(pattern, text):
            return "moderate"

    for pattern in WEAK_PATTERNS:
        if re.search(pattern, text):
            return "weak"

    return "moderate"


def normalize_confidence(concept_mentions: List[Dict]) -> List[Dict]:
    """
    Adjust confidence based on evidence wording.
    """
    for m in concept_mentions:
        evidence_text = m.get("evidence_text", "")
        rule_conf = detect_confidence(evidence_text)

        # Keep strongest value between LLM and rule
        current = m.get("confidence", "weak")

        order = {"weak": 1, "moderate": 2, "strong": 3}
        if order[rule_conf] > order.get(current, 1):
            m["confidence"] = rule_conf

    return concept_mentions

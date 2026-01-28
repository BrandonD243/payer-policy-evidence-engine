import re
from collections import defaultdict

# ===============================
# Helper: text normalization
# ===============================
def normalize_text(text: str) -> str:
    """
    Normalize text for better concept matching.
    - Lowercase everything
    - Replace common synonyms
    - Strip extra whitespace
    """
    text = text.lower()
    # Synonyms or mapping
    text = text.replace("lumbar", "back")
    text = text.replace("thoracic", "back")
    text = text.replace("lumbosacral", "back")
    return text.strip()


# ===============================
# Split document into sections
# ===============================
def split_into_sections(document_text: str):
    """
    Split a clinical note into sections using headers like '=====' or common clinical headings.
    Returns a list of (section_title, section_text) tuples.
    """
    pattern = re.compile(r"^(?P<header>[A-Z _]{3,})\n[-=]{3,}\n", re.MULTILINE)
    sections = []
    last_idx = 0
    last_header = "GENERAL"

    for match in pattern.finditer(document_text):
        start = match.start()
        if start > 0:
            section_text = document_text[last_idx:start].strip()
            sections.append((last_header, section_text))
        last_header = match.group("header").strip()
        last_idx = match.end()

    # Add remaining text
    sections.append((last_header, document_text[last_idx:].strip()))
    return sections


# ===============================
# Confidence ranking
# ===============================
def confidence_rank(level: str) -> int:
    """Assign numeric ranks to confidence levels for easy comparison."""
    rank_map = {"weak": 1, "moderate": 2, "strong": 3}
    return rank_map.get(level.lower(), 0)


# ===============================
# Section-aware concept extraction
# ===============================
def extract_concepts_from_sections(document_text: str, relevant_concepts: list, extractor_fn):
    """
    Run concept extraction on each section separately, normalize text, and aggregate results.
    - extractor_fn: a function like extract_concepts_from_text(text, relevant_concepts)
    Returns a deduplicated list of all concept mentions with max confidence per concept.
    """
    sections = split_into_sections(document_text)
    aggregated = defaultdict(lambda: {
        "concept_id": "",
        "confidence": "",
        "certainty_level": "",
        "evidence_text": "",
        "section": ""
    })


    for header, text in sections:
        if not text:
            continue
        # Normalize section text before extraction
        concepts = extractor_fn(text, relevant_concepts)

        for c in concepts:
            cid = c["concept_id"]
            current_conf = aggregated[cid]["confidence"]

            if confidence_rank(c["confidence"]) > confidence_rank(current_conf):
                aggregated[cid] = {
                    **c,
                    "section": header
                }


    return list(aggregated.values())

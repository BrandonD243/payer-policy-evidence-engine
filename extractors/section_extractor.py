from typing import List, Dict
import re
from extractors.openai_concept_extractor import extract_concepts_from_text


def normalize_section(section_name: str) -> str:
    """
    Normalize any section name generically:
    - strip whitespace
    - remove special characters
    - collapse spaces
    - Title Case (capitalized words)
    - no underscores
    """

    if not section_name:
        return "General"

    cleaned = section_name.strip()

    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r"[^\w\s]", "", cleaned)

    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Title case
    cleaned = cleaned.title()

    return cleaned if cleaned else "General"



def split_into_sections(document_text: str) -> List[Dict]:
    """
    Splits document using this structure:

    ===============================
    SECTION NAME
    ===============================

    Works for ANY section name.
    """

    pattern = re.compile(
        r"=+\n"              # opening =====
        r"([^\n]+)\n"        # section name
        r"=+\n"              # closing =====
        r"(.*?)"             # section body
        r"(?=\n=+\n|\Z)",    # stop at next section or end
        re.DOTALL
    )

    sections = []

    for match in pattern.finditer(document_text):
        raw_section_name = match.group(1)
        section_body = match.group(2).strip()

        normalized_name = normalize_section(raw_section_name)

        sections.append({
            "section": normalized_name,
            "text": section_body
        })

    # Fallback if no structured sections found
    if not sections:
        sections.append({
            "section": "general",
            "text": document_text.strip()
        })

    return sections


def extract_concepts_from_sections(
    document_text: str,
    relevant_concepts: List[Dict],
    extractor_fn=extract_concepts_from_text
) -> List[Dict]:
    """
    Extract concepts per section.
    Deterministically assigns section names.
    Fully dynamic — no hardcoded section list.
    """

    sections = split_into_sections(document_text)
    mentions: List[Dict] = []

    for section in sections:
        section_name = section["section"]
        section_text = section["text"]

        if not section_text.strip():
            continue

        section_mentions = extractor_fn(section_text, relevant_concepts)

        for m in section_mentions:
            # Force deterministic section assignment
            m["section"] = section_name

        mentions.extend(section_mentions)

    return mentions

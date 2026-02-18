from typing import List, Dict
from extractors.openai_concept_extractor import extract_concepts_from_text
import re

def normalize_section(section_text: str) -> str:
    """
    Simplify section headers to just the name in lowercase.
    """
    if not section_text:
        return "general"
    
    # Remove lines of =====, --- or whitespace, keep only text lines
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    lines = [line for line in lines if not re.fullmatch(r"[=\-]{3,}", line)]
    
    if not lines:
        return "general"
    
    # Take the first meaningful line as section name
    return lines[0].lower()

def extract_concepts_from_sections(
    document_text: str,
    relevant_concepts: List[Dict],
    extractor_fn=extract_concepts_from_text
) -> List[Dict]:
    """
    Splits a document into sections (by headings or newlines)
    and extracts concept mentions for each section.
    """

    # Split the document into chunks/sections
    sections = [s.strip() for s in document_text.split("\n\n") if s.strip()]
    mentions: List[Dict] = []

    for section_text in sections:
        section_mentions = extractor_fn(section_text, relevant_concepts)

        for m in section_mentions:
            # If extractor already set a section, normalize it
            if m.get("section"):
                m["section"] = normalize_section(m["section"])
            else:
                # Do NOT infer from paragraph text
                m["section"] = "general"

        mentions.extend(section_mentions)

    return mentions

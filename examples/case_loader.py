from pathlib import Path

SECTION_ORDER = [
    "PATIENT_DEMOGRAPHICS",
    "HISTORY",
    "CHIEF_COMPLAINT",
    "HISTORY_OF_PRESENT_ILLNESS",
    "PAST_MEDICAL_HISTORY",
    "CONSERVATIVE_MANAGEMENT",
    "PHYSICAL_EXAM",
    "IMAGING",
    "REQUEST",
    "ASSESSMENT",
    "PLAN",
]

def load_case_documents(case_dir: str) -> str:
    """
    Loads TXT files in clinical-document order and combines them
    into a single document string for the LLM.

    Assumes each TXT file already contains its own section header.
    """

    case_path = Path(case_dir)
    sections = []

    for section in SECTION_ORDER:
        file_path = case_path / f"{section}.txt"
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8").strip()
            sections.append(text)

    return "\n\n".join(sections)

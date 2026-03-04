# storage/case_store.py
import uuid
from datetime import datetime

# Simple in-memory store (swap for DB later)
_CASES = {}

def create_case(
    patient_name: str,
    payer: str,
    cpt_code: str,
    status: str = "created",
):
    case_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    case = {
        "case_id": case_id,
        "patient_name": patient_name,
        "payer": payer,
        "cpt_code": cpt_code,
        "status": status,
        "created_at": now,
    }

    _CASES[case_id] = case
    return case

def update_case_status(case_id: str, status: str):
    if case_id in _CASES:
        _CASES[case_id]["status"] = status
    else:
        raise KeyError(f"Case {case_id} not found")

def get_all_cases():
    """
    Returns all cases for dashboard listing
    """
    return list(_CASES.values())

def get_case(case_id: str):
    """
    Returns a single case or None
    """
    return _CASES.get(case_id)

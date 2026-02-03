# storage/document_store.py
import uuid
from datetime import datetime

_DOCUMENTS = []

def save_document(filename: str, text: str, case_id: str):
    doc = {
        "document_id": str(uuid.uuid4()),
        "filename": filename,
        "text": text,
        "case_id": case_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    _DOCUMENTS.append(doc)
    return doc

def get_documents_for_case(case_id: str):
    return [d for d in _DOCUMENTS if d["case_id"] == case_id]
    
def get_document(document_id: str):
    for doc in _DOCUMENTS:
        if doc["document_id"] == document_id:
            return doc
    return None


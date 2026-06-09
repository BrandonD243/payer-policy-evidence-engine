from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PriorAuthCase:
    case_id: str
    patient_name: str
    payer: str
    cpt_code: str
    provider_name: Optional[str] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, case: Dict[str, Any]) -> "PriorAuthCase":
        return cls(
            case_id=str(case.get("case_id") or ""),
            patient_name=str(case.get("patient_name") or "Unknown"),
            payer=str(case.get("payer") or "unknown"),
            cpt_code=str(case.get("cpt_code") or ""),
            provider_name=case.get("provider_name"),
            status=case.get("status"),
            metadata={
                key: value
                for key, value in case.items()
                if key not in {"case_id", "patient_name", "payer", "cpt_code", "provider_name", "status"}
            },
        )


@dataclass(frozen=True)
class PriorAuthArtifact:
    filename: str
    content: bytes
    mime_type: str = "application/pdf"


@dataclass
class SubmissionResult:
    method: str
    status: str
    case: PriorAuthCase
    message: Optional[str] = None
    artifacts: List[PriorAuthArtifact] = field(default_factory=list)
    prepared_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

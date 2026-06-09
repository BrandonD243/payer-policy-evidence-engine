from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def patient_display_name(patient: Dict[str, Any]) -> str:
    names = patient.get("name") or []
    if not names:
        return "Unknown"
    name = names[0]
    given = " ".join(name.get("given") or [])
    family = name.get("family") or ""
    full = " ".join(part for part in [given, family] if part).strip()
    return full or name.get("text") or "Unknown"


def patient_demographics(patient: Dict[str, Any], patient_id: Optional[str] = None) -> Dict[str, str]:
    demographics = {
        "id": patient_id or patient.get("id", ""),
        "name": patient_display_name(patient),
        "birthDate": patient.get("birthDate", ""),
        "gender": patient.get("gender", ""),
    }

    address = patient.get("address") or []
    if address:
        first_address = address[0]
        line = ", ".join(first_address.get("line") or [])
        city_state_zip = " ".join(
            part
            for part in [
                first_address.get("city", ""),
                first_address.get("state", ""),
                first_address.get("postalCode", ""),
            ]
            if part
        )
        demographics["address"] = ", ".join(part for part in [line, city_state_zip] if part)

    telecom = patient.get("telecom") or []
    for item in telecom:
        system = item.get("system")
        value = item.get("value")
        if system and value and system not in demographics:
            demographics[system] = value

    return {key: value for key, value in demographics.items() if value}


def coverage_payer_display(coverages: List[Dict[str, Any]]) -> Optional[str]:
    for coverage in coverages:
        for payor in coverage.get("payor") or []:
            display = payor.get("display")
            if display:
                return display
    return None


def resources_to_document_text(resource_type: str, resources: List[Dict[str, Any]]) -> str:
    lines = [f"FHIR {resource_type} Summary"]
    if not resources:
        lines.append("No resources returned.")
        return "\n".join(lines)

    for resource in resources:
        formatter = RESOURCE_FORMATTERS.get(resource_type, format_generic_resource)
        lines.extend(["", formatter(resource)])
    return "\n".join(line for line in lines if line is not None)


def patient_to_document_text(patient: Dict[str, Any]) -> str:
    lines = [
        "FHIR Patient Summary",
        f"Name: {patient_display_name(patient)}",
    ]
    if patient.get("birthDate"):
        lines.append(f"Date of birth: {patient['birthDate']}")
    if patient.get("gender"):
        lines.append(f"Gender: {patient['gender']}")
    return "\n".join(lines)


def format_condition(resource: Dict[str, Any]) -> str:
    return "\n".join([
        f"Diagnosis: {code_display(resource.get('code'))}",
        f"Clinical status: {code_display(resource.get('clinicalStatus'))}",
        f"Verification status: {code_display(resource.get('verificationStatus'))}",
        f"Recorded date: {resource.get('recordedDate', '')}",
    ])


def format_procedure(resource: Dict[str, Any]) -> str:
    return "\n".join([
        f"Procedure: {code_display(resource.get('code'))}",
        f"Status: {resource.get('status', '')}",
        f"Performed: {resource.get('performedDateTime') or resource.get('performedPeriod', '')}",
        f"Outcome: {code_display(resource.get('outcome'))}",
    ])


def format_observation(resource: Dict[str, Any]) -> str:
    value = (
        resource.get("valueString")
        or resource.get("valueQuantity", {}).get("value")
        or code_display(resource.get("valueCodeableConcept"))
        or resource.get("valueInteger")
        or resource.get("valueBoolean")
        or ""
    )
    unit = resource.get("valueQuantity", {}).get("unit", "")
    return "\n".join([
        f"Observation: {code_display(resource.get('code'))}",
        f"Status: {resource.get('status', '')}",
        f"Value: {value} {unit}".strip(),
        f"Effective: {resource.get('effectiveDateTime') or resource.get('effectivePeriod', '')}",
    ])


def format_diagnostic_report(resource: Dict[str, Any]) -> str:
    lines = [
        f"Diagnostic report: {code_display(resource.get('code'))}",
        f"Status: {resource.get('status', '')}",
        f"Effective: {resource.get('effectiveDateTime') or resource.get('effectivePeriod', '')}",
    ]
    conclusion = resource.get("conclusion")
    if conclusion:
        lines.append(f"Conclusion: {conclusion}")
    return "\n".join(lines)


def format_document_reference(resource: Dict[str, Any]) -> str:
    lines = [
        f"Document: {resource.get('description') or code_display(resource.get('type'))}",
        f"Status: {resource.get('status', '')}",
        f"Date: {resource.get('date', '')}",
    ]
    for content in resource.get("content") or []:
        attachment = content.get("attachment", {})
        if attachment.get("title"):
            lines.append(f"Attachment title: {attachment['title']}")
        if attachment.get("contentType"):
            lines.append(f"Attachment type: {attachment['contentType']}")
    return "\n".join(lines)


def format_medication_request(resource: Dict[str, Any]) -> str:
    medication = resource.get("medicationCodeableConcept") or {}
    return "\n".join([
        f"Medication request: {code_display(medication)}",
        f"Status: {resource.get('status', '')}",
        f"Intent: {resource.get('intent', '')}",
        f"Authored on: {resource.get('authoredOn', '')}",
    ])


def format_generic_resource(resource: Dict[str, Any]) -> str:
    resource_type = resource.get("resourceType", "Resource")
    resource_id = resource.get("id", "")
    return f"{resource_type}: {resource_id}"


def code_display(codeable: Optional[Dict[str, Any]]) -> str:
    if not codeable:
        return ""
    if codeable.get("text"):
        return codeable["text"]
    coding = codeable.get("coding") or []
    if coding:
        first = coding[0]
        display = first.get("display") or first.get("code") or ""
        system = first.get("system")
        return f"{display} ({system})" if system and display else display
    return ""


RESOURCE_FORMATTERS = {
    "Condition": format_condition,
    "Procedure": format_procedure,
    "Observation": format_observation,
    "DiagnosticReport": format_diagnostic_report,
    "DocumentReference": format_document_reference,
    "MedicationRequest": format_medication_request,
}

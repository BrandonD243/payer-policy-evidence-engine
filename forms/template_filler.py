from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import re

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib import colors


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "templates"


def get_single_pdf_template() -> Path:
    templates = sorted(TEMPLATE_DIR.glob("*.pdf"))
    if not templates:
        raise FileNotFoundError("No PDF template found in templates/")
    if len(templates) > 1:
        raise ValueError("Expected exactly one PDF template in templates/")
    return templates[0]


def fill_prior_auth_template(case: Dict[str, Any]) -> bytes:
    template_path = get_single_pdf_template()
    reader = PdfReader(str(template_path))
    if reader.get_fields():
        return fill_acroform_template(reader, case)

    return fill_flat_template(reader, case)


def fill_acroform_template(reader: PdfReader, case: Dict[str, Any]) -> bytes:
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.set_need_appearances_writer(True)

    field_values = build_field_values(writer.get_fields() or {}, case)
    for page in writer.pages:
        writer.update_page_form_field_values(page, field_values, auto_regenerate=False)

    overlay_reader = PdfReader(BytesIO(build_field_overlay_pdf(reader, field_values)))
    for index, overlay_page in enumerate(overlay_reader.pages):
        if index < len(writer.pages):
            writer.pages[index].merge_page(overlay_page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def build_field_values(fields: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    demographics = case.get("fhir_patient") or {}
    patient_name = format_member_name(case.get("patient_name") or "Unknown")
    cpt_code = str(case.get("cpt_code") or "").strip()
    payer = str(case.get("payer") or "").strip()
    birth_date = format_pdf_date(demographics.get("birthDate") or case.get("date_of_birth"))
    patient_id = str(demographics.get("id") or case.get("member_id") or "").strip()
    gender = str(demographics.get("gender") or case.get("gender") or "").strip().lower()
    address = str(demographics.get("address") or case.get("address") or "").strip()
    phone = str(demographics.get("phone") or case.get("phone") or "").strip()

    if "Name" in fields:
        values["Name"] = patient_name
    if "Date of Request" in fields:
        values["Date of Request"] = datetime.utcnow().strftime("%m/%d/%Y")
    if "Date of Birth" in fields and birth_date:
        values["Date of Birth"] = birth_date
    if "ID Number" in fields and patient_id:
        values["ID Number"] = patient_id
    if "Address" in fields and address:
        values["Address"] = address
    if "Telephone" in fields and phone:
        values["Telephone"] = phone
    if "Procedure Item Requested CPTHCPCS CodesRow1" in fields:
        values["Procedure Item Requested CPTHCPCS CodesRow1"] = cpt_code
    if "Check Box1" in fields and gender in {"female", "male"}:
        values["Check Box1"] = "/F" if gender == "female" else "/M"
    if "Check Box2" in fields:
        values["Check Box2"] = "/Outpatient"

    payer_field = find_first_matching_field(fields, ["payer", "health plan", "insurance", "plan name"])
    if payer_field:
        values[payer_field] = payer

    return values


def find_first_matching_field(fields: Dict[str, Any], needles: list[str]) -> str | None:
    for field_name, field in fields.items():
        haystack = f"{field_name} {field.get('/TU', '')}".lower()
        if any(needle in haystack for needle in needles):
            return field_name
    return None


def build_field_overlay_pdf(reader: PdfReader, field_values: Dict[str, str]) -> bytes:
    packet = BytesIO()
    first_page = reader.pages[0]
    default_width = float(first_page.mediabox.width)
    default_height = float(first_page.mediabox.height)
    c = canvas.Canvas(packet, pagesize=(default_width, default_height))
    c.setFillColor(colors.HexColor("#183153"))

    for page_index, page in enumerate(reader.pages):
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        if page_index > 0:
            c.setPageSize((page_width, page_height))
        for annot_ref in page.get("/Annots", []):
            annotation = annot_ref.get_object()
            field_name = annotation.get("/T")
            parent = annotation.get("/Parent")
            if parent and not field_name:
                field_name = parent.get_object().get("/T")
            if field_name not in field_values:
                continue
            if str(field_values[field_name]).startswith("/"):
                continue
            rect = annotation.get("/Rect")
            if not rect:
                continue
            x0, y0, x1, y1 = [float(value) for value in rect]
            draw_fitted_text(
                c,
                field_values[field_name],
                x=x0 + 3,
                y=y0 + 3,
                max_width=max(x1 - x0 - 6, 20),
                font_size=8.5,
            )
        c.showPage()

    c.save()
    packet.seek(0)
    return packet.getvalue()


def fill_flat_template(reader: PdfReader, case: Dict[str, Any]) -> bytes:
    writer = PdfWriter()

    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)

    overlay = build_overlay_pdf(
        case=case,
        page_width=page_width,
        page_height=page_height,
    )
    overlay_reader = PdfReader(BytesIO(overlay))
    first_page.merge_page(overlay_reader.pages[0])
    writer.add_page(first_page)

    for page in reader.pages[1:]:
        writer.add_page(page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def build_overlay_pdf(case: Dict[str, Any], page_width: float, page_height: float) -> bytes:
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_width, page_height))
    c.setFillColor(colors.HexColor("#183153"))
    c.setFont("Helvetica-Bold", 9)

    patient_name = format_member_name(case.get("patient_name") or "Unknown")
    cpt_code = case.get("cpt_code") or ""
    payer = case.get("payer") or ""

    # Coordinates are for the current flat PDF template in templates/.
    draw_fitted_text(c, patient_name, x=74, y=608, max_width=275, font_size=9)
    draw_fitted_text(c, cpt_code, x=221, y=82, max_width=31, font_size=8)
    draw_fitted_text(c, f"Payer: {payer}", x=72, y=82, max_width=136, font_size=8)

    c.save()
    packet.seek(0)
    return packet.getvalue()


def draw_fitted_text(
    c: canvas.Canvas,
    text: str,
    *,
    x: float,
    y: float,
    max_width: float,
    font_size: float,
) -> None:
    value = str(text or "").strip()
    if not value:
        return

    size = font_size
    while size > 6 and c.stringWidth(value, "Helvetica-Bold", size) > max_width:
        size -= 0.5
    c.setFont("Helvetica-Bold", size)
    c.drawString(x, y, value)


def format_member_name(raw_name: str) -> str:
    value = re.sub(r"\s+", " ", str(raw_name or "")).strip()
    if not value:
        return ""
    if "," in value:
        return value

    parts = value.split(" ")
    if len(parts) < 2:
        return value

    first = parts[0]
    last = parts[-1]
    middle = " ".join(part[0].upper() + "." for part in parts[1:-1] if part)
    if middle:
        return f"{last}, {first} {middle}"
    return f"{last}, {first}"


def format_pdf_date(raw_date: Any) -> str:
    value = str(raw_date or "").strip()
    if not value:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%m/%d/%Y")
        except ValueError:
            continue
    return value

from __future__ import annotations

from datetime import datetime, UTC
from io import BytesIO
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PAGE_WIDTH, PAGE_HEIGHT = letter
BRAND_INK = colors.HexColor("#183153")
MUTED = colors.HexColor("#617388")
LINE = colors.HexColor("#d7e1ea")
PANEL = colors.HexColor("#f7fafc")
GREEN = colors.HexColor("#2f7d4b")
AMBER = colors.HexColor("#e3a12b")
RED = colors.HexColor("#c84b44")
BLUE = colors.HexColor("#265f88")


def generate_prior_auth_form_pdf(form: Dict[str, Any]) -> bytes:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = build_styles()
    story: List[Any] = []

    story.extend(header_banner(form, generated_at, styles))
    story.append(Spacer(1, 12))
    story.extend(case_summary(form, styles))
    story.append(Spacer(1, 12))

    story.extend(section_table("Patient Information", [
        ("Patient Name", form.get("patient_name")),
        ("Date of Birth", form.get("patient_dob")),
        ("Member ID", form.get("member_id")),
        ("Payer", form.get("payer")),
    ], styles))

    story.extend(section_table("Provider Information", [
        ("Provider Name", form.get("provider_name")),
        ("Provider NPI", form.get("provider_npi")),
        ("Provider Phone", form.get("provider_phone")),
        ("Provider Fax", form.get("provider_fax")),
    ], styles))

    story.extend(section_table("Request Details", [
        ("CPT Code", form.get("cpt_code")),
        ("Diagnosis Codes", ", ".join(form.get("diagnosis_codes", []))),
        ("Requested Service", form.get("requested_service")),
        ("Place of Service", form.get("place_of_service")),
        ("Urgency", form.get("urgency")),
    ], styles))

    if form.get("evidence_review"):
        story.extend(evidence_review_table(form.get("evidence_review", []), styles))

    story.append(section_heading("Clinical Summary", styles))
    story.append(panel_paragraph(form.get("clinical_summary") or "No clinical summary provided.", styles))
    story.append(Spacer(1, 10))

    if form.get("supporting_notes"):
        story.append(section_heading("Supporting Notes", styles))
        story.append(panel_paragraph(form.get("supporting_notes"), styles))
        story.append(Spacer(1, 10))

    doc.build(
        story,
        onFirstPage=lambda canvas, doc_obj: footer(canvas, doc_obj, form, generated_at),
        onLaterPages=lambda canvas, doc_obj: footer(canvas, doc_obj, form, generated_at),
    )
    return buffer.getvalue()


def build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "brand": ParagraphStyle(
            "PABrand",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.white,
        ),
        "title": ParagraphStyle(
            "PATitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=23,
            textColor=colors.white,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "PASubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#dce8f3"),
        ),
        "section": ParagraphStyle(
            "PASection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=BRAND_INK,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "PABody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=12,
            textColor=colors.HexColor("#203042"),
        ),
        "body_bold": ParagraphStyle(
            "PABodyBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.3,
            leading=12,
            textColor=colors.HexColor("#203042"),
        ),
        "small": ParagraphStyle(
            "PASmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
        ),
        "label": ParagraphStyle(
            "PALabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=MUTED,
        ),
        "card_value": ParagraphStyle(
            "PACardValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=BRAND_INK,
        ),
        "badge": ParagraphStyle(
            "PABadge",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=colors.white,
        ),
    }


def header_banner(form: Dict[str, Any], generated_at: str, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    title_stack = [
        Paragraph("Caldarium Prior Authorization Evidence Engine", styles["brand"]),
        Paragraph("Prior Authorization Request", styles["title"]),
        Paragraph(f"Generated {safe(generated_at)}", styles["subtitle"]),
    ]
    badge_stack = []
    if form.get("coverage_status"):
        badge_stack.append(status_badge(form.get("coverage_status")))
        badge_stack.append(Spacer(1, 5))
    if form.get("review_status"):
        badge_stack.append(status_badge(form.get("review_status")))

    table = Table([[title_stack, badge_stack]], colWidths=[4.65 * inch, 1.65 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_INK),
        ("BOX", (0, 0), (-1, -1), 1, BRAND_INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))
    return [table]


def case_summary(form: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> List[Any]:
    cards = [
        summary_card("Payer", form.get("payer"), styles),
        summary_card("CPT Code", form.get("cpt_code"), styles),
        summary_card("Urgency", form.get("urgency"), styles),
        summary_card("Case ID", form.get("case_id") or "Not linked", styles),
    ]
    table = Table([cards], colWidths=[1.56 * inch] * 4)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [Paragraph("Case Summary", styles["section"]), table]


def summary_card(label: str, value: Any, styles: Dict[str, ParagraphStyle]) -> Table:
    table = Table([
        [Paragraph(safe(label).upper(), styles["label"])],
        [Paragraph(safe(value or "Not provided"), styles["card_value"])],
    ], colWidths=[1.42 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 1, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def section_heading(title: str, styles: Dict[str, ParagraphStyle]) -> Table:
    table = Table([[Paragraph(title, styles["section"])]], colWidths=[6.3 * inch])
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, BLUE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def section_table(title: str, rows: List[tuple[str, Any]], styles: Dict[str, ParagraphStyle]) -> List[Any]:
    data = [[
        Paragraph("Field", styles["label"]),
        Paragraph("Value", styles["label"]),
    ]]
    for label, value in rows:
        data.append([
            Paragraph(safe(label), styles["body_bold"]),
            Paragraph(safe(value or "Not provided"), styles["body"]),
        ])

    table = Table(data, colWidths=[2.05 * inch, 4.25 * inch], repeatRows=1)
    table.setStyle(base_table_style())
    return [section_heading(title, styles), table, Spacer(1, 10)]


def evidence_review_table(evidence_review: List[Dict[str, Any]], styles: Dict[str, ParagraphStyle]) -> List[Any]:
    data = [[
        Paragraph("Category", styles["label"]),
        Paragraph("Status", styles["label"]),
        Paragraph("Notes", styles["label"]),
    ]]
    for item in evidence_review:
        clauses = item.get("clauses") or []
        notes = "<br/>".join(
            f"<b>{safe(clause.get('status', '')).upper()}</b>: {safe(clause.get('policy_text', ''))}"
            for clause in clauses
        ) or "No clause details provided"
        data.append([
            Paragraph(safe(item.get("evidence_category", "Uncategorized")), styles["body_bold"]),
            status_badge(item.get("status", "")),
            Paragraph(notes, styles["body"]),
        ])

    table = Table(data, colWidths=[1.55 * inch, 1.35 * inch, 3.4 * inch], repeatRows=1)
    table.setStyle(base_table_style())
    return [section_heading("Evidence Review", styles), table, Spacer(1, 10)]


def panel_paragraph(text: str, styles: Dict[str, ParagraphStyle]) -> Table:
    table = Table([[Paragraph(safe(text), styles["body"])]], colWidths=[6.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return table


def status_badge(status: Optional[str]) -> Table:
    label = safe(status or "Not provided").upper()
    normalized = (status or "").strip().lower()
    bg = colors.HexColor("#8ea2b5")
    if normalized in {"covered", "ready to submit", "satisfied", "sufficient", "approved"}:
        bg = GREEN
    elif normalized in {"partial", "partially sufficient", "review before submitting", "needs review", "standard"}:
        bg = AMBER
    elif normalized in {"not covered", "do not submit", "unsatisfied", "insufficient", "denied"}:
        bg = RED

    table = Table([[Paragraph(label, build_styles()["badge"])]])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 1, bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def base_table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f8")),
        ("BOX", (0, 0), (-1, -1), 1, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]),
    ])


def footer(canvas, doc, form: Dict[str, Any], generated_at: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(0.55 * inch, 0.45 * inch, PAGE_WIDTH - 0.55 * inch, 0.45 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    case_id = form.get("case_id") or "Unlinked case"
    canvas.drawString(0.55 * inch, 0.28 * inch, f"Generated by Prior Authorization Evidence Engine | Case: {case_id}")
    canvas.drawRightString(PAGE_WIDTH - 0.55 * inch, 0.28 * inch, generated_at)
    canvas.restoreState()


def safe(value: Any) -> str:
    return escape(str(value or ""))

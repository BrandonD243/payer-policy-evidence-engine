from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from io import BytesIO
import html
import re
import textwrap
from typing import Any, Dict, Iterable, List

from jinja2 import Environment, FileSystemLoader, select_autoescape
try:
    from weasyprint import HTML
except Exception:
    HTML = None
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.graphics.shapes import Drawing, Rect, String
except Exception:
    colors = None


REPORTS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = REPORTS_DIR / "templates"

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def build_report_context(
    case: Dict[str, Any],
    documents: List[Dict[str, Any]],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    def simplify_clause(clause: Dict[str, Any]) -> Dict[str, Any]:
        matched = clause.get("matched_concepts", [])
        matched_summary = [
            {
                "concept_id": item.get("concept_id", ""),
                "evidence_text": item.get("evidence_text", ""),
                "confidence": item.get("confidence", ""),
                "certainty_level": item.get("certainty_level", ""),
            }
            for item in matched
        ]
        return {
            "clause_id": clause.get("clause_id", ""),
            "status": clause.get("status", ""),
            "status_class": status_class(clause.get("status", "")),
            "policy_text": clause.get("policy_text", ""),
            "missing_concepts": clause.get("missing_concepts", []),
            "matched_concepts": matched_summary,
            "evidence_category": clause.get("evidence_category", ""),
        }

    clauses_not_met_raw = evaluation.get("clauses_not_met", None)
    clauses_not_met = int(clauses_not_met_raw or 0) if clauses_not_met_raw not in (None, "N/A") else 0
    clause_status_counts = {
        "satisfied": int(evaluation.get("clauses_met", 0) or 0),
        "partial": int(evaluation.get("clauses_need_attention", 0) or 0),
        "unsatisfied": clauses_not_met,
    }
    approval_counts = count_statuses(evaluation.get("approval_clauses", []))
    exclusion_counts = count_statuses(evaluation.get("exclusion_clauses", []))
    total_visual_count = sum(clause_status_counts.values()) or 1
    review_status = str(evaluation.get("review", "")).upper()
    coverage_status = str(evaluation.get("coverage_status", "")).upper()
    pa_required = bool(evaluation.get("pa_required"))
    requirements_checked = str(evaluation.get("requirements_checked", ""))
    clauses_met = int(evaluation.get("clauses_met", 0) or 0)
    clauses_need_attention = int(evaluation.get("clauses_need_attention", 0) or 0)
    evidence_review = []
    for category in evaluation.get("evidence_review", []):
        evidence_review.append(
            {
                **category,
                "status_class": status_class(category.get("status", "")),
            }
        )

    return {
        "generated_at": generated_at,
        "case": case,
        "documents": documents,
        "evaluation": evaluation,
        "coverage_status_class": status_class(coverage_status),
        "review_status_class": status_class(review_status),
        "approval_clauses": [simplify_clause(c) for c in evaluation.get("approval_clauses", [])],
        "exclusion_clauses": [simplify_clause(c) for c in evaluation.get("exclusion_clauses", [])],
        "evidence_review": evidence_review,
        "metrics": [
            {
                "label": "PA Required",
                "value": "YES" if pa_required else "NO",
                "status_class": "good" if pa_required else "bad",
            },
            {
                "label": "Documents Reviewed",
                "value": str(evaluation.get("num_documents", len(documents))),
                "status_class": "neutral",
            },
            {
                "label": "Requirements Checked",
                "value": requirements_checked,
                "status_class": "good",
            },
            {
                "label": "Clauses Met",
                "value": str(clauses_met),
                "status_class": "good" if clauses_met > 0 else "neutral",
            },
            {
                "label": "Clauses Needing Attention",
                "value": str(clauses_need_attention),
                "status_class": "warn" if clauses_need_attention > 0 else "neutral",
            },
            {
                "label": "Clauses Not Met",
                "value": str(evaluation.get("clauses_not_met", "N/A")),
                "status_class": "bad" if clauses_not_met > 0 else "neutral",
            },
        ],
        "visuals": {
            "clause_status_bar_svg": build_stacked_bar_svg(
                [
                    ("Met", clause_status_counts["satisfied"], "#2f7d4b"),
                    ("Need review", clause_status_counts["partial"], "#e3a12b"),
                    ("Not met", clause_status_counts["unsatisfied"], "#c84b44"),
                ],
                width=520,
                height=30,
            ),
            "approval_status_bar_svg": build_stacked_bar_svg(
                [
                    ("Satisfied", approval_counts["satisfied"], "#2f7d4b"),
                    ("Partial", approval_counts["partial"], "#e3a12b"),
                    ("Unsatisfied", approval_counts["unsatisfied"], "#c84b44"),
                ],
                width=420,
                height=24,
            ),
            "exclusion_status_bar_svg": build_stacked_bar_svg(
                [
                    ("Triggered", exclusion_counts["satisfied"], "#c84b44"),
                    ("Partial", exclusion_counts["partial"], "#e3a12b"),
                    ("Clear", exclusion_counts["unsatisfied"], "#8ea2b5"),
                ],
                width=420,
                height=24,
            ),
            "clauses_total": total_visual_count,
            "approval_total": sum(approval_counts.values()),
            "exclusion_total": sum(exclusion_counts.values()),
            "clause_status_counts": clause_status_counts,
        },
    }


def render_report_html(context: Dict[str, Any]) -> str:
    template = jinja_env.get_template("prior_auth_packet.html")
    return template.render(**context)


def generate_prior_auth_packet_pdf(
    case: Dict[str, Any],
    documents: List[Dict[str, Any]],
    evaluation: Dict[str, Any],
) -> bytes:
    context = build_report_context(case, documents, evaluation)
    if colors is not None:
        return render_reportlab_pdf(context)
    if HTML is not None:
        html_content = render_report_html(context)
        return HTML(string=html_content, base_url=str(REPORTS_DIR)).write_pdf()
    return render_simple_pdf(context)


def status_class(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"covered", "ready to submit", "satisfied", "sufficient", "yes"}:
        return "good"
    if normalized in {"partial", "partially sufficient", "review before submitting", "needs review"}:
        return "warn"
    if normalized in {"unsatisfied", "insufficient", "not covered", "do not submit", "triggered"}:
        return "bad"
    return "neutral"


def count_statuses(clauses: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"satisfied": 0, "partial": 0, "unsatisfied": 0}
    for clause in clauses:
        status = str(clause.get("status", "")).lower()
        if status in counts:
            counts[status] += 1
    return counts


def build_stacked_bar_svg(segments: List[tuple[str, int, str]], width: int, height: int) -> str:
    total = sum(value for _, value, _ in segments) or 1
    x = 0.0
    rects: List[str] = []
    labels: List[str] = []
    for label, value, color in segments:
        segment_width = width * (value / total)
        if value > 0:
            rects.append(
                f'<rect x="{x:.2f}" y="0" width="{segment_width:.2f}" height="{height}" fill="{color}" rx="8" ry="8"></rect>'
            )
            labels.append(
                f'<text x="{x + segment_width / 2:.2f}" y="{height / 2 + 4:.2f}" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="11" text-anchor="middle" fill="white">{html.escape(str(value))}</text>'
            )
        x += segment_width
    if not rects:
        rects.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#d8e0e8" rx="8" ry="8"></rect>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(rects)
        + "".join(labels)
        + "</svg>"
    )


def render_reportlab_pdf(context: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = build_reportlab_styles()
    story = []

    story.extend(build_header_block(context, styles))
    story.append(Spacer(1, 10))
    story.extend(build_summary_block(context, styles))
    story.append(Spacer(1, 12))
    story.extend(build_metric_cards(context, styles))
    story.append(Spacer(1, 12))
    story.extend(build_visual_blocks(context, styles))
    story.append(Spacer(1, 12))
    story.extend(build_evidence_review_table(context, styles))
    story.append(Spacer(1, 12))
    story.extend(build_clause_sections(context["approval_clauses"], "Approval Clauses", styles))
    story.append(Spacer(1, 12))
    story.extend(build_clause_sections(context["exclusion_clauses"], "Exclusion Clauses", styles))
    story.append(Spacer(1, 12))
    story.extend(build_documents_section(context, styles))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Rendered from the same structured evaluation used by the API response.", styles["fine"]))

    doc.build(story)
    return buffer.getvalue()


def build_reportlab_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PacketTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#183153"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="PacketSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#183153"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="PacketBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=12,
        textColor=colors.HexColor("#203042"),
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="PacketSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#617388"),
    ))
    styles.add(ParagraphStyle(
        name="PacketLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=9,
        textColor=colors.HexColor("#617388"),
    ))
    styles.add(ParagraphStyle(
        name="PacketBadge",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="PacketFine",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=9.5,
        textColor=colors.HexColor("#617388"),
    ))
    return {
        "title": styles["PacketTitle"],
        "section": styles["PacketSection"],
        "body": styles["PacketBody"],
        "small": styles["PacketSmall"],
        "label": styles["PacketLabel"],
        "badge": styles["PacketBadge"],
        "fine": styles["PacketFine"],
    }


def build_header_block(context: Dict[str, Any], styles: Dict[str, Any]):
    case = context["case"]
    left = [
        Paragraph("Prior Authorization Review Packet", styles["title"]),
        Paragraph(
            "Structured payer-ready dashboard generated from the same clause evaluation used by the case API response.",
            styles["small"],
        ),
    ]
    right = [
        status_badge_paragraph(str(context["evaluation"].get("coverage_status", "")).upper()),
        Spacer(1, 4),
        status_badge_paragraph(str(context["evaluation"].get("review", ""))),
    ]
    meta = Table(
        [[left, right]],
        colWidths=[4.9 * inch, 1.45 * inch],
    )
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eaf2f8")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#c9d8e6")),
        ("ROUNDEDCORNERS", [14, 14, 14, 14]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    case_grid = Table([
        [
            labeled_value("Case ID", case.get("case_id", ""), styles),
            labeled_value("Patient", case.get("patient_name", ""), styles),
            labeled_value("Payer", str(case.get("payer", "")).upper(), styles),
            labeled_value("CPT", case.get("cpt_code", ""), styles),
        ]
    ], colWidths=[1.8 * inch, 1.7 * inch, 1.3 * inch, 0.8 * inch])
    case_grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d7e1ea")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return [meta, Spacer(1, 8), case_grid]


def build_summary_block(context: Dict[str, Any], styles: Dict[str, Any]):
    evaluation = context["evaluation"]
    summary_left = [
        Paragraph("Executive Summary", styles["section"]),
        Paragraph(evaluation.get("summary_why", ""), styles["body"]),
    ]
    if evaluation.get("summary_fixes"):
        summary_left.extend([
            Spacer(1, 6),
            Paragraph(f"<b>Follow-up needed:</b> {evaluation['summary_fixes']}", styles["body"]),
        ])
    summary_right = [
        Paragraph("Case Snapshot", styles["section"]),
        Paragraph(f"<b>Generated:</b> {context['generated_at']}", styles["body"]),
        Paragraph(f"<b>PA Required:</b> {'Yes' if evaluation.get('pa_required') else 'No'}", styles["body"]),
        Paragraph(f"<b>Documents Reviewed:</b> {evaluation.get('num_documents')}", styles["body"]),
        Paragraph(f"<b>Requirements Checked:</b> {evaluation.get('requirements_checked')}", styles["body"]),
        Paragraph(f"<b>Evaluation Summary:</b> {evaluation.get('evaluation_result_summary')}", styles["body"]),
    ]
    table = Table([[summary_left, summary_right]], colWidths=[4.2 * inch, 2.15 * inch])
    table.setStyle(panel_style())
    return [table]


def build_metric_cards(context: Dict[str, Any], styles: Dict[str, Any]):
    cards = []
    row = []
    widths = [1.62 * inch] * 4
    for metric in context["metrics"]:
        row.append(metric_card(metric["label"], metric["value"], metric["status_class"], styles))
        if len(row) == 4:
            cards.append(row)
            row = []
    if row:
        while len(row) < 4:
            row.append("")
        cards.append(row)
    table = Table(cards, colWidths=widths, hAlign="LEFT")
    ts = TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])
    table.setStyle(ts)
    return [Paragraph("Decision Dashboard", styles["section"]), table]


def build_visual_blocks(context: Dict[str, Any], styles: Dict[str, Any]):
    visuals = context["visuals"]
    left = [
        Paragraph("Overall Clause Status", styles["section"]),
        Paragraph("Met, review-needed, and not-met counts across the packet summary fields.", styles["small"]),
        Spacer(1, 8),
        make_bar_drawing([
            ("Met", visuals["clause_status_counts"]["satisfied"], colors.HexColor("#2f7d4b")),
            ("Need review", visuals["clause_status_counts"]["partial"], colors.HexColor("#e3a12b")),
            ("Not met", visuals["clause_status_counts"]["unsatisfied"], colors.HexColor("#c84b44")),
        ], width=300, height=22),
        Spacer(1, 8),
        Paragraph(
            f"Met: {visuals['clause_status_counts']['satisfied']} | Need review: {visuals['clause_status_counts']['partial']} | Not met: {visuals['clause_status_counts']['unsatisfied']}",
            styles["small"],
        ),
    ]
    approval_counts = count_statuses(context["evaluation"].get("approval_clauses", []))
    exclusion_counts = count_statuses(context["evaluation"].get("exclusion_clauses", []))
    right = [
        Paragraph("Approval vs Exclusion", styles["section"]),
        Paragraph("Clause status distribution split by approval criteria and exclusion review.", styles["small"]),
        Spacer(1, 8),
        Paragraph("Approval clauses", styles["label"]),
        Spacer(1, 4),
        make_bar_drawing([
            ("Satisfied", approval_counts["satisfied"], colors.HexColor("#2f7d4b")),
            ("Partial", approval_counts["partial"], colors.HexColor("#e3a12b")),
            ("Unsatisfied", approval_counts["unsatisfied"], colors.HexColor("#c84b44")),
        ], width=230, height=18),
        Spacer(1, 8),
        Paragraph("Exclusion clauses", styles["label"]),
        Spacer(1, 4),
        make_bar_drawing([
            ("Triggered", exclusion_counts["satisfied"], colors.HexColor("#c84b44")),
            ("Partial", exclusion_counts["partial"], colors.HexColor("#e3a12b")),
            ("Clear", exclusion_counts["unsatisfied"], colors.HexColor("#8ea2b5")),
        ], width=230, height=18),
    ]
    table = Table([[left, right]], colWidths=[3.55 * inch, 2.8 * inch])
    table.setStyle(panel_style())
    return [table]


def build_evidence_review_table(context: Dict[str, Any], styles: Dict[str, Any]):
    data = [[
        Paragraph("<b>Category</b>", styles["label"]),
        Paragraph("<b>Status</b>", styles["label"]),
        Paragraph("<b>Included Clauses</b>", styles["label"]),
    ]]
    for category in context["evidence_review"]:
        clauses_text = "<br/>".join(
            f"<b>{clause.get('status', '').upper()}</b>: {clause.get('policy_text', '')}"
            for clause in category.get("clauses", [])
        )
        data.append([
            Paragraph(category.get("evidence_category", ""), styles["body"]),
            status_badge_paragraph(category.get("status", "")),
            Paragraph(clauses_text or "No clauses shown", styles["body"]),
        ])
    table = Table(data, colWidths=[1.6 * inch, 1.35 * inch, 3.45 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f8")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d7e1ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7e1ea")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [Paragraph("Evidence Review", styles["section"]), table]


def build_clause_sections(clauses: List[Dict[str, Any]], title: str, styles: Dict[str, Any]):
    items = [Paragraph(title, styles["section"])]
    if not clauses:
        items.append(Paragraph("No clauses returned.", styles["body"]))
        return items
    for clause in clauses:
        content = [
            [Paragraph(f"<b>{clause.get('clause_id', '')}</b>", styles["small"]), status_badge_paragraph(clause.get("status", ""))],
            [Paragraph(clause.get("policy_text", ""), styles["body"]), ""],
        ]
        if clause.get("missing_concepts"):
            content.append([Paragraph("<b>Missing concepts:</b>", styles["small"]), Paragraph(", ".join(clause["missing_concepts"]), styles["small"])])
        matched = clause.get("matched_concepts", [])
        if matched:
            evidence_text = "<br/>".join(
                f"<b>{m.get('concept_id', '')}</b> | confidence {m.get('confidence', '')}"
                + (f" | {m.get('certainty_level', '')}" if m.get("certainty_level") else "")
                + f" | {m.get('evidence_text', '')}"
                for m in matched
            )
            content.append([Paragraph("<b>Evidence:</b>", styles["small"]), Paragraph(evidence_text, styles["small"])])
        else:
            content.append([Paragraph("<b>Evidence:</b>", styles["small"]), Paragraph("No matching evidence captured.", styles["small"])])
        table = Table(content, colWidths=[1.35 * inch, 4.95 * inch])
        ts = panel_style()
        ts.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f7fafc"))
        table.setStyle(ts)
        items.append(table)
        items.append(Spacer(1, 8))
    return items


def build_documents_section(context: Dict[str, Any], styles: Dict[str, Any]):
    items = [Paragraph("Documents Reviewed", styles["section"])]
    docs = context["documents"]
    rows = []
    row = []
    for doc in docs:
        row.append(Table([[
            Paragraph(f"<b>{doc.get('filename', '')}</b><br/>{doc.get('created_at', '')}", styles["body"])
        ]], colWidths=[3.08 * inch], style=panel_style()))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        while len(row) < 2:
            row.append("")
        rows.append(row)
    table = Table(rows, colWidths=[3.15 * inch, 3.15 * inch])
    table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    items.append(table)
    return items


def panel_style():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d7e1ea")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ])


def labeled_value(label: str, value: str, styles: Dict[str, Any]):
    return Paragraph(f"<font color='#617388'><b>{label}</b></font><br/><font size='11'><b>{value}</b></font>", styles["body"])


def metric_card(label: str, value: str, status_class: str, styles: Dict[str, Any]):
    value_color = {
        "good": "#2f7d4b",
        "warn": "#8d5c09",
        "bad": "#c84b44",
        "neutral": "#183153",
    }.get(status_class, "#183153")
    table = Table([[
        Paragraph(label, styles["label"]),
    ], [
        Paragraph(f"<font size='16' color='{value_color}'><b>{value}</b></font>", styles["body"]),
    ]], colWidths=[1.42 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d7e1ea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def status_badge_paragraph(status: str):
    text = str(status or "")
    normalized = text.strip().lower()
    bg = "#8ea2b5"
    if normalized in {"covered", "ready to submit", "satisfied", "sufficient"}:
        bg = "#2f7d4b"
    elif normalized in {"partial", "partially sufficient", "review before submitting"}:
        bg = "#e3a12b"
    elif normalized in {"unsatisfied", "insufficient", "not covered", "do not submit"}:
        bg = "#c84b44"
    return Paragraph(
        f"<para backColor='{bg}' textColor='white' borderPadding='4'><b>{text.upper()}</b></para>",
        getSampleStyleSheet()["BodyText"],
    )


def make_bar_drawing(segments: List[tuple[str, int, Any]], width: int, height: int):
    drawing = Drawing(width, height + 14)
    total = sum(value for _, value, _ in segments) or 1
    x = 0
    for label, value, color in segments:
        segment_width = width * (value / total)
        if value > 0:
            drawing.add(Rect(x, 8, segment_width, height, fillColor=color, strokeColor=color, rx=5, ry=5))
            drawing.add(String(x + segment_width / 2, 8 + height / 2 - 3, str(value), fontName="Helvetica-Bold", fontSize=8, fillColor=colors.white, textAnchor="middle"))
        x += segment_width
    if x == 0:
        drawing.add(Rect(0, 8, width, height, fillColor=colors.HexColor("#d8e0e8"), strokeColor=colors.HexColor("#d8e0e8"), rx=5, ry=5))
    return drawing


def render_simple_pdf(context: Dict[str, Any]) -> bytes:
    lines = build_packet_lines(context)
    pages = paginate_lines(lines, max_lines=46)
    return build_pdf_document(pages)


def build_packet_lines(context: Dict[str, Any]) -> List[str]:
    case = context["case"]
    evaluation = context["evaluation"]
    lines: List[str] = []

    lines.extend(
        [
            "Prior Authorization Review Packet",
            "",
            f"Generated: {context['generated_at']}",
            f"Case ID: {case.get('case_id', '')}",
            f"Patient: {case.get('patient_name', 'Unknown')}",
            f"Payer: {str(case.get('payer', '')).upper()}",
            f"CPT Code: {case.get('cpt_code', '')}",
            "",
            "Executive Summary",
            f"Coverage Status: {str(evaluation.get('coverage_status', '')).upper()}",
            f"Review Recommendation: {evaluation.get('review', '')}",
            f"PA Required: {'Yes' if evaluation.get('pa_required') else 'No'}",
            f"Summary: {evaluation.get('summary_why', '')}",
        ]
    )
    if evaluation.get("summary_fixes"):
        lines.append(f"Follow-up Needed: {evaluation['summary_fixes']}")

    lines.extend(["", "Summary Metrics"])
    for label, value in context["metrics"]:
        lines.append(f"{label}: {value}")

    lines.extend(["", "Evidence Review"])
    for category in context["evidence_review"]:
        lines.append(
            f"{category.get('evidence_category', 'uncategorized')}: {category.get('status', '')}"
        )
        for clause in category.get("clauses", []):
            lines.append(
                f"  - {clause.get('status', '').upper()}: {clause.get('policy_text', '')}"
            )

    lines.extend(["", "Approval Clauses"])
    for clause in context["approval_clauses"]:
        append_clause_block(lines, clause)

    lines.extend(["", "Exclusion Clauses"])
    if context["exclusion_clauses"]:
        for clause in context["exclusion_clauses"]:
            append_clause_block(lines, clause)
    else:
        lines.append("No exclusion clauses were returned.")

    lines.extend(["", "Documents Reviewed"])
    for doc in context["documents"]:
        lines.append(f"- {doc.get('filename', '')} ({doc.get('created_at', '')})")

    lines.extend(["", "Rendered from the same structured evaluation used by the API response."])
    return lines


def append_clause_block(lines: List[str], clause: Dict[str, Any]) -> None:
    lines.append(
        f"{clause.get('clause_id', '')} | {clause.get('status', '').upper()} | {clause.get('policy_text', '')}"
    )
    if clause.get("missing_concepts"):
        lines.append(f"  Missing: {', '.join(clause['missing_concepts'])}")
    if clause.get("matched_concepts"):
        for match in clause["matched_concepts"][:3]:
            evidence_text = match.get("evidence_text", "")
            evidence_text = re.sub(r"\s+", " ", evidence_text).strip()
            lines.append(
                "  Evidence: "
                f"{match.get('concept_id', '')} | confidence {match.get('confidence', '')} | "
                f"{evidence_text}"
            )
    else:
        lines.append("  Evidence: none captured")


def paginate_lines(lines: Iterable[str], max_lines: int) -> List[List[str]]:
    wrapped_lines: List[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=92, subsequent_indent="    ") or [""])
    return [
        wrapped_lines[index:index + max_lines]
        for index in range(0, len(wrapped_lines), max_lines)
    ] or [[]]


def pdf_escape(text: str) -> str:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf_document(pages: List[List[str]]) -> bytes:
    objects: List[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: List[int] = []
    content_ids: List[int] = []

    for page_lines in pages:
        stream_lines = ["BT", "/F1 10 Tf", "72 770 Td", "14 TL"]
        for line in page_lines:
            stream_lines.append(f"({pdf_escape(line)}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        content_id = add_object(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream"
        )
        content_ids.append(content_id)
        page_ids.append(0)

    pages_id = add_object(b"")

    for index, content_id in enumerate(content_ids):
        page_payload = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_ids[index] = add_object(page_payload)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [ {kids} ] /Count {len(page_ids)} >>".encode("latin-1")
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, payload in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        pdf.extend(payload)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)

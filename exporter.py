"""
exporter.py — Generate a one-page PDF investment advice report.

Uses ReportLab Platypus for layout.  The PDF contains:
* Header with generation timestamp and user profile summary
* Recommended portfolio table (fund / name / weight)
* Headline metrics (expected return, vol, Sharpe, utility)
* Embedded Plotly charts (frontier + pie) rendered as PNG

NOTE: Plotly-to-PNG requires the ``kaleido`` package. If it's not installed we
fall back to an image-free PDF — the demo still works, just without charts.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT
from data_loader import fund_display_name
from state_machine import SessionState, ensure_prices_loaded
from visuals import plot_efficient_frontier, plot_weights_pie

logger = logging.getLogger(__name__)

EXPORT_DIR = PROJECT_ROOT / "data" / "exports"


def build_pdf(state: SessionState) -> Path:
    """Render the final advice PDF and return its path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EXPORT_DIR / f"advice_{stamp}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleNavy", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=20, textColor=colors.HexColor("#0B2545"),
        alignment=0, spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#13315C"),
        spaceBefore=14, spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#0B2545"),
    )
    tiny = ParagraphStyle(
        "Tiny", parent=body, fontSize=8, textColor=colors.HexColor("#5A6B85"),
    )

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="MDFinTech Robo-Adviser Report",
    )
    story: list = []

    story.append(Paragraph("MDFinTech Robo-Adviser — Advice Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now():%Y-%m-%d %H:%M} · "
        f"Data source: <b>{state.data_source or 'n/a'}</b>",
        tiny,
    ))
    story.append(Spacer(1, 6))

    # --- Investor profile -------------------------------------------------
    story.append(Paragraph("1. Investor Profile", h2))
    profile_html = state.profile_text.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")
    # Strip markdown bold for reportlab (convert to <b>)
    profile_html = _md_to_reportlab(profile_html)
    story.append(Paragraph(profile_html, body))

    profile_table = Table(
        [
            ["Weighted score", f"{state.total_score} / 75"],
            ["Risk level",     f"{state.level_code} — {state.level_name}"],
            ["Risk aversion (A)", f"{state.A_value}"],
        ],
        colWidths=[50 * mm, 80 * mm],
    )
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.HexColor("#0B2545")),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D6E4")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 6))
    story.append(profile_table)

    # --- Portfolio --------------------------------------------------------
    story.append(Paragraph("2. Recommended Portfolio", h2))
    rows = [["Fund", "Name", "Weight"]]
    for code, w in sorted(state.weights.items(), key=lambda kv: -kv[1]):
        if w >= 0.001:
            rows.append([code, fund_display_name(code), f"{w * 100:.2f}%"])
    pf_table = Table(rows, colWidths=[22 * mm, 115 * mm, 25 * mm])
    pf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D6E4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(pf_table)

    # --- Metrics ----------------------------------------------------------
    story.append(Paragraph("3. Headline Metrics (annualized)", h2))
    m = state.metrics
    metrics_table = Table(
        [
            ["Expected return", f"{m.get('expected_return', 0) * 100:.2f}%"],
            ["Volatility (std)", f"{m.get('std', 0) * 100:.2f}%"],
            ["Sharpe ratio",    f"{m.get('sharpe', 0):.2f}"],
            ["Utility U",       f"{m.get('utility', 0):.4f}"],
        ],
        colWidths=[60 * mm, 40 * mm],
    )
    metrics_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D6E4")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(metrics_table)

    # --- Charts -----------------------------------------------------------
    try:
        prices, mu, sigma, _src = ensure_prices_loaded()
        user_pt = (m.get("std", 0), m.get("expected_return", 0))
        frontier_png = _fig_to_png(plot_efficient_frontier(mu, sigma, user_point=user_pt))
        pie_png = _fig_to_png(plot_weights_pie(state.weights,
                                               {c: fund_display_name(c) for c in state.weights}))
        if frontier_png:
            story.append(Paragraph("4. Efficient Frontier", h2))
            story.append(Image(io.BytesIO(frontier_png), width=165 * mm, height=95 * mm))
        if pie_png:
            story.append(Paragraph("5. Allocation Breakdown", h2))
            story.append(Image(io.BytesIO(pie_png), width=165 * mm, height=95 * mm))
    except Exception as exc:
        logger.warning("Chart embedding skipped: %s", exc)
        story.append(Paragraph(
            "_Charts unavailable (install the 'kaleido' package to include them in the PDF)._",
            tiny,
        ))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "_This report is generated by an educational Robo-Adviser for the "
        "BMD5302 Financial Modeling project. It is not financial advice. Past "
        "performance does not guarantee future results._",
        tiny,
    ))

    doc.build(story)
    logger.info("PDF exported to %s", out_path)
    return out_path


def _md_to_reportlab(text: str) -> str:
    """Convert a subset of Markdown (**bold**, *italic*) to ReportLab tags."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _fig_to_png(fig) -> bytes | None:
    """Render a Plotly figure to PNG bytes (requires kaleido)."""
    try:
        return fig.to_image(format="png", width=1000, height=580, scale=2)
    except Exception as exc:  # pragma: no cover
        logger.info("Plotly PNG export unavailable: %s", exc)
        return None


__all__ = ["build_pdf"]

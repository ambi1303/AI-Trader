"""One-page PDF summary using ReportLab.

The PDF is meant to be a *document* the recipient can save / print -- not a
heavyweight investment report. We keep it deliberately to one page so it
loads instantly on a phone email client.

ReportLab is a pure-Python renderer; no external binaries (wkhtmltopdf,
weasyprint) are required.

Security note: every text field that comes from user / DB content is passed
through ReportLab's Paragraph escape (the underlying mini-language treats
``<``/``>``/``&`` as XML, so we pre-escape them) -- this prevents broken PDFs
from a corrupt symbol field.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.notifications.report_builder import DailyReport
from src.utils.logger import get_logger

log = get_logger("notifications.pdf_writer")


_XML_BAD = re.compile(r"[<>&]")


def _escape(s: Any) -> str:
    """Escape text for ReportLab's Paragraph mini-language."""
    if s is None:
        return ""
    text = str(s)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=16,
            leading=20,
            spaceAfter=4,
            textColor=colors.HexColor("#0b3d91"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#0b3d91"),
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#555555"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=12,
        ),
    }


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2fb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e2e6")),
            ("ALIGN", (1, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ]
    )


def _signals_table(report: DailyReport, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = [Paragraph("Today's signals", styles["h2"])]
    if not report.signals:
        flow.append(
            Paragraph(
                "No symbols crossed the threshold today "
                f"({(report.threshold_used or 0):.3f}). Stay flat.",
                styles["body"],
            )
        )
        return flow
    rows = [["Symbol", "Calibrated prob", "Decision"]]
    for s in report.signals:
        rows.append(
            [
                _escape(s.symbol),
                f"{(s.calibrated_prob or 0):.4f}",
                "BUY",
            ]
        )
    table = Table(rows, colWidths=[40 * mm, 40 * mm, 30 * mm])
    table.setStyle(_table_style())
    flow.append(table)
    return flow


def _predictions_table(report: DailyReport, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = [Paragraph(f"Top {report.top_n} predictions", styles["h2"])]
    if not report.predictions:
        flow.append(Paragraph("(no predictions for this date)", styles["body"]))
        return flow
    rows = [["#", "Symbol", "Calibrated", "Raw", "Signal"]]
    for i, p in enumerate(report.predictions, 1):
        rows.append(
            [
                str(i),
                _escape(p.symbol),
                f"{(p.calibrated_prob or 0):.4f}",
                f"{(p.raw_prob or 0):.4f}",
                "YES" if p.is_signal else "",
            ]
        )
    table = Table(rows, colWidths=[10 * mm, 35 * mm, 30 * mm, 30 * mm, 20 * mm])
    table.setStyle(_table_style())
    flow.append(table)
    return flow


def _model_block(report: DailyReport, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = [Paragraph("Model snapshot", styles["h2"])]
    if not report.latest_model:
        flow.append(Paragraph("(no trained model yet)", styles["body"]))
        return flow
    m = report.latest_model
    rows = [
        ["Run ID", _escape(m.run_id)],
        ["Trained", f"{m.trained_from} \u2192 {m.trained_to}"],
        ["Threshold", f"{(report.threshold_used or 0):.3f}"],
        ["Git SHA", _escape(m.git_sha or "unknown")],
    ]
    table = Table(rows, colWidths=[35 * mm, 130 * mm])
    table.setStyle(_table_style())
    flow.append(table)
    return flow


def _backtest_block(report: DailyReport, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = [Paragraph("Latest backtest", styles["h2"])]
    if not report.latest_backtest:
        flow.append(Paragraph("(no backtest runs)", styles["body"]))
        return flow
    bt = report.latest_backtest
    m = bt.metrics
    rows = [
        ["Name", _escape(bt.name)],
        ["Window", f"{bt.start_date} \u2192 {bt.end_date}"],
        ["Sharpe / Sortino",
         f"{(m.get('sharpe', 0) or 0):.2f} / {(m.get('sortino', 0) or 0):.2f}"],
        ["Max DD", f"{(m.get('max_drawdown_pct', 0) or 0):.2f}%"],
        ["Total return", f"{(m.get('total_return_pct', 0) or 0):.2f}%"],
        ["Trades / Hit-rate",
         f"{m.get('n_trades', 0)} / {(m.get('hit_rate_pct', 0) or 0):.1f}%"],
    ]
    table = Table(rows, colWidths=[35 * mm, 130 * mm])
    table.setStyle(_table_style())
    flow.append(table)
    return flow


def _trades_block(report: DailyReport, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = [Paragraph("Recent trades (latest backtest)", styles["h2"])]
    if not report.recent_trades:
        flow.append(Paragraph("(no trades)", styles["body"]))
        return flow
    rows = [["Symbol", "Entry", "Exit", "Hold", "P&L (Rs)", "P&L %", "Reason"]]
    for t in report.recent_trades:
        rows.append(
            [
                _escape(t.symbol),
                t.entry_date,
                t.exit_date,
                f"{t.holding_days}d",
                f"{t.net_pnl:.0f}",
                f"{t.pnl_pct:.2f}%",
                _escape(t.exit_reason),
            ]
        )
    table = Table(
        rows,
        colWidths=[25 * mm, 22 * mm, 22 * mm, 14 * mm, 22 * mm, 18 * mm, 22 * mm],
    )
    table.setStyle(_table_style())
    flow.append(table)
    return flow


def _paper_block(report: DailyReport, styles: dict[str, ParagraphStyle]) -> list:
    """Open paper positions + recently closed paper trades."""
    flow: list = [Paragraph("Paper portfolio", styles["h2"])]
    p = report.paper
    summary = (
        f"Open: <b>{p.open_count}</b> &nbsp;&middot;&nbsp; "
        f"Unrealised: <b>Rs {p.unrealised_pnl:,.0f}</b> &nbsp;&middot;&nbsp; "
        f"Realised ({p.window_days}d): <b>Rs {p.realised_pnl_window:,.0f}</b> "
        f"&nbsp;&middot;&nbsp; Win-rate: {p.win_rate_pct_window:.1f}%"
    )
    flow.append(Paragraph(summary, styles["body"]))

    if p.open_positions:
        rows = [["Symbol", "Qty", "Entry", "Last", "SL", "TP", "Hold", "Unreal P&L"]]
        for op in p.open_positions:
            rows.append([
                _escape(op.symbol),
                str(op.qty),
                f"{op.entry_price:.2f}",
                f"{(op.last_close or 0):.2f}",
                f"{(op.stop_loss or 0):.2f}",
                f"{(op.take_profit or 0):.2f}",
                f"{op.holding_days}d",
                f"{op.unrealised_pnl:,.0f} ({op.unrealised_pnl_pct:+.2f}%)",
            ])
        t = Table(rows, colWidths=[22 * mm, 14 * mm, 18 * mm, 18 * mm,
                                   18 * mm, 18 * mm, 14 * mm, 33 * mm])
        t.setStyle(_table_style())
        flow.append(t)

    if p.closed_recent:
        flow.append(Paragraph(
            f"Recently closed (last {p.window_days} days)", styles["h2"]
        ))
        rows = [["Symbol", "Entry", "Exit", "Hold", "P&L (Rs)", "P&L %", "Reason"]]
        for t in p.closed_recent:
            rows.append([
                _escape(t.symbol),
                t.entry_date,
                t.exit_date,
                f"{t.holding_days}d",
                f"{t.net_pnl:.0f}",
                f"{t.pnl_pct:.2f}%",
                _escape(t.exit_reason),
            ])
        tbl = Table(
            rows,
            colWidths=[25 * mm, 22 * mm, 22 * mm, 14 * mm, 22 * mm, 18 * mm, 22 * mm],
        )
        tbl.setStyle(_table_style())
        flow.append(tbl)
    return flow


def write_pdf(report: DailyReport, output_path: Path) -> Path:
    """Render the report to ``output_path``. Returns the resolved path.

    The directory is created if missing. Existing files are overwritten.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"AI Trader Daily Report {report.report_date}",
        author="ai_trading_system",
    )
    flow: list = [
        Paragraph(f"AI Trader \u2014 Daily Report \u2014 {report.report_date}", styles["title"]),
        Paragraph(_escape(report.headline), styles["body"]),
        Spacer(1, 4),
        Paragraph(
            f"Universe: {report.universe_size} \u00b7 "
            f"Generated at {report.generated_at} (UTC) \u00b7 "
            "Auto-generated. Not investment advice.",
            styles["small"],
        ),
    ]
    flow.extend(_signals_table(report, styles))
    flow.extend(_predictions_table(report, styles))
    flow.extend(_paper_block(report, styles))
    flow.extend(_model_block(report, styles))
    flow.extend(_backtest_block(report, styles))
    flow.extend(_trades_block(report, styles))
    doc.build(flow)
    log.info("Wrote PDF report to {}", output_path.as_posix())
    return output_path


def render_pdf_bytes(report: DailyReport) -> bytes:
    """In-memory variant for cases where we don't want to touch disk."""
    buf = io.BytesIO()
    styles = _styles()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    flow: list = [
        Paragraph(f"AI Trader \u2014 Daily Report \u2014 {report.report_date}", styles["title"]),
        Paragraph(_escape(report.headline), styles["body"]),
        Spacer(1, 4),
        Paragraph(
            f"Universe: {report.universe_size} \u00b7 "
            f"Generated at {report.generated_at} (UTC) \u00b7 "
            "Auto-generated. Not investment advice.",
            styles["small"],
        ),
    ]
    flow.extend(_signals_table(report, styles))
    flow.extend(_predictions_table(report, styles))
    flow.extend(_paper_block(report, styles))
    flow.extend(_model_block(report, styles))
    flow.extend(_backtest_block(report, styles))
    flow.extend(_trades_block(report, styles))
    doc.build(flow)
    return buf.getvalue()


def default_pdf_path(output_dir: Path | str, report: DailyReport) -> Path:
    """Convenience: ``data/reports/notifications/daily_<date>.pdf``."""
    stamp = report.report_date.replace("-", "")
    suffix = datetime.utcnow().strftime("%H%M%S")
    return Path(output_dir) / f"daily_{stamp}_{suffix}.pdf"

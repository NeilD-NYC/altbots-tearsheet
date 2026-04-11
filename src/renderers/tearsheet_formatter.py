"""
src/renderers/tearsheet_formatter.py — AltBots Pipeline: Gold Copy Formatter (Step 4)

Transforms the Assembler's Gold Copy JSON into three formats:
  --format pdf   Institutional two-page PDF (ReportLab, Navy/Gold palette)
  --format md    Structured Markdown optimised for RAG and AI agents
  --format json  Pass-through (pretty-printed JSON)

The PDF mandatory four-layer disclosure block is rendered:
  1. As a compact single-line notice in the canvas footer on EVERY page.
  2. As the full four-layer KeepTogether block in the story at the end of the
     document (guaranteed never to be truncated).

Usage (standalone)
------------------
  python -m src.renderers.tearsheet_formatter gold_copy.json --format pdf
  python -m src.renderers.tearsheet_formatter gold_copy.json --format md

See also: main.py (preferred entry point).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("formatter")

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR    = _PROJECT_ROOT / "output"

# ── Brand palette ─────────────────────────────────────────────────────────────
_NAVY_HEX  = "#1B2A4A"
_GOLD_HEX  = "#C5A55A"
_GRAY_DARK = "#4A4A4A"
_GRAY_MID  = "#767676"
_GRAY_LITE = "#F2F2F2"
_GRAY_DISC = "#808080"   # legal text
_RED_HEX   = "#B22222"
_GREEN_HEX = "#2E7D32"
_AMBER_HEX = "#E65100"
_WHITE_HEX = "#FFFFFF"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (format-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_usd(value) -> str:
    if value is None:
        return "—"
    try:
        v = int(value)
    except (TypeError, ValueError):
        return str(value)
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:,}"


def _safe(val, fallback: str = "—") -> str:
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s else fallback


def _trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _recency_label(flag: str) -> str:
    return {"green": "Current", "yellow": "Approaching stale"}.get(flag, "Stale")


def _fmt_date(iso: str) -> str:
    """Convert ISO-8601 string to 'Mon DD, YYYY' or return as-is."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except Exception:
        return iso[:10] if len(iso) >= 10 else iso


def _slug(name: str) -> str:
    """'VIKING GLOBAL INVESTORS LP' → 'VIKING_GLOBAL_INVESTORS_LP'"""
    return re.sub(r"[^\w]", "_", name.strip()).rstrip("_")


def _derive_red_flags(gold: dict) -> list[dict]:
    """
    Derive red flags from the Gold Copy data:
      - ADV filing staleness
      - 13F staleness
      - Skipped modules
      - Missing team data
    Returns list of {level: red|yellow|info, message: str}.
    """
    flags = []
    inst = gold.get("institutional_numbers") or {}
    adv_flag = inst.get("adv_recency_flag", "")
    if adv_flag == "red":
        flags.append({"level": "red",
                      "message": f"ADV filing is stale — last filed {_safe(inst.get('adv_filing_date'))}."})
    elif adv_flag == "yellow":
        flags.append({"level": "yellow",
                      "message": f"ADV filing approaching stale — filed {_safe(inst.get('adv_filing_date'))}."})

    f13 = inst.get("latest_13f") or {}
    if f13:
        f13_flag = f13.get("recency_flag", "")
        if f13_flag == "red":
            flags.append({"level": "red",
                          "message": f"13F filing is stale — period {_safe(f13.get('period_of_report'))}."})
        elif f13_flag == "yellow":
            flags.append({"level": "yellow",
                          "message": f"13F approaching stale — period {_safe(f13.get('period_of_report'))}."})

    skipped = gold.get("modules_skipped") or {}
    for mod, reason in skipped.items():
        flags.append({"level": "yellow",
                      "message": f"Module '{mod}' unavailable: {reason}"})

    if not gold.get("team_roster"):
        flags.append({"level": "info",
                      "message": "No team profiles available for this firm."})

    if not flags:
        flags.append({"level": "info",
                      "message": "No material red flags identified from available data."})

    return flags


def _build_disclosure_layers(disclosure: dict) -> list[str]:
    """Return list of 4 layer strings from disclosure dict."""
    d = disclosure.get("disclaimers") or {}
    return [
        d.get("not_investment_advice", ""),
        d.get("regulatory_disclaimer", ""),
        d.get("data_limitation", ""),
        d.get("confidentiality", ""),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → Gold Copy parser  (--from-md path)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_md_to_gold(md_text: str) -> dict:
    """
    Parse a tearsheet Markdown file (produced by _render_md or hand-edited)
    back into a Gold Copy dict suitable for passing to _render_pdf.

    Special keys injected:
      _analyst_commentary  str   — body of an ## Analyst Commentary section
      _parsed_red_flags    list  — flags parsed directly from the MD
      _from_md             dict  — {"sections_present": set of normalised keys}
    """
    lines = md_text.splitlines()

    # ── Split on H2 (##) headers ──────────────────────────────────────────────
    sections_raw: dict = {}
    preamble: list = []
    cur_title: str | None = None
    cur_lines: list = []

    for line in lines:
        m2 = re.match(r"^## (.+)$", line)
        if m2:
            if cur_title is None:
                preamble = cur_lines[:]
            else:
                sections_raw[cur_title] = cur_lines[:]
            cur_title = m2.group(1).strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_title is not None:
        sections_raw[cur_title] = cur_lines[:]
    elif cur_lines:
        preamble = cur_lines[:]

    # ── Section lookup (case-insensitive keyword match) ───────────────────────
    def _find_sec(*kws):
        for title, content in sections_raw.items():
            t = title.lower()
            if all(k in t for k in kws):
                return title, content
        return None, None

    # ── Table parsers ─────────────────────────────────────────────────────────
    def _kv_table(sec_lines):
        """Parse | Key | Value | table into dict."""
        out = {}
        for ln in sec_lines:
            ln2 = ln.replace("\\|", "\x00")
            m = re.match(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", ln2.strip())
            if not m:
                continue
            k = m.group(1).replace("\x00", "|").strip()
            v = m.group(2).replace("\x00", "|").strip()
            if not k or re.match(r"^[-: ]+$", k) or k.lower() in ("field", "#"):
                continue
            out[k] = v
        return out

    def _pipe_table(sec_lines):
        """Parse a pipe table → list of rows; rows[0]=header, rows[1:]=data."""
        rows = []
        for ln in sec_lines:
            ln2 = ln.replace("\\|", "\x00")
            if not ln2.strip().startswith("|"):
                continue
            cells = [
                c.replace("\x00", "|").strip()
                for c in ln2.strip().strip("|").split("|")
            ]
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                continue  # separator row
            rows.append(cells)
        return rows

    # ── Value parsers ─────────────────────────────────────────────────────────
    def _parse_usd(s):
        if not s or str(s).strip() in ("—", "-", ""):
            return None
        s = str(s).strip().replace(",", "")
        m = re.match(r"^\$([0-9.]+)([BMKbmk]?)$", s)
        if not m:
            return None
        v = float(m.group(1))
        suf = m.group(2).upper()
        if suf == "B":
            return int(v * 1_000_000_000)
        if suf == "M":
            return int(v * 1_000_000)
        if suf == "K":
            return int(v * 1_000)
        return int(v)

    def _recency(label):
        l = re.sub(r"[\[\]]", "", label).strip().lower()
        if "current" in l:
            return "green"
        if "approach" in l:
            return "yellow"
        if "stale" in l:
            return "red"
        return ""

    # ── Preamble: firm name, date, CRD, CIK ──────────────────────────────────
    firm_name = ""
    gen_date_str = ""
    crd_pre = ""
    cik_pre = ""
    for ln in preamble:
        m = re.match(r"^# (.+)$", ln)
        if m:
            firm_name = m.group(1).strip()
        m = re.search(r"Generated:\s*(.+?)$", ln)
        if m:
            gen_date_str = m.group(1).strip()
        m = re.match(r"CRD:\s*`(.+?)`", ln)
        if m:
            crd_pre = m.group(1).strip()
        m = re.match(r"CIK:\s*`(.+?)`", ln)
        if m:
            cik_pre = m.group(1).strip()

    generated_at = ""
    if gen_date_str:
        for fmt_s in ("%b %d, %Y", "%B %d, %Y"):
            try:
                generated_at = (
                    datetime.strptime(gen_date_str, fmt_s)
                    .strftime("%Y-%m-%dT00:00:00+00:00")
                )
                break
            except ValueError:
                pass
        if not generated_at:
            generated_at = gen_date_str

    # ── Firm Overview ─────────────────────────────────────────────────────────
    _, ov_lines = _find_sec("firm", "overview")
    if ov_lines is None:
        _, ov_lines = _find_sec("overview")
    overview: dict = {}
    inst: dict = {}
    _ov_present = ov_lines is not None
    if _ov_present:
        kv = _kv_table(ov_lines)
        ln = kv.get("Legal Name", firm_name)
        if not firm_name:
            firm_name = ln
        overview["legal_name"]          = ln
        overview["crd"]                 = (
            kv.get("CRD Number") or kv.get("CRD") or crd_pre or ""
        )
        overview["cik"]                 = kv.get("CIK") or cik_pre or ""
        overview["website"]             = kv.get("Website", "")
        overview["sec_registered"]      = (
            kv.get("SEC Registered", "No").lower().strip() == "yes"
        )
        overview["registration_status"] = kv.get("Registration Status", "")
        for kk, ik in (
            ("AUM (Total)",         "aum_total_usd"),
            ("AUM (Discretionary)", "aum_discretionary_usd"),
        ):
            v = _parse_usd(kv.get(kk, ""))
            if v is not None:
                inst[ik] = v
        nc = kv.get("Number of Clients", "")
        if nc and nc not in ("—", ""):
            try:
                inst["num_clients"] = int(nc.replace(",", ""))
            except ValueError:
                pass
        ct = kv.get("Client Types", "")
        if ct and ct != "—":
            inst["client_types"] = [x.strip() for x in ct.split(";") if x.strip()]
        fs = kv.get("Fee Structures", "")
        if fs and fs != "—":
            inst["fee_structures"] = [x.strip() for x in fs.split(";") if x.strip()]
        adv_raw = kv.get("ADV Filing Date", "")
        if adv_raw and adv_raw != "—":
            m2 = re.match(r"(\S+)\s*\[([^\]]+)\]", adv_raw)
            if m2:
                inst["adv_filing_date"]  = m2.group(1)
                inst["adv_recency_flag"] = _recency(m2.group(2))
            else:
                inst["adv_filing_date"]  = adv_raw.strip()

    # ── Team Roster / Key Personnel ───────────────────────────────────────────
    _, team_lines = _find_sec("team")
    if team_lines is None:
        _, team_lines = _find_sec("roster")
    if team_lines is None:
        _, team_lines = _find_sec("personnel")
    _team_present = team_lines is not None
    roster: list = []
    if _team_present:
        rows = _pipe_table(team_lines)
        if len(rows) > 1:
            hdr = [h.lower() for h in rows[0]]
            for row in rows[1:]:
                mem: dict = {}
                for i, cell in enumerate(row):
                    if i >= len(hdr):
                        break
                    h = hdr[i]
                    if "name" in h:
                        mem["name"] = cell
                    elif "title" in h:
                        mem["title"] = cell
                    elif "bio" in h or "background" in h:
                        mem["bio"] = cell
                    elif "finra" in h or "broker" in h:
                        mem["finra_status"] = cell
                if mem.get("name") and mem["name"] not in ("", "—"):
                    roster.append(mem)

    # ── SEC Filing Snapshot ───────────────────────────────────────────────────
    _, sec_lines = _find_sec("sec", "filing")
    if sec_lines is None:
        _, sec_lines = _find_sec("sec", "snapshot")
    _sec_present = sec_lines is not None
    f13: dict = {}
    if _sec_present and sec_lines:
        sec_text = "\n".join(sec_lines)
        f13_blk = re.search(
            r"### Form 13F\s*\n(.*?)(?=^###|\Z)", sec_text,
            re.DOTALL | re.MULTILINE,
        )
        if f13_blk:
            for bm in re.finditer(
                r"^\s*[-*]\s+\*\*(.+?):\*\*\s*(.+)$",
                f13_blk.group(1), re.MULTILINE,
            ):
                key = bm.group(1).strip().lower()
                val = bm.group(2).strip()
                if "period" in key:
                    f13["period_of_report"] = val
                elif "filing date" in key:
                    m2 = re.match(r"(\S+)\s*\[([^\]]+)\]", val)
                    if m2:
                        f13["filing_date"]  = m2.group(1)
                        f13["recency_flag"] = _recency(m2.group(2))
                    else:
                        f13["filing_date"]  = val
                elif "total portfolio" in key or "portfolio value" in key:
                    v = _parse_usd(val)
                    if v is not None:
                        f13["total_portfolio_value_usd"] = v
                elif "positions" in key:
                    try:
                        f13["num_positions"] = int(val.replace(",", ""))
                    except ValueError:
                        f13["num_positions"] = val
                elif "concentration" in key:
                    pm = re.match(r"([0-9.]+)%", val)
                    if pm:
                        f13["concentration_top10_pct"] = float(pm.group(1))
        hold_blk = re.search(
            r"#### Top Holdings\s*\n(.*?)(?=^####|\Z)", sec_text,
            re.DOTALL | re.MULTILINE,
        )
        if hold_blk:
            h_rows = _pipe_table(hold_blk.group(1).splitlines())
            if len(h_rows) > 1:
                holdings = []
                for row in h_rows[1:]:
                    if len(row) < 5:
                        continue
                    pm = re.match(r"([0-9.]+)%", row[4].strip())
                    sh_str = row[3].strip().replace(",", "")
                    holdings.append({
                        "name":             row[1].strip(),
                        "value_usd":        _parse_usd(row[2]),
                        "shares":           int(sh_str) if sh_str.isdigit() else 0,
                        "pct_of_portfolio": float(pm.group(1)) if pm else 0.0,
                    })
                if holdings:
                    f13["top_holdings"] = holdings
    if f13:
        inst["latest_13f"] = f13

    # ── Social Signals ────────────────────────────────────────────────────────
    _, sig_lines = _find_sec("social", "signal")
    if sig_lines is None:
        _, sig_lines = _find_sec("social")
    _sig_present = sig_lines is not None
    signals: list = []
    if _sig_present and sig_lines:
        rows = _pipe_table(sig_lines)
        if len(rows) > 1:
            hdr = [h.lower() for h in rows[0]]
            for row in rows[1:]:
                sig: dict = {}
                for i, cell in enumerate(row):
                    if i >= len(hdr):
                        break
                    h = hdr[i]
                    if "date" in h:
                        sig["published_date"] = cell
                    elif "title" in h:
                        sig["title"] = cell
                    elif "type" in h:
                        sig["signal_type"] = cell
                    elif "source" in h:
                        sig["source"] = cell
                if sig.get("title") and sig["title"] not in ("", "—"):
                    signals.append(sig)

    # ── Red Flags ─────────────────────────────────────────────────────────────
    _, rf_lines = _find_sec("red", "flag")
    _rf_present = rf_lines is not None
    parsed_flags: list = []
    if _rf_present and rf_lines:
        for ln in rf_lines:
            m = re.match(r"^\s*[-*]\s+(.+)$", ln)
            if not m:
                continue
            text = m.group(1).strip()
            if "🔴" in text or "●" in text:
                level = "red"
            elif "🟡" in text or "▲" in text:
                level = "yellow"
            else:
                level = "info"
            text = re.sub(r"[🔴🟡🟢●▲✓]\s*", "", text).strip()
            if text:
                parsed_flags.append({"level": level, "message": text})

    # ── Analyst Commentary (custom section) ───────────────────────────────────
    _, ac_lines = _find_sec("analyst")
    analyst_commentary = ""
    if ac_lines is not None:
        ac_parts = [
            ln for ln in ac_lines
            if ln.strip() and not ln.strip().startswith(">")
        ]
        analyst_commentary = "\n".join(ac_parts).strip()

    # ── Legal Disclosures ─────────────────────────────────────────────────────
    _, disc_lines = _find_sec("disclosure")
    if disc_lines is None:
        _, disc_lines = _find_sec("legal")
    disclaimers: dict = {}
    if disc_lines:
        disc_text = "\n".join(disc_lines)
        _DISC_MAP = {
            "not investment advice":           "not_investment_advice",
            "regulatory screening limitation": "regulatory_disclaimer",
            "regulatory limitation":           "regulatory_disclaimer",
            "data limitation":                 "data_limitation",
            "confidentiality":                 "confidentiality",
        }
        for m in re.finditer(
            r"\*\*([^*]+?)\.\*\*\s+(.+?)(?=\n\*\*|\Z)", disc_text, re.DOTALL
        ):
            raw_lbl = m.group(1).strip().lower()
            text    = m.group(2).strip()
            for pat, key in _DISC_MAP.items():
                if pat in raw_lbl:
                    disclaimers[key] = text
                    break
    if not disclaimers:
        disclaimers = {
            "not_investment_advice":
                "For informational purposes only. Not investment advice.",
            "regulatory_disclaimer":
                "Based on public records. Does not replace an ODD review.",
            "data_limitation":
                "Data from public filings. No warranty of accuracy.",
            "confidentiality":
                "Intended solely for the recipient. Redistribution prohibited.",
        }

    # ── Data Provenance ───────────────────────────────────────────────────────
    _, dp_lines = _find_sec("data", "provenance")
    data_sources: list = []
    if dp_lines:
        for ln in dp_lines:
            m = re.match(r"^\s*[-*]\s+(.+)$", ln)
            if m:
                data_sources.append(m.group(1).strip())

    # ── Track which sections were present ─────────────────────────────────────
    _present: set = set()
    if _ov_present:
        _present.add("firm_overview")
    if _team_present:
        _present.add("team_roster")
    if _sec_present:
        _present.add("sec_filing_snapshot")
    if _sig_present:
        _present.add("social_signals")
    if _rf_present:
        _present.add("red_flags")
    if analyst_commentary:
        _present.add("analyst_commentary")

    # ── Assemble Gold Copy dict ───────────────────────────────────────────────
    gold: dict = {
        "firm_name":             firm_name,
        "generated_at":          generated_at,
        "is_legal_ready":        True,
        "firm_overview":         overview,
        "institutional_numbers": inst,
        "team_roster":           roster,
        "social_signals":        signals,
        "sec_enforcement":       [],
        "form_d_filings":        [],
        "sanctions_screening":   None,
        "performance":           None,
        "modules_skipped":       {},
        "data_sources":          data_sources,
        "disclosure": {
            "version":          "1.0",
            "data_sources":     data_sources,
            "modules_skipped":  {},
            "disclaimers":      disclaimers,
        },
        "_analyst_commentary":   analyst_commentary,
        "_from_md":              {"sections_present": _present},
    }
    if parsed_flags:
        gold["_parsed_red_flags"] = parsed_flags
    return gold


# ─────────────────────────────────────────────────────────────────────────────
# PDF renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_pdf(gold: dict, out_path: Path) -> Path:
    """Render Gold Copy to a two-page institutional PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.platypus import (
            BaseDocTemplate, Frame, HRFlowable, KeepTogether,
            NextPageTemplate, PageBreak, PageTemplate,
            Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(f"reportlab is required for PDF output: {exc}") from exc

    # ── Palette ───────────────────────────────────────────────────────────────
    NAVY       = colors.HexColor(_NAVY_HEX)
    GOLD       = colors.HexColor(_GOLD_HEX)
    WHITE      = colors.white
    GRAY_DARK  = colors.HexColor(_GRAY_DARK)
    GRAY_MID   = colors.HexColor(_GRAY_MID)
    GRAY_LITE  = colors.HexColor(_GRAY_LITE)
    GRAY_DISC  = colors.HexColor(_GRAY_DISC)
    RED_COL    = colors.HexColor(_RED_HEX)
    GREEN_COL  = colors.HexColor(_GREEN_HEX)
    AMBER_COL  = colors.HexColor(_AMBER_HEX)

    # ── Page geometry ─────────────────────────────────────────────────────────
    PAGE_W, PAGE_H = letter          # 612 × 792 pt
    MARGIN_H   = 0.65 * inch
    MARGIN_TOP = 0.75 * inch
    HEADER_H   = 0.52 * inch
    # Bottom margin reserves space for the compact disclosure footer
    FOOTER_H   = 0.95 * inch         # 4 disclosure lines + generation line
    CONTENT_W  = PAGE_W - 2 * MARGIN_H
    CONTENT_H  = PAGE_H - HEADER_H - MARGIN_TOP - FOOTER_H

    # ── Styles ────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    def S(name, **kw):
        parent = kw.pop("parent", "Normal")
        return ParagraphStyle(name, parent=base[parent], **kw)

    sty = {
        "h1":          S("fmt_h1",  fontName="Helvetica-Bold",   fontSize=11,
                          textColor=NAVY,      spaceAfter=2, spaceBefore=3, leading=14),
        "h2":          S("fmt_h2",  fontName="Helvetica-Bold",   fontSize=9,
                          textColor=NAVY,      spaceAfter=2, spaceBefore=3,  leading=12),
        "h3":          S("fmt_h3",  fontName="Helvetica-BoldOblique", fontSize=8,
                          textColor=GRAY_DARK, spaceAfter=2, spaceBefore=4,  leading=11),
        "body":        S("fmt_body",fontName="Helvetica",         fontSize=8.5,
                          textColor=GRAY_DARK, leading=12, spaceAfter=3),
        "body_sm":     S("fmt_bsm", fontName="Helvetica",         fontSize=7.5,
                          textColor=GRAY_DARK, leading=10.5, spaceAfter=2),
        "italic":      S("fmt_it",  fontName="Helvetica-Oblique", fontSize=8,
                          textColor=GRAY_MID,  leading=11, spaceAfter=2),
        "kv_key":      S("fmt_kvk", fontName="Helvetica-Bold",   fontSize=7,
                          textColor=NAVY,      leading=10),
        "kv_val":      S("fmt_kvv", fontName="Helvetica",         fontSize=7,
                          textColor=GRAY_DARK, leading=10),
        "tbl_hdr":     S("fmt_th",  fontName="Helvetica-Bold",   fontSize=7.5,
                          textColor=WHITE,     leading=10),
        "tbl_cell":    S("fmt_tc",  fontName="Helvetica",         fontSize=7.5,
                          textColor=GRAY_DARK, leading=10),
        "tbl_cell_sm": S("fmt_tcs", fontName="Helvetica",         fontSize=7,
                          textColor=GRAY_DARK, leading=9.5),
        "legal":       S("fmt_leg", fontName="Helvetica",         fontSize=7.5,
                          textColor=GRAY_DISC, leading=10.5, spaceAfter=2,
                          alignment=TA_JUSTIFY),
        "footer_txt":  S("fmt_ft",  fontName="Helvetica",         fontSize=7,
                          textColor=GRAY_MID,  leading=9, alignment=TA_CENTER),
    }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def gold_rule():
        return HRFlowable(width=CONTENT_W, thickness=1, color=GOLD,
                          spaceAfter=4, spaceBefore=4)
    def navy_rule():
        return HRFlowable(width=CONTENT_W, thickness=0.5, color=NAVY,
                          spaceAfter=3, spaceBefore=2)

    firm_name    = gold.get("firm_name", "Unknown Firm")
    generated_at = gold.get("generated_at", "")
    gen_date     = _fmt_date(generated_at)
    disclosure   = gold.get("disclosure") or {}
    disc_layers  = _build_disclosure_layers(disclosure)

    # ── Canvas decorator (header bar + footer on every page) ──────────────────
    def _on_page(canvas, doc):
        canvas.saveState()

        # ── Header bar ────────────────────────────────────────────────────────
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(2)
        canvas.line(0, PAGE_H - HEADER_H, PAGE_W, PAGE_H - HEADER_H)
        # Firm name
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(MARGIN_H, PAGE_H - HEADER_H + 14,
                          _trunc(firm_name, 60))
        # Strategy pill (below firm name)
        _strategy = gold.get("strategy_type") or gold.get("fund_type")
        if _strategy:
            _pill_label = _trunc(str(_strategy), 30)
            canvas.setFont("Helvetica-Bold", 6.5)
            _pill_w = canvas.stringWidth(_pill_label, "Helvetica-Bold", 6.5) + 10
            _pill_x = MARGIN_H
            _pill_y = PAGE_H - HEADER_H + 3
            canvas.setFillColor(GOLD)
            canvas.roundRect(_pill_x, _pill_y, _pill_w, 9, 2, fill=1, stroke=0)
            canvas.setFillColor(NAVY)
            canvas.drawString(_pill_x + 5, _pill_y + 2.5, _pill_label)

        # Service providers block (below strategy pill)
        _svc = gold.get("service_providers")
        if _svc:
            _MGRAY = colors.HexColor("#CCCCCC")
            canvas.setFont("Helvetica", 6.5)
            canvas.setFillColor(_MGRAY)
            _custs = _svc.get("custodians") or []
            if isinstance(_custs, list):
                _custs_str = "  |  ".join(str(c) for c in _custs if c)
            else:
                _custs_str = str(_custs)
            _auditor   = _svc.get("auditor") or ""
            _fund_admin = _svc.get("fund_administrator") or ""
            _svc_y1 = PAGE_H - HEADER_H - 9
            _svc_y2 = PAGE_H - HEADER_H - 18
            if _custs_str:
                canvas.drawString(MARGIN_H, _svc_y1,
                                  _trunc(f"Custodians: {_custs_str}", 90))
            _line2_parts = []
            if _auditor:
                _line2_parts.append(f"Auditor: {_auditor}")
            if _fund_admin:
                _line2_parts.append(f"Fund Admin: {_fund_admin}")
            if _line2_parts:
                canvas.drawString(MARGIN_H, _svc_y2,
                                  _trunc("  |  ".join(_line2_parts), 90))

        # Report type
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GOLD)
        canvas.drawRightString(PAGE_W - MARGIN_H, PAGE_H - HEADER_H + 14,
                               "Institutional Intelligence Report")
        canvas.setFont("Helvetica-Oblique", 7)
        canvas.setFillColor(colors.HexColor("#A0A8B8"))
        canvas.drawRightString(PAGE_W - MARGIN_H, PAGE_H - HEADER_H + 4,
                               "Prepared by AltBots")

        # ── Peer flags box (top-right of header) ─────────────────────────────
        _pf = gold.get("peer_flags")
        if _pf:
            _PF_LEFT   = 5.62 * inch
            _PF_W      = PAGE_W - _PF_LEFT - MARGIN_H
            _PF_X      = _PF_LEFT
            _PF_Y      = PAGE_H - HEADER_H          # bottom of header bar
            _PF_H      = HEADER_H
            # Dark navy background with gold rounded border
            canvas.setFillColor(colors.HexColor("#111E33"))
            canvas.roundRect(_PF_X, _PF_Y, _PF_W, _PF_H, 4, fill=1, stroke=0)
            canvas.setStrokeColor(GOLD)
            canvas.setLineWidth(0.5)
            canvas.roundRect(_PF_X, _PF_Y, _PF_W, _PF_H, 4, fill=0, stroke=1)
            # "PEER FLAGS" header label
            _pf_cx = _PF_X + _PF_W / 2
            canvas.setFont("Helvetica-Bold", 6.5)
            canvas.setFillColor(GOLD)
            _pf_label_y = _PF_Y + _PF_H - 8
            canvas.drawCentredString(_pf_cx, _pf_label_y, "PEER FLAGS")
            # Thin gold rule below header label
            canvas.setStrokeColor(GOLD)
            canvas.setLineWidth(0.4)
            _pf_rule_y = _PF_Y + _PF_H - 11
            canvas.line(_PF_X + 3, _pf_rule_y, _PF_X + _PF_W - 3, _pf_rule_y)
            # ABOVE PEER section
            _above = (_pf.get("above_peer") or [])[:4]
            _cy = _pf_rule_y - 7
            canvas.setFont("Helvetica-Bold", 6)
            canvas.setFillColor(colors.HexColor("#27AE60"))
            canvas.drawString(_PF_X + 4, _cy, "ABOVE PEER")
            _cy -= 7
            canvas.setFont("Helvetica", 5.8)
            canvas.setFillColor(colors.HexColor("#A8F0C0"))
            for _item in _above:
                canvas.drawString(_PF_X + 4, _cy, _trunc(f"+ {_item}", 38))
                _cy -= 6.5
            # Thin dark separator
            canvas.setStrokeColor(colors.HexColor("#2A3A55"))
            canvas.setLineWidth(0.3)
            canvas.line(_PF_X + 3, _cy, _PF_X + _PF_W - 3, _cy)
            _cy -= 5
            # BELOW PEER section
            _below = (_pf.get("below_peer") or [])[:3]
            canvas.setFont("Helvetica-Bold", 6)
            canvas.setFillColor(colors.HexColor("#C0392B"))
            canvas.drawString(_PF_X + 4, _cy, "BELOW PEER")
            _cy -= 7
            canvas.setFont("Helvetica", 5.8)
            canvas.setFillColor(colors.HexColor("#F5AAAA"))
            for _item in _below:
                canvas.drawString(_PF_X + 4, _cy, _trunc(f"- {_item}", 38))
                _cy -= 6.5

        # ── Footer: compact 4-layer disclosure + generation line ──────────────
        labels = [
            "NOT INVESTMENT ADVICE",
            "REGULATORY LIMITATION",
            "DATA LIMITATION",
            "CONFIDENTIALITY",
        ]
        summaries = [
            "For informational purposes only. Not investment advice or a recommendation to buy or sell any security.",
            "Based on public records only. Does not replace a comprehensive ODD review.",
            "Data from public filings and third-party providers. No warranty of accuracy or completeness.",
            "Intended solely for the recipient. Redistribution prohibited without written consent.",
        ]
        # Fall back to actual text from envelope (first ~120 chars) if available
        for i, layer in enumerate(disc_layers):
            if layer and len(layer) > 20:
                summaries[i] = _trunc(layer, 130)

        canvas.setFont("Helvetica-Bold", 6)
        canvas.setFillColor(GRAY_DISC)
        # Gold rule above footer
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(0.75)
        footer_top = FOOTER_H - 4
        canvas.line(MARGIN_H, footer_top, PAGE_W - MARGIN_H, footer_top)

        line_h = 9.5   # pt per disclosure line
        y = footer_top - 10
        for label, summary in zip(labels, summaries):
            canvas.setFont("Helvetica-Bold", 6)
            canvas.setFillColor(GRAY_DISC)
            canvas.drawString(MARGIN_H, y, label + ".")
            canvas.setFont("Helvetica", 6)
            canvas.drawString(MARGIN_H + 82, y, summary)
            y -= line_h

        # Generation line + page number
        y -= 3
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(GRAY_MID)
        canvas.drawCentredString(PAGE_W / 2, y,
                                 f"Generated by AltBots  ·  {gen_date}")
        canvas.drawRightString(PAGE_W - MARGIN_H, y,
                               f"Page {doc.page}")

        canvas.restoreState()

    # ── Document setup ────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=MARGIN_H,
        rightMargin=MARGIN_H,
        topMargin=MARGIN_TOP + HEADER_H,
        bottomMargin=FOOTER_H,
    )

    full_frame = Frame(
        MARGIN_H, FOOTER_H, CONTENT_W, CONTENT_H,
        id="full", leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )
    page1_tmpl = PageTemplate(id="page1", frames=[full_frame], onPage=_on_page)
    page2_tmpl = PageTemplate(id="page2", frames=[full_frame], onPage=_on_page)
    doc.addPageTemplates([page1_tmpl, page2_tmpl])

    # ══════════════════════════════════════════════════════════════════════════
    # Story
    # ══════════════════════════════════════════════════════════════════════════
    story = []

    # ── From-MD section gating ────────────────────────────────────────────────
    _from_md_info = gold.get("_from_md") or {}
    _secs_present = _from_md_info.get("sections_present")  # None → show all

    def _should_render(key: str) -> bool:
        return _secs_present is None or key in _secs_present

    # ── Analyst Commentary box (from-md only): rendered first on page 1 ───────
    _ac_text = (gold.get("_analyst_commentary") or "").strip()
    if _ac_text:
        _ac_title_sty = ParagraphStyle(
            "_ac_title", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=9,
            textColor=GOLD, leading=12, spaceAfter=4,
        )
        _ac_body_sty = ParagraphStyle(
            "_ac_body", parent=base["Normal"],
            fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=WHITE, leading=12,
        )
        _ac_box = Table(
            [
                [Paragraph("ANALYST COMMENTARY", _ac_title_sty)],
                [Paragraph(_ac_text.replace("\n", "<br/>"), _ac_body_sty)],
            ],
            colWidths=[CONTENT_W],
        )
        _ac_box.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
            ("BOX",           (0, 0), (-1, -1), 2,    GOLD),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(_ac_box)
        story.append(Spacer(1, 8))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1  — Firm Overview + Team Roster
    # ─────────────────────────────────────────────────────────────────────────

    # ── Subheader row (CRD / CIK / Generated date) ───────────────────────────
    overview = gold.get("firm_overview") or {}
    inst     = gold.get("institutional_numbers") or {}

    meta_parts = []
    if overview.get("crd"):
        meta_parts.append(f"CRD {overview['crd']}")
    if overview.get("cik"):
        meta_parts.append(f"CIK {overview['cik']}")
    meta_parts.append(f"Generated {gen_date}")
    story.append(Paragraph(" · ".join(meta_parts), sty["italic"]))
    story.append(gold_rule())

    # ── Section 1: Firm Overview ──────────────────────────────────────────────
    story.append(Paragraph("Firm Overview", sty["h1"]))
    story.append(navy_rule())

    ov_rows: list[tuple[str, str]] = []
    ov_rows.append(("Legal Name",   _safe(overview.get("legal_name"))))
    if overview.get("crd"):
        ov_rows.append(("CRD Number", overview["crd"]))
    if overview.get("cik"):
        ov_rows.append(("CIK", overview["cik"]))
    if overview.get("website"):
        ov_rows.append(("Website", _trunc(_safe(overview["website"]).lower(), 60)))
    if overview.get("inception_year"):
        ov_rows.append(("Inception Year", str(overview["inception_year"])))
    if overview.get("hq_city"):
        ov_rows.append(("HQ City", overview["hq_city"]))
    ov_rows.append(("SEC Registered",
                    "Yes" if overview.get("sec_registered") else "No"))
    ov_rows.append(("Registration Status",
                    _safe(overview.get("registration_status"))))

    if inst.get("aum_total_usd") is not None:
        ov_rows.append(("AUM (Total)",         _fmt_usd(inst["aum_total_usd"])))
        ov_rows.append(("AUM (Discretionary)",  _fmt_usd(inst.get("aum_discretionary_usd"))))
    if inst.get("num_clients") is not None:
        ov_rows.append(("Number of Clients",   str(inst["num_clients"])))
    if inst.get("client_types"):
        ov_rows.append(("Client Types",
                        _trunc(", ".join(str(c) for c in inst["client_types"]), 80)))
    if inst.get("fee_structures"):
        ov_rows.append(("Fee Structures",
                        _trunc(", ".join(str(f) for f in inst["fee_structures"]), 80)))
    if inst.get("adv_filing_date"):
        flag   = inst.get("adv_recency_flag", "")
        label  = _recency_label(flag)
        ov_rows.append(("ADV Filing Date",
                        f"{inst['adv_filing_date']}  [{label}]"))

    col_w = [1.55 * inch, CONTENT_W - 1.55 * inch]
    tdata = [
        [Paragraph(k, sty["kv_key"]), Paragraph(v, sty["kv_val"])]
        for k, v in ov_rows
    ]
    tbl = Table(tdata, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS",(0, 0),(-1, -1), [WHITE, GRAY_LITE]),
        ("GRID",         (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D0D0")),
    ]))
    story.append(tbl)

    # ── Section 2: Team Roster ────────────────────────────────────────────────
    roster = gold.get("team_roster") or []

    def _finra_style(val: str):
        """Return a ParagraphStyle with color based on FINRA status value."""
        v = val.lower()
        if val in ("No disclosures", "Clear"):
            col = colors.HexColor("#27AE60")
        elif "disclosure" in v or "flag" in v:
            col = colors.HexColor("#C0392B")
        else:
            col = colors.HexColor("#555555")
        return ParagraphStyle("finra_cell", parent=base["Normal"],
                              fontName="Helvetica", fontSize=7,
                              textColor=col, leading=10)

    if _should_render("team_roster"):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Team Roster", sty["h1"]))
        story.append(navy_rule())
        if roster:
            headers  = ["Name", "Title", "Professional Background", "FINRA / BrokerCheck"]
            col_w2   = [1.20 * inch, 1.55 * inch,
                        CONTENT_W - 2.75 * inch - 1.0 * inch, 1.0 * inch]
            tdata2   = [[Paragraph(h, sty["tbl_hdr"]) for h in headers]]
            page_rows = roster[:6]
            for member in page_rows:
                bio_text    = _trunc(_safe(member.get("bio"), "Profile available upon request."), 300)
                finra_val   = str(member.get("finra_status") or "Not queried")
                tdata2.append([
                    Paragraph(_safe(member.get("name")),  sty["tbl_cell"]),
                    Paragraph(_safe(member.get("title")), sty["tbl_cell"]),
                    Paragraph(bio_text,                   sty["tbl_cell_sm"]),
                    Paragraph(finra_val,                  _finra_style(finra_val)),
                ])
            overflow = len(roster) - 6
            if overflow > 0:
                note_sty = ParagraphStyle("roster_note", parent=base["Normal"],
                                          fontName="Helvetica-Oblique", fontSize=6.5,
                                          textColor=colors.HexColor("#555555"), leading=9)
                tdata2.append([
                    Paragraph(
                        f"+ {overflow} additional personnel on file. "
                        "Full roster available on request.",
                        note_sty,
                    ), "", "", "",
                ])
            tbl2 = Table(tdata2, colWidths=col_w2, repeatRows=1)
            _note_row = len(tdata2) - 1 if overflow > 0 else None
            _tbl2_style = [
                ("BACKGROUND",    (0, 0), (-1, 0),   NAVY),
                ("TEXTCOLOR",     (0, 0), (-1, 0),   WHITE),
                ("VALIGN",        (0, 0), (-1, -1),  "TOP"),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [WHITE, GRAY_LITE]),
                ("LEFTPADDING",   (0, 0), (-1, -1),  4),
                ("RIGHTPADDING",  (0, 0), (-1, -1),  4),
                ("TOPPADDING",    (0, 0), (-1, -1),  3),
                ("BOTTOMPADDING", (0, 0), (-1, -1),  3),
                ("GRID",          (0, 0), (-1, -1),  0.25, colors.HexColor("#C0C0C0")),
            ]
            if _note_row is not None:
                _tbl2_style += [
                    ("SPAN",          (0, _note_row), (-1, _note_row)),
                    ("BACKGROUND",    (0, _note_row), (-1, _note_row), GRAY_LITE),
                ]
            tbl2.setStyle(TableStyle(_tbl2_style))
            story.append(tbl2)
            story.append(Paragraph(
                f"Source: AltBots team profiles pipeline · {len(roster)} member(s) · "
                f"Bios synthesized by Claude Haiku",
                sty["italic"],
            ))
        else:
            _empty_sty = ParagraphStyle("roster_empty", parent=base["Normal"],
                                        fontName="Helvetica-Oblique", fontSize=6.5,
                                        textColor=colors.HexColor("#555555"), leading=9)
            headers  = ["Name", "Title", "Professional Background", "FINRA / BrokerCheck"]
            col_w2   = [1.20 * inch, 1.55 * inch,
                        CONTENT_W - 2.75 * inch - 1.0 * inch, 1.0 * inch]
            tdata2   = [[Paragraph(h, sty["tbl_hdr"]) for h in headers],
                        [Paragraph("Personnel data unavailable for this entity.",
                                   _empty_sty), "", "", ""]]
            tbl2 = Table(tdata2, colWidths=col_w2)
            tbl2.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  NAVY),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
                ("SPAN",         (0, 1), (-1, 1)),
                ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING",   (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                ("GRID",         (0, 0), (-1, -1), 0.25, colors.HexColor("#C0C0C0")),
            ]))
            story.append(tbl2)

    # ── Section 2b: Social Signals (page 1) ──────────────────────────────────
    signals = gold.get("social_signals") or []
    if _should_render("social_signals"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("Social Signals  (Exa · Claude Haiku · Last 90 Days)", sty["h1"]))
        story.append(navy_rule())
        if signals:
            s_headers = ["Date", "Title", "Source", "Type"]
            s_widths  = [0.75*inch, 3.20*inch, 0.80*inch, CONTENT_W - 4.75*inch]
            s_data    = [[Paragraph(h, sty["tbl_hdr"]) for h in s_headers]]
            for sig in signals[:5]:
                pub = sig.get("published_date", "")
                s_data.append([
                    Paragraph(_fmt_date(pub)[:10],                   sty["tbl_cell_sm"]),
                    Paragraph(_trunc(_safe(sig.get("title")), 80),   sty["tbl_cell_sm"]),
                    Paragraph(_safe(sig.get("source", "—")),         sty["tbl_cell_sm"]),
                    Paragraph(_safe(sig.get("signal_type", "—")),    sty["tbl_cell_sm"]),
                ])
            tbl_s = Table(s_data, colWidths=s_widths, repeatRows=1)
            tbl_s.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LITE]),
                ("LEFTPADDING",   (0, 0), (-1, -1), 3),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
                ("TOPPADDING",    (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#C0C0C0")),
            ]))
            story.append(tbl_s)
        else:
            story.append(Paragraph("No social signals retrieved.", sty["italic"]))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2  — SEC Snapshot + Sanctions + Performance + Red Flags + Disclosure
    # ─────────────────────────────────────────────────────────────────────────
    story.append(NextPageTemplate("page2"))
    story.append(PageBreak())

    # ── Section 3: SEC Filing Snapshot (3-panel side-by-side) ────────────────
    story.append(Paragraph("SEC Filing Snapshot", sty["h1"]))
    story.append(navy_rule())

    f13 = inst.get("latest_13f") or {}

    # Shared panel geometry and styles
    _PW  = CONTENT_W / 3          # each panel's full width
    _PKW = _PW * 0.42              # key column within a KV panel
    _PVW = _PW * 0.58              # value column within a KV panel

    _ph_sty = ParagraphStyle("_ph",  parent=base["Normal"], fontName="Helvetica-Bold",
                              fontSize=7, textColor=WHITE, leading=10)
    _pk_sty = ParagraphStyle("_pk",  parent=base["Normal"], fontName="Helvetica-Bold",
                              fontSize=6.5, textColor=NAVY, leading=9)
    _pv_sty = ParagraphStyle("_pv",  parent=base["Normal"], fontName="Helvetica",
                              fontSize=6.5, textColor=GRAY_DARK, leading=9)
    _pi_sty = ParagraphStyle("_pi",  parent=base["Normal"], fontName="Helvetica-Oblique",
                              fontSize=6.5, textColor=GRAY_MID, leading=9)

    _KV_PANEL_STYLE = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("SPAN",          (0, 0), (-1, 0)),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LITE]),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D0D0")),
    ])

    # ── Panel 1: ADV Filing Summary ──────────────────────────────────────────
    p1_data = [[Paragraph("ADV FILING SUMMARY", _ph_sty), ""]]
    if inst.get("adv_filing_date"):
        _flag = inst.get("adv_recency_flag", "")
        p1_data.append([Paragraph("ADV Filing Date", _pk_sty),
                         Paragraph(f"{inst['adv_filing_date']}  [{_recency_label(_flag)}]",
                                   _pv_sty)])
    p1_data.append([Paragraph("Reg. Status",   _pk_sty),
                     Paragraph(_safe(overview.get("registration_status")), _pv_sty)])
    p1_data.append([Paragraph("SEC Registered", _pk_sty),
                     Paragraph("Yes" if overview.get("sec_registered") else "No", _pv_sty)])
    if inst.get("aum_total_usd") is not None:
        p1_data.append([Paragraph("AUM (Total)",   _pk_sty),
                         Paragraph(_fmt_usd(inst["aum_total_usd"]), _pv_sty)])
    if inst.get("num_clients") is not None:
        p1_data.append([Paragraph("Clients",       _pk_sty),
                         Paragraph(str(inst["num_clients"]), _pv_sty)])
    if inst.get("client_types"):
        p1_data.append([Paragraph("Client Types",  _pk_sty),
                         Paragraph(_trunc(", ".join(str(c) for c in inst["client_types"]), 40),
                                   _pv_sty)])
    if inst.get("fee_structures"):
        p1_data.append([Paragraph("Fee Structures", _pk_sty),
                         Paragraph(_trunc(", ".join(str(f) for f in inst["fee_structures"]), 40),
                                   _pv_sty)])
    p1_tbl = Table(p1_data, colWidths=[_PKW, _PVW])
    p1_tbl.setStyle(_KV_PANEL_STYLE)

    # ── Panel 2: 13F Holdings Snapshot ───────────────────────────────────────
    p2_data = [[Paragraph("13F HOLDINGS SNAPSHOT", _ph_sty), ""]]
    if f13:
        for _k, _v in [
            ("Period",      _safe(f13.get("period_of_report"))),
            ("Filed",       _safe(f13.get("filing_date"))),
            ("Recency",     _recency_label(f13.get("recency_flag", ""))),
            ("Total Value", _fmt_usd(f13.get("total_portfolio_value_usd"))),
            ("Positions",   _safe(f13.get("num_positions"))),
            ("Top-10 Conc.",
             f"{f13['concentration_top10_pct']:.1f}%"
             if f13.get("concentration_top10_pct") is not None else "—"),
        ]:
            p2_data.append([Paragraph(_k, _pk_sty), Paragraph(_v, _pv_sty)])
        for _h in (f13.get("top_holdings") or [])[:3]:
            p2_data.append([
                Paragraph(_trunc(_safe(_h.get("name")), 20), _pv_sty),
                Paragraph(f"{_h.get('pct_of_portfolio', 0):.1f}%", _pv_sty),
            ])
    else:
        p2_data.append([Paragraph("No 13F data available.", _pi_sty), ""])
    p2_tbl = Table(p2_data, colWidths=[_PKW, _PVW])
    p2_tbl.setStyle(_KV_PANEL_STYLE)

    # ── Panel 3: Enforcement and Form D ──────────────────────────────────────
    _enf_list  = gold.get("sec_enforcement")  or []
    _fd_list   = gold.get("form_d_filings")   or []
    _DGRAY_COL = colors.HexColor("#555555")
    _GREEN_COL = colors.HexColor("#27AE60")

    _penf_body = ParagraphStyle("_peb", parent=base["Normal"], fontName="Helvetica",
                                 fontSize=6.5, textColor=GRAY_DARK, leading=9)
    _penf_none = ParagraphStyle("_pen", parent=base["Normal"], fontName="Helvetica-Oblique",
                                 fontSize=6.5, textColor=_GREEN_COL, leading=9)
    _pfd_name  = ParagraphStyle("_pfn", parent=base["Normal"], fontName="Helvetica-Bold",
                                 fontSize=6.5, textColor=GRAY_DARK, leading=9)
    _pfd_det   = ParagraphStyle("_pfd", parent=base["Normal"], fontName="Helvetica",
                                 fontSize=6.5, textColor=_DGRAY_COL, leading=9)
    _pfd_foot  = ParagraphStyle("_pff", parent=base["Normal"], fontName="Helvetica-Oblique",
                                 fontSize=6,   textColor=_DGRAY_COL, leading=8)
    _p3_sep    = ParagraphStyle("_sep", parent=base["Normal"], fontName="Helvetica",
                                 fontSize=3, textColor=GRAY_MID, leading=4,
                                 spaceBefore=1, spaceAfter=1)

    p3_data = [[Paragraph("ENFORCEMENT AND FORM D", _ph_sty)]]
    if _enf_list:
        for _e in _enf_list:
            _line = _trunc(
                f"{_safe(_e.get('date', ''))}  {_safe(_e.get('type', ''))}: "
                f"{_safe(_e.get('summary', ''))}",
                80)
            p3_data.append([Paragraph(_line, _penf_body)])
    else:
        p3_data.append([Paragraph("No enforcement actions identified.", _penf_none)])
    p3_data.append([Paragraph("─" * 28, _p3_sep)])   # thin separator
    if _fd_list:
        for _fd in _fd_list[:3]:
            p3_data.append([Paragraph(_trunc(_safe(_fd.get("fund_name", "")), 35), _pfd_name)])
            _det = _trunc(
                f"{_safe(_fd.get('amount', ''))}  |  "
                f"{_safe(_fd.get('exemption', ''))}  |  "
                f"{_safe(_fd.get('year', ''))}",
                55)
            p3_data.append([Paragraph(_det, _pfd_det)])
        p3_data.append([Paragraph(f"Filing count: {len(_fd_list)} on record", _pfd_foot)])
    else:
        p3_data.append([Paragraph("No Form D filings on record.", _pfd_det)])

    p3_tbl = Table(p3_data, colWidths=[_PW - 6])
    p3_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LITE]),
        ("LEFTPADDING",   (0, 0), (-1, -1), 3),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D0D0")),
    ]))

    # ── Outer 3-panel row ────────────────────────────────────────────────────
    _outer_tbl = Table([[p1_tbl, p2_tbl, p3_tbl]], colWidths=[_PW, _PW, _PW])
    _outer_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(_outer_tbl)

    # ── Section 3b: Sanctions and Adverse Screening ───────────────────────────
    story.append(Spacer(1, 4))
    story.append(Paragraph("Sanctions and Adverse Screening", sty["h1"]))
    story.append(navy_rule())

    _SS_DATABASES = [
        "OFAC SDN List",
        "EU Consolidated Sanctions",
        "UN Security Council",
        "UK HMT Sanctions",
        "FinCEN Advisories",
        "PEP Database (Tier 1+2)",
        "FINRA BrokerCheck",
        "SEC Enforcement Actions",
        "CFTC Actions Database",
        "Adverse Media (AI scan)",
    ]
    _ss_data = gold.get("sanctions_screening")
    if _ss_data is None:
        story.append(Paragraph(
            "Sanctions screening data unavailable.",
            ParagraphStyle("_ss_na", parent=base["Normal"],
                           fontName="Helvetica-Oblique", fontSize=7.5,
                           textColor=colors.HexColor("#555555"), leading=10),
        ))
    else:
        # Paired layout: 5 rows × 8 cols (two 4-col panels side-by-side)
        # Halves vertical height vs a single 10-row table
        _ss_cw = [0.81*inch, 0.87*inch, 0.95*inch, 0.51*inch]   # per panel
        _ss_hdr_widths = _ss_cw + _ss_cw                         # 8 columns total
        _ss_hdr_labels = ["Database", "Scope", "Result", "Date"]
        _ss_tdata = [[Paragraph(h, sty["tbl_hdr"]) for h in _ss_hdr_labels * 2]]

        _ss_cell = ParagraphStyle("_ss_c", parent=base["Normal"],
                                  fontName="Helvetica", fontSize=6.5,
                                  textColor=GRAY_DARK, leading=9)

        def _ss_result_para(val: str) -> "Paragraph":
            _v = str(val)
            if _v == "CLEAR" or _v.startswith("CLEAR"):
                _s = ParagraphStyle("_ssr_clr", parent=base["Normal"],
                                    fontName="Helvetica-Bold", fontSize=6.5,
                                    textColor=colors.HexColor("#27AE60"), leading=9)
            elif "FLAG" in _v or "flag" in _v:
                _s = ParagraphStyle("_ssr_flg", parent=base["Normal"],
                                    fontName="Helvetica-Bold", fontSize=6.5,
                                    textColor=colors.HexColor("#C0392B"), leading=9)
            elif _v == "Not queried":
                _s = ParagraphStyle("_ssr_nq", parent=base["Normal"],
                                    fontName="Helvetica-Oblique", fontSize=6.5,
                                    textColor=colors.HexColor("#555555"), leading=9)
            else:
                _s = _ss_cell
            return Paragraph(_v, _s)

        def _ss_row_cells(db):
            _entry  = (_ss_data.get(db) or {}) if isinstance(_ss_data, dict) else {}
            _scope  = _entry.get("scope",        "All principals")
            _result = _entry.get("result",       "Not queried")
            _date_q = _entry.get("date_queried", "")
            return [Paragraph(_trunc(db, 22), _ss_cell),
                    Paragraph(_scope,          _ss_cell),
                    _ss_result_para(_result),
                    Paragraph(_date_q,         _ss_cell)]

        _left  = _SS_DATABASES[:5]
        _right = _SS_DATABASES[5:]
        for _l, _r in zip(_left, _right):
            _ss_tdata.append(_ss_row_cells(_l) + _ss_row_cells(_r))

        _ss_tbl = Table(_ss_tdata, colWidths=_ss_hdr_widths, repeatRows=1)
        _ss_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LITE]),
            ("LEFTPADDING",   (0, 0), (-1, -1), 3),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#C0C0C0")),
            # Vertical divider between the two panels
            ("LINEAFTER",     (3, 0), (3, -1),  1.5, NAVY),
        ]))
        story.append(_ss_tbl)
        story.append(Paragraph(
            "Sources: OFAC SDN, EU/UN/HMT sanctions registers, FinCEN, FINRA BrokerCheck, "
            f"SEC/CFTC enforcement databases, adverse media AI scan  |  All queried {gen_date}",
            ParagraphStyle("_ss_attr", parent=base["Normal"],
                           fontName="Helvetica-Oblique", fontSize=6,
                           textColor=colors.HexColor("#555555"), leading=8,
                           spaceBefore=2),
        ))
        # Adverse media flag callout
        _am      = (_ss_data.get("adverse_media") or {}) if isinstance(_ss_data, dict) else {}
        _am_flag = _am.get("flag")
        if _am_flag:
            _am_note   = str(_am.get("note", ""))
            _am_red    = colors.HexColor("#C0392B")
            _am_txt_sty = ParagraphStyle("_am_call", parent=base["Normal"],
                                         fontName="Helvetica", fontSize=7.5,
                                         textColor=_am_red, leading=10)
            _am_box = Table(
                [[Paragraph(f"<b>Adverse Media: </b>{_am_note}", _am_txt_sty)]],
                colWidths=[CONTENT_W],
            )
            _am_box.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FDF2F2")),
                ("LINEBEFORE",    (0, 0), (-1, -1), 2, _am_red),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(Spacer(1, 3))
            story.append(_am_box)
            story.append(Paragraph(
                "Adverse media classification by AI. May not represent verified facts.",
                ParagraphStyle("_am_src", parent=base["Normal"],
                               fontName="Helvetica-Oblique", fontSize=6,
                               textColor=colors.HexColor("#555555"), leading=8,
                               spaceBefore=1),
            ))

    # ── Section 3c: Indicative Performance Signals ───────────────────────────
    _perf = gold.get("performance")
    if _perf is not None:
        story.append(Spacer(1, 4))
        story.append(Paragraph("Indicative Performance Signals", sty["h1"]))
        story.append(navy_rule())

        # Confidence badge
        _conf_val = str(_perf.get("confidence_level") or "Estimated")
        _conf_colors = {
            "Estimated": colors.HexColor("#E67E22"),
            "Reported":  NAVY,
            "Audited":   colors.HexColor("#27AE60"),
        }
        _conf_bg = _conf_colors.get(_conf_val, colors.HexColor("#E67E22"))
        _badge_label = f"CONFIDENCE: {_conf_val}"
        # Badge rendered as a canvas-draw won't work inside story; use a small Table pill
        _badge_sty = ParagraphStyle("_perf_badge", parent=base["Normal"],
                                    fontName="Helvetica-Bold", fontSize=7,
                                    textColor=WHITE, leading=9)
        _badge_tbl = Table([[Paragraph(_badge_label, _badge_sty)]],
                           colWidths=[1.85 * inch])
        _badge_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _conf_bg),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(_badge_tbl)
        story.append(Spacer(1, 2))

        # Performance table
        def _pf(key):
            v = _perf.get(key)
            return "N/A" if v is None else str(v)

        _perf_col_w = [1.87 * inch] * 4
        _lbl_sty = ParagraphStyle("_pfl", parent=base["Normal"],
                                  fontName="Helvetica-Bold", fontSize=7.5,
                                  textColor=NAVY, leading=10)
        _val_sty = ParagraphStyle("_pfv", parent=base["Normal"],
                                  fontName="Helvetica", fontSize=7.5,
                                  textColor=GRAY_DARK, leading=10)
        _perf_rows = [
            [Paragraph("Est. Ann. Return (3yr)", _lbl_sty),
             Paragraph(_pf("ann_return"),         _val_sty),
             Paragraph("Est. Sharpe Ratio",       _lbl_sty),
             Paragraph(_pf("sharpe"),              _val_sty)],
            [Paragraph("Est. Ann. Volatility",    _lbl_sty),
             Paragraph(_pf("ann_volatility"),      _val_sty),
             Paragraph("Max Drawdown (3yr)",      _lbl_sty),
             Paragraph(_pf("max_drawdown"),        _val_sty)],
            [Paragraph("Best Year",               _lbl_sty),
             Paragraph(_pf("best_year"),           _val_sty),
             Paragraph("Worst Year",              _lbl_sty),
             Paragraph(_pf("worst_year"),          _val_sty)],
            [Paragraph("S&P 500 Correlation",     _lbl_sty),
             Paragraph(_pf("sp500_correlation"),   _val_sty),
             Paragraph("Beta to S&P 500",         _lbl_sty),
             Paragraph(_pf("beta"),                _val_sty)],
        ]
        _perf_tbl = Table(_perf_rows, colWidths=_perf_col_w)
        _perf_tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, GRAY_LITE]),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#C0C0C0")),
        ]))
        story.append(_perf_tbl)
        story.append(Paragraph(
            "Sources: Public LP disclosures, HFR Index comparables, third-party estimates  |  "
            f"Confidence: {_conf_val}",
            ParagraphStyle("_perf_src", parent=base["Normal"],
                           fontName="Helvetica-Oblique", fontSize=6,
                           textColor=colors.HexColor("#555555"), leading=8,
                           spaceBefore=2),
        ))

    # ── Section 4 (page 2): Red Flags & Observations ─────────────────────────
    if _should_render("red_flags"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("Red Flags & Observations", sty["h1"]))
        story.append(navy_rule())
        _flags = gold.get("_parsed_red_flags") or _derive_red_flags(gold)
        flag_colors = {"red": RED_COL, "yellow": AMBER_COL, "info": GREEN_COL}
        flag_icons  = {"red": "●", "yellow": "▲", "info": "✓"}
        for flag in _flags:
            lvl   = flag["level"]
            col   = flag_colors.get(lvl, GRAY_MID)
            icon  = flag_icons.get(lvl, "·")
            hex_c = f"#{int(col.red*255):02X}{int(col.green*255):02X}{int(col.blue*255):02X}"
            story.append(Paragraph(
                f'<font color="{hex_c}"><b>{icon}</b></font>  {flag["message"]}',
                sty["body_sm"],
            ))

    # ── Disclosure attribution block ──────────────────────────────────────────
    # The full four-layer disclosure is rendered in the canvas footer on every
    # page (never truncated, always visible). Here we record data provenance.
    disc_block: list = []
    disc_block.append(Spacer(1, 10))
    disc_block.append(HRFlowable(
        width=CONTENT_W, thickness=1.5, color=GOLD,
        spaceAfter=5, spaceBefore=4,
    ))

    # Data sources line
    src_list = disclosure.get("data_sources") or []
    if src_list:
        disc_block.append(Paragraph(
            "<b>Data sources:</b>  " + " · ".join(src_list),
            sty["legal"],
        ))

    # Skipped modules
    skipped_disc = disclosure.get("modules_skipped") or {}
    if skipped_disc:
        skipped_txt = "; ".join(f"{m} ({r})" for m, r in skipped_disc.items())
        disc_block.append(Paragraph(
            f"<b>Modules unavailable:</b>  {skipped_txt}",
            sty["legal"],
        ))

    disc_block.append(Paragraph(
        f"<b>NOT INVESTMENT ADVICE</b>  ·  "
        f"<b>CONFIDENTIAL</b>  ·  "
        f"Tearsheet v{disclosure.get('version', '1.0')}  ·  "
        f"Generated {gen_date}  ·  Full disclosures in page footer.",
        sty["legal"],
    ))
    disc_block.append(Spacer(1, 4))

    story.append(KeepTogether(disc_block))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story)
    logger.info("PDF written to %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_md(gold: dict, out_path: Path) -> Path:
    """
    Render Gold Copy to structured Markdown optimised for RAG and AI agents.
    Uses # headers and | tables for maximum parsability.
    """
    lines: list[str] = []
    add = lines.append

    firm_name    = gold.get("firm_name", "Unknown Firm")
    generated_at = gold.get("generated_at", "")
    gen_date     = _fmt_date(generated_at)
    overview     = gold.get("firm_overview") or {}
    inst         = gold.get("institutional_numbers") or {}
    disclosure   = gold.get("disclosure") or {}
    disc_layers  = _build_disclosure_layers(disclosure)

    # ── Document header ───────────────────────────────────────────────────────
    add(f"# {firm_name}")
    add(f"**Institutional Intelligence Report** · Generated: {gen_date}")
    if overview.get("crd"):
        add(f"CRD: `{overview['crd']}`", )
    if overview.get("cik"):
        add(f"CIK: `{overview['cik']}`")
    add("")
    add("> **CONFIDENTIAL — NOT INVESTMENT ADVICE.** "
        "For authorized recipients only. "
        "This report does not constitute a recommendation to invest.")
    add("")

    # ── Section 1: Firm Overview ──────────────────────────────────────────────
    add("## Firm Overview")
    add("")
    add("| Field | Value |")
    add("|-------|-------|")
    add(f"| Legal Name | {_safe(overview.get('legal_name'))} |")
    add(f"| CRD Number | {_safe(overview.get('crd'))} |")
    add(f"| CIK | {_safe(overview.get('cik'))} |")
    if overview.get("website"):
        add(f"| Website | {overview['website'].lower()} |")
    add(f"| SEC Registered | {'Yes' if overview.get('sec_registered') else 'No'} |")
    add(f"| Registration Status | {_safe(overview.get('registration_status'))} |")
    if inst.get("aum_total_usd") is not None:
        add(f"| AUM (Total) | {_fmt_usd(inst['aum_total_usd'])} |")
        add(f"| AUM (Discretionary) | {_fmt_usd(inst.get('aum_discretionary_usd'))} |")
    if inst.get("num_clients") is not None:
        add(f"| Number of Clients | {inst['num_clients']} |")
    if inst.get("client_types"):
        add(f"| Client Types | {'; '.join(str(c) for c in inst['client_types'])} |")
    if inst.get("fee_structures"):
        add(f"| Fee Structures | {'; '.join(str(f) for f in inst['fee_structures'])} |")
    if inst.get("adv_filing_date"):
        flag = inst.get("adv_recency_flag", "")
        add(f"| ADV Filing Date | {inst['adv_filing_date']} [{_recency_label(flag)}] |")
    add("")

    # ── Section 2: Team Roster ────────────────────────────────────────────────
    add("## Team Roster")
    add("")
    roster = gold.get("team_roster") or []
    if roster:
        add("| Name | Title | Bio |")
        add("|------|-------|-----|")
        for m in roster:
            bio = _trunc(_safe(m.get("bio"), "—"), 200).replace("|", "\\|")
            add(f"| {_safe(m.get('name'))} | {_safe(m.get('title'))} | {bio} |")
        add("")
        add(f"_Source: AltBots team pipeline · {len(roster)} member(s) · "
            f"Bios synthesized by Claude Haiku_")
    else:
        add("_No team profiles available._")
    add("")

    # ── Section 3: SEC Filing Snapshot ───────────────────────────────────────
    add("## SEC Filing Snapshot")
    add("")

    f13 = inst.get("latest_13f") or {}
    if f13:
        add("### Form 13F")
        add("")
        add(f"- **Period of Report:** {_safe(f13.get('period_of_report'))}")
        add(f"- **Filing Date:** {_safe(f13.get('filing_date'))} "
            f"[{_recency_label(f13.get('recency_flag', ''))}]")
        add(f"- **Total Portfolio Value:** {_fmt_usd(f13.get('total_portfolio_value_usd'))}")
        add(f"- **Number of Positions:** {_safe(f13.get('num_positions'))}")
        if f13.get("concentration_top10_pct") is not None:
            add(f"- **Top-10 Concentration:** {f13['concentration_top10_pct']:.1f}%")
        add("")

        holdings = f13.get("top_holdings") or []
        if holdings:
            add("#### Top Holdings")
            add("")
            add("| # | Holding | Value | Shares | % Portfolio |")
            add("|---|---------|-------|--------|-------------|")
            for i, h in enumerate(holdings[:10], 1):
                add(f"| {i} | {_safe(h.get('name'))} | {_fmt_usd(h.get('value_usd'))} "
                    f"| {h.get('shares', 0):,} | {h.get('pct_of_portfolio', 0):.2f}% |")
            add("")
    else:
        add("_No 13F data available._")
        add("")

    # ── Section 4: Social Signals ─────────────────────────────────────────────
    add("## Social Signals")
    add("")
    add("_Source: Exa.ai · Claude Haiku · Last 90 days_")
    add("")
    signals = gold.get("social_signals") or []
    if signals:
        add("| Date | Title | Type | Source |")
        add("|------|-------|------|--------|")
        for sig in signals[:10]:
            pub   = _fmt_date(sig.get("published_date", ""))[:10]
            title = _trunc(_safe(sig.get("title")), 80).replace("|", "\\|")
            add(f"| {pub} | {title} | {_safe(sig.get('signal_type'))} "
                f"| {_safe(sig.get('source'))} |")
        add("")
        # Full summaries for RAG context
        add("### Signal Summaries")
        add("")
        for i, sig in enumerate(signals[:5], 1):
            add(f"**{i}. {_safe(sig.get('title'))}**")
            add(f"   {_safe(sig.get('summary'), '')[:300]}")
            if sig.get("url"):
                add(f"   URL: {sig['url']}")
            add("")
    else:
        add("_No social signals retrieved._")
        add("")

    # ── Section 5: Red Flags ──────────────────────────────────────────────────
    add("## Red Flags & Observations")
    add("")
    icon_map = {"red": "🔴", "yellow": "🟡", "info": "🟢"}
    for flag in _derive_red_flags(gold):
        icon = icon_map.get(flag["level"], "·")
        add(f"- {icon} {flag['message']}")
    add("")

    # ── Data Provenance ───────────────────────────────────────────────────────
    add("## Data Provenance")
    add("")
    for src in gold.get("data_sources") or []:
        add(f"- {src}")
    add("")

    # ── Full Disclosure ───────────────────────────────────────────────────────
    add("---")
    add("")
    add("## Legal Disclosures")
    add("")
    layer_labels = [
        "NOT INVESTMENT ADVICE",
        "REGULATORY SCREENING LIMITATION",
        "DATA LIMITATION",
        "CONFIDENTIALITY",
    ]
    for label, text in zip(layer_labels, disc_layers):
        if text:
            add(f"**{label}.** {text}")
            add("")
    add(f"_Tearsheet schema v{disclosure.get('version', '1.0')} · Generated {gen_date}_")
    add("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown written to %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# JSON renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_json(gold: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gold, indent=2, default=str, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("JSON written to %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# TearsheetFormatter — public API
# ─────────────────────────────────────────────────────────────────────────────

class TearsheetFormatter:
    """
    Transforms the Assembler's Gold Copy dict into PDF, Markdown, or JSON.

    Parameters
    ----------
    output_dir : Path, optional
        Directory for saved files (default: <project_root>/output).
    """

    FORMATS = ("pdf", "md", "json")

    def __init__(self, output_dir: Optional[Path] = None):
        self._output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    def format(
        self,
        gold: dict,
        fmt: str = "pdf",
        out_path: Optional[Path] = None,
    ) -> Path:
        """
        Render *gold* to *fmt*.

        Parameters
        ----------
        gold     : Gold Copy dict from TearsheetAssembler.assemble().
        fmt      : 'pdf', 'md', or 'json'.
        out_path : Explicit output path; if None, auto-generated in output_dir.

        Returns
        -------
        Path to the written file.
        """
        fmt = fmt.lower().strip()
        if fmt not in self.FORMATS:
            raise ValueError(f"Unknown format {fmt!r}. Choose from: {self.FORMATS}")

        if not gold.get("is_legal_ready"):
            raise ValueError(
                "Gold Copy is not legal-ready (is_legal_ready=False). "
                "Cannot produce output without a valid disclosure envelope."
            )

        if out_path is None:
            stem = _slug(gold.get("firm_name", "tearsheet"))
            ext  = "pdf" if fmt == "pdf" else ("md" if fmt == "md" else "json")
            out_path = self._output_dir / f"{stem}.{ext}"

        if fmt == "pdf":
            return _render_pdf(gold, Path(out_path))
        if fmt == "md":
            return _render_md(gold, Path(out_path))
        return _render_json(gold, Path(out_path))

    def parse_md(self, md_path: Path) -> dict:
        """
        Parse a tearsheet Markdown file into a Gold Copy dict.

        Parameters
        ----------
        md_path : Path to the .md file produced by _render_md (or hand-edited).

        Returns
        -------
        Gold Copy dict with ``is_legal_ready=True``.
        """
        return _parse_md_to_gold(Path(md_path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI (standalone)
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="AltBots tearsheet formatter — render a Gold Copy JSON to PDF/MD/JSON.",
    )
    p.add_argument("input", metavar="GOLD_COPY_JSON",
                   help="Path to a Gold Copy JSON file.")
    p.add_argument("--format", choices=TearsheetFormatter.FORMATS, default="pdf",
                   help="Output format (default: pdf).")
    p.add_argument("--output", metavar="FILE",
                   help="Output file path (auto-generated if omitted).")
    return p


def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    parser = _build_parser()
    args   = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"ERROR: Input file not found: {src}", file=sys.stderr)
        return 1

    try:
        gold = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse JSON: {exc}", file=sys.stderr)
        return 1

    formatter = TearsheetFormatter()
    out_path  = Path(args.output) if args.output else None

    try:
        result = formatter.format(gold, fmt=args.format, out_path=out_path)
        print(f"Written: {result}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

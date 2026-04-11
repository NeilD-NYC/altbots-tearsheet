"""
src/mcp/server.py — AltBots Research Tool Server  (port 8200)

Standalone FastAPI application exposing four research tools via a simple
JSON-over-HTTP interface. Also retains the legacy MCP SSE server objects
for backward compat with src/api/main.py.

Standalone usage
----------------
  uvicorn src.mcp.server:app --host 0.0.0.0 --port 8200 --reload

Endpoints
---------
  GET  /health       — liveness + tool count + managers indexed
  POST /messages     — {"tool": "<name>", "input": {...}} → JSON response

Tools
-----
  get_pipeline_status          — collection counts + health
  generate_manager_tearsheet   — full tearsheet (json | pdf_url | markdown)
  get_manager_coverage         — data-quality scorecard for a manager
  list_available_managers      — top 50 managers by AUM from Qdrant
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ── Legacy MCP SSE objects — kept for src/api/main.py import compat ──────────
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool

from src.mcp.disclosure import wrap_with_disclosure

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR   = _PROJECT_ROOT / "output"
_SAMPLE_DIR   = Path("/var/www/html/sample")
_SAMPLE_URL   = "https://api.altbots.io/sample"

# ─────────────────────────────────────────────────────────────────────────────
# Legacy MCP SSE server  (retained for src/api/main.py)
# ─────────────────────────────────────────────────────────────────────────────

mcp_server    = Server("altbots-manager-research-signals")
sse_transport = SseServerTransport("/messages/")


@mcp_server.list_tools()
async def _mcp_list_tools() -> list[Tool]:
    """Stub — tools are served via the REST /messages endpoint."""
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry (schema definitions used by /health + discovery)
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_DEFINITIONS = [
    {
        "name": "get_pipeline_status",
        "description": "Return health and indexed-record counts for all AltBots data collections.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "generate_manager_tearsheet",
        "description": (
            "Generate an institutional-grade tearsheet for a fund manager. "
            "Returns full JSON data, a hosted PDF URL, or structured Markdown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "manager_name":  {"type": "string",
                                  "description": "Fund manager or firm name"},
                "crd_number":    {"type": "integer",
                                  "description": "Optional SEC CRD number"},
                "output_format": {"type": "string",
                                  "enum": ["json", "pdf_url", "markdown"],
                                  "default": "json"},
            },
            "required": ["manager_name"],
        },
    },
    {
        "name": "get_manager_coverage",
        "description": (
            "Return a data-quality scorecard showing what information is "
            "currently indexed for a named fund manager."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "manager_name": {"type": "string"},
            },
            "required": ["manager_name"],
        },
    },
    {
        "name": "list_available_managers",
        "description": "List up to 50 indexed fund managers sorted by AUM descending.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(host="localhost", port=6333)


def _managers_indexed() -> int:
    try:
        return _qdrant().get_collection("fund_managers").points_count or 0
    except Exception:
        return 0


def _make_meta(manager_name: str | None = None) -> dict:
    d: dict = {
        "generated_at":           datetime.now(timezone.utc).isoformat(),
        "product_classification": "Research Signal Report",
    }
    if manager_name:
        d["manager_name"] = manager_name
    return d


def _err(code: str, message: str, manager_name: str | None = None) -> dict:
    base = wrap_with_disclosure({})
    base["_meta"] = {**_make_meta(manager_name), "error_code": code, "error": message}
    base["data"]  = None
    return base


def _wrap(data: Any, manager_name: str | None = None) -> dict:
    """Wrap data in the standard disclosure envelope."""
    result = wrap_with_disclosure(data if isinstance(data, dict) else {"result": data})
    result["_meta"] = _make_meta(manager_name)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — get_pipeline_status
# ─────────────────────────────────────────────────────────────────────────────

async def _get_pipeline_status(_inp: dict) -> dict:
    try:
        c      = _qdrant()
        counts = {}
        for col in c.get_collections().collections:
            try:
                counts[col.name] = c.get_collection(col.name).points_count or 0
            except Exception:
                counts[col.name] = -1

        return _wrap({
            "status":                   "operational",
            "collections":              counts,
            "total_managers_indexed":   counts.get("fund_managers", 0),
            "pipeline_version":         "2.0",
        })
    except Exception as exc:
        logger.exception("get_pipeline_status error")
        return _err("pipeline_error", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — generate_manager_tearsheet
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_manager_tearsheet(inp: dict) -> dict:
    manager_name  = (inp.get("manager_name") or "").strip()
    output_format = (inp.get("output_format") or "json").lower()

    if not manager_name:
        return _err("invalid_input", "manager_name is required")

    logger.info("generate_manager_tearsheet: %r  fmt=%s", manager_name, output_format)

    # ── Assemble ──────────────────────────────────────────────────────────────
    try:
        from src.core.tearsheet_assembler import TearsheetAssembler
        gold = await asyncio.to_thread(TearsheetAssembler().assemble, manager_name)
    except Exception as exc:
        logger.warning("Assembler failed for %r: %s", manager_name, exc)
        return _err("assembly_failed", str(exc), manager_name)

    # ── json ──────────────────────────────────────────────────────────────────
    if output_format == "json":
        return _wrap(gold, manager_name)

    # ── markdown ──────────────────────────────────────────────────────────────
    if output_format == "markdown":
        try:
            import tempfile
            from src.renderers.tearsheet_formatter import TearsheetFormatter
            tmp = Path(tempfile.mktemp(suffix=".md"))
            TearsheetFormatter().format(gold, fmt="md", out_path=tmp)
            md_text = tmp.read_text(encoding="utf-8")
            tmp.unlink(missing_ok=True)
            return _wrap({"markdown": md_text}, manager_name)
        except Exception as exc:
            logger.exception("Markdown render failed for %r", manager_name)
            return _err("render_failed", str(exc), manager_name)

    # ── pdf_url ───────────────────────────────────────────────────────────────
    if output_format == "pdf_url":
        try:
            from src.renderers.tearsheet_formatter import TearsheetFormatter, _slug
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            slug     = _slug(gold.get("firm_name", manager_name))
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            fname    = f"{slug}_{date_str}.pdf"
            out_path = _OUTPUT_DIR / fname

            TearsheetFormatter().format(gold, fmt="pdf", out_path=out_path)

            _SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, _SAMPLE_DIR / fname)

            pdf_url = f"{_SAMPLE_URL}/{fname}"
            logger.info("PDF hosted at %s", pdf_url)
            return _wrap({"pdf_url": pdf_url}, manager_name)
        except Exception as exc:
            logger.exception("PDF render failed for %r", manager_name)
            return _err("render_failed", str(exc), manager_name)

    return _err("invalid_input", f"Unknown output_format: {output_format!r}", manager_name)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — get_manager_coverage
# ─────────────────────────────────────────────────────────────────────────────

async def _get_manager_coverage(inp: dict) -> dict:
    manager_name = (inp.get("manager_name") or "").strip()
    if not manager_name:
        return _err("invalid_input", "manager_name is required")

    try:
        from qdrant_client.http import models as qm
        c = _qdrant()

        # ── Resolve CIK ───────────────────────────────────────────────────────
        cik        = None
        resolved   = None
        fm_pts     = c.scroll("fund_managers", limit=200, with_payload=True)[0]
        needle     = manager_name.lower()
        for pt in fm_pts:
            legal = (pt.payload.get("legal_name") or "").lower()
            if needle in legal or legal in needle:
                cik      = pt.payload.get("cik")
                resolved = pt.payload.get("legal_name")
                break

        # ── edgar_cache lookup ────────────────────────────────────────────────
        adv_data     = {}
        f13_list     = []
        last_updated = None
        if cik:
            cache_pts = c.scroll(
                "edgar_cache",
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="cik", match=qm.MatchValue(value=cik))
                ]),
                limit=1,
                with_payload=True,
            )[0]
            if cache_pts:
                payload      = cache_pts[0].payload
                last_updated = payload.get("last_updated")
                data         = payload.get("data") or {}
                adv_data     = data.get("adv") or {}
                f13_raw      = data.get("form_13f") or []
                f13_list     = f13_raw if isinstance(f13_raw, list) else [f13_raw]

        # ── team_profiles lookup ──────────────────────────────────────────────
        team_pts = c.scroll("team_profiles", limit=200, with_payload=True)[0]
        team_count = sum(
            1 for pt in team_pts
            if needle in (pt.payload.get("firm_name") or "").lower()
        )

        # ── Score each dimension ──────────────────────────────────────────────
        # SEC ADV
        if adv_data.get("aum") and adv_data.get("clients"):
            sec_adv  = "full";    adv_pts = 3
        elif adv_data:
            sec_adv  = "partial"; adv_pts = 1
        else:
            sec_adv  = "missing"; adv_pts = 0

        # 13F Holdings
        best_f13 = f13_list[0] if f13_list else {}
        if best_f13.get("num_positions") and best_f13.get("top_holdings"):
            holdings_13f = "full";    f13_pts = 3
        elif best_f13:
            holdings_13f = "partial"; f13_pts = 1
        else:
            holdings_13f = "missing"; f13_pts = 0

        # Team Roster
        if team_count >= 2:
            team_roster = "full";    team_pts = 2
        elif team_count == 1:
            team_roster = "partial"; team_pts = 1
        else:
            team_roster = "missing"; team_pts = 0

        # Social Signals — requires live scan; always "missing" in static coverage
        social_signals = "missing"
        social_pts     = 0

        overall_score = adv_pts + f13_pts + team_pts + social_pts

        if overall_score >= 7:
            recommendation = "ready"
        elif overall_score >= 4:
            recommendation = "enrich"
        else:
            recommendation = "manual"

        return _wrap({
            "manager_name":   manager_name,
            "resolved_name":  resolved,
            "cik":            cik,
            "last_updated":   last_updated,
            "sec_adv":        sec_adv,
            "holdings_13f":   holdings_13f,
            "team_roster":    team_roster,
            "social_signals": social_signals,
            "overall_score":  overall_score,
            "recommendation": recommendation,
        }, manager_name)

    except Exception as exc:
        logger.exception("get_manager_coverage failed for %r", manager_name)
        return _err("coverage_error", str(exc), manager_name)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — list_available_managers
# ─────────────────────────────────────────────────────────────────────────────

async def _list_available_managers(_inp: dict) -> dict:
    try:
        c = _qdrant()

        # All fund_managers
        fm_pts = c.scroll("fund_managers", limit=200, with_payload=True)[0]
        mgrs: dict[str, dict] = {}
        for pt in fm_pts:
            p   = pt.payload
            cik = p.get("cik", "")
            mgrs[cik] = {
                "name":         p.get("legal_name", ""),
                "crd":          p.get("crd"),
                "cik":          cik,
                "website":      p.get("website"),
                "aum":          None,
                "strategy":     None,
                "last_updated": None,
            }

        # Enrich AUM + last_updated from edgar_cache
        cache_pts = c.scroll("edgar_cache", limit=200, with_payload=True)[0]
        for pt in cache_pts:
            p   = pt.payload
            cik = p.get("cik", "")
            if cik in mgrs:
                data = p.get("data") or {}
                aum  = (data.get("adv") or {}).get("aum") or {}
                mgrs[cik]["aum"]          = aum.get("total_usd")
                mgrs[cik]["last_updated"] = p.get("last_updated")

        # Sort by AUM desc, take top 50
        ranked = sorted(
            mgrs.values(),
            key=lambda m: m["aum"] or 0,
            reverse=True,
        )[:50]

        return _wrap({
            "count":    len(ranked),
            "managers": ranked,
        })

    except Exception as exc:
        logger.exception("list_available_managers failed")
        return _err("list_error", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: dict[str, Any] = {
    "get_pipeline_status":        _get_pipeline_status,
    "generate_manager_tearsheet": _generate_manager_tearsheet,
    "get_manager_coverage":       _get_manager_coverage,
    "list_available_managers":    _list_available_managers,
}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AltBots Research Tool API",
    description="Institutional fund manager research signal tools",
    version="2.0.0",
)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe — returns tool count and managers indexed."""
    return {
        "status":           "ok",
        "tools":            len(_TOOLS),
        "managers_indexed": _managers_indexed(),
    }


@app.post("/messages", tags=["tools"])
async def handle_message(request: Request) -> JSONResponse:
    """
    Dispatch a tool call.

    Request body: ``{"tool": "<tool_name>", "input": {...}}``
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    tool_name = (body.get("tool") or "").strip()
    inp       = body.get("input") or {}

    if not tool_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'tool' field"})

    handler = _TOOLS.get(tool_name)
    if handler is None:
        return JSONResponse(
            status_code=404,
            content={
                "error":       f"Unknown tool: {tool_name!r}",
                "known_tools": list(_TOOLS.keys()),
            },
        )

    result = await handler(inp)
    return Response(
        content=json.dumps(result, default=str),
        media_type="application/json",
    )

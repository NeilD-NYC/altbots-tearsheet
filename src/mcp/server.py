"""
src/mcp/server.py — AltBots Research Tool Server  (port 8200)

Standalone FastAPI application exposing four research tools via a simple
JSON-over-HTTP interface and a full MCP SSE transport.

Standalone usage
----------------
  uvicorn src.mcp.server:app --host 0.0.0.0 --port 8200 --reload

Endpoints
---------
  GET  /health           — liveness + tool count + managers indexed
  POST /messages         — {"tool": "<name>", "input": {...}} → JSON response
  GET  /sse              — MCP SSE transport (open stream, receive endpoint event)
  POST /sse/messages     — MCP JSON-RPC messages from client → SSE stream response

Tools
-----
  get_pipeline_status          — collection counts + health
  generate_manager_tearsheet   — full tearsheet (json | pdf_url | markdown)
  get_manager_coverage         — data-quality scorecard for a manager
  list_available_managers      — top 50 managers by AUM from Qdrant
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ── MCP SDK ───────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from src.mcp.disclosure import wrap_with_disclosure

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR   = _PROJECT_ROOT / "output"
_SAMPLE_DIR   = Path("/var/www/html/sample")
_SAMPLE_URL   = "https://api.altbots.io/sample"
_KEYS_FILE    = _PROJECT_ROOT / "config" / "api_keys.json"
_CUSTOMER_LOG = _PROJECT_ROOT / "logs" / "new_customers.log"

# ── Stripe config (optional — MVP degrades gracefully without these) ──────────
_STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY", "")
_STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")

if not _STRIPE_SECRET_KEY:
    logging.warning("STRIPE_SECRET_KEY not set — Stripe API calls disabled")
if not _STRIPE_WEBHOOK_SECRET:
    logging.warning("STRIPE_WEBHOOK_SECRET not set — webhook signature verification disabled")

# ── Tier definitions ──────────────────────────────────────────────────────────
_FREE_TOOLS = {
    "list_available_managers",
    "get_manager_coverage",
    "get_pipeline_status",
}
_PRO_TOOLS = {
    "generate_manager_tearsheet",
}

# ─────────────────────────────────────────────────────────────────────────────
# API key store (config/api_keys.json)
# ─────────────────────────────────────────────────────────────────────────────

def _load_keys() -> dict:
    try:
        return json.loads(_KEYS_FILE.read_text(encoding="utf-8")) if _KEYS_FILE.exists() else {}
    except Exception:
        return {}


def _save_keys(keys: dict) -> None:
    _KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEYS_FILE.write_text(json.dumps(keys, indent=2), encoding="utf-8")


def _extract_api_key(request: Request) -> str | None:
    """Pull API key from Authorization: Bearer ak_... or x-api-key header."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate.startswith("ak_"):
            return candidate
    xkey = request.headers.get("x-api-key", "").strip()
    if xkey.startswith("ak_"):
        return xkey
    return None


def _validate_key(key: str) -> tuple[bool, dict | None]:
    """Return (is_valid, entry).  is_valid=False if missing or inactive."""
    keys = _load_keys()
    entry = keys.get(key)
    if entry is None or not entry.get("active"):
        return False, entry
    return True, entry


def _record_usage(key: str) -> None:
    """Increment usage_count and update last_used in place."""
    keys = _load_keys()
    if key in keys:
        keys[key]["usage_count"] = keys[key].get("usage_count", 0) + 1
        keys[key]["last_used"]   = datetime.now(timezone.utc).isoformat()
        _save_keys(keys)


def _provision_key(name: str, email: str, plan: str = "pro") -> str:
    """Generate and persist a new API key; returns the key string."""
    import random, string
    chars = string.ascii_letters + string.digits
    keys  = _load_keys()
    key   = "ak_" + "".join(random.choices(chars, k=12))
    while key in keys:
        key = "ak_" + "".join(random.choices(chars, k=12))
    keys[key] = {
        "name":        name,
        "email":       email,
        "plan":        plan,
        "active":      True,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "last_used":   None,
        "usage_count": 0,
    }
    _save_keys(keys)
    return key


def _deactivate_by_email(email: str) -> bool:
    """Set active=False for all keys matching email. Returns True if any found."""
    keys    = _load_keys()
    changed = False
    for entry in keys.values():
        if entry.get("email", "").lower() == email.lower() and entry.get("active"):
            entry["active"] = False
            changed = True
    if changed:
        _save_keys(keys)
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# MCP server + legacy SSE transport (retained for src/api/main.py import compat)
# ─────────────────────────────────────────────────────────────────────────────

mcp_server    = Server("altbots-manager-research-signals")
sse_transport = SseServerTransport("/messages/")   # legacy — do not remove


@mcp_server.list_tools()
async def _mcp_list_tools() -> list[Tool]:
    return [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in _TOOL_DEFINITIONS
    ]


@mcp_server.call_tool()
async def _mcp_call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = _TOOLS.get(name)
    if handler is None:
        return [TextContent(type="text", text=json.dumps({
            "error":       f"Unknown tool: {name!r}",
            "known_tools": list(_TOOLS.keys()),
        }))]
    result = await handler(arguments or {})
    return [TextContent(type="text", text=json.dumps(result, default=str))]


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
# SSE transport for the FastAPI /sse endpoint
# ─────────────────────────────────────────────────────────────────────────────

_sse_transport = SseServerTransport("/sse/messages")


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

    # ── Auth gate ─────────────────────────────────────────────────────────────
    if tool_name in _PRO_TOOLS:
        api_key = _extract_api_key(request)
        if api_key is None:
            return JSONResponse(
                status_code=401,
                content={
                    "error":       "api_key_required",
                    "message":     "This tool requires a Pro API key.",
                    "upgrade_url": "https://altbots.io/pricing",
                    "free_tools":  sorted(_FREE_TOOLS),
                },
            )
        valid, _ = _validate_key(api_key)
        if not valid:
            return JSONResponse(
                status_code=403,
                content={
                    "error":       "invalid_api_key",
                    "message":     "API key not found or inactive.",
                    "upgrade_url": "https://altbots.io/pricing",
                },
            )
        _record_usage(api_key)

    result = await handler(inp)
    return Response(
        content=json.dumps(result, default=str),
        media_type="application/json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SSE endpoints  (MCP SSE transport)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/sse", tags=["mcp"])
async def sse_endpoint(request: Request) -> None:
    """
    MCP SSE transport — open a Server-Sent Events stream.

    On connect the server emits an ``endpoint`` event containing the URL the
    client must POST JSON-RPC messages to::

        event: endpoint
        data: /sse/messages?session_id=<uuid>

    The stream then carries all MCP responses (initialize, tools/list,
    tools/call results, etc.) as ``message`` events.
    """
    async with _sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options(),
        )


@app.post("/sse/messages", tags=["mcp"])
async def sse_messages(request: Request) -> None:
    """Receive MCP JSON-RPC messages from the client and route them to the SSE stream."""
    await _sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stripe webhook  (POST /stripe/webhook)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_stripe_signature(payload: bytes, sig_header: str, secret: str) -> bool:
    """Manually verify Stripe webhook signature without requiring stripe.Webhook."""
    try:
        parts  = {k: v for k, v in (p.split("=", 1) for p in sig_header.split(",") if "=" in p)}
        ts     = parts.get("t", "")
        v1_sig = parts.get("v1", "")
        signed = f"{ts}.".encode() + payload
        expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, v1_sig)
    except Exception:
        return False


def _log_new_customer(name: str, email: str, key: str) -> None:
    _CUSTOMER_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).isoformat()
    line = f"{ts}  key={key}  name={name!r}  email={email}\n"
    with _CUSTOMER_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
    logger.info("New customer provisioned: %s <%s> → %s", name, email, key)


@app.post("/stripe/webhook", tags=["billing"])
async def stripe_webhook(request: Request) -> JSONResponse:
    """
    Receive and process Stripe webhook events.

    Handled events
    --------------
    checkout.session.completed
        Provision a new Pro API key and log to logs/new_customers.log.

    customer.subscription.deleted
        Deactivate all API keys associated with the customer's email.
    """
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # ── Signature verification ────────────────────────────────────────────────
    if _STRIPE_WEBHOOK_SECRET:
        if not _verify_stripe_signature(payload, sig_header, _STRIPE_WEBHOOK_SECRET):
            logger.warning("Stripe webhook signature verification failed")
            return JSONResponse(status_code=400, content={"error": "invalid_signature"})
    else:
        logger.warning("Stripe webhook received without secret — skipping verification")

    try:
        event = json.loads(payload)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    event_type = event.get("type", "")
    data_obj   = (event.get("data") or {}).get("object") or {}

    # ── checkout.session.completed ────────────────────────────────────────────
    if event_type == "checkout.session.completed":
        email    = (data_obj.get("customer_details") or {}).get("email") or \
                   data_obj.get("customer_email") or ""
        metadata = data_obj.get("metadata") or {}
        name     = metadata.get("firm_name") or metadata.get("name") or \
                   (data_obj.get("customer_details") or {}).get("name") or email

        if not email:
            logger.warning("checkout.session.completed missing email — skipping key provision")
            return JSONResponse(content={"status": "skipped", "reason": "no_email"})

        key = _provision_key(name=name, email=email, plan="pro")
        _log_new_customer(name=name, email=email, key=key)
        return JSONResponse(content={"status": "provisioned", "key_prefix": key[:8] + "****"})

    # ── customer.subscription.deleted ─────────────────────────────────────────
    elif event_type == "customer.subscription.deleted":
        # Fetch customer email from the subscription object
        email = ""
        # The subscription has customer ID; for MVP use metadata or billing_email
        billing_email = data_obj.get("billing_email") or ""
        metadata      = data_obj.get("metadata") or {}
        email         = metadata.get("email") or billing_email

        # Fallback: try to look up via Stripe API if secret is set
        if not email and _STRIPE_SECRET_KEY:
            try:
                import stripe as _stripe
                _stripe.api_key = _STRIPE_SECRET_KEY
                customer_id = data_obj.get("customer")
                if customer_id:
                    cust  = await asyncio.to_thread(_stripe.Customer.retrieve, customer_id)
                    email = cust.get("email", "")
            except Exception as exc:
                logger.warning("Stripe customer lookup failed: %s", exc)

        if email:
            found = _deactivate_by_email(email)
            logger.info("subscription.deleted for %s — keys deactivated=%s", email, found)
            return JSONResponse(content={"status": "deactivated", "email": email, "found": found})
        else:
            logger.warning("subscription.deleted — could not resolve customer email")
            return JSONResponse(content={"status": "skipped", "reason": "no_email"})

    # ── unhandled event ───────────────────────────────────────────────────────
    logger.debug("Unhandled Stripe event: %s", event_type)
    return JSONResponse(content={"status": "ignored", "type": event_type})


# ─────────────────────────────────────────────────────────────────────────────
# Smithery config schema  (GET /.well-known/mcp/config-schema)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/.well-known/mcp/config-schema", tags=["discovery"])
async def mcp_config_schema() -> JSONResponse:
    """Return the Smithery-compatible config schema for this MCP server."""
    return JSONResponse(content={
        "configSchema": {
            "type": "object",
            "properties": {
                "apiKey": {
                    "type":        "string",
                    "title":       "AltBots API Key",
                    "description": (
                        "Your AltBots Pro API key. "
                        "Get one at altbots.io/pricing. "
                        "Format: ak_xxxxxxxxxxxx"
                    ),
                    "format": "password",
                },
            },
        },
    })

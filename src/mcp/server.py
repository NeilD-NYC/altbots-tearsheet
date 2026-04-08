"""
src/mcp/server.py — AltBots MCP server

Exposes the `generate_manager_tearsheet` tool via the MCP protocol.
Mounted into the FastAPI app in src/api/main.py via SSE transport.
"""

from __future__ import annotations

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from src.core.assembler import TearsheetAssembler, IdentityResolutionError
from src.mcp.disclosure import wrap_with_disclosure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp_server = Server("altbots-manager-research-signals")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_manager_tearsheet",
            description=(
                "Generate an institutional-grade fund manager research signal report. "
                "Covers firm overview, key personnel, SEC filings (ADV, 13F, Form D), "
                "indicative performance, regulatory screening (BrokerCheck, enforcement), "
                "and observed social media signals. Built for allocators, family offices, "
                "fund-of-funds, and institutional ODD teams. "
                "Returns research signals with mandatory disclosure envelope."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "manager_name": {
                        "type": "string",
                        "description": "Fund manager or firm name to research",
                    },
                    "crd_number": {
                        "type": "integer",
                        "description": (
                            "Optional SEC CRD registration number for faster resolution"
                        ),
                    },
                },
                "required": ["manager_name"],
            },
        )
    ]


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "generate_manager_tearsheet":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    manager_name = (arguments.get("manager_name") or "").strip()
    if not manager_name:
        return [TextContent(type="text", text="Error: manager_name is required")]

    # crd_number → pass as cik_hint string if provided (assembler accepts it)
    crd_number = arguments.get("crd_number")
    cik_hint = str(crd_number) if crd_number else None

    logger.info("MCP generate_manager_tearsheet: %r (crd=%s)", manager_name, crd_number)

    try:
        assembler = TearsheetAssembler()
        # assemble() is synchronous — run in a thread to avoid blocking the loop
        tearsheet = await asyncio.to_thread(
            assembler.assemble,
            manager_name,
            None,       # website
            cik_hint,
        )
        tearsheet_dict = (
            tearsheet.model_dump() if hasattr(tearsheet, "model_dump")
            else tearsheet.dict() if hasattr(tearsheet, "dict")
            else tearsheet
        )
        wrapped = wrap_with_disclosure(tearsheet_dict)
        return [TextContent(type="text", text=json.dumps(wrapped, indent=2, default=str))]

    except IdentityResolutionError as exc:
        msg = (
            f"Could not resolve a SEC identity for '{manager_name}'. "
            f"Verify the firm name or supply a crd_number. Detail: {exc}"
        )
        logger.warning("MCP identity resolution failed: %s", exc)
        return [TextContent(type="text", text=f"Error: {msg}")]

    except Exception as exc:
        logger.exception("MCP call_tool unexpected error for %r", manager_name)
        return [TextContent(type="text", text=f"Error generating tearsheet: {exc}")]


# ---------------------------------------------------------------------------
# SSE transport — mounted at /messages/ in main.py
# ---------------------------------------------------------------------------

sse_transport = SseServerTransport("/messages/")

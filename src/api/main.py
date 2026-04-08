# src/api/main.py
# AltBots Tearsheet API — Phase 9
#
# Run:
#   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
#
# Endpoints
# ---------
#   POST /generate          — assemble + render a tearsheet, return download URL
#   GET  /output/{filename} — static download of generated PDFs (auto-mounted)
#   GET  /health            — liveness probe

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=True)

import logging
import uuid
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.core.assembler import TearsheetAssembler, IdentityResolutionError
from src.renderers.pdf_renderer import render_tearsheet
from src.mcp.server import mcp_server, sse_transport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AltBots Tearsheet API",
    description=(
        "Generate institutional-grade due-diligence tearsheets for fund managers. "
        "POST /generate to assemble a tearsheet and receive a download link."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Static file hosting — /output served at /output so download_url works
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    manager_name: str
    website_hint: Optional[str] = None
    cik_hint: Optional[str] = None     # bypass FormAdv for exempt advisers

    @field_validator("manager_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("manager_name must not be empty")
        return v


class GenerateResponse(BaseModel):
    job_id: str
    legal_name: str
    download_url: str


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# MCP endpoints
# ---------------------------------------------------------------------------

@app.get("/sse", tags=["mcp"])
async def handle_sse(request: Request):
    """SSE endpoint for MCP clients (Claude Desktop, Cursor, etc.)."""
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )


@app.post("/messages/", tags=["mcp"])
async def handle_messages(request: Request):
    """Post-message endpoint used by the MCP SSE transport."""
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )


@app.get("/mcp/info", tags=["mcp"])
async def mcp_info() -> dict:
    """Tool schema for agent discovery."""
    return {
        "name": "altbots-manager-research-signals",
        "version": "1.0.0",
        "description": "Institutional-grade fund manager research signal reports",
        "tools": ["generate_manager_tearsheet"],
        "disclosure": "All responses include mandatory legal disclosure envelope",
        "server_url": "https://mcp.altbots.io/sse",
    }


@app.post("/generate", response_model=GenerateResponse, tags=["tearsheet"])
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """
    Assemble a tearsheet for the named fund manager and render it to PDF.

    - **manager_name**: Plain-English name of the investment manager.
    - **website_hint**: Optional known website URL; speeds up identity resolution.

    Returns a `job_id`, the resolved `legal_name`, and a `download_url` for
    the generated PDF.

    Error responses
    ---------------
    - **422** — Identity could not be resolved (firm not found in SEC records).
    - **500** — Unexpected server-side failure during assembly or rendering.
    """
    job_id = str(uuid.uuid4())
    logger.info("[%s] /generate called for %r", job_id, req.manager_name)

    # ── Step 1: Assemble ──────────────────────────────────────────────────
    try:
        assembler = TearsheetAssembler()
        tearsheet = assembler.assemble(
            manager_name=req.manager_name,
            website=req.website_hint,
            cik_hint=req.cik_hint,
        )
    except IdentityResolutionError as exc:
        logger.warning("[%s] Identity resolution failed: %s", job_id, exc)
        return _error(
            422,
            "identity_resolution_failed",
            f"Could not resolve a SEC identity for '{req.manager_name}'. "
            f"Verify the firm name or supply a website_hint. Detail: {exc}",
        )
    except Exception as exc:
        logger.exception("[%s] Assembler raised unexpected error", job_id)
        return _error(
            500,
            "assembly_error",
            f"Tearsheet assembly failed unexpectedly. "
            f"Please try again or contact support. Detail: {exc}",
        )

    # ── Step 2: Render to PDF ─────────────────────────────────────────────
    try:
        pdf_path = render_tearsheet(tearsheet)
    except Exception as exc:
        logger.exception("[%s] PDF rendering failed", job_id)
        return _error(
            500,
            "render_error",
            f"PDF rendering failed. The tearsheet was assembled but could not "
            f"be written to disk. Detail: {exc}",
        )

    # ── Step 3: Build download URL ────────────────────────────────────────
    filename = Path(pdf_path).name
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/output/{filename}"

    legal_name = (
        getattr(tearsheet.identity, "legal_name", None) or tearsheet.manager_name
    )

    logger.info("[%s] Done — %s → %s", job_id, legal_name, download_url)

    return GenerateResponse(
        job_id=job_id,
        legal_name=legal_name,
        download_url=download_url,
    )

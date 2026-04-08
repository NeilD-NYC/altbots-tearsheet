"""
src/tools/tearsheet_mcp.py — AltBots MCP Server

Model Context Protocol server exposing three AltBots intelligence tools to
any MCP-compatible agent (Claude Desktop, Alice, custom agents).

Server name : AltBots-Intelligence
Transport   : stdio (default) — drop-in for Claude Desktop config
              Also supports SSE via --transport sse --port <N>

Tools
-----
  get_tearsheet      Assemble + render a full Markdown tearsheet for a firm.
  search_personnel   Vector-search the team_profiles collection for people.
  check_red_flags    Run RedFlagScanner and return regulatory findings.

Safety contract
---------------
Every tool response ends with the mandatory DisclosureEnvelope footer.
No tool output ever reaches the caller without it.

Usage
-----
  # stdio (Claude Desktop / agent pipe)
  python -m src.tools.tearsheet_mcp

  # SSE (browser / HTTP client)
  python -m src.tools.tearsheet_mcp --transport sse --port 8765

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "altbots": {
        "command": "/path/to/venv/bin/python",
        "args": ["-m", "src.tools.tearsheet_mcp"],
        "cwd": "/path/to/tearsheet-project"
      }
    }
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── .env — load before any AltBots imports ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# ── MCP SDK ───────────────────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

# ── AltBots pipeline imports ──────────────────────────────────────────────────
from src.core.schema import DisclosureEnvelope
from src.core.tearsheet_assembler import TearsheetAssembler, AssemblerError
from src.core.red_flags import RedFlagScanner
from src.renderers.tearsheet_formatter import _render_md, _slug

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [mcp] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,          # stderr keeps stdio transport clean
)
logger = logging.getLogger("altbots_mcp")

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="AltBots-Intelligence",
    instructions=(
        "AltBots provides institutional intelligence on investment managers. "
        "Use get_tearsheet for a complete report, search_personnel to look up "
        "team members, and check_red_flags for regulatory screening. "
        "All responses include mandatory legal disclosures — do not suppress them."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared disclosure footer builder
# ─────────────────────────────────────────────────────────────────────────────

def _disclosure_footer(
    envelope_dict: dict,
    sources: Optional[list[str]] = None,
) -> str:
    """
    Build the mandatory 4-layer DisclosureEnvelope footer as Markdown.
    Always appended to every tool response — never optional.

    Parameters
    ----------
    envelope_dict : dict
        The ``disclosure`` key from a Gold Copy, or a raw
        ``DisclosureEnvelope.to_dict()`` output.
    sources : list[str], optional
        Extra data-source strings to include if not already in the envelope.
    """
    d = envelope_dict.get("disclaimers") or {}
    gen = envelope_dict.get("generated_at", datetime.now(timezone.utc).isoformat())
    version = envelope_dict.get("version", "1.0")

    # Combine envelope sources with any extras passed in
    env_sources = list(envelope_dict.get("data_sources") or [])
    if sources:
        for s in sources:
            if s not in env_sources:
                env_sources.append(s)

    skipped = envelope_dict.get("modules_skipped") or {}

    lines = [
        "",
        "---",
        "",
        "## Legal Disclosures",
        "",
    ]

    layer_map = [
        ("NOT INVESTMENT ADVICE",           "not_investment_advice"),
        ("REGULATORY SCREENING LIMITATION", "regulatory_disclaimer"),
        ("DATA LIMITATION",                 "data_limitation"),
        ("CONFIDENTIALITY",                 "confidentiality"),
    ]
    for label, key in layer_map:
        text = d.get(key, "")
        if text:
            lines.append(f"**{label}.** {text}")
            lines.append("")

    if env_sources:
        lines.append(
            "_Data sources consulted: " + "; ".join(env_sources) + "._"
        )
        lines.append("")

    if skipped:
        skipped_txt = "; ".join(f"{m} ({r})" for m, r in skipped.items())
        lines.append(f"_Modules unavailable: {skipped_txt}._")
        lines.append("")

    try:
        dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
        gen_label = dt.strftime("%B %d, %Y at %H:%M UTC")
    except Exception:
        gen_label = gen

    lines.append(
        f"_AltBots Intelligence · Schema v{version} · Generated {gen_label}_"
    )

    return "\n".join(lines)


def _fresh_envelope() -> dict:
    """Return a minimal DisclosureEnvelope dict for error-path responses."""
    return DisclosureEnvelope().to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: get_tearsheet
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="get_tearsheet",
    description=(
        "Assemble and return a full institutional intelligence report for an "
        "investment manager as structured Markdown. Pulls from SEC filings "
        "(ADV + 13F), team profiles, and live social signals. "
        "Provide the firm name exactly as registered, e.g. "
        "'Viking Global Investors' or 'CITADEL ADVISORS LLC'."
    ),
)
def get_tearsheet(firm_name: str) -> str:
    """
    Parameters
    ----------
    firm_name : str
        Name of the investment manager to generate a tearsheet for.

    Returns
    -------
    str
        Full Markdown tearsheet followed by the mandatory disclosure footer.
    """
    logger.info("get_tearsheet called: firm_name=%r", firm_name)

    # ── Assemble Gold Copy ────────────────────────────────────────────────────
    try:
        assembler = TearsheetAssembler()
        gold = assembler.assemble(firm_name)
    except AssemblerError as exc:
        logger.warning("Assembly failed for %r: %s", firm_name, exc)
        footer = _disclosure_footer(_fresh_envelope())
        return (
            f"# {firm_name} — Assembly Failed\n\n"
            f"Could not generate a tearsheet: {exc}\n\n"
            f"Please verify the firm name is registered in the AltBots "
            f"database and try again."
            + footer
        )
    except Exception as exc:
        logger.error("Unexpected error assembling %r: %s", firm_name, exc, exc_info=True)
        footer = _disclosure_footer(_fresh_envelope())
        return (
            f"# {firm_name} — Internal Error\n\n"
            f"An unexpected error occurred during assembly. "
            f"Please try again or contact support."
            + footer
        )

    # ── Render to Markdown ────────────────────────────────────────────────────
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, dir=_PROJECT_ROOT / "output"
        ) as tmp:
            tmp_path = Path(tmp.name)

        _render_md(gold, tmp_path)
        md_content = tmp_path.read_text(encoding="utf-8")
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.error("Markdown render failed for %r: %s", firm_name, exc, exc_info=True)
        # Fall back to a minimal inline render
        inst = gold.get("institutional_numbers") or {}
        overview = gold.get("firm_overview") or {}
        md_content = (
            f"# {gold.get('firm_name', firm_name)}\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Legal Name | {overview.get('legal_name', '—')} |\n"
            f"| AUM | {_fmt_usd_inline(inst.get('aum_total_usd'))} |\n"
            f"| SEC Registered | {'Yes' if overview.get('sec_registered') else 'No'} |\n"
        )

    # The Markdown renderer already includes a Legal Disclosures section.
    # Strip it and re-append via the authoritative _disclosure_footer() so the
    # footer is always produced by the same code path for all three tools.
    if "## Legal Disclosures" in md_content:
        md_content = md_content[:md_content.index("## Legal Disclosures")].rstrip()

    footer = _disclosure_footer(gold.get("disclosure") or _fresh_envelope())
    logger.info("get_tearsheet complete: firm=%r legal_ready=%s",
                gold.get("firm_name"), gold.get("is_legal_ready"))
    return md_content + footer


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: search_personnel
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="search_personnel",
    description=(
        "Search the AltBots team profiles database for investment professionals "
        "by name or job title. Returns matching profiles with synthesized bios. "
        "Examples: 'portfolio manager', 'Ole Andreas Halvorsen', 'Head of Research'."
    ),
)
def search_personnel(query: str, limit: int = 10) -> str:
    """
    Parameters
    ----------
    query : str
        Name, title, or keyword to search for (e.g. 'CIO', 'John Smith').
    limit : int
        Maximum number of profiles to return (1–20, default 10).

    Returns
    -------
    str
        Markdown table of matching profiles followed by mandatory disclosure.
    """
    logger.info("search_personnel called: query=%r limit=%d", query, limit)
    limit = max(1, min(20, limit))

    envelope = _fresh_envelope()

    # ── Qdrant vector search against team_profiles ────────────────────────────
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchText
        from openai import OpenAI

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Check collection exists
        existing = {c.name for c in client.get_collections().collections}
        if "team_profiles" not in existing:
            footer = _disclosure_footer(envelope)
            return (
                "## Personnel Search — No Data\n\n"
                "The `team_profiles` collection does not exist yet. "
                "Run `team_scraper` → `team_profiles_ingestor` to populate it."
                + footer
            )

        # Embed the query then run a vector similarity search
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set — cannot embed query")

        oai = OpenAI(api_key=openai_key)
        embed_resp = oai.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_vector = embed_resp.data[0].embedding

        hits = client.query_points(
            collection_name="team_profiles",
            query=query_vector,
            limit=limit,
            score_threshold=0.30,     # low threshold — cast a wide net
            with_payload=True,
        )
        points = hits.points

    except Exception as exc:
        logger.warning("search_personnel Qdrant/embed failed: %s", exc)
        footer = _disclosure_footer(envelope)
        return (
            f"## Personnel Search — Error\n\n"
            f"Could not query team profiles: {exc}"
            + footer
        )

    # ── Format results ────────────────────────────────────────────────────────
    if not points:
        footer = _disclosure_footer(envelope)
        return (
            f"## Personnel Search: '{query}'\n\n"
            f"No matching profiles found. Try a broader query or check that "
            f"team data has been ingested for the relevant firms."
            + footer
        )

    lines = [
        f"## Personnel Search: '{query}'",
        f"",
        f"_{len(points)} result(s) from AltBots team profiles · "
        f"Bios synthesized by Claude Haiku_",
        f"",
        f"| Name | Firm | Title | Bio |",
        f"|------|------|-------|-----|",
    ]

    seen_ids: set[str] = set()
    for pt in points:
        p = pt.payload or {}
        uid = f"{p.get('firm_name','')}:{p.get('person_name','')}"
        if uid in seen_ids:
            continue
        seen_ids.add(uid)

        name  = p.get("person_name", "—")
        firm  = p.get("firm_name", "—")
        title = p.get("title", "—")
        bio   = p.get("bio", "—")
        # Truncate bio for table display; full bio follows
        bio_short = bio[:120] + "…" if len(bio) > 120 else bio
        bio_short = bio_short.replace("|", "\\|")
        lines.append(f"| {name} | {firm} | {title} | {bio_short} |")

    # Full bio narratives below the table (more useful for agents)
    lines += ["", "### Full Profiles", ""]
    seen_ids_full: set[str] = set()
    for pt in points:
        p = pt.payload or {}
        uid = f"{p.get('firm_name','')}:{p.get('person_name','')}"
        if uid in seen_ids_full:
            continue
        seen_ids_full.add(uid)

        name  = p.get("person_name", "—")
        firm  = p.get("firm_name", "—")
        title = p.get("title", "—")
        bio   = p.get("bio", "—")
        scraped = p.get("scraped_at", "")[:10]
        lines += [
            f"**{name}** · {firm}",
            f"*{title}*",
            f"{bio}",
            f"_(Profile scraped: {scraped})_",
            "",
        ]

    envelope["data_sources"] = ["team_profiles Qdrant (vector similarity search)"]
    footer = _disclosure_footer(envelope)
    return "\n".join(lines) + footer


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: check_red_flags
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="check_red_flags",
    description=(
        "Run a regulatory red-flag screen for an investment manager. "
        "Checks Form ADV Item 11 (disciplinary disclosures) and SEC enforcement "
        "releases. Returns findings and the mandatory regulatory disclaimer. "
        "Requires the firm's CRD number — provide the firm name and the tool "
        "will attempt to resolve it from the AltBots database."
    ),
)
def check_red_flags(firm_name: str, crd_number: Optional[str] = None) -> str:
    """
    Parameters
    ----------
    firm_name : str
        Name of the investment manager to screen.
    crd_number : str, optional
        SEC CRD number. If omitted, resolved from fund_managers collection.

    Returns
    -------
    str
        Regulatory findings in Markdown followed by mandatory disclosure footer.
    """
    logger.info("check_red_flags called: firm=%r crd=%r", firm_name, crd_number)

    envelope = _fresh_envelope()

    # ── Resolve CRD if not supplied ───────────────────────────────────────────
    resolved_name = firm_name
    if not crd_number:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchText

            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            qclient = QdrantClient(host=qdrant_host, port=qdrant_port)

            result = qclient.scroll(
                collection_name="fund_managers",
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="legal_name",
                        match=MatchText(text=firm_name.upper()),
                    )
                ]),
                limit=3,
                with_payload=True,
                with_vectors=False,
            )
            pts = result[0]
            if pts:
                payload = pts[0].payload or {}
                crd_number = payload.get("crd") or ""
                resolved_name = payload.get("legal_name", firm_name)
                envelope["data_sources"] = [
                    f"fund_managers Qdrant (identity: {resolved_name})"
                ]
                logger.info("CRD resolved: %r → CRD %s", resolved_name, crd_number)
        except Exception as exc:
            logger.warning("CRD resolution failed for %r: %s", firm_name, exc)

    if not crd_number:
        footer = _disclosure_footer(envelope)
        return (
            f"## Red Flag Check: {firm_name}\n\n"
            f"**Could not resolve CRD number** for '{firm_name}'. "
            f"Please provide the `crd_number` argument directly, or ensure "
            f"the firm is registered in the AltBots `fund_managers` collection."
            + footer
        )

    # ── Run RedFlagScanner ────────────────────────────────────────────────────
    try:
        scanner = RedFlagScanner()
        result  = scanner.scan(firm_name=resolved_name, crd_number=crd_number)
    except Exception as exc:
        logger.error("RedFlagScanner failed for CRD %s: %s", crd_number, exc, exc_info=True)
        footer = _disclosure_footer(envelope)
        return (
            f"## Red Flag Check: {firm_name} (CRD {crd_number})\n\n"
            f"Regulatory screening encountered an error: {exc}\n\n"
            f"Please retry or consult SEC IAPD directly at "
            f"https://adviserinfo.sec.gov/firm/summary/{crd_number}"
            + footer
        )

    # ── Format findings ───────────────────────────────────────────────────────
    d = result.to_dict()
    lines = [
        f"## Regulatory Screen: {d['firm_name']}",
        f"",
        f"**CRD Number:** {d['crd_number']}  "
        f"**Scan Date:** {d['scan_date']}",
        f"",
        f"### Status",
        f"",
        f"> {d['status']}",
        f"",
    ]

    if d["findings"]:
        lines += [
            f"### Findings ({len(d['findings'])})",
            f"",
        ]
        for i, finding in enumerate(d["findings"], 1):
            lines += [
                f"**{i}. {finding['source_label']}**",
                f"",
                f"{finding['summary']}",
                f"",
                f"- As of: {finding['as_of']}",
                f"- Source: [{finding['source_label']}]({finding['source_url']})",
            ]
            if finding.get("filing_date"):
                lines.append(f"- Filing date: {finding['filing_date']}")
            if finding.get("item"):
                lines.append(f"- Item: {finding['item']}")
            lines.append("")
    else:
        lines += [
            "### Findings",
            "",
            "_No regulatory findings identified from available public records._",
            "",
            "This screen covers Form ADV Item 11 (disciplinary information) and "
            "SEC enforcement releases accessible via EDGAR full-text search. "
            "It is not exhaustive.",
            "",
        ]

    lines += [
        "### Regulatory Disclaimer",
        "",
        f"> {d['disclaimer']}",
        "",
        f"For a complete regulatory history, review the firm's current Form ADV "
        f"at [SEC IAPD](https://adviserinfo.sec.gov/firm/summary/{crd_number}).",
    ]

    # Build an envelope that reflects scanner sources
    envelope["data_sources"] = list(
        set((envelope.get("data_sources") or []) + [
            "RedFlagScanner (Form ADV Item 11)",
            "SEC EDGAR full-text enforcement search",
            f"SEC IAPD (CRD {crd_number})",
        ])
    )

    footer = _disclosure_footer(envelope)
    logger.info(
        "check_red_flags complete: firm=%r crd=%s findings=%d",
        resolved_name, crd_number, len(d["findings"]),
    )
    return "\n".join(lines) + footer


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: get_pipeline_status
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="get_pipeline_status",
    description=(
        "Return the status of the last AltBots weekly data-refresh sweep. "
        "Shows when it ran, which steps succeeded or failed, and the total "
        "duration. Use this to verify the pipeline is healthy before running "
        "get_tearsheet or search_personnel."
    ),
)
def get_pipeline_status() -> str:
    """
    Returns
    -------
    str
        Markdown summary of the last weekly sweep heartbeat.
    """
    logger.info("get_pipeline_status called")
    heartbeat_path = _PROJECT_ROOT / "logs" / "heartbeat.json"

    if not heartbeat_path.exists():
        return (
            "## AltBots Pipeline Status\n\n"
            "**No heartbeat found.** The weekly sweep has not run yet, or "
            "`run_weekly_sweep.sh` has not been executed.\n\n"
            "Run `bash run_weekly_sweep.sh` manually to initialise the pipeline."
        )

    try:
        hb = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            f"## AltBots Pipeline Status\n\n"
            f"**Could not read heartbeat:** {exc}"
        )

    overall  = hb.get("overall_status", "unknown")
    last_run = hb.get("last_run", "—")
    duration = hb.get("total_duration_seconds", "—")
    steps    = hb.get("steps") or {}
    failed   = [f for f in (hb.get("failed_steps") or []) if f]

    status_icon = "✅" if overall == "success" else "⚠️"

    lines = [
        "## AltBots Pipeline Status",
        "",
        f"**Last run:** {last_run}  **Duration:** {duration}s  "
        f"**Status:** {status_icon} {overall.replace('_', ' ').title()}",
        "",
        "| Step | Status | Duration |",
        "|------|--------|----------|",
    ]

    step_icons = {"ok": "✅", "skipped": "⏭️"}
    for step_name, info in steps.items():
        st = info.get("status", "unknown")
        icon = step_icons.get(st, "❌")
        lines.append(
            f"| {step_name.replace('_', ' ')} | {icon} {st} | {info.get('duration', '—')} |"
        )

    if failed:
        lines += [
            "",
            f"**Failed steps:** {', '.join(failed)}",
            f"Check `logs/cron_errors.log` and `logs/weekly_sweep.log` for details.",
        ]
    else:
        lines += [
            "",
            "_No failed steps. Pipeline is healthy._",
        ]

    lines += [
        "",
        f"_Heartbeat file: `logs/heartbeat.json`_",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Inline helpers (avoid importing formatter just for one helper)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_usd_inline(value) -> str:
    if value is None:
        return "—"
    try:
        v = int(value)
        if v >= 1_000_000_000:
            return f"${v/1_000_000_000:.2f}B"
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        return f"${v:,}"
    except Exception:
        return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AltBots MCP server — expose intelligence tools via MCP protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transport options:
  stdio  (default) — pipe-based transport for Claude Desktop and agent SDKs
  sse              — HTTP Server-Sent Events transport for browser/HTTP clients

Examples:
  python -m src.tools.tearsheet_mcp
  python -m src.tools.tearsheet_mcp --transport sse --port 8765
  python -m src.tools.tearsheet_mcp --log-level DEBUG
        """,
    )
    p.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE transport (default: 8765)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.transport == "stdio":
        logger.info("Starting AltBots-Intelligence MCP server (stdio transport)")
        mcp.run(transport="stdio")
    else:
        logger.info(
            "Starting AltBots-Intelligence MCP server (SSE transport, port=%d)",
            args.port,
        )
        # Override port via FastMCP settings before running
        mcp.settings.port = args.port
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()

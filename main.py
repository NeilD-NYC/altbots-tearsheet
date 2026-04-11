"""
main.py — AltBots Tearsheet Pipeline: CLI Entry Point

Assembles a Gold Copy from Qdrant + live data sources, then renders it to
the requested output format.

Usage
-----
  python main.py "Viking Global Investors" --format pdf
  python main.py "Viking Global Investors" --format md
  python main.py "Viking Global Investors" --format json
  python main.py "Viking Global Investors" --format pdf --output /tmp/viking.pdf
  python main.py "Viking Global Investors" --format pdf --save-gold
  python main.py --from-md output/FIRM_NAME.md --format pdf
  python main.py "Mast Hill Fund" --enrich --manual

Environment variables
---------------------
  OPENAI_API_KEY      — OpenAI embeddings (identity resolution, search)
  ANTHROPIC_API_KEY   — Claude Haiku (social signals)
  QDRANT_HOST         — Qdrant host (default: localhost)
  QDRANT_PORT         — Qdrant port (default: 6333)
  EXA_API_KEY         — Exa.ai (social signal scanner)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ── .env ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Enrichment data directory ─────────────────────────────────────────────────
_ENRICHED_DIR = _HERE / "data" / "enriched"


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blank_gold(firm_name: str) -> dict:
    """Return a minimal legal-ready Gold Copy dict for a firm with no data."""
    return {
        "firm_name":             firm_name,
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "is_legal_ready":        True,
        "firm_overview":         {"legal_name": firm_name},
        "institutional_numbers": {},
        "team_roster":           [],
        "social_signals":        [],
        "sec_enforcement":       [],
        "form_d_filings":        [],
        "sanctions_screening":   None,
        "performance":           None,
        "modules_skipped":       {},
        "data_sources":          [],
        "disclosure": {
            "version":          "1.0",
            "data_sources":     [],
            "modules_skipped":  {},
            "disclaimers": {
                "not_investment_advice":
                    "For informational purposes only. Not investment advice.",
                "regulatory_disclaimer":
                    "Based on public records. Does not replace an ODD review.",
                "data_limitation":
                    "Data from public filings. No warranty of accuracy.",
                "confidentiality":
                    "Intended solely for the recipient. Redistribution prohibited.",
            },
        },
    }


def _print_coverage(gold: dict) -> None:
    """Print current data coverage to stdout."""
    inst    = gold.get("institutional_numbers") or {}
    f13     = inst.get("latest_13f") or {}
    roster  = gold.get("team_roster") or []
    signals = gold.get("social_signals") or []

    W = 16  # label column width (after the 2-space indent)

    def _row(label, found, count=None):
        if found:
            status = f"{count} found" if count is not None else "found"
        else:
            status = "MISSING"
        print(f"  {label:<{W}} {status}")

    print()
    _row("SEC ADV:",        bool(inst.get("adv_filing_date")))
    _row("13F Holdings:",   bool(f13),      count=f13.get("num_positions") if f13 else None)
    _row("Team Roster:",    bool(roster),   count=len(roster))
    _row("Social Signals:", bool(signals),  count=len(signals))
    print()


def _prompt_enrichment() -> dict:
    """
    Interactively prompt the user for enrichment fields.
    Returns a dict of entered values (skipped fields are absent).
    """
    enriched: dict = {}

    def _ask(prompt: str) -> str:
        try:
            return input(f"  {prompt}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return ""

    print("  Press Enter to skip any field.")
    print()

    val = _ask("CRD Number: ")
    if val:
        enriched["crd"] = val

    val = _ask("AUM (e.g. 450000000 for $450M): ")
    if val:
        try:
            enriched["aum_usd"] = int(float(val.replace(",", "").replace("$", "")))
        except ValueError:
            print(f"    (skipping AUM — could not parse {val!r})")

    val = _ask("Strategy (e.g. Long/Short Equity): ")
    if val:
        enriched["strategy"] = val

    val = _ask("Inception Year: ")
    if val:
        enriched["inception_year"] = val

    val = _ask("Website URL: ")
    if val:
        enriched["website"] = val

    val = _ask("HQ City: ")
    if val:
        enriched["hq_city"] = val

    val = _ask("Custodian(s) comma separated: ")
    if val:
        enriched["custodians"] = [c.strip() for c in val.split(",") if c.strip()]

    val = _ask("Auditor: ")
    if val:
        enriched["auditor"] = val

    val = _ask("Fund Administrator: ")
    if val:
        enriched["fund_administrator"] = val

    key_persons = []
    for i in range(1, 3):
        name = _ask(f"Key Person {i} - Name: ")
        if not name:
            break
        title = _ask(f"Key Person {i} - Title: ")
        bg    = _ask(f"Key Person {i} - Background: ")
        key_persons.append({"name": name, "title": title or "—", "bio": bg or ""})
    if key_persons:
        enriched["key_persons"] = key_persons

    val = _ask("Above-peer strengths comma separated: ")
    if val:
        enriched["above_peer"] = [x.strip() for x in val.split(",") if x.strip()]

    val = _ask("Below-peer weaknesses comma separated: ")
    if val:
        enriched["below_peer"] = [x.strip() for x in val.split(",") if x.strip()]

    print("  Analyst notes (free text, appears in")
    print("    commentary section): ", end="", flush=True)
    try:
        val = input().strip()
    except (KeyboardInterrupt, EOFError):
        val = ""
    if val:
        enriched["analyst_notes"] = val

    return enriched


def _merge_enrichment(gold: dict, enriched: dict) -> dict:
    """
    Return a new Gold Copy dict with enriched values overlaid.
    Enriched data takes precedence; absent keys leave the original intact.
    """
    gold = copy.deepcopy(gold)
    ov   = gold.setdefault("firm_overview",         {})
    inst = gold.setdefault("institutional_numbers", {})
    svc  = gold.setdefault("service_providers",     {})

    if "crd" in enriched:
        ov["crd"] = enriched["crd"]
    if "aum_usd" in enriched:
        inst["aum_total_usd"]         = enriched["aum_usd"]
        inst["aum_discretionary_usd"] = enriched["aum_usd"]
    if "strategy" in enriched:
        gold["strategy_type"] = enriched["strategy"]
    if "inception_year" in enriched:
        ov["inception_year"] = enriched["inception_year"]
    if "website" in enriched:
        ov["website"] = enriched["website"]
    if "hq_city" in enriched:
        ov["hq_city"] = enriched["hq_city"]
    if "custodians" in enriched:
        svc["custodians"] = enriched["custodians"]
    if "auditor" in enriched:
        svc["auditor"] = enriched["auditor"]
    if "fund_administrator" in enriched:
        svc["fund_administrator"] = enriched["fund_administrator"]
    if "key_persons" in enriched:
        # Prepend manually entered persons; keep existing automated roster after
        gold["team_roster"] = enriched["key_persons"] + (gold.get("team_roster") or [])
    if "above_peer" in enriched or "below_peer" in enriched:
        pf = gold.setdefault("peer_flags", {})
        if "above_peer" in enriched:
            pf["above_peer"] = enriched["above_peer"]
        if "below_peer" in enriched:
            pf["below_peer"] = enriched["below_peer"]
    if "analyst_notes" in enriched:
        gold["_analyst_commentary"] = enriched["analyst_notes"]

    return gold


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "AltBots Tearsheet — assemble and render an institutional intelligence "
            "report for a fund manager."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Viking Global Investors" --format pdf
  python main.py "CITADEL ADVISORS LLC"   --format md
  python main.py "Viking Global Investors" --format json --output output/viking.json
  python main.py "Viking Global Investors" --format pdf --save-gold
  python main.py "Mast Hill Fund" --enrich --manual
        """,
    )
    p.add_argument(
        "firm",
        metavar="FIRM_NAME",
        nargs="?",
        help=(
            'Fund manager name, e.g. "Viking Global Investors". '
            "Not required when --from-md is used."
        ),
    )
    p.add_argument(
        "--from-md",
        metavar="MD_FILE",
        help=(
            "Skip live assembly. Parse a tearsheet Markdown file and render "
            "it directly to --format. Output goes to output/<SLUG>_edited.<ext> "
            "unless --output is specified."
        ),
    )
    p.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "Overlay manual enrichment data from data/enriched/<slug>.json "
            "before rendering. Use with --manual to enter data interactively."
        ),
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Prompt interactively for enrichment fields (requires --enrich). "
            "Saves answers to data/enriched/<slug>.json and regenerates the PDF."
        ),
    )
    p.add_argument(
        "--format",
        choices=["pdf", "md", "json"],
        default="pdf",
        help="Output format (default: pdf)",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help=(
            "Explicit output file path. "
            "If omitted, auto-generated under output/<FIRM_SLUG>.<ext>"
        ),
    )
    p.add_argument(
        "--save-gold",
        action="store_true",
        help="Also save the raw Gold Copy JSON alongside the formatted output.",
    )
    p.add_argument(
        "--use-fixture",
        action="store_true",
        help=(
            "Skip live assembly. Load Gold Copy from "
            "fixtures/meridian_sample.json (or fixtures/<firm_slug>.json)."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.manual and not args.enrich:
        parser.error("--manual requires --enrich")

    fmt = args.format

    # ── --from-md path ────────────────────────────────────────────────────────
    if args.from_md:
        md_path = Path(args.from_md)
        if not md_path.exists():
            logger.error("Markdown file not found: %s", md_path)
            return 1

        logger.info("Parsing Markdown tearsheet: %s", md_path)
        try:
            from src.renderers.tearsheet_formatter import TearsheetFormatter, _slug
            formatter = TearsheetFormatter()
            gold      = formatter.parse_md(md_path)
        except Exception as exc:
            logger.error("Markdown parsing failed: %s", exc, exc_info=args.debug)
            return 1

        logger.info("Parsed firm: %r", gold.get("firm_name"))

        # Derive output path: output/<SLUG>_edited.<ext> (unless --output given)
        if args.output:
            out_path = Path(args.output)
        else:
            ext      = "pdf" if fmt == "pdf" else ("md" if fmt == "md" else "json")
            slug     = _slug(gold.get("firm_name", "tearsheet"))
            out_path = Path("output") / f"{slug}_edited.{ext}"

        try:
            result = formatter.format(gold, fmt=fmt, out_path=out_path)
        except Exception as exc:
            logger.error("Formatting failed: %s", exc, exc_info=args.debug)
            return 1

        print(f"\n  ✓  {result}\n")
        return 0

    # ── Normal / enrichment path — firm name required ─────────────────────────
    if not args.firm:
        parser.error("FIRM_NAME is required unless --from-md is specified.")

    firm_name = args.firm.strip()
    from src.renderers.tearsheet_formatter import TearsheetFormatter, _slug
    slug = _slug(firm_name)

    # ── Step 1: Assemble Gold Copy (or load fixture) ──────────────────────────
    if args.use_fixture:
        fixture_dir  = Path("fixtures")
        fixture_path = fixture_dir / f"{slug}.json"
        if not fixture_path.exists():
            fixture_path = fixture_dir / "meridian_sample.json"
        if not fixture_path.exists():
            logger.error("No fixture found at %s", fixture_path)
            return 1
        logger.info("Loading fixture: %s", fixture_path)
        gold = json.loads(fixture_path.read_text(encoding="utf-8"))
    else:
        logger.info("Assembling Gold Copy for: %r  (format=%s)", firm_name, fmt)
        try:
            from src.core.tearsheet_assembler import TearsheetAssembler
            assembler = TearsheetAssembler()
            gold      = assembler.assemble(firm_name)
            logger.info(
                "Assembly complete — is_legal_ready=%s  modules_skipped=%d",
                gold.get("is_legal_ready"),
                len(gold.get("modules_skipped") or {}),
            )
        except Exception as exc:
            if args.enrich:
                # In enrich mode we can work from a blank slate
                logger.warning(
                    "Assembly failed (%s); starting with blank slate for enrichment.", exc
                )
                gold = _blank_gold(firm_name)
            else:
                logger.error("Assembly failed: %s", exc, exc_info=args.debug)
                return 1

    # ── Step 1b: --enrich path ────────────────────────────────────────────────
    if args.enrich:
        enriched_path = _ENRICHED_DIR / f"{slug}.json"

        if args.manual:
            # ── Print coverage ────────────────────────────────────────────────
            _print_coverage(gold)

            # ── Prompt ───────────────────────────────────────────────────────
            enriched = _prompt_enrichment()

            # ── Merge with any existing enrichment (new values win) ───────────
            if enriched_path.exists():
                try:
                    old = json.loads(enriched_path.read_text(encoding="utf-8"))
                    old.update(enriched)
                    enriched = old
                except Exception:
                    pass

            # ── Stamp metadata and save ───────────────────────────────────────
            enriched["firm_name"]    = firm_name
            enriched["enriched_at"]  = datetime.now(timezone.utc).isoformat()
            _ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
            enriched_path.write_text(
                json.dumps(enriched, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Enrichment saved to %s", enriched_path)

        else:
            # Non-interactive: load existing enrichment if present
            if not enriched_path.exists():
                logger.warning(
                    "No enrichment file found at %s — run with --manual to create one.",
                    enriched_path,
                )
                enriched = {}
            else:
                logger.info("Loading enrichment from %s", enriched_path)
                enriched = json.loads(enriched_path.read_text(encoding="utf-8"))

        # ── Merge enrichment into gold ────────────────────────────────────────
        if enriched:
            gold = _merge_enrichment(gold, enriched)

    # ── Step 2 (optional): Save Gold Copy JSON ────────────────────────────────
    if args.save_gold:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        gold_path  = output_dir / f"{slug}_gold.json"
        gold_path.write_text(
            json.dumps(gold, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Gold Copy saved to %s", gold_path)

    # ── Step 3: Format output ─────────────────────────────────────────────────
    try:
        formatter = TearsheetFormatter()
        out_path  = Path(args.output) if args.output else None
        result    = formatter.format(gold, fmt=fmt, out_path=out_path)
    except Exception as exc:
        logger.error("Formatting failed: %s", exc, exc_info=args.debug)
        return 1

    print(f"\n  ✓  {result}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import json
import logging
import sys
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
        """,
    )
    p.add_argument(
        "firm",
        metavar="FIRM_NAME",
        help='Fund manager name, e.g. "Viking Global Investors"',
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


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    firm_name = args.firm.strip()
    fmt       = args.format

    # ── Step 1: Assemble Gold Copy (or load fixture) ──────────────────────────
    if args.use_fixture:
        from src.renderers.tearsheet_formatter import _slug
        fixture_dir  = Path("fixtures")
        slug         = _slug(firm_name)
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
            from src.core.tearsheet_assembler import TearsheetAssembler, AssemblerError
            assembler = TearsheetAssembler()
            gold      = assembler.assemble(firm_name)
        except Exception as exc:
            logger.error("Assembly failed: %s", exc, exc_info=args.debug)
            return 1

        logger.info(
            "Assembly complete — is_legal_ready=%s  modules_skipped=%d",
            gold.get("is_legal_ready"),
            len(gold.get("modules_skipped") or {}),
        )

    # ── Step 2 (optional): Save Gold Copy JSON ────────────────────────────────
    if args.save_gold:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        from src.renderers.tearsheet_formatter import _slug
        gold_path  = output_dir / f"{_slug(gold['firm_name'])}_gold.json"
        gold_path.write_text(
            json.dumps(gold, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Gold Copy saved to %s", gold_path)

    # ── Step 3: Format output ─────────────────────────────────────────────────
    try:
        from src.renderers.tearsheet_formatter import TearsheetFormatter
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

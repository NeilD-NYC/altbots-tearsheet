"""
src/scrapers/team_scraper.py — AltBots Pipeline: Team Page Scraper

Discovers and scrapes team/people pages from validated fund manager websites
using Firecrawl. Saves one raw-markdown record per firm to a JSONL file for
downstream processing by the team_profiles_ingestor.

Usage
-----
  python -m src.scrapers.team_scraper                         # batch from data/firms.json
  python -m src.scrapers.team_scraper --input data/firms.json --resume
  python -m src.scrapers.team_scraper --firm "Acme Capital" --url "https://acmecapital.com"
  python -m src.scrapers.team_scraper --debug

Environment variables
---------------------
  FIRECRAWL_API_KEY   — Firecrawl v1 API key
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv

# ── Project root and .env ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# ── Optional SDK imports ──────────────────────────────────────────────────────
try:
    from firecrawl import V1FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("team_scraper")

# ── Config ────────────────────────────────────────────────────────────────────
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"

FIRMS_INPUT  = _PROJECT_ROOT / "data" / "firms.json"
OUTPUT_JSONL = _PROJECT_ROOT / "data" / "team_profiles_raw.jsonl"
RUN_LOG      = _PROJECT_ROOT / "logs" / "team_scraper_log.json"

# Candidate path suffixes — used as Phase 2 fallback when /map finds nothing
TEAM_PATH_SUFFIXES = [
    "/team",
    "/our-team",
    "/people",
    "/about/team",
    "/about/people",
    "/about-us",
    "/leadership",
    "/investment-team",
    "/investment-professionals",
    "/professionals",
    "/partners",
]

# 5-second inter-firm delay (anti-scraping protection)
INTER_FIRM_DELAY_SECONDS = 5.0
RETRY_DELAY_SECONDS      = 5.0
MAX_SCRAPE_RETRIES       = 1
SCRAPE_TIMEOUT_MS        = 30_000
MIN_CONTENT_LENGTH       = 200   # chars; below this a page is considered empty


# ── Keyword regex for validating team page content ────────────────────────────
_TEAM_CONTENT_KEYWORDS = re.compile(
    r"\b(ceo|cio|cfo|coo|founder|partner|managing director|portfolio manager|"
    r"general counsel|chief|head of|vice president|principal|analyst|associate|"
    r"managing partner|co-founder|president|director)\b",
    re.IGNORECASE,
)


class TeamScraper:
    """
    Discovers and scrapes team pages via Firecrawl.
    Saves one raw-markdown record per firm to JSONL for the ingestor.
    """

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        output_path: Path = OUTPUT_JSONL,
        log_path: Path = RUN_LOG,
    ):
        self._fc_key      = firecrawl_api_key or os.environ.get(FIRECRAWL_API_KEY_ENV, "")
        self._output_path = Path(output_path)
        self._log_path    = Path(log_path)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        if not FIRECRAWL_AVAILABLE:
            raise RuntimeError("firecrawl-py not installed. Run: pip install firecrawl-py")
        if not self._fc_key:
            raise RuntimeError("FIRECRAWL_API_KEY is not set. Export it or add it to .env.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, firms: list[dict], resume: bool = False) -> dict:
        """
        Process a list of firm dicts — each must have `firm_name` and `website_url`.

        Returns run statistics dict (also written to the log file).
        """
        already_done: set[str] = set()
        if resume:
            already_done = self._load_scraped_firms()
            logger.info("[Resume] %d firm(s) already in output — skipping", len(already_done))

        stats = {
            "firms_attempted":   0,
            "firms_succeeded":   0,
            "firms_failed":      0,
            "run_timestamp":     datetime.now(timezone.utc).isoformat(),
        }

        for i, firm in enumerate(firms):
            firm_name   = firm.get("firm_name", "").strip()
            website_url = firm.get("website_url", "").strip()

            if not firm_name or not website_url:
                logger.warning("Skipping entry with missing firm_name or website_url: %r", firm)
                continue

            if resume and firm_name in already_done:
                logger.info("[%d/%d] SKIP (already done): %s", i + 1, len(firms), firm_name)
                continue

            stats["firms_attempted"] += 1
            logger.info("[%d/%d] Processing: %s (%s)", i + 1, len(firms), firm_name, website_url)

            try:
                record = self._process_firm(firm_name, website_url)
                if record is not None:
                    stats["firms_succeeded"] += 1
                    logger.info(
                        "  ✓ %s — team page scraped (%d chars)",
                        firm_name, len(record["raw_markdown"]),
                    )
                else:
                    stats["firms_failed"] += 1
                    logger.warning("  ✗ %s — team page not found", firm_name)
            except Exception as exc:
                stats["firms_failed"] += 1
                logger.error("  ✗ %s — unexpected error: %s", firm_name, exc, exc_info=True)

            # 5-second inter-firm delay (skip after last firm)
            if i < len(firms) - 1:
                logger.debug("  Sleeping %ds before next firm…", INTER_FIRM_DELAY_SECONDS)
                time.sleep(INTER_FIRM_DELAY_SECONDS)

        self._write_run_log(stats)
        logger.info(
            "Run complete — attempted=%d succeeded=%d failed=%d",
            stats["firms_attempted"], stats["firms_succeeded"], stats["firms_failed"],
        )
        return stats

    def run_single(self, firm_name: str, website_url: str) -> Optional[dict]:
        """Process a single firm. Returns the raw record or None."""
        logger.info("Single-firm run: %s (%s)", firm_name, website_url)
        return self._process_firm(firm_name, website_url)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _process_firm(self, firm_name: str, website_url: str) -> Optional[dict]:
        """
        Discover and scrape the team page for one firm.
        Writes one record to JSONL on success; returns the record or None.
        """
        team_url, markdown = self._discover_and_scrape(website_url)
        if not team_url or not markdown:
            return None

        record = {
            "firm_name":    firm_name,
            "website_url":  website_url,
            "team_page_url": team_url,
            "raw_markdown": markdown,
            "scraped_at":   datetime.now(timezone.utc).isoformat(),
        }
        self._append_jsonl(record)
        return record

    # ── Team page discovery ───────────────────────────────────────────────────

    def _discover_and_scrape(
        self, website_url: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Phase 1 — Firecrawl /map: enumerate all site URLs, filter by team keywords,
                  scrape the best match (sorted deepest path first, cap 8).
        Phase 2 — Suffix probe fallback: try TEAM_PATH_SUFFIXES in order.

        Returns (team_url, markdown) or (None, None).
        """
        base = _normalise_base(website_url)

        # Phase 1: map-based discovery
        map_candidates = self._map_team_urls(base)
        if map_candidates:
            logger.debug("  /map returned %d team candidate(s)", len(map_candidates))
            for url in map_candidates:
                markdown = self._scrape_with_retry(url)
                if markdown and len(markdown.strip()) >= MIN_CONTENT_LENGTH:
                    if _looks_like_team_page(markdown):
                        logger.debug("  /map hit: %s", url)
                        return url, markdown
            logger.debug("  /map candidates all failed content check — falling back")

        # Phase 2: suffix probe fallback
        tried = set(map_candidates)
        for suffix in TEAM_PATH_SUFFIXES:
            url = urljoin(base, suffix)
            if url in tried:
                continue
            markdown = self._scrape_with_retry(url)
            if markdown and len(markdown.strip()) >= MIN_CONTENT_LENGTH:
                if _looks_like_team_page(markdown):
                    logger.debug("  Suffix hit: %s", url)
                    return url, markdown

        return None, None

    def _map_team_urls(self, base: str) -> list[str]:
        """Call Firecrawl /map and return team-keyword URLs sorted by depth, capped at 8."""
        try:
            fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key
            fc = V1FirecrawlApp(api_key=fc_key)
            response = fc.map_url(base, limit=200, timeout=30000)
            all_urls: list[str] = response.links or []
        except Exception as exc:
            logger.debug("  /map failed for %s: %s", base, exc)
            return []

        keywords = [
            "team", "people", "leadership", "partners", "staff",
            "management", "professionals", "executives", "founders",
            "investment-team", "about/team", "about/people",
        ]
        matched = [
            u for u in all_urls
            if any(kw in u.lower() for kw in keywords)
            and not u.lower().endswith((".pdf", ".png", ".jpg", ".svg"))
        ]
        # Prefer shallow paths (team index pages like /ourteams/) over deep ones
        # (individual profile pages like /ourteams/john-smith). Index pages have
        # fewer path segments and more likely contain the full roster.
        matched.sort(key=lambda u: u.count("/"))
        return matched[:8]

    # ── Firecrawl scrape (with one retry) ─────────────────────────────────────

    def _scrape_with_retry(self, url: str) -> Optional[str]:
        """Scrape URL via Firecrawl with one retry. Returns markdown or None."""
        for attempt in range(1, MAX_SCRAPE_RETRIES + 2):
            try:
                fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key
                fc = V1FirecrawlApp(api_key=fc_key)
                response = fc.scrape_url(
                    url,
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=SCRAPE_TIMEOUT_MS,
                )
                if not response.success:
                    raise ValueError(
                        f"Firecrawl success=False: {response.error or response.warning}"
                    )
                markdown = response.markdown or ""
                return markdown if markdown.strip() else None

            except Exception as exc:
                status = _extract_http_status(exc)
                if status in (402, 429):
                    logger.error(
                        "  Firecrawl %d — quota/rate-limit; aborting scrape for %s",
                        status, url,
                    )
                    return None
                if status in (403, 404):
                    return None
                if attempt <= MAX_SCRAPE_RETRIES:
                    logger.warning(
                        "  Scrape attempt %d failed for %s: %s — retrying in %ds",
                        attempt, url, exc, RETRY_DELAY_SECONDS,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.warning("  Scrape failed for %s after %d attempt(s): %s", url, attempt, exc)
        return None

    # ── JSONL I/O ─────────────────────────────────────────────────────────────

    def _append_jsonl(self, record: dict) -> None:
        with self._output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_scraped_firms(self) -> set[str]:
        """Return the set of firm_name values already written to the output JSONL."""
        seen: set[str] = set()
        if not self._output_path.exists():
            return seen
        with self._output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fn = obj.get("firm_name", "")
                    if fn:
                        seen.add(fn)
                except json.JSONDecodeError:
                    continue
        return seen

    def _write_run_log(self, stats: dict) -> None:
        log_entries: list[dict] = []
        if self._log_path.exists():
            try:
                with self._log_path.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                    log_entries = existing if isinstance(existing, list) else [existing]
            except Exception:
                log_entries = []
        log_entries.append(stats)
        with self._log_path.open("w", encoding="utf-8") as fh:
            json.dump(log_entries, fh, indent=2, ensure_ascii=False)
        logger.info("Run log written to %s", self._log_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _looks_like_team_page(markdown: str) -> bool:
    """Return True if the markdown contains ≥ 2 person-title keyword hits."""
    return len(_TEAM_CONTENT_KEYWORDS.findall(markdown)) >= 2


def _normalise_base(url: str) -> str:
    """Strip path/query/fragment to get scheme + host."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _extract_http_status(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from an exception, or return None."""
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    msg = str(exc)
    for code in (402, 403, 404, 429, 500):
        if str(code) in msg:
            return code
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AltBots team page scraper — crawls fund manager websites and saves raw markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scrapers.team_scraper
  python -m src.scrapers.team_scraper --input data/firms.json --resume
  python -m src.scrapers.team_scraper --firm "Acme Capital" --url "https://acmecapital.com"
        """,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--input",  metavar="FILE", help="JSON file with firm list (default: data/firms.json)")
    mode.add_argument("--firm",   metavar="NAME", help="Single firm name (use with --url)")
    p.add_argument("--url",      metavar="URL",  help="Website URL for single-firm run (use with --firm)")
    p.add_argument("--resume",   action="store_true", help="Skip firms already in the JSONL output")
    p.add_argument("--output",   metavar="FILE", default=str(OUTPUT_JSONL),
                   help=f"JSONL output path (default: {OUTPUT_JSONL})")
    p.add_argument("--debug",    action="store_true", help="Enable DEBUG-level logging")
    return p


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.firm and not args.url:
        parser.error("--firm requires --url")
    if args.url and not args.firm:
        parser.error("--url requires --firm")

    try:
        scraper = TeamScraper(output_path=Path(args.output))
    except RuntimeError as exc:
        logger.error("Init failed: %s", exc)
        return 1

    if args.firm:
        record = scraper.run_single(args.firm, args.url)
        if record:
            preview = record.copy()
            preview["raw_markdown"] = preview["raw_markdown"][:300] + "…"
            print(json.dumps(preview, indent=2, ensure_ascii=False))
        else:
            logger.warning("No team page found for %s", args.firm)
        return 0

    # Batch run
    input_path = Path(args.input) if args.input else FIRMS_INPUT
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    try:
        with input_path.open("r", encoding="utf-8") as fh:
            firms = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse input JSON: %s", exc)
        return 1

    if not isinstance(firms, list):
        logger.error("Input file must contain a JSON array of firm objects.")
        return 1

    scraper.run(firms, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())

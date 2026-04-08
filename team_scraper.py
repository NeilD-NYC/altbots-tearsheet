"""
team_scraper.py — AltBots RAG Pipeline: Team Page Scraper

Discovers, scrapes, and extracts investment professionals from fund manager
websites. Output feeds the Qdrant RAG ingestion pipeline.

Usage
-----
  # Batch run from a firms JSON file
  python team_scraper.py --input firms.json

  # Single firm (one-off)
  python team_scraper.py --firm "Acme Capital" --url "https://acmecapital.com"

  # Resume a partial run (skips firms already in the JSONL output)
  python team_scraper.py --input firms.json --resume

Environment variables required
-------------------------------
  FIRECRAWL_API_KEY   — Firecrawl v1 API key
  ANTHROPIC_API_KEY   — Anthropic API key (Claude Haiku extraction)
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

# ── Load .env from the project root ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", override=True)

# ── Optional SDK imports ──────────────────────────────────────────────────────
try:
    from firecrawl import V1FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("team_scraper")

# ── Config ────────────────────────────────────────────────────────────────────

FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
HAIKU_MODEL           = "claude-haiku-4-5-20251001"

OUTPUT_JSONL = Path("/data/team_profiles_raw.jsonl")
RUN_LOG      = Path("/logs/team_scraper_log.json")

# Candidate path suffixes tried in order for team page discovery
TEAM_PATH_SUFFIXES = [
    "/team",
    "/our-team",
    "/people",
    "/about/team",
    "/about/people",
    "/about-us",
    "/leadership",
    "/investment-team",
]

# Minimum markdown length to consider a page a real hit
MIN_CONTENT_LENGTH = 200

# Delays and retries
INTER_FIRM_DELAY_SECONDS   = 2.0
RETRY_DELAY_SECONDS        = 5.0
MAX_SCRAPE_RETRIES         = 1        # one retry after initial failure
SCRAPE_TIMEOUT_MS          = 30_000   # Firecrawl timeout in ms

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = (
    "You are a data extraction engine for an investment research database. "
    "You extract structured information about investment professionals from "
    "raw website text. You return only valid JSON — no prose, no markdown "
    "fences, no commentary."
)

_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract all investment professionals from the following firm website text.\n"
    "For each person return a JSON object with these fields:\n"
    "  name        (string, required)\n"
    "  title       (string, empty string if not found)\n"
    "  bio         (full text if available, empty string if not found)\n"
    "  linkedin_url (string URL or null)\n"
    "  email       (string or null)\n\n"
    "Return a JSON array only. No commentary. "
    "If no people are found return an empty array [].\n\n"
    "Website text:\n"
    "---\n"
    "{content}\n"
    "---"
)


# ── TeamScraper ───────────────────────────────────────────────────────────────

class TeamScraper:
    """
    Discovers team pages, scrapes them via Firecrawl, and extracts
    personnel records via Claude Haiku.

    Parameters
    ----------
    firecrawl_api_key : str, optional
        Falls back to FIRECRAWL_API_KEY env var.
    anthropic_api_key : str, optional
        Falls back to ANTHROPIC_API_KEY env var.
    output_path : Path, optional
        JSONL output file. Default: /data/team_profiles_raw.jsonl
    log_path : Path, optional
        Run log JSON file. Default: /logs/team_scraper_log.json
    """

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        output_path: Path = OUTPUT_JSONL,
        log_path: Path = RUN_LOG,
    ):
        self._fc_key = firecrawl_api_key or os.environ.get(FIRECRAWL_API_KEY_ENV, "")
        self._an_key = anthropic_api_key or os.environ.get(ANTHROPIC_API_KEY_ENV, "")

        self._output_path = Path(output_path)
        self._log_path    = Path(log_path)

        # Ensure output directories exist
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialise Firecrawl client
        if not FIRECRAWL_AVAILABLE:
            raise RuntimeError(
                "firecrawl-py is not installed. Run: pip install firecrawl-py"
            )
        if not self._fc_key:
            raise RuntimeError(
                "FIRECRAWL_API_KEY is not set. "
                "Export it or add it to .env."
            )
        self._fc = V1FirecrawlApp(api_key=self._fc_key)

        # Initialise Anthropic client
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic is not installed. Run: pip install anthropic"
            )
        if not self._an_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or add it to .env."
            )
        self._haiku = anthropic.Anthropic(api_key=self._an_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        firms: list[dict],
        resume: bool = False,
    ) -> dict:
        """
        Process a list of firm dicts.

        Parameters
        ----------
        firms : list[dict]
            Each dict must have keys: ``firm_name``, ``website_url``.
        resume : bool
            If True, skip firms whose website_url already appears in the
            JSONL output file.

        Returns
        -------
        dict
            Run statistics (also written to the log file).
        """
        already_done: set[str] = set()
        if resume:
            already_done = self._load_scraped_urls()
            logger.info("[Resume] %d firm URL(s) already in output — skipping", len(already_done))

        stats = {
            "firms_attempted":       0,
            "firms_succeeded":       0,
            "firms_failed":          0,
            "total_people_extracted": 0,
            "run_timestamp":         datetime.now(timezone.utc).isoformat(),
        }

        for i, firm in enumerate(firms):
            firm_name   = firm.get("firm_name", "").strip()
            website_url = firm.get("website_url", "").strip()

            if not firm_name or not website_url:
                logger.warning("Skipping entry with missing firm_name or website_url: %r", firm)
                continue

            if resume and website_url in already_done:
                logger.info("[%d/%d] SKIP (already done): %s", i + 1, len(firms), firm_name)
                continue

            stats["firms_attempted"] += 1
            logger.info("[%d/%d] Processing: %s (%s)", i + 1, len(firms), firm_name, website_url)

            try:
                people = self._process_firm(firm_name, website_url)
                if people is not None:
                    stats["firms_succeeded"] += 1
                    stats["total_people_extracted"] += len(people)
                    logger.info(
                        "  ✓ %s — %d person(s) extracted", firm_name, len(people)
                    )
                else:
                    stats["firms_failed"] += 1
                    logger.warning("  ✗ %s — team page not found or extraction failed", firm_name)
            except Exception as exc:
                stats["firms_failed"] += 1
                logger.error("  ✗ %s — unexpected error: %s", firm_name, exc, exc_info=True)

            # Inter-firm delay (skip after last firm)
            if i < len(firms) - 1:
                time.sleep(INTER_FIRM_DELAY_SECONDS)

        self._write_run_log(stats)
        logger.info(
            "Run complete — attempted=%d succeeded=%d failed=%d people=%d",
            stats["firms_attempted"],
            stats["firms_succeeded"],
            stats["firms_failed"],
            stats["total_people_extracted"],
        )
        return stats

    def run_single(self, firm_name: str, website_url: str) -> list[dict]:
        """
        Process a single firm. Returns the list of extracted person records.
        Writes to the same JSONL output as batch runs.
        """
        logger.info("Single-firm run: %s (%s)", firm_name, website_url)
        result = self._process_firm(firm_name, website_url)
        if result is None:
            logger.warning("No people extracted for %s", firm_name)
            return []
        return result

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _process_firm(
        self, firm_name: str, website_url: str
    ) -> Optional[list[dict]]:
        """
        Full pipeline for one firm:
          1. Discover team page URL
          2. Scrape it
          3. Extract people via Haiku
          4. Write to JSONL

        Returns the list of person records, or None if the team page
        could not be found or yielded no content.
        """
        # Step 1 — discover team page
        team_url, markdown = self._discover_and_scrape(website_url)
        if not team_url or not markdown:
            logger.info("  [%s] team_page_not_found", firm_name)
            return None

        logger.info("  [%s] Team page found: %s (%d chars)", firm_name, team_url, len(markdown))

        # Step 2 — extract people via Haiku
        people_raw = self._extract_people(markdown)
        if people_raw is None:
            # Extraction hard-failed (Haiku error, not parse error)
            return None

        # Step 3 — stamp and write records
        now = datetime.now(timezone.utc).isoformat()
        records: list[dict] = []
        for person in people_raw:
            if not isinstance(person, dict):
                continue
            name = str(person.get("name", "")).strip()
            if not name:
                continue   # skip empty-name entries

            record = {
                "firm_name":    firm_name,
                "website_url":  website_url,
                "team_page_url": team_url,
                "name":         name,
                "title":        str(person.get("title", "") or "").strip(),
                "bio":          str(person.get("bio", "") or "").strip(),
                "linkedin_url": _clean_optional(person.get("linkedin_url")),
                "email":        _clean_optional(person.get("email")),
                "scraped_at":   now,
            }
            records.append(record)

        self._append_jsonl(records)
        return records

    # ── Team page discovery ───────────────────────────────────────────────────

    def _discover_and_scrape(
        self, website_url: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Two-phase discovery:

        Phase 1 — Firecrawl /map (preferred):
          Map the full site URL list, filter to team-keyword paths,
          scrape the best match.  This reliably finds non-obvious paths
          like /our-team/, /investment-professionals/, /about/people/, etc.

        Phase 2 — Suffix probe (fallback):
          If /map returns nothing or all map candidates fail, try the
          hardcoded TEAM_PATH_SUFFIXES list in order.

        Returns (team_url, markdown) for the first successful hit,
        or (None, None) if all candidates fail.
        """
        base = _normalise_base(website_url)

        # ── Phase 1: map-based discovery ─────────────────────────────────────
        map_candidates = self._map_team_urls(base)
        if map_candidates:
            logger.debug("  /map returned %d team candidate(s)", len(map_candidates))
            for url in map_candidates:
                logger.debug("  Trying (map): %s", url)
                markdown = self._scrape_with_retry(url)
                if markdown and len(markdown.strip()) >= MIN_CONTENT_LENGTH:
                    # Confirm page has person-like content before returning
                    if _looks_like_team_page(markdown):
                        return url, markdown
            logger.debug("  /map candidates all failed content check — falling back to suffix probe")

        # ── Phase 2: suffix probe fallback ────────────────────────────────────
        suffix_candidates = [urljoin(base, s) for s in TEAM_PATH_SUFFIXES]
        # Skip any already tried in Phase 1
        tried = set(map_candidates)
        for url in suffix_candidates:
            if url in tried:
                continue
            logger.debug("  Trying (suffix): %s", url)
            markdown = self._scrape_with_retry(url)
            if markdown and len(markdown.strip()) >= MIN_CONTENT_LENGTH:
                return url, markdown

        return None, None

    def _map_team_urls(self, base: str) -> list[str]:
        """
        Call Firecrawl /map to enumerate site URLs, return those that match
        team-page keywords — sorted so deeper/more specific paths come first.
        Capped at 8 candidates to limit credit spend.
        """
        try:
            fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key
            fc = V1FirecrawlApp(api_key=fc_key)
            response = fc.map_url(base, limit=200, timeout=30000)
            all_urls: list[str] = response.links or []
        except Exception as exc:
            logger.debug("  /map failed for %s: %s", base, exc)
            return []

        # Filter to URLs whose path contains a team-related keyword
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

        # Prefer deeper paths (more specific) over shallow ones
        matched.sort(key=lambda u: -u.count("/"))

        return matched[:8]

    # ── Firecrawl scrape (with one retry) ─────────────────────────────────────

    def _scrape_with_retry(self, url: str) -> Optional[str]:
        """
        Scrape a URL via Firecrawl. Retries once after RETRY_DELAY_SECONDS
        on any failure. Returns markdown string or None.
        """
        for attempt in range(1, MAX_SCRAPE_RETRIES + 2):  # attempts: 1, 2
            try:
                # Read key fresh in case env was updated after init
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
                        f"Firecrawl reported success=False: {response.error or response.warning}"
                    )

                markdown = response.markdown or ""
                return markdown if markdown.strip() else None

            except Exception as exc:
                status = _extract_http_status(exc)

                # 402 / 429 — quota or rate limit: abort immediately, no retry
                if status in (402, 429):
                    logger.error(
                        "  Firecrawl %d (%s) — aborting scrape for %s",
                        status,
                        "quota exhausted" if status == 402 else "rate limited",
                        url,
                    )
                    return None

                # 404 / 403 — page does not exist: no point retrying
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

    # ── Haiku extraction ──────────────────────────────────────────────────────

    def _extract_people(self, markdown: str) -> Optional[list]:
        """
        Send scraped markdown to Claude Haiku and parse the JSON response.

        Returns a list of raw person dicts (may be empty), or None if
        the Haiku call itself fails (not a JSON parse error).
        JSON parse errors are logged and an empty list is returned instead.
        """
        # Trim to ~12 000 chars to stay well within Haiku context window
        content = markdown[:12_000]
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(content=content)

        try:
            response = self._haiku.messages.create(
                model=HAIKU_MODEL,
                max_tokens=4096,
                system=_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.error("  Haiku API call failed: %s", exc)
            return None

        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if Haiku wrapped the JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        # Find the outermost JSON array
        start = raw_text.find("[")
        end   = raw_text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning(
                "  Haiku returned no JSON array — parse skipped. Preview: %r",
                raw_text[:200],
            )
            return []

        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("  JSON parse error from Haiku output: %s. Preview: %r", exc, raw_text[:300])
            return []

    # ── JSONL I/O ─────────────────────────────────────────────────────────────

    def _append_jsonl(self, records: list[dict]) -> None:
        """Append records to the JSONL output file (one JSON object per line)."""
        if not records:
            return
        with self._output_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_scraped_urls(self) -> set[str]:
        """
        Read the existing JSONL output and return the set of website_url
        values already present (used for --resume).
        """
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
                    url = obj.get("website_url", "")
                    if url:
                        seen.add(url)
                except json.JSONDecodeError:
                    continue
        return seen

    def _write_run_log(self, stats: dict) -> None:
        """Write (or append to) the run log JSON file."""
        log_entries: list[dict] = []
        if self._log_path.exists():
            try:
                with self._log_path.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                    if isinstance(existing, list):
                        log_entries = existing
                    else:
                        log_entries = [existing]
            except Exception:
                log_entries = []

        log_entries.append(stats)

        with self._log_path.open("w", encoding="utf-8") as fh:
            json.dump(log_entries, fh, indent=2, ensure_ascii=False)

        logger.info("Run log written to %s", self._log_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

_TEAM_CONTENT_KEYWORDS = re.compile(
    r"\b(ceo|cio|cfo|coo|founder|partner|managing director|portfolio manager|"
    r"general counsel|chief|head of|vice president|principal|analyst|associate)\b",
    re.IGNORECASE,
)


def _looks_like_team_page(markdown: str) -> bool:
    """
    Return True if the scraped markdown contains person-like content
    (titles, role keywords) rather than just navigation or cookie banners.
    Requires at least 2 keyword hits to avoid false positives.
    """
    hits = _TEAM_CONTENT_KEYWORDS.findall(markdown)
    return len(hits) >= 2


def _normalise_base(url: str) -> str:
    """
    Strip path, query, and fragment from a URL to get the scheme + host.
    e.g. "https://www.acmecapital.com/about" → "https://www.acmecapital.com"
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _clean_optional(value) -> Optional[str]:
    """Return a stripped string or None for optional fields."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("null", "none", "n/a", "") else None


def _extract_http_status(exc: Exception) -> Optional[int]:
    """
    Try to extract an HTTP status code from a Firecrawl / requests exception.
    Returns the integer status code, or None if not determinable.
    """
    # requests HTTPError carries a .response attribute
    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)
    # Some SDKs embed the status in the exception message
    msg = str(exc)
    for code in (402, 403, 404, 429, 500):
        if str(code) in msg:
            return code
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AltBots team page scraper — extracts investment professionals from fund manager websites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python team_scraper.py --input firms.json
  python team_scraper.py --input firms.json --resume
  python team_scraper.py --firm "Acme Capital" --url "https://acmecapital.com"
        """,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", metavar="FILE",
        help="Path to a JSON file containing a list of {firm_name, website_url} dicts.",
    )
    mode.add_argument(
        "--firm", metavar="NAME",
        help="Firm name for a single-firm run (use with --url).",
    )
    p.add_argument(
        "--url", metavar="URL",
        help="Website URL for a single-firm run (use with --firm).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip firms whose website_url already appears in the JSONL output.",
    )
    p.add_argument(
        "--output", metavar="FILE", default=str(OUTPUT_JSONL),
        help=f"JSONL output path (default: {OUTPUT_JSONL})",
    )
    p.add_argument(
        "--log", metavar="FILE", default=str(RUN_LOG),
        help=f"Run log JSON path (default: {RUN_LOG})",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate single-firm args
    if args.firm and not args.url:
        parser.error("--firm requires --url")
    if args.url and not args.firm:
        parser.error("--url requires --firm")

    try:
        scraper = TeamScraper(
            output_path=Path(args.output),
            log_path=Path(args.log),
        )
    except RuntimeError as exc:
        logger.error("Initialisation failed: %s", exc)
        return 1

    # ── Single firm ───────────────────────────────────────────────────────────
    if args.firm:
        records = scraper.run_single(args.firm, args.url)
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return 0

    # ── Batch from file ───────────────────────────────────────────────────────
    input_path = Path(args.input)
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

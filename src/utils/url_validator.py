"""
src/utils/url_validator.py — AltBots URL Validator (Step 5)

Exports all firm names and website URLs from the `fund_managers` Qdrant
collection, validates each URL, and writes a clean data/firms.json that
serves as the master input for team_scraper.py.

Validation pipeline per URL
---------------------------
  1. Normalise — lowercase scheme+host, strip auth/fragment, fix common
     issues (missing scheme, uppercase, trailing slash)
  2. Domain sanity check — reject URLs pointing to social platforms,
     code forges, and media sites that can't be fund manager homepages
  3. HTTP probe — HEAD request (timeout 8s, follow redirects up to 5 hops)
     • 2xx / 3xx that stays on same domain → VALID
     • 4xx / 5xx / timeout → INVALID
     • Redirect to a different apex domain → INVALID (wrong URL)
  4. Deduplication — one canonical entry per legal_name (latest wins)

Output: data/firms.json
  [{"firm_name": "...", "website_url": "https://..."}, ...]

Usage
-----
  python -m src.utils.url_validator                      # run validation
  python -m src.utils.url_validator --dry-run            # print without writing
  python -m src.utils.url_validator --collection NAME    # override collection
  python -m src.utils.url_validator --add-13f            # also include edgar_13f managers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from dotenv import load_dotenv

# ── .env ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parents[2]  # project root
load_dotenv(_HERE / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("url_validator")

# ── Qdrant ────────────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

QDRANT_HOST            = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT            = int(os.getenv("QDRANT_PORT", "6333"))
FUND_MANAGERS_COL      = "fund_managers"
EDGAR_13F_COL          = "edgar_13f"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_PATH = Path("/data/firms.json")

# ── Validation config ─────────────────────────────────────────────────────────
REQUEST_TIMEOUT   = 8        # seconds per probe
MAX_REDIRECTS     = 5
MAX_WORKERS       = 12       # parallel HEAD requests
INTER_REQUEST_DELAY = 0.05   # throttle (seconds between submissions)

# Domains that are obviously NOT fund manager websites.
# A URL whose apex domain matches any of these is rejected immediately.
_INVALID_DOMAINS = frozenset({
    # Social / media
    "linkedin.com", "twitter.com", "x.com", "instagram.com",
    "facebook.com", "youtube.com", "soundcloud.com", "spotify.com",
    "tiktok.com", "reddit.com",
    # Code / tech
    "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    # General web
    "wikipedia.org", "bloomberg.com", "reuters.com", "ft.com",
    "wsj.com", "google.com", "bing.com", "yahoo.com",
    # File storage
    "dropbox.com", "drive.google.com", "docs.google.com",
    "sharepoint.com", "box.com",
})

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AltBots-URLValidator/1.0; "
        "+https://altbots.io)"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apex_domain(url: str) -> str:
    """Return the apex (eTLD+1) of a URL's netloc, e.g. 'www.citadel.com' → 'citadel.com'."""
    host = urlparse(url).netloc.lower().split(":")[0]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def normalise_url(raw: str) -> Optional[str]:
    """
    Return a normalised https:// URL or None if the raw string is unusable.

    Steps:
      • Strip whitespace
      • Add https:// if scheme missing
      • Lowercase scheme and host (path is case-sensitive on some servers)
      • Strip auth info, fragment, and trailing slash from path
    """
    if not raw or not raw.strip():
        return None

    url = raw.strip()

    # Add scheme if missing
    if not url.startswith(("http://", "https://", "HTTP://", "HTTPS://")):
        url = "https://" + url

    try:
        p = urlparse(url)
        # Normalise scheme and host to lowercase; preserve path case
        normalised = urlunparse((
            p.scheme.lower(),
            p.netloc.lower().rstrip("/"),
            p.path.rstrip("/") or "/",
            "",   # strip params
            "",   # strip query
            "",   # strip fragment
        ))
        return normalised
    except Exception:
        return None


def _is_domain_sane(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Rejects known non-fund-manager apex domains."""
    try:
        apex = _apex_domain(url)
    except Exception:
        return False, "unparseable URL"

    if not apex or "." not in apex:
        return False, f"no valid apex domain in {url!r}"

    if apex in _INVALID_DOMAINS:
        return False, f"domain {apex!r} is a non-fund-manager platform"

    return True, ""


def probe_url(firm_name: str, url: str) -> dict:
    """
    HTTP HEAD probe with redirect following.

    Returns a result dict:
      {
        "firm_name":   str,
        "website_url": str,          # final URL after redirects (normalised)
        "status":      "valid" | "invalid" | "error",
        "http_code":   int | None,
        "reason":      str,
        "redirected_to": str | None, # if redirect crossed domain boundary
      }
    """
    result = {
        "firm_name":    firm_name,
        "website_url":  url,
        "status":       "invalid",
        "http_code":    None,
        "reason":       "",
        "redirected_to": None,
    }

    # ── Domain sanity ─────────────────────────────────────────────────────────
    ok, reason = _is_domain_sane(url)
    if not ok:
        result["reason"] = reason
        return result

    origin_apex = _apex_domain(url)

    # ── HTTP probe ────────────────────────────────────────────────────────────
    try:
        _session = requests.Session()
        _session.max_redirects = MAX_REDIRECTS
        resp = _session.head(
            url,
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        final_url   = resp.url
        final_apex  = _apex_domain(final_url)
        http_code   = resp.status_code
        result["http_code"] = http_code

        # Cross-domain redirect — the URL is a placeholder / wrong entry
        if final_apex != origin_apex:
            result["status"]       = "invalid"
            result["reason"]       = f"redirected from {origin_apex} to {final_apex}"
            result["redirected_to"] = final_url
            return result

        # Check for non-fund-manager destination after redirect
        ok2, reason2 = _is_domain_sane(final_url)
        if not ok2:
            result["status"]       = "invalid"
            result["reason"]       = f"redirect destination: {reason2}"
            result["redirected_to"] = final_url
            return result

        # Accept 2xx and most 3xx; reject client/server errors
        if http_code < 400:
            result["status"]  = "valid"
            result["reason"]  = f"HTTP {http_code}"
            result["website_url"] = normalise_url(final_url) or url
        else:
            result["status"] = "invalid"
            result["reason"] = f"HTTP {http_code}"

    except requests.exceptions.Timeout:
        result["status"] = "error"
        result["reason"] = f"timeout after {REQUEST_TIMEOUT}s"
    except requests.exceptions.TooManyRedirects:
        result["status"] = "error"
        result["reason"] = f"exceeded {MAX_REDIRECTS} redirects"
    except requests.exceptions.SSLError as exc:
        # Some fund sites have expired certs; retry with http:// as fallback
        http_url = url.replace("https://", "http://", 1)
        try:
            resp2 = requests.head(
                http_url, headers=_HEADERS, timeout=REQUEST_TIMEOUT,
                allow_redirects=True, verify=False,
            )
            if resp2.status_code < 400:
                result["status"]      = "valid"
                result["reason"]      = f"HTTP {resp2.status_code} (SSL fallback)"
                result["website_url"] = normalise_url(http_url) or http_url
            else:
                result["status"] = "invalid"
                result["reason"] = f"SSL error + HTTP {resp2.status_code}"
        except Exception:
            result["status"] = "error"
            result["reason"] = f"SSL error: {exc}"
    except requests.exceptions.ConnectionError as exc:
        result["status"] = "error"
        result["reason"] = f"connection error: {type(exc).__name__}"
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"unexpected: {exc}"

    return result


# ── Qdrant extraction ─────────────────────────────────────────────────────────

def _get_qdrant() -> Optional["QdrantClient"]:
    if not QDRANT_AVAILABLE:
        logger.error("qdrant-client not installed")
        return None
    try:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    except Exception as exc:
        logger.error("Qdrant connection failed: %s", exc)
        return None


def export_from_fund_managers(client: "QdrantClient") -> list[dict]:
    """
    Scroll all points from the fund_managers collection.
    Returns list of {firm_name, website_url} dicts (raw, un-validated).
    Skips entries with no website.
    """
    entries: list[dict] = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=FUND_MANAGERS_COL,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in batch:
            p = pt.payload
            name    = (p.get("legal_name") or "").strip()
            website = (p.get("website") or "").strip()
            if name and website:
                norm = normalise_url(website)
                if norm:
                    entries.append({"firm_name": name, "website_url": norm})
                else:
                    logger.debug("Skipping %r — URL not normalisable: %r", name, website)
            else:
                logger.debug("Skipping entry with missing name or website: %s", p)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("fund_managers: exported %d entries with websites", len(entries))
    return entries


def export_from_edgar_13f(client: "QdrantClient") -> list[dict]:
    """
    Pull unique manager names from edgar_13f.
    Returns {firm_name, website_url: None} — no website in this collection.
    Used to surface managers who aren't yet in fund_managers.
    """
    managers: set[str] = set()
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=EDGAR_13F_COL,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in batch:
            m = (pt.payload.get("manager") or "").strip()
            if m:
                managers.add(m)
        if next_offset is None:
            break
        offset = next_offset

    logger.info("edgar_13f: found %d unique manager names (no website data)", len(managers))
    return [{"firm_name": m, "website_url": None} for m in sorted(managers)]


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(entries: list[dict]) -> list[dict]:
    """
    One entry per firm_name (case-insensitive). When duplicates exist,
    prefer the one with a non-None website_url.
    """
    seen: dict[str, dict] = {}
    for e in entries:
        key = e["firm_name"].upper().strip()
        existing = seen.get(key)
        if existing is None:
            seen[key] = e
        elif e["website_url"] and not existing["website_url"]:
            seen[key] = e  # upgrade a no-website entry
    return list(seen.values())


# ── Main validation run ───────────────────────────────────────────────────────

def run(
    collection: str = FUND_MANAGERS_COL,
    include_13f: bool = False,
    dry_run: bool = False,
    output_path: Path = OUTPUT_PATH,
    workers: int = MAX_WORKERS,
) -> list[dict]:
    """
    Full validation pipeline. Returns the list of verified firm dicts
    and (unless dry_run) writes them to output_path.
    """
    client = _get_qdrant()
    if client is None:
        sys.exit(1)

    # ── 1. Extract raw entries ────────────────────────────────────────────────
    raw: list[dict] = []

    if collection == FUND_MANAGERS_COL or collection == "both":
        raw += export_from_fund_managers(client)

    if include_13f or collection == "both":
        raw += export_from_edgar_13f(client)

    raw = _deduplicate(raw)
    logger.info("After dedup: %d unique firms", len(raw))

    # ── 2. Separate into probeable vs. no-website ─────────────────────────────
    to_probe   = [e for e in raw if e["website_url"]]
    no_website = [e for e in raw if not e["website_url"]]

    if no_website:
        logger.info(
            "%d firms have no website and will be excluded from output: %s",
            len(no_website),
            [e["firm_name"] for e in no_website],
        )

    # ── 3. Parallel URL probing ───────────────────────────────────────────────
    logger.info("Probing %d URLs (workers=%d, timeout=%ds)…", len(to_probe), workers, REQUEST_TIMEOUT)

    results: list[dict] = []
    stats = {"valid": 0, "invalid": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(probe_url, e["firm_name"], e["website_url"]): e
            for e in to_probe
        }
        for future in as_completed(future_map):
            try:
                r = future.result()
            except Exception as exc:
                e = future_map[future]
                r = {
                    "firm_name": e["firm_name"],
                    "website_url": e["website_url"],
                    "status": "error",
                    "reason": str(exc),
                }

            status = r.get("status", "error")
            stats[status] = stats.get(status, 0) + 1

            icon = "✓" if status == "valid" else "✗"
            logger.info(
                "  %s %-45s %s  %s",
                icon,
                r["firm_name"][:45],
                r.get("http_code") or "   ",
                r["reason"],
            )
            if status == "valid":
                results.append({
                    "firm_name":   r["firm_name"],
                    "website_url": r["website_url"],
                })

    # ── 4. Sort and report ────────────────────────────────────────────────────
    results.sort(key=lambda x: x["firm_name"])

    logger.info(
        "\nResults — valid: %d  invalid: %d  error: %d  (of %d probed)",
        stats["valid"], stats["invalid"], stats.get("error", 0), len(to_probe),
    )

    # ── 5. Write output ───────────────────────────────────────────────────────
    if dry_run:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote %d verified firms to %s", len(results), output_path)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export firm URLs from Qdrant, validate each one, "
            "and write data/firms.json for team_scraper.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.utils.url_validator
  python -m src.utils.url_validator --dry-run
  python -m src.utils.url_validator --add-13f
  python -m src.utils.url_validator --collection fund_managers --output /tmp/test.json
        """,
    )
    p.add_argument(
        "--collection", default=FUND_MANAGERS_COL,
        help=f"Qdrant collection to export from (default: {FUND_MANAGERS_COL})",
    )
    p.add_argument(
        "--add-13f", action="store_true",
        help="Also include manager names from edgar_13f (no website — shown in log only).",
    )
    p.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print validated output to stdout; do not write file.",
    )
    p.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Parallel probe threads (default: {MAX_WORKERS})",
    )
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    return p


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(
        collection=args.collection,
        include_13f=args.add_13f,
        dry_run=args.dry_run,
        output_path=Path(args.output),
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

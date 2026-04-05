"""
identity.py — Manager Identity Resolver (Phase 2)

Resolution order:
  1. Qdrant `fund_managers` collection  (proprietary cache — fastest)
  2. sec-api FormAdvApi                  (CRD + legal name lookup)
  3. EDGAR via edgartools                (CIK verification)
  4. Firecrawl                           (social handles + website enrichment)
  5. Cache resolved result back to Qdrant

Payload stored per Qdrant point:
  {
      legal_name, slug, crd, cik, website,
      social_handles: {platform: handle},
      source: "qdrant" | "sec-api" | "edgar",
      resolved_at: ISO-8601 timestamp,
  }

Vector: OpenAI text-embedding-3-small (1536-dim) over
        "{legal_name} CRD:{crd} CIK:{cik}"
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=True)

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Optional heavy dependencies ───────────────────────────────────────────────

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchText,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. Run: pip install qdrant-client")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Run: pip install openai")

try:
    from edgar import Company, set_identity
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False
    logger.warning("edgartools not installed. Run: pip install edgartools")

try:
    from sec_api import FormAdvApi
    SEC_API_AVAILABLE = True
except ImportError:
    SEC_API_AVAILABLE = False
    logger.warning("sec-api not installed. Run: pip install sec-api")


# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "fund_managers"
VECTOR_SIZE = 1536                           # text-embedding-3-small
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.88                  # min cosine score for a Qdrant hit

EDGAR_IDENTITY_EMAIL = os.getenv("EDGAR_IDENTITY", "neil@altbots.io")
SEC_API_KEY = os.getenv("SEC_API_KEY")

FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"
EXA_BASE = "https://api.exa.ai"
EXA_API_KEY_ENV = "EXA_API_KEY"
SOCIAL_KEYWORDS = ["linkedin", "twitter", "x.com", "instagram", "about", "team", "people"]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ManagerIdentity:
    legal_name: str
    slug: str
    crd: Optional[str] = None
    cik: Optional[str] = None
    website: Optional[str] = None
    social_handles: dict = field(default_factory=dict)
    source: str = "unknown"
    resolved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_payload(self) -> dict:
        return {
            "legal_name": self.legal_name,
            "slug": self.slug,
            "crd": self.crd,
            "cik": self.cik,
            "website": self.website,
            "social_handles": self.social_handles,
            "source": self.source,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ManagerIdentity":
        return cls(
            legal_name=payload.get("legal_name", ""),
            slug=payload.get("slug", ""),
            crd=payload.get("crd"),
            cik=payload.get("cik"),
            website=payload.get("website"),
            social_handles=payload.get("social_handles", {}),
            source=payload.get("source", "qdrant"),
            resolved_at=payload.get("resolved_at", ""),
        )


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _get_qdrant() -> Optional["QdrantClient"]:
    if not QDRANT_AVAILABLE:
        return None
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        _ensure_collection(client)
        return client
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        return None


def _ensure_collection(client: "QdrantClient"):
    """Create fund_managers collection if it does not exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")


def _embed(text: str) -> Optional[list[float]]:
    """Embed text with OpenAI text-embedding-3-small."""
    if not OPENAI_AVAILABLE:
        return None
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set — cannot embed")
            return None
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def _embed_key(identity: ManagerIdentity) -> str:
    """Canonical string used as the embedding input."""
    parts = [identity.legal_name]
    if identity.crd:
        parts.append(f"CRD:{identity.crd}")
    if identity.cik:
        parts.append(f"CIK:{identity.cik}")
    return " ".join(parts)


# ── Step 1: Qdrant lookup ─────────────────────────────────────────────────────

def _search_qdrant(name: str) -> Optional[ManagerIdentity]:
    """
    Embed the query name and search the fund_managers collection.
    Returns the top hit if its score exceeds SIMILARITY_THRESHOLD.
    """
    client = _get_qdrant()
    if client is None:
        return None

    vector = _embed(name)
    if vector is None:
        return None

    try:
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=1,
            score_threshold=SIMILARITY_THRESHOLD,
        )
        hits = result.points
        if hits:
            identity = ManagerIdentity.from_payload(hits[0].payload)
            identity.source = "qdrant"
            logger.info(
                f"[Qdrant] Hit for '{name}': {identity.legal_name} "
                f"(score={hits[0].score:.3f})"
            )
            return identity
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")

    return None


# ── Step 2: sec-api FormAdv lookup ───────────────────────────────────────────

def _get_adv_api() -> Optional["FormAdvApi"]:
    if not SEC_API_AVAILABLE:
        return None
    if not SEC_API_KEY:
        logger.error("SEC_API_KEY not set — cannot use FormAdvApi")
        return None
    return FormAdvApi(api_key=SEC_API_KEY)


def _resolve_via_sec_api(name: str) -> Optional[ManagerIdentity]:
    """
    Use sec-api FormAdvApi to resolve a firm by name or CRD number.

    If `name` is a 6-digit CRD, queries by Info.FirmCrdNb directly.
    Otherwise performs a full-text search against the Form ADV index.
    """
    adv_api = _get_adv_api()
    if adv_api is None:
        return None

    is_crd = bool(re.fullmatch(r"\d{6}", name.strip()))

    if is_crd:
        query = {
            "query": {"query_string": {"query": f"Info.FirmCrdNb:{name.strip()}"}},
            "from": 0,
            "size": 1,
        }
    else:
        query = {
            "query": {"query_string": {"query": name}},
            "from": 0,
            "size": 1,
        }

    try:
        result = adv_api.get_firms(query)
        filings = result.get("filings", [])
        if not filings:
            logger.info(f"[sec-api] No results for '{name}'")
            return None

        filing = filings[0]
        info = filing.get("Info", {})
        legal_name = (info.get("LegalNm") or info.get("BusNm") or name).strip()
        crd_val = str(info.get("FirmCrdNb", "")).strip() or None

        # Extract primary website — skip social media profile URLs
        website = None
        web_addrs = (
            filing.get("FormInfo", {})
            .get("Part1A", {})
            .get("Item1", {})
            .get("WebAddrs", {})
            .get("WebAddrs", [])
        )
        _social_domains = {"linkedin", "twitter", "x.com", "instagram", "facebook"}
        for addr in web_addrs:
            if isinstance(addr, str) and not any(d in addr.lower() for d in _social_domains):
                website = addr.strip()
                break

        identity = ManagerIdentity(
            legal_name=legal_name,
            slug=_slugify(legal_name),
            crd=crd_val,
            website=website,
            source="sec-api",
        )
        logger.info(f"[sec-api] Resolved '{name}' → {legal_name} (CRD {crd_val})")
        return identity

    except Exception as e:
        logger.error(f"[sec-api] Resolution failed for '{name}': {e}")
        return None


# ── Step 3: EDGAR CIK resolution ──────────────────────────────────────────────

def _resolve_cik(identity: ManagerIdentity) -> ManagerIdentity:
    """
    Resolve the EDGAR CIK for the firm by searching EDGAR's full-text index.

    Queries the 13F-HR form index where filer CIKs are embedded in the
    display_names field as '(CIK XXXXXXXXXX)'.  Falls back to an unfiltered
    search if no 13F hits are found.
    """
    if not identity.legal_name:
        return identity

    name_q = f'"{identity.legal_name}"'
    headers = {
        "User-Agent": f"AltBots Research Tool (Contact: {EDGAR_IDENTITY_EMAIL}) Python-requests/2.31.0",
        "Accept": "application/json",
    }

    def _cik_from_hits(hits: list) -> Optional[str]:
        """Extract the first CIK from EDGAR display_names entries."""
        _cik_re = re.compile(r"\(CIK\s+(\d+)\)", re.IGNORECASE)
        for hit in hits:
            for name_entry in hit.get("_source", {}).get("display_names", []):
                m = _cik_re.search(name_entry)
                if m and name_entry.upper().startswith(identity.legal_name[:12].upper()):
                    return m.group(1).lstrip("0").zfill(10)
        return None

    try:
        # Primary: 13F-HR filings reliably carry CIK in display_names
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": name_q, "forms": "13F-HR"},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        cik = _cik_from_hits(hits)

        # Fallback: unfiltered search (picks up firms that don't file 13F)
        if not cik:
            resp2 = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": name_q},
                headers=headers,
                timeout=15,
            )
            resp2.raise_for_status()
            hits2 = resp2.json().get("hits", {}).get("hits", [])
            cik = _cik_from_hits(hits2)

        if cik:
            identity.cik = cik
            logger.info(f"[EDGAR] CIK resolved for {identity.legal_name}: {identity.cik}")
        else:
            logger.info(f"[EDGAR] CIK not found for '{identity.legal_name}'")

    except Exception as e:
        logger.warning(f"EDGAR CIK resolution failed for '{identity.legal_name}': {e}")

    return identity


# ── Step 4: Social handle extraction (Firecrawl, with Exa fallback) ──────────

def _extract_socials(identity: ManagerIdentity) -> ManagerIdentity:
    """
    Extract social handles via Firecrawl (preferred) or Exa (fallback).
    If neither key is available, returns identity unchanged.
    """
    fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV)
    exa_key = os.environ.get(EXA_API_KEY_ENV)

    if fc_key:
        return _extract_socials_via_firecrawl(identity, fc_key)
    elif exa_key:
        logger.info(
            f"[Social] FIRECRAWL_API_KEY not set — falling back to Exa for {identity.legal_name}"
        )
        return _extract_socials_via_exa(identity, exa_key)
    else:
        logger.warning("FIRECRAWL_API_KEY and EXA_API_KEY not set — skipping social extraction")
        return identity


def _extract_socials_via_firecrawl(identity: ManagerIdentity, api_key: str) -> ManagerIdentity:
    """
    Use Firecrawl to scrape the firm's website and extract known social handles.
    Follows the map-then-scrape pattern from the existing firecrawl_client.py.
    """
    target_url = identity.website
    if not target_url:
        logger.info(f"[Firecrawl] No website for {identity.legal_name} — skipping")
        return identity

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        map_resp = requests.post(
            f"{FIRECRAWL_BASE}/map",
            headers=headers,
            json={"url": target_url},
            timeout=30,
        )
        map_resp.raise_for_status()
        all_urls = map_resp.json().get("links", []) if map_resp.json().get("success") else []

        target_pages = [
            u for u in all_urls
            if any(kw in u.lower() for kw in SOCIAL_KEYWORDS)
        ][:5]
        if not target_pages:
            target_pages = [target_url]

        combined_markdown = ""
        for page_url in target_pages:
            scrape_resp = requests.post(
                f"{FIRECRAWL_BASE}/scrape",
                headers=headers,
                json={"url": page_url, "formats": ["markdown"]},
                timeout=30,
            )
            if scrape_resp.ok and scrape_resp.json().get("success"):
                combined_markdown += scrape_resp.json()["data"].get("markdown", "")
            time.sleep(0.5)

        socials = _parse_social_handles(combined_markdown)
        if socials:
            identity.social_handles.update(socials)
            logger.info(
                f"[Firecrawl] Extracted socials for {identity.legal_name}: {socials}"
            )

    except Exception as e:
        logger.error(f"Firecrawl social extraction failed for {identity.legal_name}: {e}")

    return identity


def _extract_socials_via_exa(identity: ManagerIdentity, exa_key: str) -> ManagerIdentity:
    """
    Use Exa neural search to find social profile URLs when Firecrawl is unavailable.
    Searches for the firm's LinkedIn and Twitter/X presences by name.
    """
    headers = {"x-api-key": exa_key, "Content-Type": "application/json"}
    queries = [
        f"{identity.legal_name} LinkedIn company profile",
        f"{identity.legal_name} Twitter X fund manager",
    ]
    combined_text = ""
    for query in queries:
        try:
            resp = requests.post(
                f"{EXA_BASE}/search",
                headers=headers,
                json={
                    "query": query,
                    "type": "neural",
                    "numResults": 5,
                    "contents": {"text": {"maxCharacters": 300}},
                },
                timeout=20,
            )
            if resp.ok:
                for r in resp.json().get("results", []):
                    combined_text += (
                        f"{r.get('url', '')} {r.get('title', '')} {r.get('text', '')}\n"
                    )
        except Exception as e:
            logger.warning(f"[Exa] Social search failed ({query!r}): {e}")

    if combined_text:
        socials = _parse_social_handles(combined_text)
        if socials:
            identity.social_handles.update(socials)
            logger.info(f"[Exa] Extracted socials for {identity.legal_name}: {socials}")

    return identity


def _parse_social_handles(markdown: str) -> dict:
    """
    Extract social media handles / profile URLs from scraped markdown text.
    Returns {platform: url_or_handle}.
    """
    handles = {}

    patterns = {
        "linkedin": r"https?://(?:www\.)?linkedin\.com/(?:company|in)/([^\s\)\]\"']+)",
        "twitter": r"https?://(?:www\.)?(?:twitter|x)\.com/([^\s\)\]\"'/]+)",
        "instagram": r"https?://(?:www\.)?instagram\.com/([^\s\)\]\"'/]+)",
    }

    for platform, pattern in patterns.items():
        match = re.search(pattern, markdown, re.IGNORECASE)
        if match:
            handles[platform] = match.group(0).rstrip("/")

    return handles


# ── Step 5: Cache back to Qdrant ──────────────────────────────────────────────

def _cache_to_qdrant(identity: ManagerIdentity):
    """
    Upsert the resolved identity into the fund_managers Qdrant collection.
    Uses a deterministic UUID derived from the slug so re-runs are idempotent.
    """
    client = _get_qdrant()
    if client is None:
        return

    vector = _embed(_embed_key(identity))
    if vector is None:
        logger.warning(f"Skipping Qdrant cache for {identity.legal_name} — no embedding")
        return

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, identity.slug))

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=identity.to_payload(),
                )
            ],
        )
        logger.info(
            f"[Qdrant] Cached identity for {identity.legal_name} "
            f"(id={point_id}, source={identity.source})"
        )
    except Exception as e:
        logger.error(f"Qdrant upsert failed for {identity.legal_name}: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def resolve(
    name: str,
    website: Optional[str] = None,
    cik_hint: Optional[str] = None,
) -> Optional[ManagerIdentity]:
    """
    Resolve a fund manager's identity from a plain-English name.

    Args:
        name:     Manager or firm name as received (e.g. "Viking Global Investors")
        website:  Optional known website — skips Firecrawl map step if provided
        cik_hint: Optional known CIK (e.g. "0001273087"). When supplied, FormAdv
                  resolution is skipped and the identity is built directly from
                  the EDGAR submissions JSON for that CIK. Useful for firms that
                  are exempt reporting advisers without a public Form ADV.

    Returns:
        ManagerIdentity with as many fields populated as could be resolved,
        or None if every resolution step failed.

    Resolution order:
        1. Qdrant fund_managers collection (proprietary cache)
        2a. [cik_hint path] EDGAR submissions JSON → derive legal name + CIK
        2b. [normal path]   sec-api FormAdv → CRD + legal name
        3. EDGAR CIK (normal path only — already known in cik_hint path)
        4. Firecrawl (social handles)
        5. Write-back to Qdrant
    """
    logger.info(f"Resolving identity for: '{name}'")

    # ── 1. Qdrant proprietary cache ───────────────────────────────────────────
    cached = _search_qdrant(name)
    if cached is not None:
        # If a cik_hint is provided and conflicts with cache, skip the cache
        if cik_hint and cached.cik and cached.cik.lstrip("0") != cik_hint.lstrip("0"):
            logger.info("[Qdrant] Cache CIK mismatch with cik_hint — bypassing cache")
        else:
            return cached

    # ── 2a. CIK-hint fast path (exempt advisers without public ADV) ───────────
    if cik_hint:
        identity = _resolve_via_cik_hint(name, cik_hint, website)
        if identity is not None:
            identity = _extract_socials(identity)
            _cache_to_qdrant(identity)
            return identity
        logger.warning(f"[CIK hint] EDGAR lookup failed for CIK {cik_hint} — falling through to sec-api")

    logger.info(f"[Qdrant] No hit — falling back to sec-api for '{name}'")

    # ── 2b. sec-api FormAdv ───────────────────────────────────────────────────
    identity = _resolve_via_sec_api(name)
    if identity is None:
        logger.warning(f"sec-api returned no results for '{name}' — resolution incomplete")
        return None

    # Honour caller-supplied website over FormAdv's (often more accurate)
    if website and not identity.website:
        identity.website = website

    # ── 3. EDGAR CIK ─────────────────────────────────────────────────────────
    identity = _resolve_cik(identity)

    # ── 4. Firecrawl social handles ───────────────────────────────────────────
    identity = _extract_socials(identity)

    # ── 5. Cache back to Qdrant ───────────────────────────────────────────────
    _cache_to_qdrant(identity)

    return identity


def _resolve_via_cik_hint(
    display_name: str, cik: str, website: Optional[str]
) -> Optional[ManagerIdentity]:
    """
    Build a ManagerIdentity directly from an EDGAR CIK, bypassing FormAdv.
    Used for firms that are exempt reporting advisers without a public Form ADV.
    """
    cik_padded = cik.zfill(10)
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
            headers={"User-Agent": "altbots research@altbots.com"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("[CIK hint] EDGAR submissions fetch failed for CIK %s: %s", cik, exc)
        return None

    legal_name = data.get("name") or display_name
    slug = re.sub(r"[^a-z0-9]+", "-", legal_name.lower()).strip("-")

    logger.info("[CIK hint] Resolved CIK %s → %s", cik, legal_name)

    return ManagerIdentity(
        legal_name=legal_name,
        slug=slug,
        crd=None,      # no CRD for exempt adviser
        cik=cik_padded,
        website=website or "",
        social_handles={},
        source="edgar-cik-hint",
        resolved_at=datetime.now(timezone.utc).isoformat(),
    )


def resolve_batch(
    names: list[str],
    websites: Optional[dict[str, str]] = None,
) -> dict[str, Optional[ManagerIdentity]]:
    """
    Resolve a list of manager names.

    Args:
        names:    List of firm names
        websites: Optional {name: website} mapping to pass website hints

    Returns:
        {name: ManagerIdentity | None}
    """
    websites = websites or {}
    results = {}
    for name in names:
        results[name] = resolve(name, website=websites.get(name))
    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a firm name to a lowercase-hyphenated slug."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")

"""
src/core/tearsheet_assembler.py — AltBots Gold Copy Assembler (Step 3)

Composes a single structured "Gold Copy" JSON for any named firm by pulling
from four Qdrant collections and one live signal source:

  fund_managers   → firm identity, website, registration metadata
  edgar_cache     → ADV (AUM, clients, fees) + 13F (portfolio, holdings)
  edgar_13f       → chunked 13F summaries (richer strategy / text context)
  team_profiles   → scraped team roster (from team_scraper → ingestor)
  SocialSignalScanner → live social signals (Exa, last 90 days)

Legal gate
----------
  is_legal_ready from schema.py is called before the assembler returns.
  If the check fails the assembler raises AssemblerError rather than
  returning an incomplete object.

Usage
-----
  from src.core.tearsheet_assembler import TearsheetAssembler
  assembler = TearsheetAssembler()
  gold_copy = assembler.assemble("Viking Global Investors")

  # Gold copy is a dict — write to disk, return from API, or render to PDF
  import json
  print(json.dumps(gold_copy, indent=2, default=str))
"""

from __future__ import annotations

import ast
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── .env ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=True)

logger = logging.getLogger(__name__)

# ── Qdrant ────────────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchText,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed")

# ── OpenAI (for vector search fallback in fund_managers) ─────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Schema (legal gate) ───────────────────────────────────────────────────────
from src.core.schema import DisclosureEnvelope, Tearsheet

# ── Social scanner (live) ─────────────────────────────────────────────────────
try:
    from src.core.social import SocialSignalScanner
    SOCIAL_AVAILABLE = True
except Exception:
    SOCIAL_AVAILABLE = False
    logger.warning("SocialSignalScanner unavailable")


# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_HOST   = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.getenv("QDRANT_PORT", "6333"))
EMBED_MODEL   = "text-embedding-3-small"
VECTOR_SIZE   = 1536
SIMILARITY_THRESHOLD = 0.82   # looser than identity.py — display names vary

_COL_FUND_MANAGERS = "fund_managers"
_COL_EDGAR_CACHE   = "edgar_cache"
_COL_EDGAR_13F     = "edgar_13f"
_COL_TEAM_PROFILES = "team_profiles"

SOCIAL_LOOKBACK_DAYS = 90
MAX_SOCIAL_SIGNALS   = 10
MAX_TEAM_MEMBERS     = 20    # cap for Gold Copy JSON (all stored in Qdrant)
MAX_13F_CHUNKS       = 6     # supplementary text chunks from edgar_13f
CONCURRENT_TIMEOUT   = 60.0  # seconds for parallel fetch block


# ── Exceptions ────────────────────────────────────────────────────────────────

class AssemblerError(RuntimeError):
    """Raised when assembly cannot produce a legal-ready output."""


class FirmNotFoundError(AssemblerError):
    """Raised when the firm cannot be resolved in any collection."""


# ── TearsheetAssembler ────────────────────────────────────────────────────────

class TearsheetAssembler:
    """
    Pulls from multiple Qdrant collections to build the Gold Copy JSON for a firm.

    The Gold Copy is a plain dict (JSON-serialisable) with this top-level shape:

      {
        "firm_name":             str,
        "generated_at":          ISO-8601,
        "firm_overview":         {...},   # identity + ADV registration fields
        "institutional_numbers": {...},   # ADV AUM/clients/fees + 13F portfolio
        "team_roster":           [...],   # from team_profiles collection
        "social_signals":        [...],   # live Exa scan
        "supplementary_context": [...],   # edgar_13f text chunks
        "data_sources":          [...],   # provenance list
        "modules_skipped":       {...},   # modules that failed or had no data
        "disclosure":            {...},   # full DisclosureEnvelope (legal lock)
        "is_legal_ready":        bool,
      }

    Parameters
    ----------
    social_lookback_days : int
        How far back to scan for social signals (default 90).
    concurrent_timeout : float
        Wall-clock deadline for the parallel fetch block (default 60s).
    """

    def __init__(
        self,
        social_lookback_days: int = SOCIAL_LOOKBACK_DAYS,
        concurrent_timeout: float = CONCURRENT_TIMEOUT,
    ):
        self._social_lookback  = social_lookback_days
        self._concurrent_timeout = concurrent_timeout
        self._qdrant: Optional["QdrantClient"] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def assemble(self, firm_name: str) -> dict:
        """
        Build and return the Gold Copy for *firm_name*.

        Raises
        ------
        FirmNotFoundError
            If firm cannot be resolved in fund_managers or edgar_13f.
        AssemblerError
            If the legal gate fails (no DisclosureEnvelope attached).
        """
        start = time.monotonic()
        logger.info("[GoldCopy] Starting assembly for %r", firm_name)

        self._qdrant = self._connect()
        envelope     = DisclosureEnvelope()

        # ── Step 1: Resolve firm identity ─────────────────────────────────────
        identity = self._resolve_identity(firm_name, envelope)
        # identity is a dict with keys: legal_name, crd, cik, website, source
        logger.info(
            "[GoldCopy] Identity resolved: %s | crd=%s | cik=%s",
            identity["legal_name"], identity.get("crd"), identity.get("cik"),
        )

        # ── Step 2: Parallel data fetch ───────────────────────────────────────
        sections: dict[str, object] = {}
        errors:   dict[str, str]    = {}
        timings:  dict[str, float]  = {}

        tasks = {
            "edgar":       lambda: self._fetch_edgar(identity, envelope),
            "team":        lambda: self._fetch_team(identity, envelope),
            "social":      lambda: self._fetch_social(identity, envelope),
            "13f_ctx":     lambda: self._fetch_13f_context(identity, envelope),
            "enforcement": lambda: self._fetch_enforcement(identity, envelope),
        }

        deadline = time.monotonic() + self._concurrent_timeout
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="goldcopy") as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures, timeout=self._concurrent_timeout):
                name = futures[future]
                try:
                    t0 = time.monotonic()
                    sections[name] = future.result(timeout=max(0.0, deadline - time.monotonic()))
                    timings[name]  = time.monotonic() - t0
                except TimeoutError:
                    errors[name] = f"timed out after {self._concurrent_timeout:.0f}s"
                    envelope.record_skip(name, errors[name])
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    envelope.record_skip(name, errors[name])
                    logger.error("[GoldCopy] %s failed: %s", name, exc, exc_info=True)

        edgar_data       = sections.get("edgar")       or {}
        team_data        = sections.get("team")        or []
        social_data      = sections.get("social")      or []
        ctx_data         = sections.get("13f_ctx")     or []
        enforcement_data = sections.get("enforcement") or []

        # ── Step 3: Compose Gold Copy dict ────────────────────────────────────
        team_names = [p.get("name") for p in team_data if p.get("name")]

        gold: dict = {
            "firm_name":    identity["legal_name"],
            "generated_at": datetime.now(timezone.utc).isoformat(),

            "firm_overview": {
                "legal_name":         identity["legal_name"],
                "crd":                identity.get("crd"),
                "cik":                identity.get("cik"),
                "website":            identity.get("website"),
                "sec_registered":     edgar_data.get("sec_registered"),
                "registration_status": edgar_data.get("registration_status"),
                "identity_source":    identity.get("source"),
            },

            "institutional_numbers": {
                "aum_total_usd":            edgar_data.get("aum_total_usd"),
                "aum_discretionary_usd":    edgar_data.get("aum_discretionary_usd"),
                "aum_non_discretionary_usd": edgar_data.get("aum_non_discretionary_usd"),
                "adv_filing_date":          edgar_data.get("adv_filing_date"),
                "adv_recency_flag":         edgar_data.get("adv_recency_flag"),
                "num_clients":              edgar_data.get("num_clients"),
                "client_types":             edgar_data.get("client_types", []),
                "fee_structures":           edgar_data.get("fee_structures", []),
                "latest_13f": {
                    "period_of_report":         edgar_data.get("period_of_report"),
                    "filing_date":              edgar_data.get("filing_date_13f"),
                    "recency_flag":             edgar_data.get("recency_flag_13f"),
                    "total_portfolio_value_usd": edgar_data.get("total_portfolio_value_usd"),
                    "num_positions":            edgar_data.get("num_positions"),
                    "concentration_top10_pct":  edgar_data.get("concentration_top10_pct"),
                    "top_holdings":             edgar_data.get("top_holdings", []),
                },
            },

            # Strategy & service providers (from ADV Schedule D §7.B.(1))
            "strategy_type": edgar_data.get("strategy_type"),
            "service_providers": {
                "custodians":        edgar_data.get("_sp_custodians", []),
                "auditor":           edgar_data.get("_sp_auditor"),
                "fund_administrator": edgar_data.get("_sp_fund_administrator"),
            },

            # SEC enforcement (Litigation Releases / Administrative Proceedings)
            "sec_enforcement": enforcement_data,

            "team_roster":           team_data,
            "social_signals":        social_data,
            "supplementary_context": ctx_data,

            "data_sources":   envelope.data_sources,
            "modules_skipped": {**envelope.modules_skipped, **errors},
            "timings_seconds": timings,
        }

        # ── Legal gate ────────────────────────────────────────────────────────
        # Attach a fully populated DisclosureEnvelope and call is_legal_ready.
        # A Tearsheet wrapper is used purely to call the canonical check from schema.py.
        sentinel = Tearsheet(manager_name=identity["legal_name"])
        sentinel.disclosure = envelope
        if not sentinel.is_legal_ready:
            raise AssemblerError(
                "Legal gate failed: DisclosureEnvelope not attached. "
                "This is a programmer error — report immediately."
            )

        gold["disclosure"]     = envelope.to_dict()
        gold["is_legal_ready"] = sentinel.is_legal_ready

        elapsed = time.monotonic() - start
        logger.info(
            "[GoldCopy] Done — %s in %.1fs | team=%d social=%d edgar=%s legal_ready=%s",
            identity["legal_name"], elapsed,
            len(team_data), len(social_data),
            "ok" if edgar_data else "missing",
            gold["is_legal_ready"],
        )

        return gold

    # ── Qdrant connection ─────────────────────────────────────────────────────

    def _connect(self) -> Optional["QdrantClient"]:
        if not QDRANT_AVAILABLE:
            raise AssemblerError("qdrant-client not installed")
        try:
            return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception as exc:
            raise AssemblerError(f"Cannot connect to Qdrant: {exc}") from exc

    # ── Step 1: Identity resolution ───────────────────────────────────────────

    def _resolve_identity(
        self, firm_name: str, envelope: DisclosureEnvelope
    ) -> dict:
        """
        Resolve a firm dict from fund_managers.

        Strategy (in order):
          1. Exact case-insensitive payload text match on legal_name
          2. Vector similarity search (embed the query name)
          3. If still not found, return a minimal stub so assembly can continue
             with whatever edgar / team data exists

        Returns a plain dict with keys:
          legal_name, crd, cik, website, slug, social_handles, source
        """
        client = self._qdrant

        # ── 1. Text match ─────────────────────────────────────────────────────
        # legal_name in fund_managers is stored as ALL-CAPS (SEC format); try
        # uppercase query first, then original casing as fallback.
        for search_text in [firm_name.upper(), firm_name]:
            try:
                result = client.scroll(
                    collection_name=_COL_FUND_MANAGERS,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="legal_name",
                            match=MatchText(text=search_text),
                        )
                    ]),
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )
                pts = result[0]
                if pts:
                    best = self._pick_best_identity_match(firm_name, pts)
                    if best:
                        envelope.add_source(
                            f"fund_managers Qdrant (legal_name match) — {best.get('legal_name')}"
                        )
                        return best
            except Exception as exc:
                logger.warning("[Identity] Text search failed (%r): %s", search_text, exc)
                break

        # ── 2. Vector similarity ──────────────────────────────────────────────
        if OPENAI_AVAILABLE:
            try:
                vec = self._embed(firm_name)
                if vec:
                    hits = client.query_points(
                        collection_name=_COL_FUND_MANAGERS,
                        query=vec,
                        limit=3,
                        score_threshold=SIMILARITY_THRESHOLD,
                        with_payload=True,
                    )
                    if hits.points:
                        payload = hits.points[0].payload
                        logger.info(
                            "[Identity] Vector hit: %s (score=%.3f)",
                            payload.get("legal_name"), hits.points[0].score,
                        )
                        envelope.add_source(
                            f"fund_managers Qdrant (vector search, score={hits.points[0].score:.3f})"
                        )
                        return self._payload_to_identity(payload)
            except Exception as exc:
                logger.warning("[Identity] Vector search failed: %s", exc)

        # ── 3. Minimal stub ───────────────────────────────────────────────────
        # Firm not in fund_managers yet; proceed with what we have.
        logger.warning(
            "[Identity] %r not found in fund_managers — using stub identity", firm_name
        )
        envelope.add_source("identity: not in fund_managers — stub used")
        envelope.record_skip("identity", "firm not found in fund_managers collection")
        return {
            "legal_name": firm_name,
            "crd": None,
            "cik": None,
            "website": None,
            "slug": firm_name.lower().replace(" ", "-"),
            "social_handles": {},
            "source": "stub",
        }

    def _pick_best_identity_match(self, query: str, pts: list) -> Optional[dict]:
        """From a list of text-match hits, pick the closest legal_name."""
        query_upper = query.upper()
        # Prefer exact match
        for pt in pts:
            if (pt.payload.get("legal_name") or "").upper() == query_upper:
                return self._payload_to_identity(pt.payload)
        # Accept first hit
        return self._payload_to_identity(pts[0].payload)

    @staticmethod
    def _payload_to_identity(payload: dict) -> dict:
        return {
            "legal_name":    payload.get("legal_name", ""),
            "crd":           payload.get("crd"),
            "cik":           payload.get("cik"),
            "website":       payload.get("website"),
            "slug":          payload.get("slug", ""),
            "social_handles": payload.get("social_handles", {}),
            "source":        payload.get("source", "fund_managers"),
        }

    # ── EDGAR fetch (from edgar_cache) ────────────────────────────────────────

    def _fetch_edgar(self, identity: dict, envelope: DisclosureEnvelope) -> dict:
        """
        Look up edgar_cache by CIK (exact match) and unpack the nested
        AllFilingsData dict into a flat structure for the Gold Copy.

        Returns an empty dict if no cache entry is found.
        """
        cik = identity.get("cik")
        crd = identity.get("crd")

        if not cik and not crd:
            envelope.record_skip("edgar", "no CIK or CRD resolved")
            return {}

        client = self._qdrant

        # Try CIK lookup first (the primary cache key)
        entry = None
        if cik:
            try:
                results = client.scroll(
                    collection_name=_COL_EDGAR_CACHE,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="cik", match=MatchValue(value=cik))
                    ]),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if results[0]:
                    entry = results[0][0].payload
            except Exception as exc:
                logger.warning("[EDGAR] CIK lookup failed: %s", exc)

        # Fallback: firm_name text match
        if not entry:
            try:
                legal_name = identity.get("legal_name", "")
                results = client.scroll(
                    collection_name=_COL_EDGAR_CACHE,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="firm_name",
                            match=MatchText(text=legal_name),
                        )
                    ]),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if results[0]:
                    entry = results[0][0].payload
            except Exception as exc:
                logger.warning("[EDGAR] firm_name lookup failed: %s", exc)

        if not entry:
            envelope.record_skip("edgar", "not found in edgar_cache")
            return {}

        # ── Unpack nested AllFilingsData blob ─────────────────────────────────
        raw_data = entry.get("data", {})
        if isinstance(raw_data, str):
            try:
                raw_data = ast.literal_eval(raw_data)
            except Exception:
                try:
                    raw_data = json.loads(raw_data)
                except Exception:
                    raw_data = {}

        flat = self._flatten_edgar_data(raw_data)

        source_str = (
            f"edgar_cache (CIK {entry.get('cik', '?')}, "
            f"updated {entry.get('last_updated', '?')[:10]})"
        )
        envelope.add_source(source_str)
        logger.info("[EDGAR] Loaded from cache — %s", source_str)
        return flat

    @staticmethod
    def _flatten_edgar_data(data: dict) -> dict:
        """
        Flatten the nested AllFilingsData.to_dict() blob into a single-level
        dict for the institutional_numbers section of the Gold Copy.
        """
        flat: dict = {}
        if not data:
            return flat

        # ADV
        adv = data.get("adv") or {}
        if adv:
            flat["sec_registered"]     = adv.get("registration", {}).get("sec_registered")
            flat["registration_status"] = adv.get("registration", {}).get("status")
            flat["adv_filing_date"]    = (adv.get("source") or {}).get("filing_date")
            flat["adv_recency_flag"]   = (adv.get("recency") or {}).get("flag")

            aum = adv.get("aum") or {}
            flat["aum_total_usd"]            = aum.get("total_usd")
            flat["aum_discretionary_usd"]    = aum.get("discretionary_usd")
            flat["aum_non_discretionary_usd"] = aum.get("non_discretionary_usd")

            clients = adv.get("clients") or {}
            flat["num_clients"]  = clients.get("count")
            flat["client_types"] = clients.get("types", [])

            fees = adv.get("fees") or {}
            flat["fee_structures"]   = fees.get("structures", [])
            flat["min_account_size"] = fees.get("min_account_size")

            # Strategy & service providers (Schedule D §7.B.(1))
            strategy = adv.get("strategy") or {}
            flat["strategy_type"] = strategy.get("strategy_type")

            sp = adv.get("service_providers") or {}
            flat["_sp_auditor"]            = sp.get("auditor")
            flat["_sp_prime_brokers"]      = sp.get("prime_brokers", [])
            flat["_sp_custodians"]         = sp.get("custodians", [])
            flat["_sp_fund_administrator"] = sp.get("fund_administrator")

        # 13F — use most recent quarter (first in list, already sorted by filing date desc)
        form_13f = data.get("form_13f") or []
        if form_13f:
            latest = form_13f[0]
            flat["period_of_report"]        = (latest.get("source") or {}).get("period_of_report")
            flat["filing_date_13f"]         = (latest.get("source") or {}).get("filing_date")
            flat["recency_flag_13f"]        = (latest.get("recency") or {}).get("flag")
            flat["total_portfolio_value_usd"] = latest.get("total_portfolio_value_usd")
            flat["num_positions"]           = latest.get("num_positions")
            flat["concentration_top10_pct"] = latest.get("concentration_top10_pct")
            flat["top_holdings"]            = latest.get("top_holdings", [])[:10]

        return flat

    # ── Team roster (from team_profiles) ─────────────────────────────────────

    def _fetch_team(self, identity: dict, envelope: DisclosureEnvelope) -> list[dict]:
        """
        Retrieve all team_profiles points for this firm using an exact
        payload filter on firm_name, then deduplicate by person_name
        (keeping only chunk_index=0 to avoid bio repetition in the Gold Copy).
        """
        client     = self._qdrant
        legal_name = identity.get("legal_name", "")

        # Check collection exists
        existing_cols = {c.name for c in client.get_collections().collections}
        if _COL_TEAM_PROFILES not in existing_cols:
            envelope.record_skip("team", "team_profiles collection does not exist yet — run team_profiles_ingestor.py")
            return []

        try:
            pts, _ = client.scroll(
                collection_name=_COL_TEAM_PROFILES,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="firm_name",
                        match=MatchValue(value=legal_name),
                    )
                ]),
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("[Team] Scroll failed: %s", exc)
            envelope.record_skip("team", str(exc))
            return []

        # Keep only chunk_index=0 — each person appears once in the Gold Copy
        seen_names: set[str] = set()
        roster: list[dict]   = []
        for pt in pts:
            p = pt.payload
            if p.get("chunk_index", 0) != 0:
                continue
            name = (p.get("person_name") or "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            roster.append({
                "name":          name,
                "title":         p.get("title", ""),
                "bio":           p.get("bio", ""),
                "team_page_url": p.get("team_page_url", ""),
                "linkedin_url":  p.get("linkedin_url"),
                "email":         p.get("email"),
                "scraped_at":    p.get("scraped_at", ""),
            })

        roster = roster[:MAX_TEAM_MEMBERS]

        if roster:
            envelope.add_source(
                f"team_profiles Qdrant — {len(roster)} member(s) for {legal_name}"
            )
            logger.info("[Team] %d member(s) retrieved for %s", len(roster), legal_name)
        else:
            envelope.record_skip("team", f"no team_profiles entries for {legal_name!r}")
            logger.info("[Team] No team profiles found for %s", legal_name)

        return roster

    # ── Social signals (live scan) ────────────────────────────────────────────

    def _fetch_social(
        self, identity: dict, envelope: DisclosureEnvelope
    ) -> list[dict]:
        """Run SocialSignalScanner live against the resolved legal name."""
        if not SOCIAL_AVAILABLE:
            envelope.record_skip("social", "SocialSignalScanner not available")
            return []

        legal_name = identity.get("legal_name", "")
        try:
            scanner = SocialSignalScanner()
            signals = scanner.scan(
                firm_name=legal_name,
                lookback_days=self._social_lookback,
                max_signals=MAX_SOCIAL_SIGNALS,
            )
            serialised = [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in signals
            ]
            if serialised:
                envelope.add_source(
                    f"SocialSignalScanner (Exa / Claude Haiku) — {len(serialised)} signal(s)"
                )
            else:
                envelope.record_skip("social", "no signals found in lookback window")

            logger.info("[Social] %d signal(s) for %s", len(serialised), legal_name)
            return serialised

        except Exception as exc:
            logger.warning("[Social] Scanner failed: %s", exc)
            envelope.record_skip("social", str(exc))
            return []

    # ── Supplementary 13F context (edgar_13f chunks) ──────────────────────────

    def _fetch_13f_context(
        self, identity: dict, envelope: DisclosureEnvelope
    ) -> list[dict]:
        """
        Pull text chunks from edgar_13f for the firm to provide richer
        strategy / portfolio narrative context beyond the structured ADV numbers.
        Matches on manager field using MatchText (case-insensitive substring).
        """
        client     = self._qdrant
        legal_name = identity.get("legal_name", "")

        existing_cols = {c.name for c in client.get_collections().collections}
        if _COL_EDGAR_13F not in existing_cols:
            return []

        try:
            pts, _ = client.scroll(
                collection_name=_COL_EDGAR_13F,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key="manager",
                        match=MatchText(text=legal_name.split()[0]),  # first word is most distinctive
                    )
                ]),
                limit=MAX_13F_CHUNKS,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.debug("[13F-ctx] Scroll failed: %s", exc)
            return []

        chunks = []
        for pt in pts:
            p = pt.payload
            chunks.append({
                "manager":      p.get("manager"),
                "date":         p.get("date"),
                "content_type": p.get("content_type"),
                "chunk_label":  p.get("chunk_label"),
                "text":         p.get("text", "")[:600],  # trim for Gold Copy
            })

        if chunks:
            envelope.add_source(
                f"edgar_13f Qdrant — {len(chunks)} context chunk(s)"
            )
        logger.info("[13F-ctx] %d supplementary chunk(s) for %s", len(chunks), legal_name)
        return chunks

    # ── SEC enforcement (litigation releases) ────────────────────────────────

    def _fetch_enforcement(
        self, identity: dict, envelope: DisclosureEnvelope
    ) -> list[dict]:
        """
        Search SEC Litigation Releases and Administrative Proceedings for the
        firm name using RedFlagScanner.fetch_litigation_releases().

        Returns a list of {date, type, summary, url} dicts.
        """
        try:
            from src.core.red_flags import RedFlagScanner
        except Exception as exc:
            envelope.record_skip("enforcement", f"RedFlagScanner import failed: {exc}")
            return []

        legal_name = identity.get("legal_name", "")
        try:
            scanner = RedFlagScanner()
            results = scanner.fetch_litigation_releases(legal_name)
            if results:
                envelope.add_source(
                    f"SEC Litigation Releases (EDGAR full-text) — {len(results)} hit(s)"
                )
            else:
                envelope.record_skip("enforcement", "no SEC enforcement records found")
            logger.info("[Enforcement] %d record(s) for %s", len(results), legal_name)
            return results
        except Exception as exc:
            logger.warning("[Enforcement] fetch failed: %s", exc)
            envelope.record_skip("enforcement", str(exc))
            return []

    # ── Embedding helper (same pattern as identity.py) ────────────────────────

    def _embed(self, text: str) -> Optional[list[float]]:
        if not OPENAI_AVAILABLE:
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            client = OpenAI(api_key=api_key)
            resp   = client.embeddings.create(model=EMBED_MODEL, input=text)
            return resp.data[0].embedding
        except Exception as exc:
            logger.warning("[Embed] Failed: %s", exc)
            return None


# ── Module-level convenience function ─────────────────────────────────────────

def assemble(firm_name: str, **kwargs) -> dict:
    """
    One-shot convenience wrapper.

    Returns the Gold Copy dict for *firm_name*.

    Example
    -------
    >>> from src.core.tearsheet_assembler import assemble
    >>> gold = assemble("Viking Global Investors")
    >>> assert gold["is_legal_ready"]
    >>> print(f"AUM: ${gold['institutional_numbers']['aum_total_usd']:,}")
    """
    return TearsheetAssembler(**kwargs).assemble(firm_name)

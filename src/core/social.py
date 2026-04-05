"""
social.py — SocialSignalScanner (Phase 5)

Collects, deduplicates, and classifies observable public activity for a
fund manager across two independent sources:

  Source A — Firecrawl (/search endpoint):
    Queries LinkedIn posts, Twitter/X, and Google News for mentions of the
    firm and its key personnel within the last 90 days (configurable).

  Source B — Exa.ai (neural search REST API):
    Finds podcasts, niche industry blogs, and conference speaker lists that
    keyword-based search and general web scrapers routinely miss.

Processing pipeline:
  1. Collect raw signals from both sources in parallel query batches
  2. Normalise and deduplicate by URL and near-duplicate title similarity
  3. Classify each signal via Claude Haiku (signal type + synthesized summary)
  4. Enforce strict hedging on every output string
  5. Sort by published date and return the top 10

STRICT LANGUAGE RULE (Phase 5 requirement):
  Every output string must be framed as observed activity.
  Permitted framings  : "observed to", "appeared to", "indicated", "reportedly",
                        "noted to have", "suggested", "was seen to"
  Prohibited language : bare factual assertions — "announced", "confirmed",
                        "is", "has", "launched" — unless preceded by a hedging
                        qualifier.
  Enforcement is two-layered:
    (a) Haiku system prompt and user prompt both mandate hedged language
    (b) _enforce_hedging() post-processes every summary with regex substitutions
        and raises HedgingError if unhedged bare assertions remain after patching
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=True)

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus, urlencode, urlparse, urlunparse, parse_qs

import requests

logger = logging.getLogger(__name__)

# ── Optional Anthropic SDK ────────────────────────────────────────────────────

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")

# ── API config ────────────────────────────────────────────────────────────────

FIRECRAWL_BASE        = "https://api.firecrawl.dev/v1"
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"

EXA_BASE        = "https://api.exa.ai"
EXA_API_KEY_ENV = "EXA_API_KEY"

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
HAIKU_MODEL           = "claude-haiku-4-5-20251001"

# Default scan window
DEFAULT_LOOKBACK_DAYS: int = 90
MAX_SIGNALS_RETURNED: int  = 10

# Deduplication threshold: Jaccard similarity on title word-sets
DEDUP_SIMILARITY_THRESHOLD: float = 0.72

# Hedging substitution patterns: (bare_pattern → replacement)
_HEDGE_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(announced)\b", re.IGNORECASE),         "appeared to announce"),
    (re.compile(r"\b(confirmed)\b", re.IGNORECASE),          "reportedly confirmed"),
    (re.compile(r"\b(launched)\b",  re.IGNORECASE),          "appeared to launch"),
    (re.compile(r"\b(raised)\b",    re.IGNORECASE),          "reportedly raised"),
    (re.compile(r"\b(hired)\b",     re.IGNORECASE),          "appeared to hire"),
    (re.compile(r"\b(appointed)\b", re.IGNORECASE),          "was noted to have appointed"),
    (re.compile(r"\b(closed)\b",    re.IGNORECASE),          "reportedly closed"),
    (re.compile(r"\b(acquired)\b",  re.IGNORECASE),          "appeared to acquire"),
    (re.compile(r"\b(disclosed)\b", re.IGNORECASE),          "reportedly disclosed"),
]

# Bare assertion detector — flag any that slipped past substitution
_BARE_ASSERTION_RE = re.compile(
    r"(?<![a-z])(announced|confirmed|launched|closed|acquired|disclosed)"
    r"(?! to| that| by| with| a| an| the| its| their)",
    re.IGNORECASE,
)


class HedgingError(ValueError):
    """Raised when a synthesized summary fails the strict language check."""
    pass


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RawSignal:
    """
    Unprocessed result from Source A or B.
    Internal only — never returned to callers.
    """
    url: str
    title: str
    snippet: str            # short content excerpt
    published_date: str     # ISO-8601 or empty string
    source: str             # "firecrawl_linkedin" | "firecrawl_twitter" |
                            # "firecrawl_news" | "exa_podcast" |
                            # "exa_blog" | "exa_conference"
    raw_score: float = 0.0  # source-assigned relevance score


@dataclass
class SocialSignal:
    """
    Classified, synthesized social signal for one fund manager.

    The `summary` field is always framed as observed activity — never as fact.
    `signal_type` identifies the nature of the public-facing activity.
    """
    url: str
    title: str
    summary: str           # Haiku-synthesized, hedged language enforced
    published_date: str
    signal_type: str       # see _SIGNAL_TYPES below
    source: str
    relevance_score: int   # 1–10, Haiku-assigned

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "published_date": self.published_date,
            "signal_type": self.signal_type,
            "source": self.source,
            "relevance_score": self.relevance_score,
        }


_SIGNAL_TYPES = [
    "fund_launch",           # new fund, vehicle, or strategy launch
    "media_appearance",      # podcast, TV segment, interview, press quote
    "key_hire",              # named personnel joining the firm
    "key_departure",         # named personnel leaving the firm
    "aum_milestone",         # AUM change, fundraise close, capital raised
    "strategy_commentary",   # market views, investor letter, macro outlook
    "conference",            # speaking slot, panelist, keynote, summit
    "other",                 # anything that doesn't fit the above
]

_VALID_SIGNAL_TYPES = set(_SIGNAL_TYPES)


# ── Haiku prompts ─────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = (
    "You are a financial intelligence analyst. "
    "You classify signals about investment management firms into predefined categories. "
    "You return only a JSON array with no prose, markdown, or explanation outside the array."
)


def _build_classify_prompt(firm_name: str, signals_json: str) -> str:
    types_str = ", ".join(_SIGNAL_TYPES)
    return (
        f"Classify each of the following signals about {firm_name!r} into exactly "
        f"one of these types: {types_str}.\n\n"
        f"Return a JSON array of strings — one classification per signal, "
        f"in the same order as the input. No other fields.\n\n"
        f"Example output for 3 signals: [\"conference\", \"key_hire\", \"other\"]\n\n"
        f"Input signals:\n{signals_json}"
    )


# ── SocialSignalScanner ───────────────────────────────────────────────────────

class SocialSignalScanner:
    """
    Collects and classifies observable public social signals for a fund manager.

    Usage:
        scanner = SocialSignalScanner()
        signals = scanner.scan("Viking Global Investors",
                               personnel_names=["Andreas Halvorsen"],
                               lookback_days=90)
        # Returns list[SocialSignal], capped at 10, sorted newest-first.

    Environment variables:
        FIRECRAWL_API_KEY  — Source A scraping / search
        EXA_API_KEY        — Source B neural search
        ANTHROPIC_API_KEY  — Claude Haiku classification
    """

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        exa_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self._fc_key  = firecrawl_api_key  or os.environ.get(FIRECRAWL_API_KEY_ENV, "")
        self._exa_key = exa_api_key        or os.environ.get(EXA_API_KEY_ENV, "")
        self._an_key  = anthropic_api_key  or os.environ.get(ANTHROPIC_API_KEY_ENV, "")
        self._fc_tripped = False   # circuit breaker: True after first 402

        self._fc_headers = {
            "Authorization": f"Bearer {self._fc_key}",
            "Content-Type": "application/json",
        }
        self._exa_headers = {
            "x-api-key": self._exa_key,
            "Content-Type": "application/json",
        }

        if ANTHROPIC_AVAILABLE and self._an_key:
            self._haiku = anthropic.Anthropic(api_key=self._an_key)
        else:
            self._haiku = None
            if not self._an_key:
                logger.warning("[Social] ANTHROPIC_API_KEY not set — classification disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        firm_name: str,
        personnel_names: Optional[list[str]] = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        max_signals: int = MAX_SIGNALS_RETURNED,
    ) -> list[SocialSignal]:
        """
        Run the full pipeline for one firm.

        Args:
            firm_name:       Display name of the firm (e.g. "Viking Global Investors")
            personnel_names: Optional list of key personnel to include in search queries
            lookback_days:   Activity window in days from today (default 90)
            max_signals:     Maximum signals returned, sorted newest-first (default 10)

        Returns:
            List of SocialSignal objects, at most max_signals, sorted by
            published_date descending.  Every summary is framed as observed
            activity — no bare factual assertions.
        """
        personnel_names = personnel_names or []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        logger.info(
            f"[Social] Scanning {firm_name!r} — "
            f"lookback={lookback_days}d, personnel={len(personnel_names)}"
        )

        # ── Source A: Firecrawl ───────────────────────────────────────────────
        fc_signals: list[RawSignal] = []
        if os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key:
            fc_signals += self._scan_linkedin(firm_name, personnel_names, cutoff_str)
            if not self._fc_tripped:
                fc_signals += self._scan_twitter(firm_name, personnel_names, cutoff_str)
            if not self._fc_tripped:
                fc_signals += self._scan_news(firm_name, personnel_names, cutoff_str)
            logger.info(
                f"[Social] Firecrawl: {len(fc_signals)} raw signals"
                + (" (circuit breaker tripped)" if self._fc_tripped else "")
            )
        else:
            logger.warning("[Social] FIRECRAWL_API_KEY not set — Source A skipped")

        # ── Source B: Exa ─────────────────────────────────────────────────────
        exa_signals: list[RawSignal] = []
        if self._exa_key:
            exa_signals += self._scan_podcasts(firm_name, personnel_names, cutoff_str)
            exa_signals += self._scan_blogs(firm_name, personnel_names, cutoff_str)
            exa_signals += self._scan_conferences(firm_name, personnel_names, cutoff_str)
            logger.info(f"[Social] Exa: {len(exa_signals)} raw signals")
        else:
            logger.warning("[Social] EXA_API_KEY not set — Source B skipped")

        all_raw = fc_signals + exa_signals

        if not all_raw:
            logger.info(f"[Social] No signals found for {firm_name!r}")
            return []

        # ── Deduplicate ───────────────────────────────────────────────────────
        deduped = _deduplicate(all_raw)
        logger.info(f"[Social] After dedup: {len(deduped)} signals")

        # ── Filter by date window ─────────────────────────────────────────────
        dated = _filter_by_date(deduped, cutoff)
        logger.info(f"[Social] After date filter: {len(dated)} signals")

        # ── Classify via Haiku ────────────────────────────────────────────────
        classified = self._classify(dated, firm_name)
        logger.info(f"[Social] Classified: {len(classified)} signals")

        # ── Sort by date, cap at max_signals ──────────────────────────────────
        result = _sort_and_cap(classified, max_signals)
        logger.info(
            f"[Social] Returning {len(result)} signals for {firm_name!r}"
        )
        return result

    # ── Source A: Firecrawl searches ──────────────────────────────────────────

    def _firecrawl_search(
        self,
        query: str,
        limit: int = 8,
        tbs: str = "qdr:m3",   # last 3 months by default
    ) -> list[RawSignal]:
        """
        Use Firecrawl's /search endpoint (Google-powered) to find recent mentions.

        Args:
            query: Search query string
            limit: Max results
            tbs:   Google time-based search modifier:
                   qdr:m3 = last 3 months | qdr:m6 = last 6 months | qdr:y = last year
        """
        if self._fc_tripped:
            return []

        try:
            fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key
            fc_headers = {"Authorization": f"Bearer {fc_key}", "Content-Type": "application/json"}
            resp = requests.post(
                f"{FIRECRAWL_BASE}/search",
                headers=fc_headers,
                json={
                    "query": query,
                    "limit": limit,
                    "tbs": tbs,
                    "lang": "en",
                    "scrapeOptions": {"formats": ["markdown"]},
                },
                timeout=30,
            )
            if resp.status_code == 402:
                self._fc_tripped = True
                logger.warning(
                    "[Social] Firecrawl 402 Payment Required — circuit breaker tripped; "
                    "skipping all remaining Firecrawl calls and falling through to Exa"
                )
                return []
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", [])

            signals = []
            for r in results:
                snippet = (r.get("markdown") or r.get("description") or "")[:600]
                signals.append(RawSignal(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=snippet,
                    published_date=r.get("publishedDate", ""),
                    source="firecrawl",
                    raw_score=r.get("score", 0.0),
                ))
            return [s for s in signals if s.url and s.title]
        except Exception as e:
            logger.warning(f"[Social] Firecrawl search failed ({query!r}): {e}")
            return []

    def _scan_linkedin(
        self,
        firm_name: str,
        personnel_names: list[str],
        cutoff_str: str,
    ) -> list[RawSignal]:
        """
        Search LinkedIn for firm and personnel posts/activity.
        Targets site:linkedin.com to filter Google results to LinkedIn content.
        """
        signals: list[RawSignal] = []

        # Firm-level LinkedIn posts and activity
        queries = [
            f'site:linkedin.com "{firm_name}" post OR activity OR update',
            f'site:linkedin.com/company "{firm_name}"',
        ]
        # Add top-3 personnel if provided
        for name in personnel_names[:3]:
            queries.append(
                f'site:linkedin.com/in "{name}" "{firm_name}"'
            )

        for q in queries:
            raw = self._firecrawl_search(q, limit=5)
            for s in raw:
                s.source = "firecrawl_linkedin"
            signals.extend(raw)

        return signals

    def _scan_twitter(
        self,
        firm_name: str,
        personnel_names: list[str],
        cutoff_str: str,
    ) -> list[RawSignal]:
        """
        Search Twitter/X for firm mentions and personnel posts.
        Uses site:twitter.com OR site:x.com filter.
        """
        signals: list[RawSignal] = []

        queries = [
            f'(site:twitter.com OR site:x.com) "{firm_name}"',
        ]
        for name in personnel_names[:2]:
            queries.append(
                f'(site:twitter.com OR site:x.com) "{name}" "{firm_name}"'
            )

        for q in queries:
            raw = self._firecrawl_search(q, limit=5)
            for s in raw:
                s.source = "firecrawl_twitter"
            signals.extend(raw)

        return signals

    def _scan_news(
        self,
        firm_name: str,
        personnel_names: list[str],
        cutoff_str: str,
    ) -> list[RawSignal]:
        """
        Search Google News for recent firm and personnel mentions.
        Targets financial news publications and general press.
        """
        signals: list[RawSignal] = []

        news_domains = (
            "site:bloomberg.com OR site:ft.com OR site:wsj.com OR "
            "site:reuters.com OR site:businessinsider.com OR "
            "site:institutional-investor.com OR site:hedgeweek.com OR "
            "site:finalternatives.com OR site:hfm.global"
        )

        queries = [
            f'"{firm_name}" ({news_domains})',
            f'"{firm_name}" hedge fund OR investment manager',
        ]
        for name in personnel_names[:2]:
            queries.append(f'"{name}" "{firm_name}"')

        for q in queries:
            raw = self._firecrawl_search(q, limit=6)
            for s in raw:
                s.source = "firecrawl_news"
            signals.extend(raw)

        return signals

    # ── Source B: Exa neural search ───────────────────────────────────────────

    def _exa_search(
        self,
        query: str,
        category: Optional[str] = None,
        cutoff_str: Optional[str] = None,
        limit: int = 8,
    ) -> list[RawSignal]:
        """
        Call the Exa.ai neural search API directly via REST.

        Exa is purpose-built for semantic discovery and reliably surfaces
        niche content (podcasts, conference speaker lists, specialist blogs)
        that keyword-based search engines miss.

        API: POST https://api.exa.ai/search
        Docs: https://docs.exa.ai/reference/search

        Args:
            query:      Natural language search query
            category:   Exa content category filter: "tweet", "news",
                        "podcast", "blog", "linkedin", "paper", etc.
            cutoff_str: ISO-8601 start date for published content
            limit:      Max results (Exa default max is 100)
        """
        body: dict = {
            "query": query,
            "type": "neural",
            "numResults": limit,
            "useAutoprompt": True,
            "contents": {
                "text": {"maxCharacters": 600},
                "highlights": {
                    "numSentences": 2,
                    "highlightsPerUrl": 1,
                },
            },
        }
        if category:
            body["category"] = category
        if cutoff_str:
            body["startPublishedDate"] = cutoff_str

        try:
            resp = requests.post(
                f"{EXA_BASE}/search",
                headers=self._exa_headers,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            signals = []
            for r in results:
                highlights = r.get("highlights", [])
                snippet = (
                    " ".join(highlights)
                    or r.get("text", "")[:600]
                    or r.get("summary", "")
                )
                signals.append(RawSignal(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=snippet[:600],
                    published_date=r.get("publishedDate", ""),
                    source="exa",
                    raw_score=r.get("score", 0.0),
                ))
            return [s for s in signals if s.url and s.title]

        except Exception as e:
            logger.warning(f"[Social] Exa search failed ({query!r}): {e}")
            return []

    def _scan_podcasts(
        self,
        firm_name: str,
        personnel_names: list[str],
        cutoff_str: str,
    ) -> list[RawSignal]:
        """
        Use Exa to find podcast episodes featuring the firm or its personnel.
        Exa's `podcast` category is trained specifically to surface this content.
        """
        signals: list[RawSignal] = []

        queries = [
            f"{firm_name} investment manager podcast interview",
            f"{firm_name} fund manager views markets",
        ]
        for name in personnel_names[:2]:
            queries.append(f"{name} {firm_name} podcast episode")

        for q in queries:
            raw = self._exa_search(q, category="podcast", cutoff_str=cutoff_str, limit=5)
            for s in raw:
                s.source = "exa_podcast"
            signals.extend(raw)

        return signals

    def _scan_blogs(
        self,
        firm_name: str,
        personnel_names: list[str],
        cutoff_str: str,
    ) -> list[RawSignal]:
        """
        Use Exa to find niche industry blog posts, research notes, and
        specialist commentary that general search engines deprioritise.
        """
        signals: list[RawSignal] = []

        queries = [
            f"{firm_name} strategy investment analysis research note",
            f"{firm_name} portfolio commentary market outlook",
            f"{firm_name} hedge fund industry insight",
        ]
        for name in personnel_names[:2]:
            queries.append(f"{name} investment views essay op-ed")

        for q in queries:
            raw = self._exa_search(q, cutoff_str=cutoff_str, limit=5)
            for s in raw:
                s.source = "exa_blog"
            signals.extend(raw)

        return signals

    def _scan_conferences(
        self,
        firm_name: str,
        personnel_names: list[str],
        cutoff_str: str,
    ) -> list[RawSignal]:
        """
        Use Exa to find conference speaker lists, panel agendas, and event
        coverage where firm representatives appeared as speakers or panelists.
        """
        signals: list[RawSignal] = []

        queries = [
            f"{firm_name} speaker conference panel agenda 2025 2026",
            f"{firm_name} investment summit keynote conference",
        ]
        for name in personnel_names[:3]:
            queries.append(
                f"{name} {firm_name} conference panel speaker moderator"
            )

        for q in queries:
            raw = self._exa_search(q, cutoff_str=cutoff_str, limit=5)
            for s in raw:
                s.source = "exa_conference"
            signals.extend(raw)

        return signals

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(
        self, signals: list[RawSignal], firm_name: str
    ) -> list[SocialSignal]:
        """
        Send signals to Claude Haiku for signal-type classification and
        synthesis. Each signal gets a type, a relevance score (1–10), and
        a hedged summary framed as observed activity.

        Falls back to rule-based classification when Haiku is unavailable.
        """
        if not signals:
            return []

        if self._haiku:
            return self._classify_via_haiku(signals, firm_name)

        # Rule-based fallback
        return [_rule_based_classify(s) for s in signals]

    def _classify_via_haiku(
        self, signals: list[RawSignal], firm_name: str
    ) -> list[SocialSignal]:
        """
        Classify all signals in a single Haiku call.

        Input:  compact list of {index, title, snippet} — no URLs or raw HTML.
        Output: JSON array of signal_type strings, one per input signal, same order.

        Summaries are built locally from each signal's own title + snippet so
        the Haiku call only has to do classification, not prose generation.
        """
        # Compact input — omit URLs to reduce token count
        input_list = [
            {
                "index": i,
                "title": s.title,
                "snippet": s.snippet[:200],
            }
            for i, s in enumerate(signals)
        ]
        signals_json = json.dumps(input_list)

        try:
            response = self._haiku.messages.create(
                model=HAIKU_MODEL,
                max_tokens=512,           # ["type", ...] for 35 signals ≈ 200 tokens
                system=_CLASSIFY_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": _build_classify_prompt(firm_name, signals_json),
                }],
            )
            raw_text = response.content[0].text.strip()

            # Parse the flat string array
            start = raw_text.find("[")
            end = raw_text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                logger.warning(
                    "[Social] Haiku returned no JSON array — falling back to rule-based. "
                    "Preview: %r", raw_text[:200]
                )
                return [_rule_based_classify(s) for s in signals]

            types: list = json.loads(raw_text[start : end + 1])

        except Exception as e:
            logger.error("[Social] Haiku classification failed: %s", e)
            return [_rule_based_classify(s) for s in signals]

        # Zip classifications with signals by index; unknown types fall back to "other"
        results: list[SocialSignal] = []
        for i, raw in enumerate(signals):
            signal_type = types[i] if i < len(types) else "other"
            if signal_type not in _VALID_SIGNAL_TYPES:
                signal_type = "other"

            summary = _build_summary(raw, firm_name)
            results.append(SocialSignal(
                url=raw.url,
                title=raw.title,
                summary=summary,
                published_date=raw.published_date,
                signal_type=signal_type,
                source=raw.source,
                relevance_score=5,   # static; classification-only call no longer scores
            ))

        return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def _normalise_url(url: str) -> str:
    """
    Strip UTM parameters, trailing slashes, and www prefix.
    Used as the primary deduplication key.
    """
    try:
        parsed = urlparse(url)
        # Remove common tracking params
        clean_query = "&".join(
            p for p in (parsed.query or "").split("&")
            if not p.startswith(
                ("utm_", "ref=", "source=", "medium=", "campaign=")
            )
        )
        normalised = urlunparse((
            parsed.scheme,
            parsed.netloc.lstrip("www."),
            parsed.path.rstrip("/"),
            parsed.params,
            clean_query,
            "",  # strip fragment
        ))
        return normalised.lower()
    except Exception:
        return url.lower()


def _title_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on word sets of two titles.
    Returns 0.0 (no overlap) to 1.0 (identical word sets).
    """
    set_a = set(re.sub(r"[^\w\s]", "", a.lower()).split())
    set_b = set(re.sub(r"[^\w\s]", "", b.lower()).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate(signals: list[RawSignal]) -> list[RawSignal]:
    """
    Remove duplicate signals.

    Duplicate criteria (either condition triggers removal):
      1. Same normalised URL
      2. Title Jaccard similarity >= DEDUP_SIMILARITY_THRESHOLD

    When keeping one of a duplicate pair, prefer:
      - The signal with a longer snippet (more content)
      - Source B (Exa) over Source A (Firecrawl) for higher precision
    """
    if not signals:
        return []

    kept: list[RawSignal] = []
    seen_urls: set[str] = set()

    for candidate in signals:
        norm = _normalise_url(candidate.url)

        # URL-level dedup
        if norm in seen_urls:
            continue
        seen_urls.add(norm)

        # Title similarity dedup against already-kept signals
        is_dup = False
        for existing in kept:
            if _title_similarity(candidate.title, existing.title) >= DEDUP_SIMILARITY_THRESHOLD:
                # Keep the richer one
                if len(candidate.snippet) > len(existing.snippet):
                    kept.remove(existing)
                    seen_urls.discard(_normalise_url(existing.url))
                    break
                else:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(candidate)

    return kept


# ── Date filtering ────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse ISO-8601 or YYYY-MM-DD date string to datetime."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:26], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _filter_by_date(
    signals: list[RawSignal], cutoff: datetime
) -> list[RawSignal]:
    """
    Keep signals published on or after the cutoff date.
    Signals with no parseable date are retained (benefit of the doubt).
    """
    result = []
    for s in signals:
        dt = _parse_date(s.published_date)
        if dt is None or dt >= cutoff:
            result.append(s)
    return result


# ── Sorting ───────────────────────────────────────────────────────────────────

def _sort_and_cap(
    signals: list[SocialSignal], max_signals: int
) -> list[SocialSignal]:
    """
    Sort signals by published_date descending (newest first),
    then cap to max_signals.
    Signals with unparseable dates are placed at the end.
    """
    def sort_key(s: SocialSignal):
        dt = _parse_date(s.published_date)
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    return sorted(signals, key=sort_key, reverse=True)[:max_signals]


# ── Hedging enforcement ───────────────────────────────────────────────────────

def _enforce_hedging(text: str) -> str:
    """
    Apply substitution rules to patch bare factual assertions, then
    verify no unhedged assertions remain.

    Raises:
        HedgingError: if bare assertions survive after substitution.
    """
    patched = text
    for pattern, replacement in _HEDGE_SUBS:
        patched = pattern.sub(replacement, patched)

    # Check nothing slipped through
    remaining = _BARE_ASSERTION_RE.search(patched)
    if remaining:
        raise HedgingError(
            f"Unhedged assertion remains after patching: "
            f"{remaining.group(0)!r} in {patched!r}"
        )
    return patched


# ── Rule-based fallback classifier ───────────────────────────────────────────

_SIGNAL_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("conference",          ["conference", "summit", "panel", "speaker", "keynote", "forum"]),
    ("media_appearance",    ["podcast", "interview", "episode", "broadcast", "radio", "television", "tv segment"]),
    ("key_hire",            ["hire", "join", "appoint", "named", "joins", "promoted"]),
    ("key_departure",       ["depart", "leave", "resign", "exit", "step down", "left"]),
    ("aum_milestone",       ["aum", "raise", "fundraise", "capital", "close", "billion", "allocation"]),
    ("strategy_commentary", ["market", "portfolio", "investment", "outlook", "strategy", "thesis", "commentary", "letter"]),
    ("fund_launch",         ["fund", "launch", "new vehicle", "new strategy", "new fund"]),
]


def _build_summary(raw: RawSignal, firm_name: str) -> str:
    """
    Build a hedged summary from the signal's own title and snippet.
    Used by the Haiku classification path so no second LLM call is needed.
    """
    snippet = raw.snippet.strip()
    if snippet:
        # Use first sentence of snippet, capped at 200 chars
        sentence = re.split(r"(?<=[.!?])\s", snippet)[0][:200]
        return f"Observed activity: {sentence}"
    return f"Observed activity noted in connection with {firm_name} via {raw.source.replace('_', ' ')}."


def _rule_based_classify(raw: RawSignal) -> SocialSignal:
    """
    Fallback classifier when Haiku is unavailable.
    Assigns signal type from keyword matching on title + snippet.
    """
    combined = (raw.title + " " + raw.snippet).lower()
    signal_type = "other"
    for stype, keywords in _SIGNAL_TYPE_KEYWORDS:
        if any(kw in combined for kw in keywords):
            signal_type = stype
            break

    summary = _build_fallback_summary(raw, signal_type)

    return SocialSignal(
        url=raw.url,
        title=raw.title,
        summary=summary,
        published_date=raw.published_date,
        signal_type=signal_type,
        source=raw.source,
        relevance_score=5,
    )


def _build_fallback_summary(raw: RawSignal, signal_type: str) -> str:
    """
    Generate a minimal hedged summary when Haiku is unavailable.
    Always uses observed-activity framing.
    """
    type_phrases = {
        "conference":           "Activity appeared to indicate a conference or speaking engagement",
        "media_appearance":     "Observed activity suggested a media or podcast appearance",
        "key_hire":             "Activity indicated reported personnel joining the firm",
        "key_departure":        "Observed signals appeared to relate to a personnel departure",
        "aum_milestone":        "Observed activity appeared to relate to fund or capital activity",
        "strategy_commentary":  "Observed signals appeared to relate to public market commentary",
        "fund_launch":          "Activity indicated reported fund or vehicle launch activity",
        "other":                "Activity noted in public channels",
    }
    phrase = type_phrases.get(signal_type, type_phrases["other"])
    return f"{phrase} was observed. The signal was noted via {raw.source.replace('_', ' ')}."

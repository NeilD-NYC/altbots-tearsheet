"""
personnel.py — PersonnelEnricher (Phase 4)

Builds professional profiles for key fund manager personnel using a
two-tier enrichment pipeline:

  Tier 1 — Firecrawl:
    Map firm website → filter to team/people pages → scrape markdown →
    extract names, titles, and LinkedIn URLs.
    Delay: 5 seconds between individual page scrapes (respectful of firm sites).

  Tier 2 — Proxycurl (fallback):
    If Firecrawl returns a person without enough data (or no LinkedIn URL),
    call the Proxycurl LinkedIn API to fetch structured professional history.

  Bio synthesis — Claude Haiku:
    Every profile's bio is synthesized from scratch by Claude Haiku.
    Raw scraped text NEVER appears in the final output.

STRICT RULES (Phase 4 requirements):
  - Never copy text verbatim; always transform via Haiku.
  - Raw scraped content must never reach the final PersonProfile.
  - No reverse traceability: bios must read as synthesized summaries,
    not as paraphrases with detectable phrase patterns from the source.
  - A 7-word sliding-window verbatim check is run on every bio before
    it is accepted. Detection triggers a re-generation with a stricter prompt.
  - `RawPerson` objects are internal only and are never serialized.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. Run: pip install anthropic")


# ── Config ────────────────────────────────────────────────────────────────────

FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"

PROXYCURL_BASE = "https://nubela.co/proxycurl/api/v2/linkedin"
PROXYCURL_API_KEY_ENV = "PROXYCURL_API_KEY"

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# 5-second delay between firm website scrapes (Phase 4 requirement)
SCRAPE_DELAY_SECONDS: int = 5

# Proxycurl rate limit: default tier is ~300 credits/min; one call per profile
PROXYCURL_DELAY_SECONDS: float = 0.5

# Pages with these keywords are likely to contain team bios
TEAM_PAGE_KEYWORDS = [
    "/team", "/people", "/about", "/leadership", "/partners",
    "/staff", "/management", "/who-we-are", "/our-team",
    "/founders", "/executives",
]

# Verbatim detection window: flag if any N consecutive words appear in bio
VERBATIM_WINDOW = 7

# Max Haiku bio generation attempts per person
MAX_BIO_ATTEMPTS = 2


# ── Internal data structure (never serialised) ────────────────────────────────

@dataclass
class _RawPerson:
    """
    Internal-only container for scraped data before bio synthesis.
    Fields here MUST NOT appear in PersonProfile.
    """
    name: str
    title: str = ""
    linkedin_url: Optional[str] = None
    raw_snippet: str = ""        # raw text fragment — stays internal
    proxycurl_data: dict = field(default_factory=dict)
    source_tier: int = 1         # 1 = Firecrawl, 2 = Proxycurl


# ── Public data structure ─────────────────────────────────────────────────────

@dataclass
class PersonProfile:
    """
    Synthesized profile for one fund manager team member.
    All text is Haiku-generated — no verbatim scraped content.
    """
    name: str
    title: str
    firm: str
    linkedin_url: Optional[str]
    bio: str                     # synthesized by Claude Haiku
    source_tier: int             # 1 = Firecrawl only, 2 = Proxycurl used
    enriched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "firm": self.firm,
            "linkedin_url": self.linkedin_url,
            "bio": self.bio,
            "source_tier": self.source_tier,
            "enriched_at": self.enriched_at,
        }


# ── PersonnelEnricher ─────────────────────────────────────────────────────────

class PersonnelEnricher:
    """
    Builds synthesized professional profiles for key fund manager personnel.

    Usage:
        enricher = PersonnelEnricher()
        profiles = enricher.enrich("Viking Global Investors",
                                   "https://www.vikingglobal.com")

    Environment variables required:
        FIRECRAWL_API_KEY   — Firecrawl scraping API
        PROXYCURL_API_KEY   — Proxycurl LinkedIn enrichment API
        ANTHROPIC_API_KEY   — Anthropic API for Claude Haiku bio synthesis
    """

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        proxycurl_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self._fc_key = firecrawl_api_key or os.environ.get(FIRECRAWL_API_KEY_ENV, "")
        self._px_key = proxycurl_api_key or os.environ.get(PROXYCURL_API_KEY_ENV, "")
        self._an_key = anthropic_api_key or os.environ.get(ANTHROPIC_API_KEY_ENV, "")

        self._fc_headers = {
            "Authorization": f"Bearer {self._fc_key}",
            "Content-Type": "application/json",
        }
        self._px_headers = {
            "Authorization": f"Bearer {self._px_key}",
        }

        if ANTHROPIC_AVAILABLE and self._an_key:
            self._haiku = anthropic.Anthropic(api_key=self._an_key)
        else:
            self._haiku = None
            if not self._an_key:
                logger.warning("ANTHROPIC_API_KEY not set — bio synthesis disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(
        self,
        firm_name: str,
        website: str,
        max_people: int = 10,
    ) -> list[PersonProfile]:
        """
        Build synthesized profiles for up to `max_people` key personnel.

        Args:
            firm_name:   Display name of the fund manager firm
            website:     Firm's primary website URL
            max_people:  Cap on profiles returned (default 10)

        Returns:
            List of PersonProfile objects with synthesized bios.
            Returns [] if no team pages are discoverable.
        """
        logger.info(f"[Personnel] Starting enrichment for {firm_name} ({website})")

        # Tier 1: scrape firm website
        raw_people = self._scrape_team_pages(website)
        logger.info(f"[Personnel] Firecrawl found {len(raw_people)} candidates")

        # Tier 2: Proxycurl fallback for anyone with a LinkedIn URL but thin data
        for person in raw_people:
            if person.linkedin_url and not _has_enough_data(person):
                person = self._enrich_via_proxycurl(person)

        # Synthesize bios and build final profiles
        profiles: list[PersonProfile] = []
        for raw in raw_people[:max_people]:
            profile = self._build_profile(raw, firm_name)
            if profile:
                profiles.append(profile)

        logger.info(
            f"[Personnel] {len(profiles)} profiles built for {firm_name} "
            f"(tier-1: {sum(p.source_tier == 1 for p in profiles)}, "
            f"tier-2: {sum(p.source_tier == 2 for p in profiles)})"
        )
        return profiles

    # ── Tier 1: Firecrawl ─────────────────────────────────────────────────────

    def _scrape_team_pages(self, website: str) -> list[_RawPerson]:
        """
        Discover and scrape team/people pages on the firm website.

        Steps (mirrors firecrawl_client.py crawl_relevant_pages pattern):
          1. Map the site to get all URLs
          2. Filter to pages that look like team/people pages
          3. Scrape each page with a 5-second inter-scrape delay
          4. Extract person records from the combined markdown
        """
        if not self._fc_key:
            logger.warning("[Personnel] FIRECRAWL_API_KEY not set — skipping Tier 1")
            return []

        team_urls = self._discover_team_urls(website)
        if not team_urls:
            logger.info(f"[Personnel] No team pages found at {website}")
            return []

        logger.info(
            f"[Personnel] Found {len(team_urls)} team page(s): "
            + ", ".join(team_urls[:3])
        )

        all_people: list[_RawPerson] = []
        seen_names: set[str] = set()

        for i, url in enumerate(team_urls):
            if i > 0:
                # 5-second delay between scrapes — Phase 4 requirement
                logger.debug(f"[Personnel] Waiting {SCRAPE_DELAY_SECONDS}s before next scrape")
                time.sleep(SCRAPE_DELAY_SECONDS)

            markdown = self._scrape_page(url)
            if not markdown:
                continue

            people = _extract_people_from_markdown(markdown, url)
            for p in people:
                norm = p.name.lower().strip()
                if norm not in seen_names and norm:
                    seen_names.add(norm)
                    all_people.append(p)

        return all_people

    def _discover_team_urls(self, website: str) -> list[str]:
        """
        Use Firecrawl /map to enumerate site URLs, filter to team pages.
        Falls back to guessing common paths if map returns nothing.
        """
        mapped: list[str] = []
        try:
            fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key
            fc_headers = {"Authorization": f"Bearer {fc_key}", "Content-Type": "application/json"}
            resp = requests.post(
                f"{FIRECRAWL_BASE}/map",
                headers=fc_headers,
                json={"url": website},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                mapped = data.get("links", [])
        except Exception as e:
            logger.warning(f"[Personnel] Firecrawl map failed for {website}: {e}")

        # Filter to URLs that look like team pages
        team_urls = [
            u for u in mapped
            if any(kw in u.lower() for kw in TEAM_PAGE_KEYWORDS)
        ]

        # Always include the homepage itself as a fallback
        if not team_urls:
            team_urls = [website]

        # Deduplicate, cap at 5 pages to limit credits consumed
        seen: set[str] = set()
        result: list[str] = []
        for u in team_urls:
            if u not in seen:
                seen.add(u)
                result.append(u)
            if len(result) >= 5:
                break

        return result

    def _scrape_page(self, url: str) -> Optional[str]:
        """Scrape a single URL via Firecrawl /scrape. Returns markdown or None."""
        try:
            fc_key = os.environ.get(FIRECRAWL_API_KEY_ENV, "") or self._fc_key
            fc_headers = {"Authorization": f"Bearer {fc_key}", "Content-Type": "application/json"}
            resp = requests.post(
                f"{FIRECRAWL_BASE}/scrape",
                headers=fc_headers,
                json={"url": url, "formats": ["markdown"]},
                timeout=45,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                return data["data"].get("markdown", "")
        except Exception as e:
            logger.warning(f"[Personnel] Scrape failed for {url}: {e}")
        return None

    # ── Tier 2: Proxycurl ─────────────────────────────────────────────────────

    def _enrich_via_proxycurl(self, person: _RawPerson) -> _RawPerson:
        """
        Call the Proxycurl LinkedIn API for enrichment.
        Updates person.proxycurl_data and sets source_tier = 2.

        Endpoint: GET https://nubela.co/proxycurl/api/v2/linkedin
                  ?url={linkedin_url}&use_cache=if-present&fallback_to_cache=on-error
        """
        if not self._px_key:
            logger.warning("[Personnel] PROXYCURL_API_KEY not set — skipping Tier 2")
            return person

        if not person.linkedin_url:
            return person

        try:
            time.sleep(PROXYCURL_DELAY_SECONDS)
            resp = requests.get(
                PROXYCURL_BASE,
                headers=self._px_headers,
                params={
                    "url": person.linkedin_url,
                    "use_cache": "if-present",
                    "fallback_to_cache": "on-error",
                    "skills": "exclude",            # reduce payload size
                    "inferred_salary": "exclude",
                    "personal_email": "exclude",
                    "personal_contact_number": "exclude",
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                person.proxycurl_data = data
                person.source_tier = 2
                logger.info(
                    f"[Personnel] Proxycurl enriched: {person.name} "
                    f"({data.get('headline', 'no headline')})"
                )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.info(f"[Personnel] Proxycurl 404 for {person.linkedin_url}")
            else:
                logger.warning(
                    f"[Personnel] Proxycurl failed for {person.name}: {e}"
                )
        except Exception as e:
            logger.warning(f"[Personnel] Proxycurl error for {person.name}: {e}")

        return person

    # ── Bio synthesis ─────────────────────────────────────────────────────────

    def _build_profile(
        self, raw: _RawPerson, firm_name: str
    ) -> Optional[PersonProfile]:
        """
        Synthesize a PersonProfile from a _RawPerson.

        The bio is generated by Claude Haiku from a structured briefing.
        Raw scraped text is used as input to Haiku but is not included
        in the returned PersonProfile.
        """
        if not raw.name:
            return None

        # Resolve title: prefer Proxycurl headline over scraped title
        title = (
            raw.proxycurl_data.get("headline")
            or raw.proxycurl_data.get("occupation")
            or raw.title
            or "Investment Professional"
        )

        # Build the structured briefing for Haiku (never exposed in output)
        briefing = _build_briefing(raw, firm_name)

        # Generate synthesized bio
        bio = self._generate_bio(
            briefing=briefing,
            person_name=raw.name,
            firm_name=firm_name,
            source_text=briefing,
        )

        return PersonProfile(
            name=raw.name,
            title=title,
            firm=firm_name,
            linkedin_url=raw.linkedin_url,
            bio=bio,
            source_tier=raw.source_tier,
        )

    def _generate_bio(
        self,
        briefing: str,
        person_name: str,
        firm_name: str,
        source_text: str,
    ) -> str:
        """
        Generate a professional bio using Claude Haiku.

        Enforces strict no-verbatim rules:
        - Prompt instructs Haiku to synthesize, never quote
        - After generation, a 7-word sliding-window check detects
          any verbatim phrase carryover from the source
        - Detection triggers one re-attempt with a stronger prompt
        - If both attempts fail the verbatim check, a safe fallback
          bio is returned (name + title only)

        Raw `briefing` is used as Haiku input ONLY — it never appears
        in the returned string.
        """
        if not self._haiku:
            return self._fallback_bio(person_name, firm_name)

        for attempt in range(1, MAX_BIO_ATTEMPTS + 1):
            prompt = _build_bio_prompt(
                briefing=briefing,
                person_name=person_name,
                firm_name=firm_name,
                strict=(attempt > 1),  # escalate strictness on retry
            )
            try:
                response = self._haiku.messages.create(
                    model=HAIKU_MODEL,
                    max_tokens=300,
                    system=_BIO_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                bio = response.content[0].text.strip()

                if _has_verbatim(bio, source_text):
                    logger.warning(
                        f"[Personnel] Verbatim detected in bio for {person_name} "
                        f"(attempt {attempt}) — "
                        + ("using fallback" if attempt == MAX_BIO_ATTEMPTS else "retrying")
                    )
                    continue

                return bio

            except Exception as e:
                logger.error(
                    f"[Personnel] Haiku bio generation failed for {person_name}: {e}"
                )
                return self._fallback_bio(person_name, firm_name)

        # Both attempts failed verbatim check — return minimal safe bio
        return self._fallback_bio(person_name, firm_name)

    def _fallback_bio(self, person_name: str, firm_name: str) -> str:
        """Safe fallback bio used when Haiku is unavailable or verbatim check fails."""
        return (
            f"{person_name} is an investment professional at {firm_name}. "
            f"Full profile information is not publicly available."
        )


# ── Markdown extraction ───────────────────────────────────────────────────────

# Regex patterns for people extraction from firm website markdown
_LINKEDIN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/([\w\-%.]+)", re.IGNORECASE
)
_NAME_HEADING_RE = re.compile(
    r"^#{1,4}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\s*$", re.MULTILINE
)
_BOLD_NAME_RE = re.compile(
    r"\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\*\*"
)
_TITLE_AFTER_NAME_RE = re.compile(
    r"(?:^|\n)([A-Z][a-z]+(?: [A-Z][a-z]+)*)\n([A-Z][a-zA-Z &,/\-]+(?:Partner|Principal|Director|Officer|Analyst|Associate|Founder|Manager|President|Chairman|Head|VP|MD|CIO|CEO|CFO|COO|Managing)[A-Za-z &,/\-]*)",
    re.MULTILINE,
)


def _extract_people_from_markdown(markdown: str, source_url: str) -> list[_RawPerson]:
    """
    Extract candidate person records from a scraped team page markdown.

    Strategy:
      1. Find LinkedIn URLs — each likely corresponds to one person
      2. Look for name headings (## Name or **Name**) near the LinkedIn URL
      3. Try to extract a title from the lines immediately following the name
      4. Store raw_snippet as context window (internal only, never exported)
    """
    people: list[_RawPerson] = []
    seen_names: set[str] = set()

    # Strategy A: walk the document line by line looking for name → title → LinkedIn blocks
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match a name heading pattern
        name_match = (
            _NAME_HEADING_RE.match(line + "\n")
            or re.match(r"^\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\*\*$", line)
        )

        if name_match:
            name = name_match.group(1).strip()
            if name.lower() in seen_names:
                i += 1
                continue

            # Collect the next 15 lines as context for title + LinkedIn
            context_lines = lines[i : min(i + 15, len(lines))]
            context = "\n".join(context_lines)

            title = _extract_title(context_lines[1:6])
            linkedin_url = _find_linkedin_url(context)
            raw_snippet = context  # internal only

            person = _RawPerson(
                name=name,
                title=title,
                linkedin_url=linkedin_url,
                raw_snippet=raw_snippet,
                source_tier=1,
            )
            people.append(person)
            seen_names.add(name.lower())

        i += 1

    # Strategy B: find orphaned LinkedIn URLs that weren't caught by Strategy A
    for match in _LINKEDIN_RE.finditer(markdown):
        url = match.group(0)
        username = match.group(1)

        # Try to infer name from the username (heuristic)
        name_candidate = _linkedin_username_to_name(username)
        if not name_candidate or name_candidate.lower() in seen_names:
            continue

        # Grab surrounding context (100 chars before the URL)
        start = max(0, match.start() - 200)
        context = markdown[start : match.end() + 100]
        title = _extract_title(context.split("\n"))

        people.append(_RawPerson(
            name=name_candidate,
            title=title,
            linkedin_url=url,
            raw_snippet=context,
            source_tier=1,
        ))
        seen_names.add(name_candidate.lower())

    return people


def _extract_title(lines: list[str]) -> str:
    """
    Find a likely job title in the first few lines after a name.
    Titles are typically short and contain role-indicating words.
    """
    title_keywords = re.compile(
        r"partner|principal|director|officer|analyst|associate|founder|"
        r"manager|president|chairman|head|vice|managing|portfolio|general|"
        r"senior|chief|cio|ceo|cfo|coo|co-founder|co founder",
        re.IGNORECASE,
    )
    for line in lines[:5]:
        clean = re.sub(r"[*#\[\]]", "", line).strip()
        if clean and title_keywords.search(clean) and len(clean) < 80:
            return clean
    return ""


def _find_linkedin_url(text: str) -> Optional[str]:
    """Extract the first LinkedIn profile URL from a text block."""
    m = _LINKEDIN_RE.search(text)
    return m.group(0) if m else None


def _linkedin_username_to_name(username: str) -> Optional[str]:
    """
    Convert a LinkedIn username to a probable display name.
    e.g. "john-smith-12345" → "John Smith"
    Only returns a result if the pattern looks like a real name.
    """
    # Strip trailing numeric IDs (e.g. john-smith-123abc → john-smith)
    clean = re.sub(r"[-_]?[a-z0-9]{6,}$", "", username.lower())
    parts = re.split(r"[-_]", clean)
    parts = [p.capitalize() for p in parts if p.isalpha() and len(p) > 1]
    if 2 <= len(parts) <= 4:
        return " ".join(parts)
    return None


# ── Briefing construction (Haiku input, never in output) ─────────────────────

def _build_briefing(raw: _RawPerson, firm_name: str) -> str:
    """
    Assemble a structured briefing from all available data sources.
    This string goes into the Haiku prompt as input ONLY.
    It is NEVER included in any returned value.
    """
    parts = [f"Person: {raw.name}"]

    if raw.title:
        parts.append(f"Role: {raw.title}")

    parts.append(f"Firm: {firm_name}")

    px = raw.proxycurl_data
    if px:
        if px.get("summary"):
            parts.append(f"Professional summary: {px['summary'][:600]}")

        experiences = px.get("experiences", [])
        if experiences:
            exp_lines = []
            for exp in experiences[:5]:
                company = exp.get("company", "")
                title = exp.get("title", "")
                duration = ""
                if exp.get("starts_at"):
                    start_year = exp["starts_at"].get("year", "")
                    end_year = (exp.get("ends_at") or {}).get("year", "present")
                    duration = f" ({start_year}–{end_year})"
                if company and title:
                    exp_lines.append(f"  - {title} at {company}{duration}")
            if exp_lines:
                parts.append("Career history:\n" + "\n".join(exp_lines))

        education = px.get("education", [])
        if education:
            edu_lines = []
            for edu in education[:3]:
                school = edu.get("school", "")
                degree = edu.get("degree_name", "")
                field = edu.get("field_of_study", "")
                if school:
                    edu_lines.append(f"  - {degree} {field} at {school}".strip())
            if edu_lines:
                parts.append("Education:\n" + "\n".join(edu_lines))

    elif raw.raw_snippet:
        # Scrape-derived context — redact URLs before passing to Haiku
        snippet = re.sub(r"https?://\S+", "", raw.raw_snippet)
        snippet = re.sub(r"\s{2,}", " ", snippet).strip()[:500]
        parts.append(f"Background context: {snippet}")

    return "\n".join(parts)


# ── Haiku prompt construction ─────────────────────────────────────────────────

_BIO_SYSTEM_PROMPT = (
    "You are a professional investment research writer. "
    "You write concise, factual, third-person biographical summaries for "
    "fund manager tearsheets. Your writing is always original — you summarize "
    "and synthesize information in your own words. You NEVER quote the source "
    "directly, NEVER copy phrases verbatim, and NEVER use quotation marks around "
    "text from the source. You never mention LinkedIn, websites, or data sources "
    "in the bio itself. Write exactly 3 sentences, present tense, formal tone."
)


def _build_bio_prompt(
    briefing: str,
    person_name: str,
    firm_name: str,
    strict: bool = False,
) -> str:
    """Build the Haiku user prompt. `strict=True` adds extra no-copy instructions."""
    strictness = (
        "\n\nCRITICAL: Every word in your output must be your own. "
        "Do NOT reproduce any phrase, sequence, or sentence from the briefing. "
        "Completely rephrase every idea using different vocabulary and sentence structure."
        if strict else ""
    )
    return (
        f"Summarize the following bio for {person_name} at {firm_name} in your own words. "
        f"Do not quote the source directly. "
        f"Provide a professional 3-sentence summary.\n\n"
        f"Source material (for reference only — do not copy):\n"
        f"---\n{briefing}\n---\n\n"
        f"Requirements:\n"
        f"- Exactly 3 sentences, third person, present tense, formal tone.\n"
        f"- Focus on their role, seniority, and professional expertise.\n"
        f"- Do not reproduce any phrase from the source material above.\n"
        f"- Do not mention LinkedIn, websites, or data sources.{strictness}"
    )


# ── Verbatim detection ────────────────────────────────────────────────────────

def _has_verbatim(bio: str, source_text: str) -> bool:
    """
    Check whether the bio contains any verbatim phrase from the source.

    Uses a VERBATIM_WINDOW-word sliding window: if any N consecutive words
    from source_text appear in the same order in bio, returns True.

    This is intentionally strict — any 7-word match is flagged.
    """
    bio_words = re.sub(r"[^\w\s]", "", bio.lower()).split()
    source_words = re.sub(r"[^\w\s]", "", source_text.lower()).split()

    if len(bio_words) < VERBATIM_WINDOW or len(source_words) < VERBATIM_WINDOW:
        return False

    # Build a set of all N-grams from the bio for O(1) lookup
    bio_ngrams = {
        " ".join(bio_words[i : i + VERBATIM_WINDOW])
        for i in range(len(bio_words) - VERBATIM_WINDOW + 1)
    }

    for i in range(len(source_words) - VERBATIM_WINDOW + 1):
        ngram = " ".join(source_words[i : i + VERBATIM_WINDOW])
        if ngram in bio_ngrams:
            return True

    return False


# ── Helper ────────────────────────────────────────────────────────────────────

def _has_enough_data(raw: _RawPerson) -> bool:
    """
    Return True if a Tier-1 record has enough data to skip Proxycurl enrichment.
    Criteria: a non-empty title AND a raw snippet of at least 100 characters.
    """
    return bool(raw.title) and len(raw.raw_snippet) >= 100

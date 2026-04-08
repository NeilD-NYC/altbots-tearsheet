# src/core/red_flags.py
# Red flag detection via Form ADV Item 11 and SEC enforcement releases

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import urllib.request
import urllib.parse
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=True)

logger = logging.getLogger(__name__)

try:
    from sec_api import FormAdvApi
    _SEC_API_AVAILABLE = True
except ImportError:
    _SEC_API_AVAILABLE = False
    logger.warning("sec-api not installed — FormAdv Item 11 check disabled")

# ---------------------------------------------------------------------------
# Language constants — enforced project-wide
# ---------------------------------------------------------------------------

MANDATORY_DISCLAIMER = (
    "This screening is based on public records and is not exhaustive. "
    "It does not replace a comprehensive Operational Due Diligence (ODD) review."
)

_FORBIDDEN_WORDS = re.compile(
    r"\b(guilty|fraudulent|criminal)\b", re.IGNORECASE
)

# SEC public endpoints
_EDGAR_FULL_TEXT = (
    "https://efts.sec.gov/LATEST/search-index?q={query}&forms=ADV&hits.hits._source=period_of_report,file_date,entity_name,file_num"
)
_SEC_LITIGATION_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index?q={query}&forms=34-{year}"
)
_SEC_ENFORCE_RSS = "https://www.sec.gov/litigation/litreleases.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single regulatory disclosure finding."""
    summary: str          # Plain-language description (no forbidden words)
    source_label: str     # Human-readable label, e.g. "Form ADV Item 11"
    source_url: str       # Direct link to the SEC filing or record
    filing_date: Optional[str] = None   # ISO date string if available
    item: Optional[str] = None          # e.g. "Item 11.A"

    def as_of_phrase(self, as_of: date) -> str:
        return f"Public disclosure identified as of {as_of.strftime('%B %d, %Y')}."


@dataclass
class ScanResult:
    """Full output of a RedFlagScanner run."""
    firm_name: str
    crd_number: str
    scan_date: date
    has_disclosures: bool
    findings: list[Finding] = field(default_factory=list)
    disclaimer: str = MANDATORY_DISCLAIMER

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def status_phrase(self) -> str:
        d = self.scan_date.strftime("%B %d, %Y")
        if self.has_disclosures:
            return f"Public disclosure identified as of {d}."
        return f"No public regulatory disclosures identified as of {d}."

    def to_dict(self) -> dict:
        return {
            "firm_name": self.firm_name,
            "crd_number": self.crd_number,
            "scan_date": self.scan_date.isoformat(),
            "status": self.status_phrase,
            "has_disclosures": self.has_disclosures,
            "findings": [
                {
                    "summary": f.summary,
                    "as_of": f.as_of_phrase(self.scan_date),
                    "source_label": f.source_label,
                    "source_url": f.source_url,
                    "filing_date": f.filing_date,
                    "item": f.item,
                }
                for f in self.findings
            ],
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_get(url: str, timeout: int = 10) -> Optional[dict]:
    """Fetch a JSON URL, returning None on any error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TearsheetProject/1.0 compliance-screening"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("HTTP fetch failed for %s: %s", url, exc)
        return None


def _guard_language(text: str) -> str:
    """Raise if forbidden words appear; return text unchanged if clean."""
    match = _FORBIDDEN_WORDS.search(text)
    if match:
        raise ValueError(
            f"Forbidden word '{match.group()}' detected in output text. "
            "Use neutral regulatory language only."
        )
    return text


def _edgar_adv_url(crd: str) -> str:
    """Return an EDGAR search URL scoped to Form ADV for a given CRD/firm."""
    query = urllib.parse.quote(f'"{crd}"')
    return f"https://efts.sec.gov/LATEST/search-index?q={query}&forms=ADV"


def _sec_adv_url(crd: str) -> str:
    return f"https://adviserinfo.sec.gov/firm/summary/{crd}"


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

class RedFlagScanner:
    """
    Screens investment advisers for regulatory disclosures by:
      1. Querying Form ADV Item 11 (Disciplinary Information) via SEC IAPD.
      2. Searching SEC enforcement releases for firm mentions.

    Language policy
    ---------------
    - Uses 'Public disclosure identified as of [Date].' for any finding.
    - Uses 'No public regulatory disclosures identified as of [Date].' for clean records.
    - Never outputs the words 'guilty', 'fraudulent', or 'criminal'.
    - Appends MANDATORY_DISCLAIMER to every result.
    - Every finding carries a source_url pointing to the specific SEC record.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def scan(self, firm_name: str, crd_number: str) -> ScanResult:
        """
        Run a full red-flag screen for the named firm.

        Parameters
        ----------
        firm_name : str
            Display name of the investment adviser.
        crd_number : str
            SEC CRD (Central Registration Depository) number.

        Returns
        -------
        ScanResult
            Always contains a disclaimer; findings list is empty if clean.
        """
        scan_date = date.today()
        findings: list[Finding] = []

        adv_findings = self._check_form_adv_item11(firm_name, crd_number)
        findings.extend(adv_findings)

        enforcement_findings = self._search_enforcement_releases(firm_name, crd_number)
        findings.extend(enforcement_findings)

        # Language guard: every summary must pass
        for f in findings:
            _guard_language(f.summary)

        return ScanResult(
            firm_name=firm_name,
            crd_number=crd_number,
            scan_date=scan_date,
            has_disclosures=bool(findings),
            findings=findings,
            disclaimer=MANDATORY_DISCLAIMER,
        )

    # ------------------------------------------------------------------
    # Form ADV Item 11 — Disciplinary Information
    # ------------------------------------------------------------------

    def _check_form_adv_item11(
        self, firm_name: str, crd_number: str
    ) -> list[Finding]:
        """
        Fetch Form ADV via sec-api FormAdvApi and inspect Item 11
        (Disciplinary Information) sub-sections A–H.

        Sub-sections covered:
          A  — Regulatory/administrative proceedings
          B  — Self-regulatory organisation actions
          C  — Civil proceedings
          D  — Arbitration awards
          E  — Bankruptcy or financial matters
          F  — Bond denial, revocation, or suspension
          G  — Unsatisfied judgments or liens
          H  — Investment-related civil actions
        """
        findings: list[Finding] = []
        source_url = _sec_adv_url(crd_number)

        if not _SEC_API_AVAILABLE:
            logger.warning("sec-api unavailable — skipping Item 11 check for CRD %s", crd_number)
            return findings

        sec_api_key = os.getenv("SEC_API_KEY")
        if not sec_api_key:
            logger.warning("SEC_API_KEY not set — skipping Item 11 check")
            return findings

        try:
            adv_api = FormAdvApi(api_key=sec_api_key)
            result = adv_api.get_firms({
                "query": {"query_string": {"query": f"Info.FirmCrdNb:{crd_number}"}},
                "from": 0,
                "size": 1,
            })
            filings = result.get("filings", [])
            if not filings:
                logger.info("sec-api returned no filings for CRD %s", crd_number)
                return findings

            part1a = filings[0].get("FormInfo", {}).get("Part1A", {})

            # Master flag: Q11 == 'N' means no disclosures at all
            if part1a.get("Item11", {}).get("Q11", "N") == "N":
                logger.info("Form ADV Item 11 Q11=N (no disclosures) for CRD %s", crd_number)
                return findings

            # Item 11 sub-section labels
            section_labels = {
                "Item11A": "Item 11.A — Regulatory action or administrative proceeding",
                "Item11B": "Item 11.B — Self-regulatory organisation proceeding",
                "Item11C": "Item 11.C — Civil proceeding",
                "Item11D": "Item 11.D — Arbitration award",
                "Item11E": "Item 11.E — Bankruptcy or financial matter",
                "Item11F": "Item 11.F — Bond denial, revocation, or suspension",
                "Item11G": "Item 11.G — Unsatisfied judgment or lien",
                "Item11H": "Item 11.H — Investment-related civil action",
            }

            for item_key, label in section_labels.items():
                section = part1a.get(item_key, {})
                if not section:
                    continue
                # Any Q-flag set to "Y" in a sub-section signals a disclosure
                if any(v == "Y" for v in section.values()):
                    findings.append(Finding(
                        summary=(
                            f"Form ADV {label} indicates a disclosure on record. "
                            f"Review the current filing for full details."
                        ),
                        source_label=f"Form ADV {label}",
                        source_url=source_url,
                        item=item_key.replace("Item", "Item "),
                    ))

        except Exception as exc:
            logger.warning("sec-api Item 11 check failed for CRD %s: %s", crd_number, exc)

        return findings

    # ------------------------------------------------------------------
    # SEC enforcement release search (public API)
    # ------------------------------------------------------------------

    def fetch_litigation_releases(self, firm_name: str) -> list[dict]:
        """
        Search SEC Litigation Releases (34-LR), Administrative Proceedings
        (34-AP), and cease-and-desist orders (IC-AP) for the firm name.

        Returns a list of dicts with keys: date, type, summary, url.
        Returns an empty list if nothing is found or the search fails.
        """
        results: list[dict] = []
        enforcement_forms = "34-AP,34-LR,IC-AP"
        query = urllib.parse.quote(f'"{firm_name}"')
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q={query}&forms={enforcement_forms}&dateRange=custom"
            f"&startdt=2000-01-01"
        )
        data = _safe_get(url, timeout=self.timeout)
        if not data:
            return results

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:10]:
            src        = hit.get("_source", {})
            file_date  = str(src.get("file_date") or "")[:10]
            form_type  = str(src.get("form_type") or "SEC enforcement record")
            entity     = str(src.get("entity_name") or firm_name)
            accession  = hit.get("_id", "").replace(":", "/")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{accession}"
                if accession
                else f"https://efts.sec.gov/LATEST/search-index?q={query}&forms={enforcement_forms}"
            )
            summary = (
                f"SEC {form_type} references '{entity}'. "
                f"Review the original filing for full context."
            )
            results.append({
                "date":    file_date,
                "type":    form_type,
                "summary": _guard_language(summary),
                "url":     filing_url,
            })

        logger.info(
            "[RedFlags] fetch_litigation_releases(%r): %d hits", firm_name, len(results)
        )
        return results

    def _search_enforcement_releases(
        self, firm_name: str, crd_number: str
    ) -> list[Finding]:
        """
        Search SEC Litigation Releases and Administrative Proceedings for
        mentions of the firm name or CRD number.
        """
        findings: list[Finding] = []

        for query_term in [firm_name, crd_number]:
            findings.extend(
                self._query_sec_full_text_enforcement(
                    query_term, firm_name, crd_number
                )
            )

        # Deduplicate by source URL
        seen: set[str] = set()
        unique: list[Finding] = []
        for f in findings:
            if f.source_url not in seen:
                seen.add(f.source_url)
                unique.append(f)

        return unique

    def _query_sec_full_text_enforcement(
        self, query_term: str, firm_name: str, crd_number: str
    ) -> list[Finding]:
        """Query EDGAR full-text search for enforcement-related forms."""
        findings: list[Finding] = []

        # Litigation releases (LR), administrative proceedings (AP), and
        # cease-and-desist orders (IC) are filed under specific form types.
        enforcement_forms = "34-AP,34-LR,IC-AP"
        query = urllib.parse.quote(f'"{query_term}"')
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q={query}&forms={enforcement_forms}"
        )
        data = _safe_get(url, timeout=self.timeout)
        if not data:
            return findings

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            entity = src.get("entity_name", firm_name)
            file_date = src.get("file_date", "")
            form_type = src.get("form_type", "SEC enforcement record")
            accession = hit.get("_id", "").replace(":", "/")
            if accession:
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{accession}"
            else:
                filing_url = (
                    f"https://efts.sec.gov/LATEST/search-index"
                    f"?q={query}&forms={enforcement_forms}"
                )

            summary = (
                f"SEC enforcement record ({form_type}) references '{entity}'. "
                f"Review the original filing for full context."
            )
            findings.append(Finding(
                summary=_guard_language(summary),
                source_label=f"SEC Enforcement Release — {form_type}",
                source_url=filing_url,
                filing_date=file_date or None,
            ))

        return findings


# ---------------------------------------------------------------------------
# Language sanitiser (last-resort, not a censor — logs a warning)
# ---------------------------------------------------------------------------

_TERM_REPLACEMENTS = {
    "guilty": "subject to a regulatory finding",
    "not guilty": "resolved without a regulatory finding",
    "fraudulent": "involving alleged regulatory violations",
    "fraud": "alleged regulatory violation",
    "criminal": "subject to regulatory action",
}

_REPLACE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _TERM_REPLACEMENTS) + r")\b",
    re.IGNORECASE,
)


def _sanitise_term(text: str) -> str:
    """
    Replace forbidden words in raw data from external sources with neutral
    regulatory language. Logs a warning so reviewers are aware of substitutions.
    """
    def _replace(match: re.Match) -> str:
        original = match.group(0)
        replacement = _TERM_REPLACEMENTS[original.lower()]
        logger.warning(
            "Replaced forbidden term '%s' with '%s' in output text.",
            original, replacement,
        )
        return replacement

    return _REPLACE_PATTERN.sub(_replace, text)

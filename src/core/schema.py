# src/core/schema.py
# Pydantic models and data schemas

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Disclosure text — must appear on every tearsheet output
# ---------------------------------------------------------------------------

_NOT_INVESTMENT_ADVICE = (
    "This report is produced for informational and due diligence purposes only. "
    "It does not constitute investment advice, a recommendation to invest, or a "
    "solicitation of any offer to buy or sell any security or fund interest. "
    "Recipients should conduct their own independent analysis before making any "
    "investment decision."
)

_DATA_LIMITATION = (
    "Data is sourced from publicly available filings and third-party providers. "
    "AltBots makes no representation as to the accuracy, completeness, or "
    "timeliness of any information contained herein. Information may be subject "
    "to change without notice."
)

_REGULATORY_DISCLAIMER = (
    "This screening is based on public records and is not exhaustive. "
    "It does not replace a comprehensive Operational Due Diligence (ODD) review."
)

_CONFIDENTIALITY = (
    "This document is intended solely for the use of the recipient and may contain "
    "information that is confidential or privileged. Redistribution or reproduction "
    "is not permitted without prior written consent."
)

TEARSHEET_VERSION = "1.0"


# ---------------------------------------------------------------------------
# DisclosureEnvelope — the legal lock that must wrap every tearsheet
# ---------------------------------------------------------------------------

@dataclass
class DisclosureEnvelope:
    """
    Legal wrapper that must be attached to every tearsheet output.

    No Tearsheet object is permitted to leave the assembler without
    an instance of this class fully populated and attached.

    Fields
    ------
    generated_at       : ISO-8601 UTC timestamp of report generation
    version            : Tearsheet schema version string
    not_investment_advice : Core disclaimer — this is not investment advice
    regulatory_disclaimer : ODD caveat — screening is not exhaustive
    data_limitation    : Data accuracy and timeliness caveat
    confidentiality    : Redistribution restriction notice
    data_sources       : List of data providers consulted in this report
    modules_skipped    : Modules that were unavailable and their reason
    """

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: str = TEARSHEET_VERSION
    not_investment_advice: str = _NOT_INVESTMENT_ADVICE
    regulatory_disclaimer: str = _REGULATORY_DISCLAIMER
    data_limitation: str = _DATA_LIMITATION
    confidentiality: str = _CONFIDENTIALITY
    data_sources: list[str] = field(default_factory=list)
    modules_skipped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "version": self.version,
            "disclaimers": {
                "not_investment_advice": self.not_investment_advice,
                "regulatory_disclaimer": self.regulatory_disclaimer,
                "data_limitation": self.data_limitation,
                "confidentiality": self.confidentiality,
            },
            "data_sources": self.data_sources,
            "modules_skipped": self.modules_skipped,
        }

    def add_source(self, source: str) -> None:
        if source not in self.data_sources:
            self.data_sources.append(source)

    def record_skip(self, module: str, reason: str) -> None:
        self.modules_skipped[module] = reason


# ---------------------------------------------------------------------------
# Tearsheet — top-level output object
# ---------------------------------------------------------------------------

@dataclass
class Tearsheet:
    """
    Complete, legal-ready tearsheet for a single fund manager.

    This is the sole object returned by TearsheetAssembler.assemble().
    It is guaranteed to carry a DisclosureEnvelope — the assembler enforces
    this as an absolute final step before returning.

    Fields
    ------
    manager_name   : Canonical display name of the manager
    identity       : Resolved identity (CRD, CIK, website, socials)
    edgar          : SEC filing data (ADV, 13F, Form D)
    personnel      : Synthesized profiles for key team members
    social_signals : Classified public activity signals (last 90 days)
    red_flags      : Regulatory disclosure scan result
    disclosure     : Legal envelope — ALWAYS present, attached last
    errors         : {module_name: error_description} for failed modules
    """

    manager_name: str
    identity: Optional[object] = None          # ManagerIdentity
    edgar: Optional[object] = None             # AllFilingsData
    personnel: list = field(default_factory=list)       # list[PersonProfile]
    social_signals: list = field(default_factory=list)  # list[SocialSignal]
    red_flags: Optional[object] = None         # ScanResult
    disclosure: Optional[DisclosureEnvelope] = None     # set last, always
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        from src.core.identity import ManagerIdentity

        return {
            "manager_name": self.manager_name,
            "identity": (
                self.identity.to_payload()
                if self.identity and hasattr(self.identity, "to_payload")
                else None
            ),
            "edgar": (
                self.edgar.to_dict()
                if self.edgar and hasattr(self.edgar, "to_dict")
                else None
            ),
            "personnel": [
                p.to_dict() for p in self.personnel
                if hasattr(p, "to_dict")
            ],
            "social_signals": [
                s.to_dict() for s in self.social_signals
                if hasattr(s, "to_dict")
            ],
            "red_flags": (
                self.red_flags.to_dict()
                if self.red_flags and hasattr(self.red_flags, "to_dict")
                else None
            ),
            "disclosure": (
                self.disclosure.to_dict()
                if self.disclosure
                else None
            ),
            "errors": self.errors,
        }

    @property
    def is_legal_ready(self) -> bool:
        """True only when a DisclosureEnvelope is attached."""
        return self.disclosure is not None

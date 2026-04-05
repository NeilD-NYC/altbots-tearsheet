"""
edgar.py — EDGARFetcher: SEC Filing Retrieval for Tearsheets (Phase 3)

Covers:
  - Form ADV Part 1A  → AUM, client types, fee structures, registration status
  - Form 13F-HR       → Quarterly holdings, concentration, portfolio value
  - Form D            → Private placement raises, fund name, amount sold

Data source: sec-api.io (FormAdvApi, Form13FHoldingsApi, Form13FCoverPagesApi, FormDApi)
Authentication: SEC_API_KEY environment variable.

Source attribution:
  Every data field carries a FilingSource reference: form type, accession number,
  filing date, period of report, and direct SEC URL. Nothing is returned without
  knowing exactly which document it came from.

Recency flags:
  Green  < 90 days  — data is current
  Yellow  90-180    — approaching stale
  Red    >180 days  — stale; tearsheet should surface a warning
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=True)

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── sec-api ───────────────────────────────────────────────────────────────────

try:
    from sec_api import (
        FormAdvApi,
        Form13FHoldingsApi,
        Form13FCoverPagesApi,
        FormDApi,
    )
    SEC_API_AVAILABLE = True
except ImportError:
    SEC_API_AVAILABLE = False
    logger.warning("sec-api not installed. Run: pip install sec-api")

# ── Qdrant (cache) ────────────────────────────────────────────────────────────

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed — EDGAR cache disabled")


# ── Constants ─────────────────────────────────────────────────────────────────

EDGAR_IDENTITY_EMAIL: str = os.getenv("EDGAR_IDENTITY", "neil@altbots.io")
SEC_API_KEY_ENV   = "SEC_API_KEY"
QDRANT_URL_ENV    = "QDRANT_URL"
QDRANT_API_KEY_ENV = "QDRANT_API_KEY"

# Qdrant cache
EDGAR_CACHE_COLLECTION = "edgar_cache"
CACHE_TTL_DAYS: int = 90          # all three form types share the same TTL
_CACHE_VECTOR_DIM: int = 1        # dummy vector — lookups use payload filters

# Recency thresholds (days)
RECENCY_GREEN_DAYS: int = 90
RECENCY_YELLOW_DAYS: int = 180


# ── Recency logic ─────────────────────────────────────────────────────────────

def filing_recency_flag(filing_date_str: str) -> tuple[str, int]:
    """
    Evaluate how fresh a filing is relative to today.

    Args:
        filing_date_str: ISO-8601 date string (e.g. "2024-11-14")

    Returns:
        (flag, days_since) where flag is one of:
            "green"  — filed < 90 days ago
            "yellow" — filed 90-180 days ago
            "red"    — filed > 180 days ago (or unparseable)
    """
    try:
        filing_dt = date.fromisoformat(str(filing_date_str)[:10])
        days = (date.today() - filing_dt).days
    except (ValueError, TypeError):
        return "red", -1

    if days < RECENCY_GREEN_DAYS:
        return "green", days
    elif days < RECENCY_YELLOW_DAYS:
        return "yellow", days
    return "red", days


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FilingSource:
    """
    Attribution block carried on every extracted data field.
    Answers: "where exactly did this number come from?"
    """
    form_type: str
    accession_number: str
    filing_date: str
    period_of_report: str
    source_url: str
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "form_type": self.form_type,
            "accession_number": self.accession_number,
            "filing_date": self.filing_date,
            "period_of_report": self.period_of_report,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
        }


@dataclass
class RecencyStatus:
    """Filing freshness assessment."""
    flag: str           # "green" | "yellow" | "red"
    days_since_filing: int
    filing_date: str

    _DESCRIPTIONS = {
        "green": "Current (filed <90 days ago)",
        "yellow": "Aging (filed 90-180 days ago) — verify with issuer",
        "red": "Stale (filed >180 days ago) — data may be outdated",
    }

    def description(self) -> str:
        return self._DESCRIPTIONS.get(self.flag, "Unknown")

    def to_dict(self) -> dict:
        return {
            "flag": self.flag,
            "days_since_filing": self.days_since_filing,
            "filing_date": self.filing_date,
            "description": self.description(),
        }


@dataclass
class ADVData:
    """
    Structured data extracted from Form ADV Part 1A.

    Source attribution is carried at both the top level (source/recency)
    and on each logical group (aum_source, client_source, fee_source)
    so downstream renderers can cite the exact field origin.
    """
    source: FilingSource
    recency: RecencyStatus

    # ── AUM (Item 5.F) ────────────────────────────────────────────────────────
    aum_total_usd: Optional[int] = None
    aum_discretionary_usd: Optional[int] = None
    aum_non_discretionary_usd: Optional[int] = None
    aum_source: str = ""           # e.g. "Form ADV Part 1A Item 5.F, filed 2024-03-31"

    # ── Clients (Item 5.D) ────────────────────────────────────────────────────
    num_clients: Optional[int] = None
    client_types: list = field(default_factory=list)
    client_source: str = ""

    # ── Fees (Item 5.E) ───────────────────────────────────────────────────────
    fee_structures: list = field(default_factory=list)
    min_account_size: Optional[str] = None
    fee_source: str = ""

    # ── Registration ──────────────────────────────────────────────────────────
    sec_registered: bool = True
    registration_status: str = ""
    crd_number: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "recency": self.recency.to_dict(),
            "aum": {
                "total_usd": self.aum_total_usd,
                "discretionary_usd": self.aum_discretionary_usd,
                "non_discretionary_usd": self.aum_non_discretionary_usd,
                "source": self.aum_source,
            },
            "clients": {
                "count": self.num_clients,
                "types": self.client_types,
                "source": self.client_source,
            },
            "fees": {
                "structures": self.fee_structures,
                "min_account_size": self.min_account_size,
                "source": self.fee_source,
            },
            "registration": {
                "sec_registered": self.sec_registered,
                "status": self.registration_status,
                "crd_number": self.crd_number,
            },
        }


@dataclass
class Holding:
    """Single equity position from a 13F-HR filing."""
    name: str
    value_usd: int
    cusip: Optional[str] = None
    shares: Optional[int] = None
    pct_of_portfolio: Optional[float] = None
    source: Optional[FilingSource] = None   # filing this holding came from

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cusip": self.cusip,
            "value_usd": self.value_usd,
            "shares": self.shares,
            "pct_of_portfolio": self.pct_of_portfolio,
            "source": self.source.to_dict() if self.source else None,
        }


@dataclass
class ThirteenFData:
    """
    Quarterly 13F-HR holdings report.

    Values: all USD figures are in whole dollars (13F raw values are $000s;
    this class multiplies by 1000 before storing).
    """
    source: FilingSource
    recency: RecencyStatus

    total_portfolio_value_usd: int = 0
    num_positions: int = 0
    concentration_top10_pct: Optional[float] = None
    top_holdings: list = field(default_factory=list)   # list[Holding], top 25

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "recency": self.recency.to_dict(),
            "total_portfolio_value_usd": self.total_portfolio_value_usd,
            "num_positions": self.num_positions,
            "concentration_top10_pct": self.concentration_top10_pct,
            "top_holdings": [h.to_dict() for h in self.top_holdings],
        }


@dataclass
class FormDData:
    """
    Form D — Notice of Exempt Offering of Securities (private placement).
    Captures fund name, offering type, amount raised, and investor count.
    """
    source: FilingSource
    recency: RecencyStatus

    fund_name: str = ""
    offering_type: str = ""
    amount_raised_usd: Optional[int] = None
    total_offering_amount_usd: Optional[int] = None
    date_of_first_sale: Optional[str] = None
    investors_count: Optional[int] = None
    exemption_types: list = field(default_factory=list)  # e.g. ["Rule 506(b)"]

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "recency": self.recency.to_dict(),
            "fund_name": self.fund_name,
            "offering_type": self.offering_type,
            "amount_raised_usd": self.amount_raised_usd,
            "total_offering_amount_usd": self.total_offering_amount_usd,
            "date_of_first_sale": self.date_of_first_sale,
            "investors_count": self.investors_count,
            "exemption_types": self.exemption_types,
        }


@dataclass
class AllFilingsData:
    """Top-level container — all filing types for one manager."""
    firm_name: str
    cik: str
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    adv: Optional[ADVData] = None
    form_13f: list = field(default_factory=list)   # list[ThirteenFData]
    form_d: list = field(default_factory=list)     # list[FormDData]

    def to_dict(self) -> dict:
        return {
            "firm_name": self.firm_name,
            "cik": self.cik,
            "fetched_at": self.fetched_at,
            "adv": self.adv.to_dict() if self.adv else None,
            "form_13f": [f.to_dict() for f in self.form_13f],
            "form_d": [f.to_dict() for f in self.form_d],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_int(value) -> Optional[int]:
    """Safe coercion to int; returns None for falsy / non-numeric values."""
    if value is None:
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except (ValueError, TypeError):
        return None


def _browse_url(cik: str, form: str) -> str:
    return (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type={form}"
    )


def _dict_to_all_filings_data(d: dict) -> "AllFilingsData":
    """
    Reconstruct a fully-typed AllFilingsData tree from its serialized dict
    (the format produced by AllFilingsData.to_dict() and stored in Qdrant).
    """
    def _source(s: dict) -> FilingSource:
        return FilingSource(
            form_type=s.get("form_type", ""),
            accession_number=s.get("accession_number", ""),
            filing_date=s.get("filing_date", ""),
            period_of_report=s.get("period_of_report", ""),
            source_url=s.get("source_url", ""),
            fetched_at=s.get("fetched_at", ""),
        )

    def _recency(r: dict) -> RecencyStatus:
        return RecencyStatus(
            flag=r.get("flag", "red"),
            days_since_filing=r.get("days_since_filing", -1),
            filing_date=r.get("filing_date", ""),
        )

    def _adv(a: dict) -> ADVData:
        aum     = a.get("aum", {})
        clients = a.get("clients", {})
        fees    = a.get("fees", {})
        reg     = a.get("registration", {})
        obj = ADVData(source=_source(a.get("source", {})),
                      recency=_recency(a.get("recency", {})))
        obj.aum_total_usd             = aum.get("total_usd")
        obj.aum_discretionary_usd     = aum.get("discretionary_usd")
        obj.aum_non_discretionary_usd = aum.get("non_discretionary_usd")
        obj.aum_source                = aum.get("source", "")
        obj.num_clients               = clients.get("count")
        obj.client_types              = clients.get("types", [])
        obj.client_source             = clients.get("source", "")
        obj.fee_structures            = fees.get("structures", [])
        obj.min_account_size          = fees.get("min_account_size")
        obj.fee_source                = fees.get("source", "")
        obj.registration_status       = reg.get("status", "")
        obj.crd_number                = reg.get("crd_number")
        return obj

    def _holding(h: dict, filing_source: FilingSource) -> Holding:
        return Holding(
            name=h.get("name", ""),
            value_usd=h.get("value_usd", 0),
            cusip=h.get("cusip"),
            shares=h.get("shares"),
            pct_of_portfolio=h.get("pct_of_portfolio"),
            source=filing_source,
        )

    def _13f(t: dict) -> ThirteenFData:
        src = _source(t.get("source", {}))
        obj = ThirteenFData(source=src, recency=_recency(t.get("recency", {})))
        obj.total_portfolio_value_usd = t.get("total_portfolio_value_usd", 0)
        obj.num_positions             = t.get("num_positions", 0)
        obj.concentration_top10_pct   = t.get("concentration_top10_pct")
        obj.top_holdings              = [_holding(h, src) for h in t.get("top_holdings", [])]
        return obj

    def _form_d(fd: dict) -> FormDData:
        obj = FormDData(source=_source(fd.get("source", {})),
                        recency=_recency(fd.get("recency", {})))
        obj.fund_name                 = fd.get("fund_name", "")
        obj.offering_type             = fd.get("offering_type", "")
        obj.amount_raised_usd         = fd.get("amount_raised_usd")
        obj.total_offering_amount_usd = fd.get("total_offering_amount_usd")
        obj.date_of_first_sale        = fd.get("date_of_first_sale")
        obj.investors_count           = fd.get("investors_count")
        obj.exemption_types           = fd.get("exemption_types", [])
        return obj

    result = AllFilingsData(
        firm_name=d.get("firm_name", ""),
        cik=d.get("cik", ""),
        fetched_at=d.get("fetched_at", ""),
    )
    if d.get("adv"):
        result.adv = _adv(d["adv"])
    result.form_13f = [_13f(t) for t in d.get("form_13f", [])]
    result.form_d   = [_form_d(fd) for fd in d.get("form_d", [])]
    return result


# ── EDGARFetcher ──────────────────────────────────────────────────────────────

class EDGARFetcher:
    """
    Fetches SEC filings for fund manager tearsheets via sec-api.io.

    Requires SEC_API_KEY in the environment (or .env file).
    The sec-api library handles authentication, rate limiting, and
    403/retry logic natively — no manual HTTP management needed.

    Public interface (unchanged from prior implementation):
        fetch_adv(cik, firm_name)           → Optional[ADVData]
        fetch_13f(cik, firm_name, ...)      → list[ThirteenFData]
        fetch_form_d(cik, firm_name, ...)   → list[FormDData]
        fetch_all(cik, firm_name, ...)      → AllFilingsData
    """

    def __init__(
        self,
        identity_email: str = EDGAR_IDENTITY_EMAIL,
        request_delay: float = 0.0,   # kept for signature compatibility; sec-api handles rate limiting
    ):
        self.identity_email = identity_email

        api_key = os.getenv(SEC_API_KEY_ENV, "")
        if not api_key:
            logger.warning(
                "SEC_API_KEY not set — EDGAR data will be unavailable. "
                "Add SEC_API_KEY to your .env file."
            )

        if SEC_API_AVAILABLE and api_key:
            self._adv_api      = FormAdvApi(api_key=api_key)
            self._holdings_api = Form13FHoldingsApi(api_key=api_key)
            self._cover_api    = Form13FCoverPagesApi(api_key=api_key)
            self._form_d_api   = FormDApi(api_key=api_key)
            logger.debug("[EDGAR] sec-api clients initialised")
        else:
            self._adv_api = self._holdings_api = self._cover_api = self._form_d_api = None

        # Qdrant cache
        self._qdrant = self._init_qdrant(
            url=os.getenv(QDRANT_URL_ENV, ""),
            api_key=os.getenv(QDRANT_API_KEY_ENV, "") or None,
        )

    # ── Qdrant cache helpers ──────────────────────────────────────────────────

    def _init_qdrant(self, url: str, api_key: Optional[str]) -> Optional["QdrantClient"]:
        if not QDRANT_AVAILABLE:
            return None
        if not url:
            logger.warning("[Cache] QDRANT_URL not set — EDGAR cache disabled")
            return None
        try:
            client = QdrantClient(url=url, api_key=api_key)
            self._ensure_cache_collection(client)
            return client
        except Exception as e:
            logger.warning(f"[Cache] Qdrant connection failed: {e}")
            return None

    def _ensure_cache_collection(self, client: "QdrantClient") -> None:
        existing = {c.name for c in client.get_collections().collections}
        if EDGAR_CACHE_COLLECTION not in existing:
            client.create_collection(
                collection_name=EDGAR_CACHE_COLLECTION,
                vectors_config=VectorParams(
                    size=_CACHE_VECTOR_DIM, distance=Distance.COSINE
                ),
            )
            logger.info(f"[Cache] Created Qdrant collection: {EDGAR_CACHE_COLLECTION}")

    def _cache_get(self, cik: str) -> Optional["AllFilingsData"]:
        """
        Look up a cached AllFilingsData for `cik`.

        Uses query_points() with a payload filter — the dummy [1.0] vector
        ensures every point scores equally so the filter is the sole selector.

        Returns the cached object if it exists and is within CACHE_TTL_DAYS,
        otherwise returns None (caller should fetch fresh data).
        """
        if not self._qdrant:
            return None
        try:
            result = self._qdrant.query_points(
                collection_name=EDGAR_CACHE_COLLECTION,
                query=[1.0],
                query_filter=Filter(
                    must=[FieldCondition(key="cik", match=MatchValue(value=cik))]
                ),
                limit=1,
                with_payload=True,
            )
            if not result.points:
                logger.debug(f"[Cache] MISS for CIK {cik}")
                return None

            payload = result.points[0].payload or {}
            last_updated = payload.get("last_updated", "")
            if not last_updated:
                return None

            updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - updated_dt).days

            if age_days > CACHE_TTL_DAYS:
                logger.info(f"[Cache] STALE ({age_days}d > {CACHE_TTL_DAYS}d) for CIK {cik}")
                return None

            data_dict = payload.get("data")
            if not data_dict:
                return None

            logger.info(f"[Cache] HIT for CIK {cik} (age={age_days}d)")
            return _dict_to_all_filings_data(data_dict)

        except Exception as e:
            logger.warning(f"[Cache] Qdrant read failed for CIK {cik}: {e}")
            return None

    def _cache_put(self, cik: str, data: "AllFilingsData") -> None:
        """
        Upsert `data` into the edgar_cache collection keyed by `cik`.

        Uses a deterministic UUID so repeated upserts for the same CIK
        overwrite rather than accumulate. Payload includes a last_updated
        ISO-8601 timestamp for TTL evaluation on subsequent reads.
        """
        if not self._qdrant:
            return
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"edgar_cache:{cik}"))
        payload = {
            "cik": cik,
            "firm_name": data.firm_name,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "data": data.to_dict(),
        }
        try:
            self._qdrant.upsert(
                collection_name=EDGAR_CACHE_COLLECTION,
                points=[
                    PointStruct(id=point_id, vector=[1.0], payload=payload)
                ],
            )
            logger.info(
                f"[Cache] Upserted CIK {cik} → {EDGAR_CACHE_COLLECTION} (id={point_id})"
            )
        except Exception as e:
            logger.warning(f"[Cache] Qdrant write failed for CIK {cik}: {e}")

    # ── Form ADV ──────────────────────────────────────────────────────────────

    def fetch_adv(
        self, cik: str, firm_name: str, crd: Optional[str] = None
    ) -> Optional[ADVData]:
        """
        Fetch the latest Form ADV filing for a firm via sec-api FormAdvApi.

        FormAdv is indexed by CRD number (Info.FirmCrdNb), not EDGAR CIK.
        Fallback chain:
          1. Info.FirmCrdNb:{crd}          — direct CRD hit (most reliable)
          2. cik:"{10-digit CIK}"          — CIK (unlikely to match FormAdv)
          3. cik:"{int CIK}"               — CIK without leading zeros
          4. "{firm_name}"                 — quoted keyword search

        Returns an ADVData with AUM, client types, fee structures, and
        registration metadata, or None if no filing is found.
        """
        if not self._adv_api:
            return None

        filings = []

        # 1. CRD-based lookup — definitive match for FormAdv
        if crd:
            filings = self._query_adv(f"Info.FirmCrdNb:{crd}", size=1)
            if not filings:
                logger.debug(f"[EDGAR] ADV CRD lookup returned nothing for CRD {crd}")

        # 2. 10-digit CIK
        if not filings:
            filings = self._query_adv(f'cik:"{cik}"', size=1)

        # 3. Integer CIK (strip leading zeros)
        if not filings:
            int_cik = str(int(cik)) if cik.isdigit() else cik.lstrip("0")
            if int_cik != cik:
                filings = self._query_adv(f'cik:"{int_cik}"', size=1)

        # 4. Quoted firm-name keyword search
        if not filings:
            escaped = firm_name.replace('"', '\\"')
            filings = self._query_adv(f'"{escaped}"', size=1)

        if not filings:
            logger.warning(f"[EDGAR] No ADV filing found for {firm_name} (CIK {cik}, CRD {crd})")
            return None

        f = filings[0]

        # FormAdv API returns data nested under Info / FormInfo.Part1A / Filing
        info    = f.get("Info", {})
        part1a  = f.get("FormInfo", {}).get("Part1A", {})
        item5f  = part1a.get("Item5F", {})   # AUM
        item5d  = part1a.get("Item5D", {})   # Client types
        item5e  = part1a.get("Item5E", {})   # Fee structures
        filings_list = f.get("Filing", [])
        rgstn   = f.get("Rgstn", [{}])

        filing_date = str(filings_list[0].get("Dt", "") if filings_list else "")[:10]
        acc_no      = str(f.get("accessionNo") or f.get("id") or "")
        period      = filing_date
        crd_val     = str(info.get("FirmCrdNb") or "")

        source = FilingSource(
            form_type="ADV",
            accession_number=acc_no,
            filing_date=filing_date,
            period_of_report=period,
            source_url=_browse_url(cik, "ADV"),
        )
        flag, days = filing_recency_flag(filing_date)
        recency = RecencyStatus(flag=flag, days_since_filing=days, filing_date=filing_date)

        adv = ADVData(source=source, recency=recency)

        # AUM — Item 5.F: Q5F2C = total, Q5F2A = discretionary, Q5F2B = non-disc
        adv.aum_total_usd             = _to_int(item5f.get("Q5F2C"))
        adv.aum_discretionary_usd     = _to_int(item5f.get("Q5F2A"))
        adv.aum_non_discretionary_usd = _to_int(item5f.get("Q5F2B"))

        # Clients — Item 5.D: Q5DX1 fields are client counts per type
        _client_type_labels = {
            "Q5DA1": "Individuals",
            "Q5DB1": "High Net Worth Individuals",
            "Q5DC1": "Banking/Thrift Institutions",
            "Q5DD1": "Investment Companies",
            "Q5DE1": "Business Development Companies",
            "Q5DF1": "Pooled Investment Vehicles",
            "Q5DG1": "Pension/Profit Sharing Plans",
            "Q5DH1": "Charitable Organizations",
            "Q5DI1": "Corporations/Business Entities",
            "Q5DJ1": "International Clients",
            "Q5DK1": "Other",
        }
        total_clients = 0
        for key, label in _client_type_labels.items():
            n = _to_int(item5d.get(key))
            if n:
                adv.client_types.append(f"{label}: {n}")
                total_clients += n
        adv.num_clients = total_clients or None

        # Fees — Item 5.E: Q5E1-Q5E7 are Y/N flags for compensation types
        _fee_labels = {
            "Q5E1": "Percentage of AUM",
            "Q5E2": "Hourly charges",
            "Q5E3": "Subscription fees",
            "Q5E4": "Fixed fees",
            "Q5E5": "Commissions",
            "Q5E6": "Performance-based fees",
            "Q5E7": "Other",
        }
        adv.fee_structures = [
            label for key, label in _fee_labels.items()
            if item5e.get(key) == "Y"
        ]

        # Registration
        adv.registration_status = str(rgstn[0].get("St", "") if rgstn else "")
        adv.crd_number = crd_val or None

        attribution = f"Form ADV Part 1A, filed {filing_date} (CRD {crd_val})"
        adv.aum_source = adv.client_source = adv.fee_source = attribution

        logger.info(
            f"[EDGAR] ADV: {firm_name} | "
            + (f"AUM=${adv.aum_total_usd:,}" if adv.aum_total_usd else "AUM=unknown")
            + f" | recency={flag.upper()}"
        )
        return adv

    def _query_adv(self, query_string: str, size: int = 1) -> list[dict]:
        """Execute a FormAdvApi.get_firms() query and return the filings list.

        Note: FormAdv does not support sorting by filingDate — omitting sort
        returns results in default (most-recent-first) order from the index.
        """
        try:
            resp = self._adv_api.get_firms({
                "query": {"query_string": {"query": query_string}},
                "from": 0,
                "size": size,
            })
            return resp.get("data") or resp.get("filings") or []
        except Exception as e:
            logger.error(
                f"[EDGAR] FormAdvApi query failed ({query_string!r}): {type(e).__name__}: {e}"
            )
            return []

    # ── Form 13F-HR ───────────────────────────────────────────────────────────

    def fetch_13f(
        self, cik: str, firm_name: str, num_quarters: int = 4
    ) -> list[ThirteenFData]:
        """
        Fetch quarterly 13F-HR holdings via Form13FCoverPagesApi and
        Form13FHoldingsApi.

        Cover pages supply per-quarter metadata (total value, position count,
        period). Holdings API supplies individual position detail for the top 25
        by value. All USD figures are converted from $000s to whole dollars.

        Returns list[ThirteenFData], most recent first.
        """
        if not self._cover_api or not self._holdings_api:
            return []

        # 13F indexes CIK without leading zeros
        int_cik = str(int(cik)) if cik.isdigit() else cik.lstrip("0") or cik
        results: list[ThirteenFData] = []

        try:
            cover_resp = self._cover_api.get_data({
                "query": {
                    "query_string": {
                        "query": f'cik:{int_cik} AND formType:"13F-HR"'
                    }
                },
                "from": 0,
                "size": num_quarters,
                "sort": [{"periodOfReport": {"order": "desc"}}],
            })
            covers = cover_resp.get("filings") or cover_resp.get("data") or []
            logger.debug(f"[EDGAR] 13F cover pages returned {len(covers)} records for CIK {int_cik}")
        except Exception as e:
            logger.error(
                f"[EDGAR] 13F cover page fetch failed for {firm_name} (CIK {int_cik}): "
                f"{type(e).__name__}: {e}"
            )
            return []

        for cover in covers:
            filing_date = str(cover.get("filedAt") or cover.get("filingDate") or "")[:10]
            period      = str(cover.get("periodOfReport") or filing_date)[:10]
            acc_no      = str(cover.get("accessionNo") or cover.get("accessionNumber") or "")

            source = FilingSource(
                form_type="13F-HR",
                accession_number=acc_no,
                filing_date=filing_date,
                period_of_report=period,
                source_url=_browse_url(cik, "13F-HR"),
            )
            flag, days = filing_recency_flag(filing_date)
            recency = RecencyStatus(flag=flag, days_since_filing=days, filing_date=filing_date)
            data = ThirteenFData(source=source, recency=recency)

            # Cover page totals — sec-api returns full dollar values (already converted from $000s)
            cover_val = _to_int(cover.get("tableValueTotal") or cover.get("totalValue"))
            data.total_portfolio_value_usd = cover_val or 0
            data.num_positions = (
                _to_int(cover.get("tableEntryTotal") or cover.get("totalEntries")) or 0
            )

            # Individual holdings — Form13FHoldingsApi returns one filing record
            # per 13F filing; individual positions live in filing["holdings"].
            if acc_no:
                try:
                    h_resp = self._holdings_api.get_data({
                        "query": {
                            "query_string": {
                                "query": f"accessionNo:{acc_no} AND cik:{int_cik}"
                            }
                        },
                        "from": 0,
                        "size": 1,
                    })
                    filing_records = h_resp.get("data") or h_resp.get("filings") or []

                    # Extract the nested holdings array from the filing record
                    raw_holdings: list = []
                    if filing_records:
                        raw_holdings = filing_records[0].get("holdings", [])

                    # Sort by value descending (already full dollars in this API)
                    raw_holdings.sort(
                        key=lambda h: _to_int(h.get("value")) or 0,
                        reverse=True,
                    )

                    holdings: list[Holding] = []
                    for h in raw_holdings[:25]:
                        val_usd = _to_int(h.get("value")) or 0
                        shares  = _to_int(
                            (h.get("shrsOrPrnAmt") or {}).get("sshPrnamt")
                            or h.get("sshPrnamt")
                        )
                        holdings.append(Holding(
                            name=str(h.get("nameOfIssuer") or "").strip(),
                            cusip=h.get("cusip") or None,
                            value_usd=val_usd,
                            shares=shares,
                            pct_of_portfolio=(
                                round((val_usd / data.total_portfolio_value_usd) * 100, 2)
                                if data.total_portfolio_value_usd else None
                            ),
                            source=source,
                        ))
                    data.top_holdings = holdings

                    top10_val = sum(h.value_usd for h in holdings[:10])
                    if data.total_portfolio_value_usd:
                        data.concentration_top10_pct = round(
                            (top10_val / data.total_portfolio_value_usd) * 100, 1
                        )

                    if not data.num_positions and raw_holdings:
                        data.num_positions = len(raw_holdings)

                    logger.debug(
                        f"[EDGAR] Holdings loaded for {period}: "
                        f"{len(raw_holdings)} total, {len(holdings)} kept"
                    )

                except Exception as e:
                    logger.error(
                        f"[EDGAR] Holdings detail failed for {firm_name} ({period}): "
                        f"{type(e).__name__}: {e}"
                    )

            results.append(data)

        return results

    # ── Form D ────────────────────────────────────────────────────────────────

    def fetch_form_d(
        self, cik: str, firm_name: str, limit: int = 5
    ) -> list[FormDData]:
        """
        Fetch Form D private placement filings via sec-api FormDApi.

        Returns list[FormDData] sorted newest first, capped at `limit`.
        """
        if not self._form_d_api:
            return []

        int_cik = str(int(cik)) if cik.isdigit() else cik.lstrip("0") or cik
        results: list[FormDData] = []

        try:
            resp = self._form_d_api.get_data({
                "query": {
                    "query_string": {
                        "query": f'cik:{int_cik}'
                    }
                },
                "from": 0,
                "size": limit,
                "sort": [{"filedAt": {"order": "desc"}}],
            })
            filings = resp.get("filings") or resp.get("data") or []
            logger.debug(f"[EDGAR] Form D returned {len(filings)} records for {firm_name} (CIK {int_cik})")
        except Exception as e:
            logger.error(
                f"[EDGAR] Form D fetch failed for {firm_name} (CIK {int_cik}): "
                f"{type(e).__name__}: {e}"
            )
            return []

        for f in filings:
            filing_date     = str(f.get("filedAt") or f.get("filingDate") or "")[:10]
            first_sale_date = str(f.get("dateOfFirstSale") or "")[:10]
            acc_no          = str(f.get("accessionNo") or f.get("accessionNumber") or "")

            source = FilingSource(
                form_type="D",
                accession_number=acc_no,
                filing_date=filing_date,
                period_of_report=first_sale_date or filing_date,
                source_url=_browse_url(cik, "D"),
            )
            flag, days = filing_recency_flag(filing_date)
            recency = RecencyStatus(flag=flag, days_since_filing=days, filing_date=filing_date)

            fd = FormDData(source=source, recency=recency, fund_name=firm_name)

            # Issuer name — Form D may nest it in an issuers array
            issuers = f.get("issuers") or []
            if issuers and isinstance(issuers, list):
                fd.fund_name = str(issuers[0].get("entityName") or firm_name)
            else:
                fd.fund_name = str(f.get("issuerName") or f.get("nameOfIssuer") or firm_name)

            fd.offering_type = str(
                f.get("investmentFundType") or f.get("offeringType") or ""
            )
            fd.amount_raised_usd = (
                _to_int(f.get("totalAmountSold"))
                or _to_int(f.get("amountSold"))
            )
            fd.total_offering_amount_usd = _to_int(f.get("totalOfferingAmount"))
            fd.date_of_first_sale = first_sale_date or None
            fd.investors_count = (
                _to_int(f.get("totalNumberAlreadyInvested"))
                or _to_int(f.get("numberOfPersonsInvested"))
            )

            exemptions = f.get("exemptionsReliedUpon") or []
            if isinstance(exemptions, list):
                fd.exemption_types = [str(e) for e in exemptions]

            results.append(fd)

        return results

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def fetch_all(
        self,
        cik: str,
        firm_name: str,
        num_quarters: int = 4,
        crd: Optional[str] = None,
    ) -> AllFilingsData:
        """
        Fetch all filing types (ADV, 13F, Form D) for a single manager.

        Checks the edgar_cache Qdrant collection first. Returns the cached
        AllFilingsData immediately if it exists and is within CACHE_TTL_DAYS
        (90 days). On a miss or stale entry, fetches fresh data from sec-api.io
        and upserts the result back into the cache.

        This is the primary entry point for the tearsheet assembler.
        """
        # ── Cache check ───────────────────────────────────────────────────────
        cached = self._cache_get(cik)
        if cached is not None:
            logger.info(
                f"[EDGAR] Returning cached data for {firm_name} (CIK {cik})"
            )
            return cached

        # ── Live fetch from sec-api.io ────────────────────────────────────────
        logger.info(
            f"[EDGAR] Cache miss — fetching from sec-api.io for {firm_name} (CIK {cik})"
        )
        result = AllFilingsData(firm_name=firm_name, cik=cik)

        # Form ADV
        result.adv = self.fetch_adv(cik, firm_name, crd=crd)
        if result.adv:
            aum_str = (
                f"AUM=${result.adv.aum_total_usd:,}"
                if result.adv.aum_total_usd else "AUM=unknown"
            )
            logger.info(
                f"[EDGAR] ADV: {result.adv.recency.flag.upper()} "
                f"({result.adv.recency.days_since_filing}d) {aum_str}"
            )

        # 13F-HR
        result.form_13f = self.fetch_13f(cik, firm_name, num_quarters=num_quarters)
        if result.form_13f:
            latest = result.form_13f[0]
            logger.info(
                f"[EDGAR] 13F: {len(result.form_13f)} quarters | "
                f"latest {latest.recency.flag.upper()} "
                f"({latest.recency.days_since_filing}d) "
                + (
                    f"${latest.total_portfolio_value_usd:,} portfolio"
                    if latest.total_portfolio_value_usd else "(no portfolio value)"
                )
            )

        # Form D
        result.form_d = self.fetch_form_d(cik, firm_name)
        if result.form_d:
            logger.info(f"[EDGAR] Form D: {len(result.form_d)} filings")

        logger.info(f"[EDGAR] Fetch complete for {firm_name}")

        # ── Cache write ───────────────────────────────────────────────────────
        self._cache_put(cik, result)

        return result

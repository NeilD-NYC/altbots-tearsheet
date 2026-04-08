# src/core/assembler.py
# Tearsheet assembly — orchestrates all data modules into a single legal-ready output

"""
TearsheetAssembler — Phase 7 core orchestrator for AltBots tearsheets.

Pipeline (enforced order)
--------------------------
  Step 1  IdentityResolver   — resolve firm CRD / CIK from a plain name
  Step 2  Concurrent fetch   — EDGARFetcher + PersonnelEnricher + SocialSignalScanner
                               fired simultaneously (target: ≤60 s total)
  Step 3  RedFlagScanner     — final verification layer (regulatory disclosures)
  Final   DisclosureEnvelope — legal lock; NO Tearsheet exits this module without it

Graceful degradation
--------------------
  Each module runs in a try/except.  On failure the error is logged and stored
  in Tearsheet.errors, but the remaining modules continue.  The only hard
  failure is identity resolution: without a CRD/CIK no downstream fetch is
  possible and the function raises IdentityResolutionError.

  The DisclosureEnvelope is ALWAYS attached — even when every data module fails
  — because the legal disclaimers must appear on any output regardless of data
  completeness.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime, timezone
from typing import Optional

from src.core.identity import resolve as resolve_identity, ManagerIdentity
from src.core.edgar import EDGARFetcher, AllFilingsData
from src.core.personnel import PersonnelEnricher, PersonProfile
from src.core.social import SocialSignalScanner, SocialSignal
from src.core.red_flags import RedFlagScanner, ScanResult
from src.core.schema import Tearsheet, DisclosureEnvelope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IdentityResolutionError(RuntimeError):
    """Raised when Step 1 cannot resolve a CRD or CIK for the named firm."""


# ---------------------------------------------------------------------------
# Module labels — used consistently in logging and error keys
# ---------------------------------------------------------------------------

_MODULE_EDGAR     = "edgar"
_MODULE_PERSONNEL = "personnel"
_MODULE_SOCIAL    = "social"
_MODULE_RED_FLAGS = "red_flags"


# ---------------------------------------------------------------------------
# TearsheetAssembler
# ---------------------------------------------------------------------------

class TearsheetAssembler:
    """
    Orchestrates all data modules and returns a complete, legal-ready Tearsheet.

    Usage
    -----
        assembler = TearsheetAssembler()
        tearsheet = assembler.assemble("Viking Global Investors")
        assert tearsheet.is_legal_ready
        data = tearsheet.to_dict()

    Parameters
    ----------
    social_lookback_days : int
        Activity window passed to SocialSignalScanner (default 90 days).
    max_personnel : int
        Maximum personnel profiles to build (default 10).
    concurrent_timeout : float
        Per-future timeout in seconds for the concurrent Step 2 block.
        Futures that exceed this are cancelled and their module is recorded
        as skipped in the DisclosureEnvelope (default 55 s).
    """

    def __init__(
        self,
        social_lookback_days: int = 90,
        max_personnel: int = 10,
        concurrent_timeout: float = 90.0,
    ):
        self._social_lookback = social_lookback_days
        self._max_personnel = max_personnel
        self._concurrent_timeout = concurrent_timeout

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def assemble(
        self,
        manager_name: str,
        website: Optional[str] = None,
        cik_hint: Optional[str] = None,
    ) -> Tearsheet:
        """
        Build a complete tearsheet for the named fund manager.

        Parameters
        ----------
        manager_name : str
            Plain-English name of the manager (e.g. "Viking Global Investors").
        website : str, optional
            Known website URL; passed to IdentityResolver to skip Firecrawl map.

        Returns
        -------
        Tearsheet
            Fully assembled, always carries a DisclosureEnvelope.

        Raises
        ------
        IdentityResolutionError
            If Step 1 cannot resolve any CRD or CIK for the manager name.
            No tearsheet can be produced without an identity anchor.
        """
        start_time = time.monotonic()
        timings: dict[str, float] = {}
        logger.info("[Assembler] Starting tearsheet for %r", manager_name)

        tearsheet = Tearsheet(manager_name=manager_name)
        envelope = DisclosureEnvelope()

        # ── Step 1: Identity resolution ────────────────────────────────────────
        _t = time.monotonic()
        identity = self._step1_resolve_identity(manager_name, website, tearsheet, envelope, cik_hint)
        timings["identity"] = time.monotonic() - _t
        # Raises IdentityResolutionError if unresolvable — no point continuing.

        tearsheet.identity = identity
        if identity.crd:
            envelope.add_source(
                f"sec-api.io FormAdv (CRD {identity.crd}) — adviserinfo.sec.gov/firm/summary/{identity.crd}"
            )
        if identity.cik:
            envelope.add_source(
                f"SEC EDGAR (CIK {identity.cik}) — data.sec.gov/submissions/CIK{identity.cik}.json"
            )

        logger.info(
            "[Assembler] Identity resolved: %s | CRD=%s | CIK=%s",
            identity.legal_name, identity.crd, identity.cik,
        )

        # ── Step 2: Concurrent fetch ───────────────────────────────────────────
        _t = time.monotonic()
        self._step2_concurrent_fetch(identity, tearsheet, envelope, timings)
        timings["step2_wall"] = time.monotonic() - _t

        # ── Step 3: RedFlagScanner ─────────────────────────────────────────────
        _t = time.monotonic()
        self._step3_red_flags(identity, tearsheet, envelope)
        timings["red_flags"] = time.monotonic() - _t

        # ── Data-state audit — logged before PDF render ───────────────────────
        holdings_count = 0
        if tearsheet.edgar and tearsheet.edgar.form_13f:
            holdings_count = sum(
                len(getattr(q, "top_holdings", [])) for q in tearsheet.edgar.form_13f
            )
        logger.info(
            "[Assembler] Data audit — "
            "edgar=%s | adv_aum=%s | 13f_quarters=%d | holdings_count=%d | "
            "personnel=%d | social=%d | red_flags=%s | errors=%s",
            "ok" if tearsheet.edgar else "MISSING",
            (
                f"${tearsheet.edgar.adv.aum_total_usd:,}"
                if tearsheet.edgar and tearsheet.edgar.adv and tearsheet.edgar.adv.aum_total_usd
                else "none"
            ),
            len(tearsheet.edgar.form_13f) if tearsheet.edgar else 0,
            holdings_count,
            len(tearsheet.personnel),
            len(tearsheet.social_signals),
            "ok" if tearsheet.red_flags else "none",
            list(tearsheet.errors.keys()) or "none",
        )

        # ── Final: attach DisclosureEnvelope ──────────────────────────────────
        # This MUST be the last operation.  No Tearsheet exits this method
        # without a populated DisclosureEnvelope.
        tearsheet.disclosure = envelope

        elapsed = time.monotonic() - start_time
        timings["TOTAL"] = elapsed

        # ── Timing breakdown ───────────────────────────────────────────────────
        concurrent_modules = (_MODULE_EDGAR, _MODULE_PERSONNEL, _MODULE_SOCIAL)
        slowest = max(
            ((k, v) for k, v in timings.items() if k in concurrent_modules),
            key=lambda kv: kv[1],
            default=(None, 0.0),
        )
        lines = ["[Assembler] ── Timing Breakdown ──────────────────────────"]
        for label, key, note in [
            ("identity  ", "identity",    ""),
            ("edgar     ", _MODULE_EDGAR,     " (concurrent)"),
            ("personnel ", _MODULE_PERSONNEL, " (concurrent)"),
            ("social    ", _MODULE_SOCIAL,    " (concurrent)"),
            ("step2_wall", "step2_wall",  " (concurrent block wall time)"),
            ("red_flags ", "red_flags",   ""),
            ("TOTAL     ", "TOTAL",        ""),
        ]:
            t = timings.get(key, -1)
            marker = " ← SLOWEST" if key == slowest[0] else ""
            lines.append(
                f"[Assembler]   {label}: {t:6.1f}s{note}{marker}"
                if t >= 0 else
                f"[Assembler]   {label}:    n/a{note}"
            )
        lines.append("[Assembler] ──────────────────────────────────────────────")
        for line in lines:
            logger.info(line)

        logger.info(
            "[Assembler] Tearsheet complete for %r in %.1fs | legal_ready=%s | errors=%s",
            manager_name, elapsed, tearsheet.is_legal_ready, list(tearsheet.errors.keys()),
        )

        assert tearsheet.is_legal_ready, (
            "FATAL: Tearsheet exited assembler without DisclosureEnvelope. "
            "This is a programmer error — report immediately."
        )

        return tearsheet

    # ------------------------------------------------------------------
    # Step 1 — Identity resolution
    # ------------------------------------------------------------------

    def _step1_resolve_identity(
        self,
        manager_name: str,
        website: Optional[str],
        tearsheet: Tearsheet,
        envelope: DisclosureEnvelope,
        cik_hint: Optional[str] = None,
    ) -> ManagerIdentity:
        """
        Resolve the manager's canonical identity (CRD, CIK, website, socials).

        Resolution chain (delegated to identity.resolve()):
          1. Qdrant proprietary cache   (fastest)
          2a. cik_hint path             (EDGAR submissions JSON, skips FormAdv)
          2b. sec-api FormAdvApi        (CRD + legal name)
          3. EDGAR full-text search     (CIK)
          4. Firecrawl                  (social handles)

        Raises IdentityResolutionError if the chain returns None, because
        without at least a CRD or CIK nothing downstream can be fetched.
        """
        logger.info("[Assembler:Step1] Resolving identity for %r", manager_name)
        try:
            identity = resolve_identity(manager_name, website=website, cik_hint=cik_hint)
        except Exception as exc:
            logger.exception("[Assembler:Step1] Identity resolver raised unexpectedly")
            raise IdentityResolutionError(
                f"Identity resolution failed for {manager_name!r}: {exc}"
            ) from exc

        if identity is None:
            raise IdentityResolutionError(
                f"Could not resolve a CRD or CIK for {manager_name!r}. "
                "Check the firm name spelling or provide a CRD number directly."
            )

        envelope.add_source(f"IdentityResolver ({identity.source})")
        return identity

    # ------------------------------------------------------------------
    # Step 2 — Concurrent fetch (EDGAR + Personnel + Social)
    # ------------------------------------------------------------------

    def _step2_concurrent_fetch(
        self,
        identity: ManagerIdentity,
        tearsheet: Tearsheet,
        envelope: DisclosureEnvelope,
        timings: dict,
    ) -> None:
        """
        Fire EDGARFetcher, PersonnelEnricher, and SocialSignalScanner
        concurrently in a ThreadPoolExecutor.

        All three are submitted simultaneously.  Each runs independently:
        - A module that raises an exception is caught; the error is stored in
          tearsheet.errors and recorded as skipped in the envelope.
        - The other two modules are unaffected.
        - Futures that exceed concurrent_timeout are also treated as skipped.

        This is the key speed mechanism: all three I/O-heavy modules overlap
        rather than running sequentially, targeting ≤60 seconds end-to-end.
        """
        logger.info("[Assembler:Step2] Launching concurrent fetch (timeout=%.0fs)", self._concurrent_timeout)

        def _timed(name: str, fn):
            """Wrap fn so its wall time is recorded in timings[name]."""
            def wrapper():
                t0 = time.monotonic()
                result = fn()
                timings[name] = time.monotonic() - t0
                return result
            return wrapper

        # Map each module label to its (timed) callable
        tasks: dict[str, object] = {
            _MODULE_EDGAR:     _timed(_MODULE_EDGAR,     lambda: self._run_edgar(identity)),
            _MODULE_PERSONNEL: _timed(_MODULE_PERSONNEL, lambda: self._run_personnel(identity)),
            _MODULE_SOCIAL:    _timed(_MODULE_SOCIAL,    lambda: self._run_social(identity, tearsheet)),
        }

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="tearsheet") as pool:
            future_to_module: dict[Future, str] = {
                pool.submit(fn): name  # type: ignore[arg-type]
                for name, fn in tasks.items()
            }

            # Collect results as they complete; enforce per-future timeout.
            # as_completed() itself raises TimeoutError when the wall-clock
            # deadline expires before all futures finish — catch that too.
            pending = dict(future_to_module)  # future → module name
            deadline = time.monotonic() + self._concurrent_timeout

            try:
                for future in as_completed(pending, timeout=self._concurrent_timeout):
                    module = pending[future]
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        result = future.result(timeout=remaining)
                        self._apply_result(module, result, tearsheet, envelope)
                    except TimeoutError:
                        msg = f"Module timed out after {self._concurrent_timeout:.0f}s"
                        logger.warning("[Assembler:Step2] %s — %s", module, msg)
                        tearsheet.errors[module] = msg
                        envelope.record_skip(module, msg)
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        logger.error("[Assembler:Step2] %s failed — %s", module, msg, exc_info=True)
                        tearsheet.errors[module] = msg
                        envelope.record_skip(module, msg)
            except TimeoutError:
                # Overall deadline expired — record any futures that never completed
                for future, module in pending.items():
                    if module not in tearsheet.errors and not future.done():
                        msg = f"Module timed out (wall-clock {self._concurrent_timeout:.0f}s exceeded)"
                        logger.warning("[Assembler:Step2] %s — %s", module, msg)
                        tearsheet.errors[module] = msg
                        envelope.record_skip(module, msg)
                        future.cancel()

        logger.info(
            "[Assembler:Step2] Complete — edgar=%s personnel=%d social=%d skipped=%s",
            "ok" if tearsheet.edgar else "none",
            len(tearsheet.personnel),
            len(tearsheet.social_signals),
            list({k for k in tearsheet.errors if k != _MODULE_RED_FLAGS}),
        )

    def _run_edgar(self, identity: ManagerIdentity) -> Optional[AllFilingsData]:
        """Run EDGARFetcher.fetch_all(); requires a CIK."""
        cik = identity.cik
        if not cik:
            logger.warning("[Assembler] No CIK — EDGARFetcher skipped")
            return None

        fetcher = EDGARFetcher()
        return fetcher.fetch_all(
            cik=cik,
            firm_name=identity.legal_name,
            crd=identity.crd,
        )

    def _run_personnel(self, identity: ManagerIdentity) -> list[PersonProfile]:
        """Run PersonnelEnricher.enrich(); requires a website."""
        website = identity.website
        if not website:
            logger.warning("[Assembler] No website — PersonnelEnricher skipped")
            return []

        enricher = PersonnelEnricher()
        return enricher.enrich(
            firm_name=identity.legal_name,
            website=website,
            max_people=self._max_personnel,
        )

    def _run_social(
        self, identity: ManagerIdentity, tearsheet: Tearsheet
    ) -> list[SocialSignal]:
        """
        Run SocialSignalScanner.scan().

        Personnel names are passed if Step 2's personnel future has already
        resolved — but because all three run concurrently, personnel may not
        yet be available.  The scanner accepts an empty list gracefully.
        """
        # Use any personnel already on tearsheet (may be empty at this point)
        personnel_names = [p.name for p in tearsheet.personnel if hasattr(p, "name")]

        scanner = SocialSignalScanner()
        return scanner.scan(
            firm_name=identity.legal_name,
            personnel_names=personnel_names,
            lookback_days=self._social_lookback,
        )

    def _apply_result(
        self,
        module: str,
        result: object,
        tearsheet: Tearsheet,
        envelope: DisclosureEnvelope,
    ) -> None:
        """Write a successful module result into the tearsheet and envelope."""
        if module == _MODULE_EDGAR:
            tearsheet.edgar = result
            if result is not None:
                envelope.add_source("SEC EDGAR — EDGARFetcher (ADV / 13F-HR / Form D)")
            logger.info("[Assembler:Step2] EDGAR fetch complete")

        elif module == _MODULE_PERSONNEL:
            tearsheet.personnel = result or []
            if tearsheet.personnel:
                tier2 = sum(1 for p in tearsheet.personnel if getattr(p, "source_tier", 1) == 2)
                source_detail = (
                    f"Firecrawl / Proxycurl ({tier2} via Proxycurl) / Claude Haiku"
                    if tier2 else "Firecrawl / Claude Haiku"
                )
                envelope.add_source(
                    f"PersonnelEnricher ({source_detail}) "
                    f"— {len(tearsheet.personnel)} profile(s)"
                )
            logger.info(
                "[Assembler:Step2] Personnel enrichment complete — %d profile(s) "
                "(%d tier-1 Firecrawl, %d tier-2 Proxycurl)",
                len(tearsheet.personnel),
                sum(1 for p in tearsheet.personnel if getattr(p, "source_tier", 1) == 1),
                sum(1 for p in tearsheet.personnel if getattr(p, "source_tier", 1) == 2),
            )

        elif module == _MODULE_SOCIAL:
            tearsheet.social_signals = result or []
            if tearsheet.social_signals:
                envelope.add_source(
                    f"SocialSignalScanner (Firecrawl / Exa / Claude Haiku) "
                    f"— {len(tearsheet.social_signals)} signal(s)"
                )
            logger.info("[Assembler:Step2] Social scan complete — %d signal(s)", len(tearsheet.social_signals))

    # ------------------------------------------------------------------
    # Step 3 — RedFlagScanner
    # ------------------------------------------------------------------

    def _step3_red_flags(
        self,
        identity: ManagerIdentity,
        tearsheet: Tearsheet,
        envelope: DisclosureEnvelope,
    ) -> None:
        """
        Run RedFlagScanner as the final verification layer.

        Requires a CRD number.  If no CRD was resolved, the module is
        recorded as skipped with an explanatory reason.

        The red-flag scan runs AFTER Step 2 (not concurrently with it)
        because it is the verification layer — it needs the identity to
        be fully resolved and should be the last data-gathering step
        before the legal lock is applied.
        """
        logger.info("[Assembler:Step3] Running RedFlagScanner")

        crd = identity.crd
        if not crd:
            reason = "No CRD resolved — RedFlagScanner requires a CRD number"
            logger.warning("[Assembler:Step3] %s", reason)
            tearsheet.errors[_MODULE_RED_FLAGS] = reason
            envelope.record_skip(_MODULE_RED_FLAGS, reason)
            return

        try:
            scanner = RedFlagScanner()
            result: ScanResult = scanner.scan(
                firm_name=identity.legal_name,
                crd_number=crd,
            )
            tearsheet.red_flags = result
            envelope.add_source(
                f"RedFlagScanner — Form ADV Item 11 + SEC enforcement releases "
                f"(CRD {crd})"
            )
            logger.info(
                "[Assembler:Step3] Red flag scan complete — has_disclosures=%s findings=%d",
                result.has_disclosures,
                len(result.findings),
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("[Assembler:Step3] RedFlagScanner failed — %s", msg, exc_info=True)
            tearsheet.errors[_MODULE_RED_FLAGS] = msg
            envelope.record_skip(_MODULE_RED_FLAGS, msg)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def assemble(
    manager_name: str,
    website: Optional[str] = None,
    social_lookback_days: int = 90,
    max_personnel: int = 10,
) -> Tearsheet:
    """
    Module-level convenience function for one-shot tearsheet generation.

    Equivalent to::

        TearsheetAssembler().assemble(manager_name, website)

    Parameters
    ----------
    manager_name : str
        Plain-English manager or firm name.
    website : str, optional
        Known primary website — skip Firecrawl map step if provided.
    social_lookback_days : int
        Activity window for SocialSignalScanner (default 90 days).
    max_personnel : int
        Cap on personnel profiles built (default 10).

    Returns
    -------
    Tearsheet
        Complete, legal-ready data object.

    Raises
    ------
    IdentityResolutionError
        If no CRD/CIK can be resolved for the given name.

    Example
    -------
    >>> from src.core.assembler import assemble
    >>> ts = assemble("Viking Global Investors")
    >>> assert ts.is_legal_ready
    >>> print(ts.red_flags.status_phrase)
    >>> import json, sys
    >>> json.dump(ts.to_dict(), sys.stdout, indent=2, default=str)
    """
    assembler = TearsheetAssembler(
        social_lookback_days=social_lookback_days,
        max_personnel=max_personnel,
    )
    return assembler.assemble(manager_name, website=website)

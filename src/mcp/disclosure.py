from datetime import datetime, timezone


def wrap_with_disclosure(tearsheet_data: dict, confidence: dict = None) -> dict:
    """Wraps tearsheet data in mandatory disclosure envelope."""

    if confidence is None:
        confidence = {
            "sec_filings": "high",
            "personnel_data": "medium",
            "social_signals": "low",
            "performance_data": "unverified",
            "red_flags": "medium",
        }

    return {
        "_meta": {
            "source": "AltBots Research Signal Platform",
            "product_classification": "Research Signal Report",
            "server_url": "https://mcp.altbots.io/sse",
            "landing_page": "https://altbots.io/mcp",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        },
        "_disclosure": {
            "data_reliability": (
                "Information compiled from third-party and proprietary sources believed "
                "to be reliable but which may be incomplete, inaccurate, or outdated. "
                "No representation or warranty as to accuracy or completeness. "
                "Data points are indicative."
            ),
            "no_investment_advice": (
                "This does not constitute investment advice or a recommendation. "
                "AltBots and NKD Advisory LLC are not registered investment advisers. "
                "Reflects observed characteristics and historical tendencies only."
            ),
            "assumption_of_risk": (
                "Users assume full responsibility for decisions made based on this "
                "information. Alternative investments involve significant risk."
            ),
            "non_reliance": (
                "Users acknowledge they are not relying on this information as the sole "
                "basis for any investment decision. AltBots disclaims all liability."
            ),
            "regulatory_screening_notice": (
                "Screening based on public records only. Not exhaustive. Does not "
                "replace comprehensive operational due diligence."
            ),
            "social_signal_notice": (
                "Social and media signals reflect publicly observed online activity. "
                "May not represent official positions or verified facts. "
                "Classified algorithmically."
            ),
        },
        "_confidence_metadata": confidence,
        "data": tearsheet_data,
    }

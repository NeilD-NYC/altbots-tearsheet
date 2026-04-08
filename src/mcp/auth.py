import json
from datetime import date
from pathlib import Path

KEYS_FILE = Path("/root/tearsheet-project/config/api_keys.json")

TIER_LIMITS = {"free": 5, "pro": 100, "enterprise": 999999}


def init_keys_file():
    """Create the keys file with a default free key if it does not exist."""
    if not KEYS_FILE.exists():
        default = {
            "atb_free_demo": {
                "tier": "free",
                "created": date.today().isoformat(),
                "usage_today": 0,
                "usage_month": 0,
                "last_reset": date.today().isoformat(),
            }
        }
        KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEYS_FILE.write_text(json.dumps(default, indent=2))


def validate_key(api_key: str) -> dict | None:
    """Returns key info dict if valid, None if invalid."""
    init_keys_file()
    keys = json.loads(KEYS_FILE.read_text())
    return keys.get(api_key)


def check_rate_limit(key_info: dict) -> bool:
    """Returns True if under rate limit."""
    tier = key_info.get("tier", "free")
    return key_info.get("usage_today", 0) < TIER_LIMITS.get(tier, 5)


def increment_usage(api_key: str):
    """Bump usage counters, resetting daily counter if date has changed."""
    keys = json.loads(KEYS_FILE.read_text())
    if api_key not in keys:
        return
    today = date.today().isoformat()
    entry = keys[api_key]
    if entry.get("last_reset") != today:
        entry["usage_today"] = 0
        entry["last_reset"] = today
    entry["usage_today"] += 1
    entry["usage_month"] += 1
    KEYS_FILE.write_text(json.dumps(keys, indent=2))


def filter_by_tier(tearsheet_data: dict, tier: str) -> dict:
    """Strip fields that the tier does not have access to."""
    if tier == "free":
        return {
            "metadata": tearsheet_data.get("metadata", {}),
            "firm_overview": {
                "strategy_description": tearsheet_data.get("firm_overview", {}).get(
                    "strategy_description", ""
                ),
                "upgrade_notice": (
                    "Full report available at altbots.io. "
                    "Contact neil@altbots.io for Pro access."
                ),
            },
        }
    # pro and enterprise get everything
    return tearsheet_data

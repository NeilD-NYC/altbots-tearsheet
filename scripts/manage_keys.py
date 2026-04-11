#!/usr/bin/env python3
"""
scripts/manage_keys.py — AltBots API key management CLI

Usage
-----
  python scripts/manage_keys.py --add "Firm Name" email@domain.com [plan]
  python scripts/manage_keys.py --list
  python scripts/manage_keys.py --revoke ak_xxxxxxxxxxxx
  python scripts/manage_keys.py --check  ak_xxxxxxxxxxxx

Plans: free | pro | enterprise  (default: pro)
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "api_keys.json"
_VALID_PLANS = {"free", "pro", "enterprise"}


# ── persistence ───────────────────────────────────────────────────────────────

def _load() -> dict:
    if not _CONFIG.exists():
        return {}
    try:
        return json.loads(_CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(keys: dict) -> None:
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG.write_text(json.dumps(keys, indent=2), encoding="utf-8")


# ── key generation ────────────────────────────────────────────────────────────

def generate_key() -> str:
    """Return a fresh unique-looking ak_ key (12 random alphanumeric chars)."""
    chars = string.ascii_letters + string.digits
    return "ak_" + "".join(random.choices(chars, k=12))


def add_key(name: str, email: str, plan: str = "pro") -> str:
    """Create a new API key entry and persist it. Returns the generated key."""
    if plan not in _VALID_PLANS:
        sys.exit(f"Unknown plan '{plan}'. Valid plans: {', '.join(sorted(_VALID_PLANS))}")

    keys = _load()
    # Guarantee uniqueness (collision astronomically unlikely but be safe)
    key = generate_key()
    while key in keys:
        key = generate_key()

    keys[key] = {
        "name":        name,
        "email":       email,
        "plan":        plan,
        "active":      True,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "last_used":   None,
        "usage_count": 0,
    }
    _save(keys)
    return key


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_add(parts: list[str]) -> None:
    if len(parts) < 2:
        sys.exit("--add requires at least: NAME EMAIL  (plan is optional, default: pro)")
    name  = parts[0]
    email = parts[1]
    plan  = parts[2] if len(parts) > 2 else "pro"
    key = add_key(name, email, plan)
    print(f"Created [{plan}] key for {name} <{email}>:")
    print(f"  {key}")


def cmd_list() -> None:
    keys = _load()
    if not keys:
        print("No API keys found.")
        return
    header = f"{'KEY':<22} {'NAME':<24} {'EMAIL':<32} {'PLAN':<12} {'ACTIVE':<8} {'USES'}"
    print(header)
    print("-" * len(header))
    for k, v in sorted(keys.items(), key=lambda x: x[1]["created_at"]):
        status = "yes" if v.get("active") else "NO"
        print(
            f"{k:<22} {v['name']:<24} {v['email']:<32} "
            f"{v['plan']:<12} {status:<8} {v['usage_count']}"
        )


def cmd_revoke(key: str) -> None:
    keys = _load()
    if key not in keys:
        sys.exit(f"Key not found: {key}")
    if not keys[key]["active"]:
        print(f"Already revoked: {key}")
        return
    keys[key]["active"] = False
    _save(keys)
    print(f"Revoked: {key}  ({keys[key]['name']})")


def cmd_check(key: str) -> None:
    keys = _load()
    if key not in keys:
        print("INVALID — key not found")
        return
    v = keys[key]
    status = "ACTIVE" if v.get("active") else "REVOKED"
    last = v["last_used"] or "never"
    print(
        f"{status}: {v['name']} <{v['email']}> "
        f"plan={v['plan']}  uses={v['usage_count']}  last_used={last}"
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="AltBots API key management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--add", nargs="+", metavar="ARG",
        help='Add key: --add "Firm Name" email@domain.com [plan]',
    )
    grp.add_argument("--list",   action="store_true", help="List all keys")
    grp.add_argument("--revoke", metavar="KEY",       help="Deactivate a key")
    grp.add_argument("--check",  metavar="KEY",       help="Check key status")

    args = ap.parse_args()

    if args.add:
        cmd_add(args.add)
    elif args.list:
        cmd_list()
    elif args.revoke:
        cmd_revoke(args.revoke)
    elif args.check:
        cmd_check(args.check)


if __name__ == "__main__":
    main()

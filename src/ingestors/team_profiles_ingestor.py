"""
src/ingestors/team_profiles_ingestor.py — AltBots Pipeline: Team Profiles Ingestor

Reads data/team_profiles_raw.jsonl (raw markdown output from team_scraper),
uses Claude Haiku to extract structured personnel records with synthesized bios,
embeds via OpenAI text-embedding-3-small, and upserts into the Qdrant
`team_profiles` collection.

Strict Rule: Bios in Qdrant are ALWAYS Haiku-synthesized — no verbatim text
from the source page is stored.

Usage
-----
  python -m src.ingestors.team_profiles_ingestor --ingest
  python -m src.ingestors.team_profiles_ingestor --ingest --input data/team_profiles_raw.jsonl
  python -m src.ingestors.team_profiles_ingestor --reingest-firm "VIKING GLOBAL INVESTORS LP"
  python -m src.ingestors.team_profiles_ingestor --stats
  python -m src.ingestors.team_profiles_ingestor --dry-run

Environment variables
---------------------
  ANTHROPIC_API_KEY   — Claude Haiku (extraction + synthesis)
  OPENAI_API_KEY      — OpenAI embeddings
  QDRANT_HOST         — Qdrant host (default: localhost)
  QDRANT_PORT         — Qdrant port (default: 6333)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Project root and .env ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# ── Optional dependencies ─────────────────────────────────────────────────────
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

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("team_ingestor")

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
EMBED_MODEL    = "text-embedding-3-small"
HAIKU_MODEL    = "claude-haiku-4-5-20251001"
VECTOR_SIZE    = 1536
COLLECTION     = "team_profiles"

BATCH_SIZE        = 50
INTER_BATCH_DELAY = 1.0

INPUT_JSONL  = _PROJECT_ROOT / "data" / "team_profiles_raw.jsonl"
STATE_FILE   = _PROJECT_ROOT / "data" / "team_ingestor_state.json"
FAILED_JSONL = _PROJECT_ROOT / "logs" / "team_ingestor_failed.jsonl"

# Max raw markdown chars sent to Haiku per firm (stays well within context)
MAX_MARKDOWN_CHARS = 14_000


# ── Haiku extraction prompt ───────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a professional financial researcher building an investment personnel database. "
    "You extract and synthesize information about investment professionals. "
    "You return only valid JSON — no prose, no markdown fences, no commentary."
)

_EXTRACTION_PROMPT = """\
Extract all investment professionals from the following fund manager team page.

For each person return a JSON object with these exact fields:
  name   (string, required — full name only)
  title  (string — their role or title; empty string if not found)
  bio    (string — a synthesized 2-4 sentence professional biography written
          entirely in your own words in third-person. Cover their role,
          background, and areas of expertise.
          STRICT RULE: Do NOT copy any text verbatim from the source.
          Transform all information. If there is insufficient information,
          write: "Investment professional at the firm.")

Return a JSON array of person objects. If no investment professionals are \
found, return an empty array [].

Team page content:
---
{content}
---"""


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def _get_qdrant() -> Optional["QdrantClient"]:
    if not QDRANT_AVAILABLE:
        logger.error("qdrant-client not installed.")
        return None
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        _ensure_collection(client)
        return client
    except Exception as exc:
        logger.error("Qdrant connection failed: %s", exc)
        return None


def _ensure_collection(client: "QdrantClient") -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", COLLECTION)


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> Optional[list[float]]:
    if not OPENAI_AVAILABLE:
        logger.error("openai not installed.")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return None


# ── Haiku extraction + synthesis ──────────────────────────────────────────────

def _extract_people(
    raw_markdown: str,
    haiku_client: "anthropic.Anthropic",
) -> list[dict]:
    """
    Call Claude Haiku to extract people from raw markdown and synthesize bios.
    Returns a list of {name, title, bio} dicts (may be empty).
    Bio field is always Haiku-synthesized — never verbatim source text.
    """
    content = raw_markdown[:MAX_MARKDOWN_CHARS]
    prompt  = _EXTRACTION_PROMPT.format(content=content)

    try:
        response = haiku_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.error("Haiku call failed: %s", exc)
        return []

    raw_text = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    start = raw_text.find("[")
    end   = raw_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("Haiku returned no JSON array. Preview: %r", raw_text[:200])
        return []

    try:
        people = json.loads(raw_text[start:end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error from Haiku: %s. Preview: %r", exc, raw_text[:300])
        return []

    # Validate and clean up each person dict
    cleaned = []
    for p in people:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "") or "").strip()
        if not name:
            continue
        cleaned.append({
            "name":  name,
            "title": str(p.get("title", "") or "").strip(),
            "bio":   str(p.get("bio", "") or "").strip(),
        })

    return cleaned


# ── Point ID (deterministic UUID5, matches existing pipeline pattern) ─────────

def _point_id(firm_name: str, person_name: str, chunk_index: int) -> str:
    key = f"team_profile:{firm_name}:{person_name}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ── Dedup check ───────────────────────────────────────────────────────────────

def _is_duplicate(
    client: "QdrantClient",
    firm_name: str,
    person_name: str,
    new_scraped_at: str,
) -> bool:
    """Return True if the existing record is at least as recent as new_scraped_at."""
    point_id = _point_id(firm_name, person_name, 0)
    try:
        results = client.retrieve(
            collection_name=COLLECTION,
            ids=[point_id],
            with_payload=True,
        )
    except Exception as exc:
        logger.warning("Dedup check failed for %s/%s: %s — ingesting", firm_name, person_name, exc)
        return False

    if not results:
        return False
    existing_ts = results[0].payload.get("scraped_at", "")
    return existing_ts >= new_scraped_at


# ── Batch upsert ──────────────────────────────────────────────────────────────

def _upsert_batch(
    client: "QdrantClient",
    points: list["PointStruct"],
    dry_run: bool,
) -> tuple[int, list["PointStruct"]]:
    if dry_run:
        return len(points), []

    for attempt in range(1, 3):
        try:
            client.upsert(collection_name=COLLECTION, points=points)
            return len(points), []
        except Exception as exc:
            if attempt == 1:
                logger.warning("Batch upsert failed (attempt 1): %s — retrying in 3s", exc)
                time.sleep(3)
            else:
                logger.error("Batch upsert failed (attempt 2): %s", exc)
                return 0, points
    return 0, []


def _log_failed(failed_points: list["PointStruct"]) -> None:
    if not failed_points:
        return
    FAILED_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_JSONL.open("a", encoding="utf-8") as fh:
        for pt in failed_points:
            entry = {
                "point_id":  str(pt.id),
                "payload":   pt.payload,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── State file ────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
                state.setdefault("last_run", None)
                state.setdefault("processed_firms", [])
                state.setdefault("total_points_upserted", 0)
                return state
        except Exception as exc:
            logger.warning("State file corrupt (%s) — starting fresh", exc)
    return {"last_run": None, "processed_firms": [], "total_points_upserted": 0}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


# ── JSONL reader ──────────────────────────────────────────────────────────────

def _read_jsonl(path: Path, limit: Optional[int] = None) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping bad JSONL line: %s", exc)
            if limit and len(records) >= limit:
                break
    return records


# ── Core ingest logic ─────────────────────────────────────────────────────────

def run_ingest(
    input_path: Path,
    reingest_firm: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """
    Main ingestion pipeline.

    Reads raw-markdown records from JSONL, extracts + synthesizes person
    records via Haiku, embeds them, and upserts to Qdrant team_profiles.
    """
    if not ANTHROPIC_AVAILABLE:
        logger.error("anthropic not installed — cannot extract people. Run: pip install anthropic")
        return {}

    an_key = os.getenv("ANTHROPIC_API_KEY")
    if not an_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return {}

    haiku_client = anthropic.Anthropic(api_key=an_key)

    state             = _load_state()
    processed_firms   = set(state["processed_firms"])

    stats = {
        "firms_read":                0,
        "firms_processed":           0,
        "people_extracted":          0,
        "records_skipped_duplicate": 0,
        "points_upserted":           0,
        "failed":                    0,
        "run_timestamp":             datetime.now(timezone.utc).isoformat(),
    }

    # Connect to Qdrant
    qdrant = None
    if not dry_run:
        qdrant = _get_qdrant()
        if qdrant is None:
            logger.error("Cannot connect to Qdrant — aborting. Use --dry-run to test.")
            return stats

    # Read JSONL (one record per firm, containing raw_markdown)
    all_records = _read_jsonl(input_path, limit=limit)
    stats["firms_read"] = len(all_records)
    logger.info("Read %d firm record(s) from %s", len(all_records), input_path)

    # Filter
    if reingest_firm:
        to_process = [
            r for r in all_records
            if r.get("firm_name", "").strip().lower() == reingest_firm.strip().lower()
        ]
        logger.info("--reingest-firm %r: %d record(s) matched", reingest_firm, len(to_process))
    else:
        to_process = [
            r for r in all_records
            if r.get("firm_name", "").strip() not in processed_firms
        ]
        skipped = len(all_records) - len(to_process)
        if skipped:
            logger.info(
                "%d firm(s) already processed — skipping (use --reingest-firm to force)", skipped
            )

    if not to_process:
        logger.info("Nothing to process.")
        _print_summary(stats)
        return stats

    points_batch:  list["PointStruct"] = []
    firms_seen:    set[str]            = set()

    for raw_rec in to_process:
        firm_name    = raw_rec.get("firm_name", "").strip()
        team_url     = raw_rec.get("team_page_url", "")
        raw_markdown = raw_rec.get("raw_markdown", "")
        scraped_at   = raw_rec.get("scraped_at", datetime.now(timezone.utc).isoformat())

        if not firm_name or not raw_markdown:
            logger.debug("Skipping record with missing firm_name or raw_markdown")
            continue

        firms_seen.add(firm_name)

        # Haiku extraction + bio synthesis
        logger.info("  Extracting people for: %s", firm_name)
        people = _extract_people(raw_markdown, haiku_client)

        if not people:
            logger.warning("  No people extracted for %s", firm_name)
            # Still mark as processed so we don't retry on every run
            stats["firms_processed"] += 1
            continue

        logger.info("  %d person(s) extracted for %s", len(people), firm_name)
        stats["firms_processed"]  += 1
        stats["people_extracted"] += len(people)

        for person in people:
            name  = person["name"]
            title = person["title"]
            bio   = person["bio"]

            # Dedup check (skip if not force-reingest and record is current)
            if not reingest_firm and qdrant:
                if _is_duplicate(qdrant, firm_name, name, scraped_at):
                    stats["records_skipped_duplicate"] += 1
                    logger.debug("  SKIP duplicate: %s / %s", firm_name, name)
                    continue

            # Build chunk text for embedding:
            # "[FIRM: ...] [NAME: ...] [TITLE: ...] {synthesized bio}"
            chunk_text = f"[FIRM: {firm_name}] [NAME: {name}] [TITLE: {title}] {bio}"

            payload = {
                "firm_name":     firm_name,
                "person_name":   name,
                "title":         title,
                "bio":           bio,
                "team_page_url": team_url,
                "scraped_at":    scraped_at,
                "chunk_index":   0,
                "total_chunks":  1,
                "source":        "team_page",
                "bio_source":    "haiku_synthesized",
            }

            if dry_run:
                _print_dry_run_record(firm_name, name, title, bio, chunk_text, payload)
                stats["points_upserted"] += 1
                continue

            # Embed
            vector = _embed(chunk_text)
            if vector is None:
                logger.warning("  Embedding failed for %s / %s — skipping", firm_name, name)
                stats["failed"] += 1
                continue

            point_id = _point_id(firm_name, name, 0)
            points_batch.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )

            # Flush batch
            if len(points_batch) >= BATCH_SIZE:
                n_ok, failed = _upsert_batch(qdrant, points_batch, dry_run)
                stats["points_upserted"] += n_ok
                stats["failed"]          += len(failed)
                _log_failed(failed)
                points_batch = []
                logger.info("  Upserted batch of %d point(s)", n_ok)
                time.sleep(INTER_BATCH_DELAY)

    # Flush remainder
    if points_batch and not dry_run:
        n_ok, failed = _upsert_batch(qdrant, points_batch, dry_run)
        stats["points_upserted"] += n_ok
        stats["failed"]          += len(failed)
        _log_failed(failed)
        logger.info("Upserted final batch of %d point(s)", n_ok)

    # Update state
    if not dry_run and not reingest_firm:
        state["processed_firms"]        = sorted(processed_firms | firms_seen)
        state["total_points_upserted"] += stats["points_upserted"]
        state["last_run"]               = stats["run_timestamp"]
        _save_state(state)
        logger.info("State saved to %s", STATE_FILE)

    _print_summary(stats)
    return stats


# ── Stats command ─────────────────────────────────────────────────────────────

def run_stats() -> None:
    qdrant = _get_qdrant()
    if qdrant is None:
        return
    try:
        info = qdrant.get_collection(COLLECTION)
    except Exception as exc:
        logger.error("Failed to get collection info: %s", exc)
        return

    state = _load_state()
    print(f"\n{'─' * 50}")
    print(f"  Collection : {COLLECTION}")
    print(f"  Points     : {info.points_count:,}")
    print(f"  Status     : {info.status}")
    print(f"  Last run   : {state['last_run'] or 'never'}")
    print(f"  Firms done : {len(state['processed_firms'])}")
    print(f"  Total pts  : {state['total_points_upserted']:,} (lifetime)")
    print(f"{'─' * 50}\n")


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_dry_run_record(
    firm_name: str,
    name: str,
    title: str,
    bio: str,
    chunk_text: str,
    payload: dict,
) -> None:
    print(f"\n{'═' * 70}")
    print(f"  Firm   : {firm_name}")
    print(f"  Name   : {name}")
    print(f"  Title  : {title}")
    print(f"  Bio    : {bio[:200]}{'…' if len(bio) > 200 else ''}")
    print(f"\n  Chunk text (~{len(chunk_text) // 4} tokens):")
    for i in range(0, min(len(chunk_text), 400), 80):
        print(f"    {chunk_text[i:i+80]}")
    print(f"\n  bio_source : {payload['bio_source']}")
    print(f"  point_id   : {_point_id(firm_name, name, 0)}")


def _print_summary(stats: dict) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Firms read                : {stats['firms_read']}")
    print(f"  Firms processed           : {stats['firms_processed']}")
    print(f"  People extracted          : {stats['people_extracted']}")
    print(f"  Skipped (duplicate)       : {stats['records_skipped_duplicate']}")
    print(f"  Points upserted           : {stats['points_upserted']}")
    print(f"  Failed                    : {stats['failed']}")
    print(f"{'─' * 50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AltBots team profiles ingestor — extract, synthesize, and embed team data into Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ingestors.team_profiles_ingestor --ingest
  python -m src.ingestors.team_profiles_ingestor --reingest-firm "VIKING GLOBAL INVESTORS LP"
  python -m src.ingestors.team_profiles_ingestor --stats
  python -m src.ingestors.team_profiles_ingestor --dry-run
        """,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ingest",        action="store_true", help="Ingest all unprocessed records.")
    mode.add_argument("--reingest-firm", metavar="FIRM",      help="Force re-ingest all records for this firm.")
    mode.add_argument("--stats",         action="store_true", help="Print collection stats; no ingestion.")
    mode.add_argument("--dry-run",       action="store_true", help="Run extraction; skip embedding and Qdrant writes.")

    p.add_argument("--input",  metavar="FILE", default=str(INPUT_JSONL),
                   help=f"JSONL input file (default: {INPUT_JSONL})")
    p.add_argument("--limit",  metavar="N", type=int, default=None,
                   help="Cap total firm records read.")
    p.add_argument("--debug",  action="store_true", help="Enable DEBUG-level logging.")
    return p


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.stats:
        run_stats()
        return 0

    input_path = Path(args.input)

    if args.dry_run:
        run_ingest(input_path, dry_run=True, limit=args.limit)
        return 0

    if args.reingest_firm:
        run_ingest(input_path, reingest_firm=args.reingest_firm)
        return 0

    if args.ingest:
        run_ingest(input_path, limit=args.limit)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

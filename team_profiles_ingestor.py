"""
team_profiles_ingestor.py — AltBots RAG Pipeline: Team Profiles Ingestor

Reads /data/team_profiles_raw.jsonl (output of team_scraper.py), chunks each
person record, embeds via OpenAI text-embedding-3-small, and upserts into the
Qdrant `team_profiles` collection.

Embedding model and Qdrant connection patterns are taken verbatim from
src/core/identity.py to stay consistent with the rest of the pipeline.

Usage
-----
  python team_profiles_ingestor.py --ingest                     # process all unprocessed
  python team_profiles_ingestor.py --ingest --input /path/file  # explicit input file
  python team_profiles_ingestor.py --reingest-firm "Acme Capital"
  python team_profiles_ingestor.py --stats
  python team_profiles_ingestor.py --dry-run
  python team_profiles_ingestor.py --dry-run --limit 5          # first N records only

Environment variables (same as rest of pipeline)
-------------------------------------------------
  OPENAI_API_KEY    — OpenAI embeddings
  QDRANT_HOST       — Qdrant host (default: localhost)
  QDRANT_PORT       — Qdrant port (default: 6333)
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

# ── Load .env ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env", override=True)

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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingestor")

# ── Config — mirrors identity.py exactly ─────────────────────────────────────
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", "6333"))
EMBED_MODEL    = "text-embedding-3-small"    # same as identity.py
VECTOR_SIZE    = 1536                        # same as identity.py

COLLECTION_NAME = "team_profiles"

# Chunking parameters (token approximation: 1 token ≈ 4 chars for English)
_CHARS_PER_TOKEN  = 4
MAX_BIO_TOKENS    = 800
CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS    = 100
MIN_BIO_CHARS     = 50   # below this, use title as content body

# Batch upsert settings
BATCH_SIZE        = 50
INTER_BATCH_DELAY = 1.0  # seconds between batches

# File paths
INPUT_JSONL     = Path("/data/team_profiles_raw.jsonl")
STATE_FILE      = Path("/data/team_ingestor_state.json")
FAILED_JSONL    = Path("/logs/ingestor_failed.jsonl")


# ── Qdrant helpers (verbatim pattern from identity.py) ────────────────────────

def _get_qdrant() -> Optional["QdrantClient"]:
    """Return a connected QdrantClient with the team_profiles collection ensured."""
    if not QDRANT_AVAILABLE:
        logger.error("qdrant-client not installed. Run: pip install qdrant-client")
        return None
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        _ensure_collection(client)
        return client
    except Exception as exc:
        logger.error("Qdrant connection failed: %s", exc)
        return None


def _ensure_collection(client: "QdrantClient") -> None:
    """Create team_profiles collection if it does not exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", COLLECTION_NAME)


# ── Embedding (verbatim pattern from identity.py) ─────────────────────────────

def _embed(text: str) -> Optional[list[float]]:
    """Embed text with OpenAI text-embedding-3-small (1536-dim)."""
    if not OPENAI_AVAILABLE:
        logger.error("openai not installed. Run: pip install openai")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set — cannot embed")
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return None


# ── Chunking ──────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 chars (sufficient for chunking decisions)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _build_header(firm_name: str, name: str, title: str) -> str:
    return f"[FIRM: {firm_name}] [NAME: {name}] [TITLE: {title}]"


def chunk_person(record: dict) -> list[dict]:
    """
    Produce one or more chunk dicts from a person record.

    Text block format:
      "[FIRM: {firm_name}] [NAME: {name}] [TITLE: {title}] {body}"

    Where body is:
      - The bio if len(bio) >= MIN_BIO_CHARS
      - The title repeated as body if bio is absent / too short

    Long bios (> MAX_BIO_TOKENS) are split into overlapping segments of
    CHUNK_SIZE_TOKENS with OVERLAP_TOKENS overlap; each segment keeps the header.

    Returns a list of chunk dicts:
      {
        "text":        str,   # the full text block for this chunk
        "chunk_index": int,
        "total_chunks": int,
        "metadata":    dict,  # Qdrant payload fields
      }
    """
    firm_name = record.get("firm_name", "").strip()
    name      = record.get("name", "").strip()
    title     = record.get("title", "").strip()
    bio       = record.get("bio", "").strip()

    header = _build_header(firm_name, name, title)

    # Decide body text
    if len(bio) >= MIN_BIO_CHARS:
        body = bio
    else:
        # Bio absent or too short — use title as content body so the record
        # is still searchable; bio field may be empty string here.
        body = title if title else name

    # Base metadata (chunk_index / total_chunks filled in after splitting)
    base_meta = {
        "firm_name":     firm_name,
        "person_name":   name,
        "title":         title,
        "team_page_url": record.get("team_page_url", ""),
        "linkedin_url":  record.get("linkedin_url"),
        "email":         record.get("email"),
        "scraped_at":    record.get("scraped_at", ""),
        "source":        "team_page",
    }

    # Decide whether to split
    bio_tokens = _approx_tokens(body)
    if bio_tokens <= MAX_BIO_TOKENS:
        # Single chunk
        text = f"{header} {body}"
        meta = {**base_meta, "chunk_index": 0, "total_chunks": 1}
        return [{"text": text, "chunk_index": 0, "total_chunks": 1, "metadata": meta}]

    # Split into overlapping char-based windows
    chunk_chars   = CHUNK_SIZE_TOKENS * _CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS    * _CHARS_PER_TOKEN

    segments: list[str] = []
    start = 0
    while start < len(body):
        end = start + chunk_chars
        segments.append(body[start:end])
        if end >= len(body):
            break
        start = end - overlap_chars

    total = len(segments)
    chunks = []
    for idx, seg in enumerate(segments):
        text = f"{header} {seg}"
        meta = {**base_meta, "chunk_index": idx, "total_chunks": total}
        chunks.append({"text": text, "chunk_index": idx, "total_chunks": total, "metadata": meta})

    return chunks


# ── Deterministic point IDs (matches identity.py UUID5 pattern) ───────────────

def _point_id(firm_name: str, person_name: str, chunk_index: int) -> str:
    """
    Deterministic UUID5 from firm_name + person_name + chunk_index.
    Idempotent: the same person always maps to the same Qdrant point IDs,
    so upsert is naturally idempotent.
    """
    key = f"team_profile:{firm_name}:{person_name}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# ── Deduplication ─────────────────────────────────────────────────────────────

def _should_skip(
    client: "QdrantClient",
    firm_name: str,
    person_name: str,
    new_scraped_at: str,
) -> bool:
    """
    Check chunk_index=0 for an existing point with this firm+person.
    Returns True (skip) if the existing scraped_at is >= new_scraped_at.
    Returns False (ingest) if not found or existing record is older.
    """
    point_id = _point_id(firm_name, person_name, 0)
    try:
        results = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
    except Exception as exc:
        logger.warning("Dedup check failed for %s / %s: %s — will ingest", firm_name, person_name, exc)
        return False

    if not results:
        return False  # not found → ingest

    existing_scraped_at = results[0].payload.get("scraped_at", "")
    if existing_scraped_at >= new_scraped_at:
        return True   # same or older → skip
    return False      # newer data → replace


# ── Batch upsert ──────────────────────────────────────────────────────────────

def _upsert_batch(
    client: "QdrantClient",
    points: list["PointStruct"],
    dry_run: bool,
) -> tuple[int, list["PointStruct"]]:
    """
    Upsert a batch of points. Retries once on failure.
    Returns (upserted_count, failed_points).
    """
    if dry_run:
        return len(points), []

    for attempt in range(1, 3):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            return len(points), []
        except Exception as exc:
            if attempt == 1:
                logger.warning("Batch upsert failed (attempt 1): %s — retrying in 3s", exc)
                time.sleep(3)
            else:
                logger.error("Batch upsert failed (attempt 2): %s — logging to failed file", exc)
                return 0, points

    return 0, []


# ── Failed records log ────────────────────────────────────────────────────────

def _log_failed(failed_points: list["PointStruct"]) -> None:
    """Append failed PointStruct payloads to the failed JSONL log."""
    if not failed_points:
        return
    FAILED_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_JSONL.open("a", encoding="utf-8") as fh:
        for pt in failed_points:
            entry = {
                "point_id": str(pt.id),
                "payload":  pt.payload,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── State file ────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load state from disk; return a fresh state if file absent or corrupt."""
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
                # Ensure expected keys exist (forward-compatible)
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
    """Read records from a JSONL file. Skips blank lines and bad JSON."""
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

    Parameters
    ----------
    input_path:     JSONL file to read.
    reingest_firm:  If set, force-reingest only records for this firm
                    (bypasses processed_firms state check and scraped_at dedup).
    dry_run:        Parse + chunk everything; skip embedding and Qdrant writes.
    limit:          Cap total records read (useful for dry-run previews).

    Returns
    -------
    dict with run statistics.
    """
    state = _load_state()
    processed_firms: set[str] = set(state["processed_firms"])

    stats = {
        "records_read":              0,
        "records_skipped_duplicate": 0,
        "records_ingested":          0,
        "points_upserted":           0,
        "failed":                    0,
        "run_timestamp":             datetime.now(timezone.utc).isoformat(),
    }

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    qdrant = None
    if not dry_run:
        qdrant = _get_qdrant()
        if qdrant is None:
            logger.error("Cannot connect to Qdrant — aborting. Use --dry-run to test without Qdrant.")
            return stats

    # ── Read JSONL ────────────────────────────────────────────────────────────
    all_records = _read_jsonl(input_path, limit=limit)
    stats["records_read"] = len(all_records)
    logger.info("Read %d record(s) from %s", len(all_records), input_path)

    # ── Filter by processed_firms (skip already-done firms unless reingest) ───
    if reingest_firm:
        records_to_process = [
            r for r in all_records
            if r.get("firm_name", "").strip().lower() == reingest_firm.strip().lower()
        ]
        logger.info(
            "--reingest-firm %r: %d record(s) matched", reingest_firm, len(records_to_process)
        )
    else:
        records_to_process = [
            r for r in all_records
            if r.get("firm_name", "").strip() not in processed_firms
        ]
        skipped_by_state = len(all_records) - len(records_to_process)
        if skipped_by_state:
            logger.info(
                "%d record(s) from already-processed firms skipped "
                "(use --reingest-firm to force)", skipped_by_state
            )

    if not records_to_process:
        logger.info("Nothing to process.")
        _print_summary(stats)
        return stats

    # ── Build points ──────────────────────────────────────────────────────────
    points_batch:   list["PointStruct"] = []
    failed_points:  list["PointStruct"] = []
    firms_seen:     set[str]            = set()

    for rec in records_to_process:
        firm_name   = rec.get("firm_name", "").strip()
        person_name = rec.get("name", "").strip()
        scraped_at  = rec.get("scraped_at", "")

        if not firm_name or not person_name:
            logger.debug("Skipping record with missing firm_name or name: %r", rec)
            continue

        firms_seen.add(firm_name)

        # ── Dedup check (skip if not force-reingest) ──────────────────────────
        if not reingest_firm and qdrant:
            if _should_skip(qdrant, firm_name, person_name, scraped_at):
                stats["records_skipped_duplicate"] += 1
                logger.debug("SKIP duplicate: %s / %s", firm_name, person_name)
                continue

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunks = chunk_person(rec)

        if dry_run:
            # In dry-run mode: print the chunk and continue
            stats["records_ingested"] += 1
            stats["points_upserted"] += len(chunks)
            _print_dry_run_chunks(firm_name, person_name, chunks)
            continue

        # ── Embed + build PointStructs ────────────────────────────────────────
        for chunk in chunks:
            vector = _embed(chunk["text"])
            if vector is None:
                logger.warning(
                    "Embedding failed for %s / %s chunk %d — skipping chunk",
                    firm_name, person_name, chunk["chunk_index"]
                )
                stats["failed"] += 1
                continue

            point_id = _point_id(firm_name, person_name, chunk["chunk_index"])
            points_batch.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk["metadata"],
                )
            )

            # ── Flush batch when full ─────────────────────────────────────────
            if len(points_batch) >= BATCH_SIZE:
                n_ok, failed = _upsert_batch(qdrant, points_batch, dry_run)
                stats["points_upserted"] += n_ok
                stats["failed"]          += len(failed)
                _log_failed(failed)
                failed_points.extend(failed)
                points_batch = []
                logger.info("Upserted batch of %d point(s)", n_ok)
                time.sleep(INTER_BATCH_DELAY)

        stats["records_ingested"] += 1

    # ── Flush remaining points ────────────────────────────────────────────────
    if points_batch and not dry_run:
        n_ok, failed = _upsert_batch(qdrant, points_batch, dry_run)
        stats["points_upserted"] += n_ok
        stats["failed"]          += len(failed)
        _log_failed(failed)
        logger.info("Upserted final batch of %d point(s)", n_ok)

    # ── Update state file ─────────────────────────────────────────────────────
    if not dry_run and not reingest_firm:
        state["processed_firms"] = sorted(processed_firms | firms_seen)
        state["total_points_upserted"] += stats["points_upserted"]
        state["last_run"] = stats["run_timestamp"]
        _save_state(state)
        logger.info("State file updated: %s", STATE_FILE)

    _print_summary(stats)
    return stats


# ── Stats command ─────────────────────────────────────────────────────────────

def run_stats() -> None:
    """Print collection statistics without ingesting anything."""
    qdrant = _get_qdrant()
    if qdrant is None:
        logger.error("Cannot connect to Qdrant")
        return

    try:
        info = qdrant.get_collection(COLLECTION_NAME)
    except Exception as exc:
        logger.error("Failed to get collection info: %s", exc)
        return

    state = _load_state()

    print(f"\n{'─' * 50}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Points     : {info.points_count:,}")
    print(f"  Vectors    : {info.vectors_count:,}")
    print(f"  Status     : {info.status}")
    print(f"  Last run   : {state['last_run'] or 'never'}")
    print(f"  Firms done : {len(state['processed_firms'])}")
    print(f"  Total pts  : {state['total_points_upserted']:,} (lifetime)")
    print(f"{'─' * 50}\n")


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_dry_run_chunks(firm_name: str, person_name: str, chunks: list[dict]) -> None:
    print(f"\n{'═' * 70}")
    print(f"  Person  : {person_name}")
    print(f"  Firm    : {firm_name}")
    print(f"  Chunks  : {len(chunks)}")
    for ch in chunks:
        print(f"\n  ── Chunk {ch['chunk_index'] + 1}/{ch['total_chunks']} "
              f"(~{_approx_tokens(ch['text'])} tokens) {'─' * 30}")
        # Show full text block, wrapping at 80 chars for readability
        text = ch["text"]
        for i in range(0, len(text), 80):
            print(f"  {text[i:i+80]}")
        print(f"\n  Payload:")
        for k, v in ch["metadata"].items():
            if v is not None and v != "":
                print(f"    {k:<18} {v}")
        point_id = _point_id(firm_name, person_name, ch["chunk_index"])
        print(f"    {'point_id':<18} {point_id}")


def _print_summary(stats: dict) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Records read              : {stats['records_read']}")
    print(f"  Records skipped (dup)     : {stats['records_skipped_duplicate']}")
    print(f"  Records ingested          : {stats['records_ingested']}")
    print(f"  Points upserted to Qdrant : {stats['points_upserted']}")
    print(f"  Failed                    : {stats['failed']}"
          + (f"  (see {FAILED_JSONL})" if stats["failed"] else ""))
    print(f"{'─' * 50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AltBots team profiles ingestor — push team_scraper output into Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python team_profiles_ingestor.py --ingest
  python team_profiles_ingestor.py --ingest --input /data/team_profiles_raw.jsonl
  python team_profiles_ingestor.py --reingest-firm "Bridgewater Associates"
  python team_profiles_ingestor.py --stats
  python team_profiles_ingestor.py --dry-run --limit 5
        """,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ingest",        action="store_true", help="Ingest all unprocessed records.")
    mode.add_argument("--reingest-firm", metavar="FIRM",      help="Force re-ingest all records for this firm name.")
    mode.add_argument("--stats",         action="store_true", help="Print collection stats; no ingestion.")
    mode.add_argument("--dry-run",       action="store_true", help="Parse + chunk; skip embedding and Qdrant writes.")

    p.add_argument(
        "--input", metavar="FILE", default=str(INPUT_JSONL),
        help=f"JSONL input file (default: {INPUT_JSONL})",
    )
    p.add_argument(
        "--limit", metavar="N", type=int, default=None,
        help="Cap total records read (useful with --dry-run).",
    )
    p.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
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

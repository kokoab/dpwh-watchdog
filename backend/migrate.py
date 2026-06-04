# migrate.py
"""
ChromaDB → PostgreSQL + pgvector migration.
- Zero re-embedding: vectors are extracted directly from ChromaDB.
- Cursor-based pagination: no offset penalty on large collections.
- Resume-safe: checkpoints last processed ID to disk on every batch.
- Limit-aware: set MIGRATE_LIMIT to test with a subset, None for all.
- Idempotent: safe to re-run; uses upserts throughout.
"""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import chromadb
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ── Configuration ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "dpwh_contracts"
BATCH_SIZE = 256
LOG_EVERY = 10

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  MIGRATION LIMIT                                                         │
# │  Set to an integer to process only that many contracts (e.g. 5000).     │
# │  Set to None to migrate everything.                                      │
# └─────────────────────────────────────────────────────────────────────────┘
MIGRATE_LIMIT: int | None = 5000

CHECKPOINT_FILE = "./migration_checkpoint.txt"

PG_DSN = os.environ.get("PG_DSN") or (
    f"host={os.environ['PG_HOST']} "
    f"port={os.environ.get('PG_PORT', 5432)} "
    f"dbname={os.environ['PG_DB']} "
    f"user={os.environ['PG_USER']} "
    f"password={os.environ['PG_PASSWORD']}"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ── Checkpoint helpers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def load_checkpoint() -> str | None:
    """Return the last successfully processed contract_id, or None."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            val = f.read().strip()
            if val:
                log.info(f"Resuming from checkpoint: '{val}'")
                return val
    return None


def save_checkpoint(last_id: str):
    """Persist the last successfully processed contract_id."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(last_id)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Checkpoint cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# ── Data class ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MigratedContract:
    contract_id: str
    chunk_text: str
    embedding: list[float]
    metadata: dict
    bidders: list[dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ── ChromaDB helpers ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def open_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def _extract_bidders(raw_json: dict) -> list[dict]:
    return [
        {
            "pcab_id": b.get("pcabId"),
            "name": b.get("name"),
            "is_winner": bool(b.get("isWinner")),
        }
        for b in raw_json.get("bidders", [])
    ]


def iter_chroma_batches(
    collection,
    batch_size: int,
    resume_after: str | None = None,
    limit: int | None = None,
) -> Iterator[list[MigratedContract]]:
    """
    Cursor-based pagination over ChromaDB.

    Instead of offset=N (which forces Chroma to scan N rows every call),
    we fetch ALL ids once (cheap — ids only, no vectors), sort them, find
    our cursor position, then slice. Vectors are fetched only for the
    current page using `ids=` which is an O(1) lookup in Chroma.

    Args:
        resume_after:  contract_id of the last successfully processed record.
                       Iteration starts from the record AFTER this id.
        limit:         Maximum total contracts to yield across all batches.
                       None means no limit.
    """
    log.info("Fetching all IDs from ChromaDB for cursor pagination...")
    all_ids: list[str] = sorted(collection.get(include=[])["ids"])
    total_in_chroma = len(all_ids)
    log.info(f"ChromaDB collection '{COLLECTION_NAME}': {total_in_chroma:,} documents.")

    # ── Find resume cursor ────────────────────────────────────────────────────
    start_index = 0
    if resume_after:
        try:
            start_index = all_ids.index(resume_after) + 1
            log.info(f"Skipping {start_index:,} already-processed records.")
        except ValueError:
            log.warning(
                f"Checkpoint ID '{resume_after}' not found in collection. Starting from beginning."
            )

    # ── Apply limit ───────────────────────────────────────────────────────────
    ids_to_process = all_ids[start_index:]
    if limit is not None:
        ids_to_process = ids_to_process[:limit]
        log.info(
            f"MIGRATE_LIMIT={limit:,} → will process {len(ids_to_process):,} contracts."
        )
    else:
        log.info(f"No limit → will process {len(ids_to_process):,} contracts.")

    # ── Page through using id slices (no offset penalty) ─────────────────────
    for page_start in range(0, len(ids_to_process), batch_size):
        page_ids = ids_to_process[page_start : page_start + batch_size]

        result = collection.get(
            ids=page_ids,
            include=["documents", "embeddings", "metadatas"],
        )

        batch: list[MigratedContract] = []
        for cid, doc, emb, meta in zip(
            result["ids"],
            result["documents"],
            result["embeddings"],
            result["metadatas"],
        ):
            raw_json_str = meta.get("raw_json")
            raw = json.loads(raw_json_str) if isinstance(raw_json_str, str) else {}
            batch.append(
                MigratedContract(
                    contract_id=cid,
                    chunk_text=doc,
                    embedding=emb,
                    metadata=meta,
                    bidders=_extract_bidders(raw),
                )
            )

        yield batch


# ══════════════════════════════════════════════════════════════════════════════
# ── Postgres helpers ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def safe_date(val):
    return val if isinstance(val, str) and val.strip() else None


def safe_int(val):
    try:
        return int(val)
    except:
        return None


def safe_float(val):
    try:
        return float(val)
    except:
        return None


def upsert_batch(conn, batch: list[MigratedContract]) -> dict:
    counts = {"inserted": 0, "updated": 0, "skipped": 0}

    with conn.cursor() as cur:
        cids = [c.contract_id for c in batch]
        cur.execute(
            "SELECT contract_id, has_detail FROM contracts WHERE contract_id = ANY(%s)",
            (cids,),
        )
        existing = {row[0]: row[1] for row in cur.fetchall()}

        contracts_data = []
        embeddings_data = []
        bidder_deletes = []
        bidder_rows = []

        for c in batch:
            meta = c.metadata
            has_detail = bool(meta.get("hasDetail", False))
            already_has = existing.get(c.contract_id)

            if already_has and not has_detail:
                counts["skipped"] += 1
                continue

            if c.contract_id in existing:
                counts["updated"] += 1
            else:
                counts["inserted"] += 1

            contracts_data.append(
                (
                    c.contract_id,
                    meta.get("description"),
                    meta.get("category"),
                    meta.get("status"),
                    safe_float(meta.get("budget")),
                    safe_float(meta.get("amountPaid")),
                    safe_int(meta.get("progress")),
                    meta.get("region"),
                    meta.get("province"),
                    safe_float(meta.get("latitude")),
                    safe_float(meta.get("longitude")),
                    (meta.get("contractor") or "")[:500],
                    safe_date(meta.get("startDate")),
                    safe_date(meta.get("completionDate")),
                    safe_int(meta.get("infraYear")),
                    meta.get("programName"),
                    meta.get("sourceOfFunds"),
                    has_detail,
                    json.dumps({}),
                )
            )

            embeddings_data.append(
                (
                    c.contract_id,
                    c.chunk_text,
                    c.embedding,
                )
            )

            if c.bidders:
                bidder_deletes.append(c.contract_id)
                for b in c.bidders:
                    bidder_rows.append(
                        (
                            c.contract_id,
                            b["pcab_id"],
                            b["name"],
                            b["is_winner"],
                        )
                    )

        # ── 1. Upsert contracts ───────────────────────────────────────────────
        if contracts_data:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO contracts (
                    contract_id, description, category, status,
                    budget, amount_paid, progress,
                    region, province, latitude, longitude,
                    contractor, start_date, completion_date,
                    infra_year, program_name, source_of_funds,
                    has_detail, raw_json
                ) VALUES %s
                ON CONFLICT (contract_id) DO UPDATE SET
                    description     = EXCLUDED.description,
                    category        = EXCLUDED.category,
                    status          = EXCLUDED.status,
                    budget          = EXCLUDED.budget,
                    amount_paid     = EXCLUDED.amount_paid,
                    progress        = EXCLUDED.progress,
                    region          = EXCLUDED.region,
                    province        = EXCLUDED.province,
                    latitude        = EXCLUDED.latitude,
                    longitude       = EXCLUDED.longitude,
                    contractor      = EXCLUDED.contractor,
                    start_date      = EXCLUDED.start_date,
                    completion_date = EXCLUDED.completion_date,
                    infra_year      = EXCLUDED.infra_year,
                    program_name    = EXCLUDED.program_name,
                    source_of_funds = EXCLUDED.source_of_funds,
                    has_detail      = EXCLUDED.has_detail,
                    raw_json        = EXCLUDED.raw_json
            """,
                contracts_data,
            )

        # ── 2. Bidders: delete-then-insert ────────────────────────────────────
        if bidder_deletes:
            cur.execute(
                "DELETE FROM contract_bidders WHERE contract_id = ANY(%s)",
                (bidder_deletes,),
            )
        if bidder_rows:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO contract_bidders (contract_id, pcab_id, name, is_winner)
                VALUES %s
            """,
                bidder_rows,
            )

        # ── 3. Upsert embeddings ──────────────────────────────────────────────
        if embeddings_data:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO contract_embeddings (contract_id, chunk_text, embedding)
                VALUES %s
                ON CONFLICT (contract_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding  = EXCLUDED.embedding
            """,
                embeddings_data,
            )

    conn.commit()
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# ── Main ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def run_migration():
    resume_after = load_checkpoint()

    log.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False

    log.info("Opening ChromaDB collection...")
    collection = open_chroma_collection()

    totals = {"inserted": 0, "updated": 0, "skipped": 0, "failed_batches": 0}
    batch_num = 0
    last_id = resume_after
    start = time.time()

    try:
        batch_iter = iter_chroma_batches(
            collection,
            batch_size=BATCH_SIZE,
            resume_after=resume_after,
            limit=MIGRATE_LIMIT,
        )

        for batch in batch_iter:
            batch_num += 1
            try:
                counts = upsert_batch(conn, batch)
                for k in ("inserted", "updated", "skipped"):
                    totals[k] += counts[k]

                # Checkpoint the last ID in this batch
                last_id = batch[-1].contract_id
                save_checkpoint(last_id)

                if batch_num % LOG_EVERY == 0:
                    processed = batch_num * BATCH_SIZE
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else 0
                    log.info(
                        f"Batch {batch_num:>5} | "
                        f"~{processed:>7,} processed | "
                        f"+{counts['inserted']} new | "
                        f"{rate:.0f} rows/s | "
                        f"last_id={last_id}"
                    )

            except Exception as e:
                conn.rollback()
                log.error(f"Batch {batch_num} failed — {e}. Retrying after 2s...")
                totals["failed_batches"] += 1
                time.sleep(2)
                # Checkpoint is NOT updated — this batch will be retried on resume
                continue

            finally:
                del batch
                gc.collect()

    finally:
        elapsed = time.time() - start
        log.info("=" * 60)
        log.info(f"Migration finished in {elapsed:.1f}s")
        log.info(f"  Inserted:       {totals['inserted']:>8,}")
        log.info(f"  Updated:        {totals['updated']:>8,}")
        log.info(f"  Skipped:        {totals['skipped']:>8,}")
        log.info(f"  Failed batches: {totals['failed_batches']:>8,}")

        if totals["failed_batches"] == 0 and MIGRATE_LIMIT is None:
            clear_checkpoint()
            log.info("Full migration succeeded — checkpoint cleared.")
        else:
            log.info(f"Checkpoint preserved at: '{last_id}'")

        conn.close()


if __name__ == "__main__":
    run_migration()

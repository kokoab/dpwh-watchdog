#!/usr/bin/env python3
"""
ingest_pipeline.py

A unified, high-performance ingestion engine for DPWH Contracts into PostgreSQL.
Integrates:
- Database schema initialization with high-performance pgvector HNSW indexing.
- Memory-safe directory streaming using Python generators.
- Exact contract-level processing limit matching target configurations.
- Fast indexed database checkpointing to bypass pre-existing records.
- Text chunk generation matching your custom RAG context layout.
- Microservice calls to your Dockerized FastAPI sentence-transformers server.
- Recursive binary splitting (divide-and-conquer) to isolate toxic records.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

# .env
load_dotenv()

# CONSTANTS
DATA_DIR: str = os.environ.get("DATA_DIR", "./data")
BATCH_SIZE: int = 128
EMBED_URL: str = os.environ.get("EMBED_URL", "http://127.0.0.1:8000/embed")
POISON_PILL_LOG: str = "./poison_pills.log"

# CHANGE LIMIT HERE DEPENDING ON HOW MANY CONTRACTS YOU WANT TO INGEST
PROCESS_LIMIT: Optional[int] = 5000

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)

# Configure Logging Streams
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_pipeline")


# Database schema setup
def initialize_database_schema(conn) -> None:
    """
    Ensures that the required database tables, extensions, and the high-performance
    HNSW vector index are completely built before starting ingestion.
    """
    logger.info("Verifying database extensions and tables schema...")
    with conn.cursor() as cur:
        # import pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # core contracts table (important ones for faster lookup)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                contract_id VARCHAR(50) PRIMARY KEY,
                description TEXT,
                category VARCHAR(100),
                status VARCHAR(50),
                budget NUMERIC(15, 2),
                amount_paid NUMERIC(15, 2),
                award_amount NUMERIC(15, 2),
                progress INT,
                region VARCHAR(100),
                province VARCHAR(100),
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                contractor TEXT,
                advertisement_date DATE,
                expiry_date DATE,  
                bid_submission_deadline DATE, 
                start_date DATE,
                completion_date DATE,
                infra_year INT,
                program_name TEXT,
                source_of_funds TEXT,
                has_detail BOOLEAN NOT NULL DEFAULT false,
                raw_json JSONB,
                fts_vector tsvector
            );
        """)

        # setweight based on most important ones
        cur.execute("""
            UPDATE contracts
            SET fts_vector =
                setweight(to_tsvector('english', COALESCE(description, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(contractor, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(category, '')), 'B') ||
                setweight(to_tsvector('english', COALESCE(program_name, '')), 'B') ||
                setweight(to_tsvector('english', COALESCE(province, '')), 'B') ||
                setweight(to_tsvector('english', COALESCE(region, '')), 'B');
        """)

        # use GIN infex so text search is fast (sort of a dictionary hashmap)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_contracts_fts
            ON contracts USING GIN(fts_vector);
        """)

        # to keep it updated on update/insert
        cur.execute("""
            CREATE OR REPLACE FUNCTION contracts_fts_update() RETURNS trigger AS $$
            BEGIN
                NEW.fts_vector :=
                    setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(NEW.contractor, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(NEW.category, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.program_name, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.province, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.region, '')), 'B');
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)

        cur.execute("""
            DROP TRIGGER IF EXISTS contracts_fts_trigger ON contracts;
            CREATE TRIGGER contracts_fts_trigger
            BEFORE INSERT OR UPDATE ON contracts
            FOR EACH ROW EXECUTE FUNCTION contracts_fts_update();
        """)

        # relational child table for contract_bidders
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contract_bidders (
                id SERIAL PRIMARY KEY, 
                contract_id VARCHAR(50) REFERENCES contracts(contract_id) ON DELETE CASCADE,
                pcab_id VARCHAR(50),
                name TEXT,
                is_winner BOOLEAN
            );
        """)

        # vector embeddings table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contract_embeddings (
                id SERIAL PRIMARY KEY,
                contract_id VARCHAR(50) NOT NULL UNIQUE REFERENCES contracts(contract_id) ON DELETE CASCADE,
                chunk_text TEXT,
                embedding vector(384)
            );
        """)

        # 4. standard indexes for fast search
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_contracts_budget ON contracts(budget);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_contracts_region ON contracts(region);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_contracts_status ON contracts(status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_contracts_category ON contracts(category);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_contracts_infra_year ON contracts(infra_year);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_bidders_contract_id ON contract_bidders(contract_id);"
        )

        # 5. pgvector HNSW for vector embeddings index
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
            ON contract_embeddings 
            USING hnsw (embedding vector_cosine_ops);
        """)

    conn.commit()
    logger.info(
        "Database schema checks complete. Relational and vector tables are ready."
    )


def get_json_file_paths(directory: str) -> Iterator[Path]:
    """
    Yields paths to JSON files iteratively to preserve system RAM.
    """
    path_obj = Path(directory)
    if not path_obj.exists():
        logger.error(f"Target system data directory does not exist: {directory}")
        return
    for path in path_obj.glob("**/*.json"):
        yield path


def compile_rag_chunk_text(contract: Dict[str, Any], contract_id: str) -> str:
    """
    Compiles text representations matching structural RAG expectations.
    """
    location: Dict[str, Any] = contract.get("location", {}) or {}
    procurement: Dict[str, Any] = contract.get("procurement", {}) or {}
    bidders: List[Dict[str, Any]] = contract.get("bidders", []) or []
    components: List[Dict[str, Any]] = contract.get("components", []) or []

    lines: List[str] = [
        "passage:",
        f"Contract ID: {contract_id}",
        f"Description: {components.get('description', 'N/A')}",
        f"Category: {contract.get('category', 'N/A')}",
        f"Status: {contract.get('status', 'N/A')}",
        f"Region: {location.get('region', 'N/A')}",
        f"Province: {components.get('province', 'N/A')}",
        f"Contractor: {contract.get('contractor', 'N/A')}",
        f"Infrastructure Year: {contract.get('infraYear', 'N/A')}",
        f"Budget: PHP {contract.get('budget', 0.0):,.2f}",
        f"Amount Paid: PHP {contract.get('amountPaid', 0.0):,.2f}",
        f"Progress: {contract.get('progress', 0)}%",
        f"Program Name: {contract.get('programName', 'N/A')}",
        f"Source of Funds: {contract.get('sourceOfFunds', 'N/A')}",
        f"Approved Budget for Contract (ABC): PHP {procurement.get('abc', 'N/A')}",
        f"Award Amount: PHP {procurement.get('awardAmount', 'N/A')}",
        f"Advertisement Date: {procurement.get('advertisementDate', 'N/A')}",
        f"Bid Submission Deadline: {procurement.get('bidSubmissionDeadline', 'N/A')}",
        f"Date of Award: {procurement.get('dateOfAward', 'N/A')}",
        f"Start Date: {contract.get('startDate', 'N/A')}",
        f"Completion Date: {contract.get('completionDate', 'N/A')}",
        f"Expiry Date: {contract.get('expiryDate', 'N/A')}",
        f"Funding Instrument: {procurement.get('fundingInstrument', 'N/A')}",
    ]

    if bidders:
        lines.append(f"Number of Bidders: {len(bidders)}")
        for b in bidders:
            tag = " [WINNER]" if b.get("isWinner") else ""
            lines.append(f"  Bidder: {b.get('name', 'N/A')}{tag}")

    return "\n".join(lines)


def extract_contract_payload(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses JSON files and standardizes data models for ingestion.
    Supports single contracts or array dump file collections.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.warning(f"Skipping corrupt JSON structure at: {file_path}. Error: {e}")
        return []

    inner = raw_data.get("data", {}) if isinstance(raw_data, dict) else {}

    if isinstance(inner, dict) and isinstance(inner.get("data"), list):
        contract_list = inner["data"]
    elif isinstance(inner, dict) and inner.get("contractId"):
        contract_list = [inner]
    elif isinstance(raw_data, dict) and raw_data.get("contractId"):
        contract_list = [raw_data]
    else:
        contract_list = []

    parsed_records: List[Dict[str, Any]] = []
    for item in contract_list:
        contract_id = item.get("contractId")
        if not contract_id:
            continue

        location: Dict[str, Any] = item.get("location", {}) or {}
        coordinates: Dict[str, Any] = location.get("coordinates", {}) or {}

        # Safely parse structural data model fields
        infra_year_raw = item.get("infraYear")
        infra_year_val: Optional[int] = None
        if infra_year_raw is not None:
            try:
                infra_year_val = int(float(infra_year_raw))
            except (ValueError, TypeError):
                infra_year_val = None

        contract_row: Tuple[Any, ...] = (
            str(contract_id),
            item.get("description"),
            item.get("category"),
            item.get("status"),
            float(item.get("budget") or 0.0),
            float(item.get("amountPaid") or 0.0),
            int(item.get("progress") or 0),
            location.get("region"),
            location.get("province"),
            coordinates.get("latitude")
            if coordinates.get("latitude") is not None
            else None,
            coordinates.get("longitude")
            if coordinates.get("longitude") is not None
            else None,
            item.get("contractor"),
            item.get("startDate") if item.get("startDate") else None,
            item.get("completionDate") if item.get("completionDate") else None,
            infra_year_val,
            item.get("programName"),
            item.get("sourceOfFunds"),
            bool(item.get("hasDetail", False)),
            json.dumps(item),
        )

        # Parse child array bidders
        bidders_rows: List[Tuple[str, Optional[str], Optional[str], bool]] = []
        for bidder in item.get("bidders", []) or []:
            bidders_rows.append(
                (
                    str(contract_id),
                    bidder.get("pcabId") or bidder.get("pcab_id"),
                    bidder.get("name"),
                    bool(bidder.get("isWinner", False)),
                )
            )

        chunk_text: str = compile_rag_chunk_text(item, str(contract_id))

        parsed_records.append(
            {
                "contract_id": str(contract_id),
                "contract_row": contract_row,
                "bidders_rows": bidders_rows,
                "chunk_text": chunk_text,
            }
        )

    return parsed_records


def retrieve_vector_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Sends batch requests to your local Dockerized FastAPI embedding server.
    """
    if not texts:
        return []
    response = requests.post(EMBED_URL, json={"inputs": texts}, timeout=120)
    response.raise_for_status()

    response_data = response.json()
    if isinstance(response_data, dict) and "embedding" in response_data:
        return response_data["embedding"]
    elif isinstance(response_data, list):
        return response_data
    else:
        raise ValueError(
            f"Unexpected response structure from API server: {response_data}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Core Database Bulk Execution Engine ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def filter_existing_contracts(
    conn, records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Filters out records that already exist in the database based on contract_id.
    Ensures we have absolute, instant checkpoint tracking without text files.
    """
    if not records:
        return []

    contract_ids: List[str] = [r["contract_id"] for r in records]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT contract_id FROM contracts WHERE contract_id = ANY(%s);",
            (contract_ids,),
        )
        existing_ids = {row[0] for row in cur.fetchall()}

    return [r for r in records if r["contract_id"] not in existing_ids]


def execute_db_bulk_insert(conn, batch: List[Dict[str, Any]]) -> None:
    """
    Performs atomic transactions into PostgreSQL across all target tables.
    """
    if not batch:
        return

    contracts_data: List[Tuple[Any, ...]] = [item["contract_row"] for item in batch]

    # Flatten child bidders collection array rows
    bidders_data: List[Tuple[str, Optional[str], Optional[str], bool]] = []
    for item in batch:
        bidders_data.extend(item["bidders_rows"])

    embeddings_data: List[Tuple[str, str, List[float]]] = [
        (item["contract_id"], item["chunk_text"], item["embedding"]) for item in batch
    ]

    with conn.cursor() as cur:
        # 1. Populate Core Relational Structural Profiles
        contracts_query = """
            INSERT INTO contracts (
                contract_id, description, category, status, budget, amount_paid,
                progress, region, province, latitude, longitude, contractor,
                start_date, completion_date, infra_year, program_name, source_of_funds,
                has_detail, raw_json
            ) VALUES %s
            ON CONFLICT (contract_id) DO UPDATE SET
                description = EXCLUDED.description,
                status = EXCLUDED.status,
                progress = EXCLUDED.progress,
                amount_paid = EXCLUDED.amount_paid,
                has_detail = EXCLUDED.has_detail,
                raw_json = EXCLUDED.raw_json;
        """
        psycopg2.extras.execute_values(cur, contracts_query, contracts_data)

        # 2. Clear out any existing bidders for these contracts to preserve absolute integrity
        contract_ids: List[str] = [item["contract_id"] for item in batch]
        cur.execute(
            "DELETE FROM contract_bidders WHERE contract_id = ANY(%s);", (contract_ids,)
        )

        # 3. Populate Associated Child Bidder Records
        if bidders_data:
            bidders_query = """
                INSERT INTO contract_bidders (contract_id, pcab_id, name, is_winner)
                VALUES %s;
            """
            psycopg2.extras.execute_values(cur, bidders_query, bidders_data)

        # 4. Ingest Calculated Vectors Natively inside pgvector Structures
        embeddings_query = """
            INSERT INTO contract_embeddings (contract_id, chunk_text, embedding)
            VALUES %s
            ON CONFLICT (contract_id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding;
        """
        psycopg2.extras.execute_values(cur, embeddings_query, embeddings_data)


# ══════════════════════════════════════════════════════════════════════════════
# ── Recursive Binary Splitting Strategy ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def process_and_retry_batch_recursive(
    conn, batch: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """
    Divide-and-Conquer Engine.
    Isolates single row failures down to a specific index while saving the rest.
    """
    if not batch:
        return 0, 0

    try:
        # Generate the text embeddings for this specific sub-segment
        texts_to_embed: List[str] = [
            item["chunk_text"] for item in batch if "embedding" not in item
        ]
        if texts_to_embed:
            vectors: List[List[float]] = retrieve_vector_embeddings(texts_to_embed)

            vector_idx = 0
            for item in batch:
                if "embedding" not in item:
                    item["embedding"] = vectors[vector_idx]
                    vector_idx += 1

        # Execute transaction block isolation parameters
        execute_db_bulk_insert(conn, batch)
        conn.commit()
        return len(batch), 0

    except Exception as batch_error:
        conn.rollback()

        # If the batch size is exactly 1 row, we have isolated the toxic file
        if len(batch) == 1:
            poison_record = batch[0]
            cid: str = poison_record["contract_id"]
            logger.error(
                f"[POISON PILL ISOLATED] Record {cid} failed insertion. Logging entry."
            )

            try:
                with open(POISON_PILL_LOG, "a", encoding="utf-8") as pf:
                    pf.write(
                        f"Timestamp: {time.time()} | ID: {cid} | Error: {str(batch_error)}\n"
                    )
            except Exception as log_err:
                logger.critical(
                    f"Failed writing to poison pill file tracking logs: {log_err}"
                )

            return 0, 1

        # Calculate midpoint boundary variables
        mid: int = len(batch) // 2
        left_branch: List[Dict[str, Any]] = batch[:mid]
        right_branch: List[Dict[str, Any]] = batch[mid:]

        logger.warning(
            f"Sub-batch transaction execution failed (Size: {len(batch)}). "
            f"Splitting matrix pipeline: Left ({len(left_branch)}) | Right ({len(right_branch)})."
        )

        success_left, failed_left = process_and_retry_batch_recursive(conn, left_branch)
        success_right, failed_right = process_and_retry_batch_recursive(
            conn, right_branch
        )

        return (success_left + success_right), (failed_left + failed_right)


# ══════════════════════════════════════════════════════════════════════════════
# ── Ingestion Driver Routine ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def execute_pipeline() -> None:
    """
    The orchestrator for data processing, checkpoint filtering, and metrics aggregation.
    """
    start_time: float = time.time()
    logger.info("Initializing Postgres Vector Integration Pipeline Engine...")
    if PROCESS_LIMIT is not None:
        logger.info(
            f"Target process limit scale applied: {PROCESS_LIMIT:,} contracts max."
        )

    try:
        conn = psycopg2.connect(PG_DSN)
    except Exception as e:
        logger.critical(f"Failed to connect to the PostgreSQL cluster instance: {e}")
        return

    # Initialize the complete schema and HNSW configurations upfront
    try:
        initialize_database_schema(conn)
    except Exception as e:
        logger.critical(f"Failed to initialize database structures: {e}")
        conn.close()
        return

    total_processed: int = 0
    total_skipped: int = 0
    total_inserted: int = 0
    total_poison_pills: int = 0

    accumulated_records: List[Dict[str, Any]] = []
    stop_scanning: bool = False

    # Process files sequentially through our streaming generator
    for file_path in get_json_file_paths(DATA_DIR):
        if stop_scanning:
            break

        records: List[Dict[str, Any]] = extract_contract_payload(file_path)
        if not records:
            continue

        for rec in records:
            # Check hard boundary limits before stacking items into the batch array
            if PROCESS_LIMIT is not None and total_processed >= PROCESS_LIMIT:
                logger.info(
                    f"Exact contract execution cap ({PROCESS_LIMIT:,}) met. Closing down discovery scan stream."
                )
                stop_scanning = True
                break

            accumulated_records.append(rec)
            total_processed += 1

        # Once the queue reaches our batch limit size, run database sync executions
        while len(accumulated_records) >= BATCH_SIZE:
            current_chunk: List[Dict[str, Any]] = accumulated_records[:BATCH_SIZE]
            accumulated_records = accumulated_records[BATCH_SIZE:]

            # 1. Deduplicate records against pre-existing items in the DB
            unprocessed_records: List[Dict[str, Any]] = filter_existing_contracts(
                conn, current_chunk
            )
            skipped_count: int = len(current_chunk) - len(unprocessed_records)
            total_skipped += skipped_count

            if not unprocessed_records:
                del current_chunk
                continue

            # 2. Process records using the recursive recovery function
            inserted, failed = process_and_retry_batch_recursive(
                conn, unprocessed_records
            )
            total_inserted += inserted
            total_poison_pills += failed

            logger.info(
                f"Metrics View | Total Evaluated: {total_processed:,} | "
                f"Synced: {total_inserted:,} | Skipped: {total_skipped:,} | "
                f"Poison Pills: {total_poison_pills:,}"
            )

            # Flush batch references out of system memory context
            del current_chunk, unprocessed_records
            gc.collect()

    # Process any remaining records left over in the queue after stream breakdown
    if accumulated_records:
        unprocessed_records = filter_existing_contracts(conn, accumulated_records)
        total_skipped += len(accumulated_records) - len(unprocessed_records)

        if unprocessed_records:
            inserted, failed = process_and_retry_batch_recursive(
                conn, unprocessed_records
            )
            total_inserted += inserted
            total_poison_pills += failed

        del accumulated_records, unprocessed_records
        gc.collect()

    conn.close()
    duration: float = time.time() - start_time

    logger.info("═" * 60)
    logger.info("INGESTION PROCESSING PIPELINE TERMINATION ANALYSIS METRICS:")
    logger.info(f" - Execution Duration Space  : {duration:.2f} seconds")
    logger.info(f" - Scanned Records Evaluated : {total_processed:,}")
    logger.info(f" - New Rows Committed        : {total_inserted:,}")
    logger.info(f" - Checkpoint Skips Found    : {total_skipped:,}")
    logger.info(f" - Core Poison Pills Logged  : {total_poison_pills:,}")
    logger.info("═" * 60)


if __name__ == "__main__":
    execute_pipeline()

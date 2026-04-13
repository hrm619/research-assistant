"""Read content from knowledge-base's ChromaDB and kb.db.

Thin access layer — no imports from the knowledge_base package.
Mirrors the pattern where KB reads RA's SQLite directly.
"""

import re
import sqlite3
from pathlib import Path

import chromadb


_HEADER_RE = re.compile(r"^\[SOURCE: .+ \| DATE: .+ \| TYPE: .+\]\n\n", re.DOTALL)


def get_kb_connection(kb_db_path: str) -> sqlite3.Connection:
    """Read-only connection to kb.db."""
    path = Path(kb_db_path)
    if not path.exists():
        raise FileNotFoundError(f"KB database not found: {kb_db_path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def get_chroma_client(chroma_persist_dir: str) -> chromadb.ClientAPI:
    """Connect to KB's ChromaDB."""
    path = Path(chroma_persist_dir)
    if not path.exists():
        raise FileNotFoundError(f"ChromaDB directory not found: {chroma_persist_dir}")
    return chromadb.PersistentClient(path=str(path))


def get_content_record(
    kb_conn: sqlite3.Connection, content_id: str
) -> dict | None:
    """Fetch a content_record row from kb.db by content_id."""
    row = kb_conn.execute(
        "SELECT * FROM content_record WHERE content_id = ?", (content_id,)
    ).fetchone()
    return dict(row) if row else None


def list_kb_content(
    kb_conn: sqlite3.Connection, domain: str
) -> list[dict]:
    """List all content_records for a domain, ordered by ingested_at desc."""
    rows = kb_conn.execute(
        "SELECT * FROM content_record WHERE domain = ? ORDER BY ingested_at DESC",
        (domain,),
    ).fetchall()
    return [dict(r) for r in rows]


def _strip_header(text: str) -> str:
    """Remove the [SOURCE: ... | DATE: ... | TYPE: ...] header from a chunk."""
    return _HEADER_RE.sub("", text)


def reconstruct_transcript(
    chroma_client: chromadb.ClientAPI,
    collection_name: str,
    content_id: str,
) -> str:
    """Reconstruct full transcript from ChromaDB chunks.

    Gets all chunks for content_id, sorts by chunk_index,
    strips metadata headers, and joins into continuous text.
    """
    collection = chroma_client.get_collection(collection_name)
    result = collection.get(
        where={"content_id": content_id},
        include=["documents", "metadatas"],
    )

    if not result["documents"]:
        raise ValueError(
            f"No chunks found for content_id={content_id} "
            f"in collection={collection_name}"
        )

    # Pair documents with their chunk_index for sorting
    pairs = list(zip(result["documents"], result["metadatas"]))
    pairs.sort(key=lambda p: p[1].get("chunk_index", 0))

    # Strip headers and join
    body_parts = [_strip_header(doc) for doc, _ in pairs]
    return "\n\n".join(part.strip() for part in body_parts if part.strip())

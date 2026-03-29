import json
import logging
import sqlite3

from research_assistant.config import Settings
from research_assistant.db import get_row, insert_row, list_rows, resolve_domain
from research_assistant.extractors.youtube import detect_source_type, extract_youtube
from research_assistant.schemas import ContentItem, Source, _now_iso, _uuid

logger = logging.getLogger(__name__)


def register_source(
    source_type: str,
    url: str,
    author: str,
    domain_id: str,
    trust_tier: str,
    conn: sqlite3.Connection,
) -> str:
    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        raise ValueError(f"Domain not found: {domain_id}")

    source = Source(
        source_type=source_type,
        url=url,
        author=author,
        domain_id=resolved,
        trust_tier=trust_tier,
    )
    insert_row(conn, "source", {
        "source_id": source.source_id,
        "source_type": source.source_type,
        "url": source.url,
        "author": source.author,
        "domain_id": source.domain_id,
        "trust_tier": source.trust_tier,
        "added_at": source.added_at,
        "active": source.active,
    })
    logger.info("Registered source %s (%s) for domain %s", source.source_id, url, resolved)
    return source.source_id


def ingest_content(
    source_id: str,
    conn: sqlite3.Connection,
    settings: Settings,
) -> ContentItem:
    source_row = get_row(conn, "source", "source_id", source_id)
    if not source_row:
        raise ValueError(f"Source not found: {source_id}")

    source_type = source_row["source_type"]
    url = source_row["url"]

    if source_type == "youtube":
        content = extract_youtube(url, source_id)
    else:
        raise NotImplementedError(
            f"Ingestion for source type '{source_type}' is not implemented in MVP. "
            "Only YouTube sources are supported."
        )

    insert_row(conn, "content_item", {
        "content_id": content.content_id,
        "source_id": content.source_id,
        "ingested_at": content.ingested_at,
        "content_type": content.content_type,
        "title": content.title,
        "author": content.author,
        "published_at": content.published_at,
        "raw_text": content.raw_text,
        "word_count": content.word_count,
        "format_metadata": content.format_metadata.model_dump_json(),
        "processing_status": content.processing_status,
        "error_detail": content.error_detail,
    })
    logger.info(
        "Ingested content %s from source %s (status: %s)",
        content.content_id, source_id, content.processing_status,
    )
    return content


def ingest_batch(
    source_file: str,
    domain_id: str,
    conn: sqlite3.Connection,
    settings: Settings,
) -> list[ContentItem]:
    with open(source_file) as f:
        sources = json.load(f)

    results = []
    for entry in sources:
        try:
            source_type = entry.get("source_type") or detect_source_type(entry["url"])
            sid = register_source(
                source_type=source_type,
                url=entry["url"],
                author=entry.get("author", "Unknown"),
                domain_id=domain_id,
                trust_tier=entry.get("trust_tier", "supplementary"),
                conn=conn,
            )
            content = ingest_content(sid, conn, settings)
            results.append(content)
        except Exception:
            logger.exception("Failed to ingest source: %s", entry.get("url", "unknown"))
    return results


def list_content(
    domain_id: str,
    conn: sqlite3.Connection,
    filters: dict | None = None,
) -> list[dict]:
    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        return []
    # Join content_item with source to filter by domain
    query = """
        SELECT ci.* FROM content_item ci
        JOIN source s ON ci.source_id = s.source_id
        WHERE s.domain_id = ?
    """
    params: list = [resolved]
    if filters:
        for col, val in filters.items():
            query += f" AND ci.{col} = ?"
            params.append(val)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]

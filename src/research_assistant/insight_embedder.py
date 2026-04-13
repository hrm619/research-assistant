"""Embed distilled insights into chroma for semantic retrieval in Translate."""

import json
import logging
import sqlite3
from datetime import datetime, timezone

import chromadb
import openai

from research_assistant.config import Settings
from research_assistant.db import get_row, insert_row, list_rows, update_row
from research_assistant.schemas import Insight

logger = logging.getLogger(__name__)


def build_embedding_text(insight: Insight) -> str:
    parts = [
        f"[TYPE: {insight.insight_type}]",
        f"[ANALYST: {insight.analyst or 'unknown'}]",
        f"[TRUST: {insight.trust_tier or 'unknown'}]",
    ]

    if insight.insight_type == "framework" and insight.framework:
        fw = insight.framework
        parts.append(f"{fw.name}: {fw.mechanism}")
        if fw.conditions:
            parts.append(f"Conditions: {'; '.join(fw.conditions)}")
        if fw.predictions:
            parts.append(f"Predictions: {'; '.join(fw.predictions)}")
    elif insight.insight_type == "claim" and insight.claim:
        cl = insight.claim
        parts.append(cl.statement)
        parts.append(f"Reasoning: {cl.reasoning}")
        if cl.timeframe:
            parts.append(f"Timeframe: {cl.timeframe}")

    return "\n".join(parts)


def build_chroma_metadata(insight: Insight, domain: str) -> dict:
    return {
        "insight_id": insight.insight_id,
        "insight_type": insight.insight_type,
        "analyst": insight.analyst or "",
        "trust_tier": insight.trust_tier or "",
        "domain": domain,
        "content_item_ref": insight.content_item_ref or insight.content_id,
        "status": insight.status,
        "content_source": insight.content_source,
    }


def get_or_create_insights_collection(
    client: chromadb.ClientAPI, domain: str,
) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=f"insights_{domain}",
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(
    texts: list[str],
    openai_client: openai.OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = openai_client.embeddings.create(input=batch, model=model)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def embed_and_store_insights(
    insights: list[Insight],
    domain: str,
    conn: sqlite3.Connection,
    chroma_client: chromadb.ClientAPI,
    openai_client: openai.OpenAI,
    settings: Settings,
) -> tuple[int, int]:
    if not insights:
        return 0, 0

    collection = get_or_create_insights_collection(chroma_client, domain)
    now = datetime.now(timezone.utc).isoformat()
    collection_name = f"insights_{domain}"
    success_count = 0
    fail_count = 0

    texts = [build_embedding_text(i) for i in insights]
    ids = [i.insight_id for i in insights]
    metadatas = [build_chroma_metadata(i, domain) for i in insights]

    try:
        embeddings = embed_texts(texts, openai_client, settings.insight_embedding_model)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        for insight in insights:
            _upsert_embedding_status(conn, insight.insight_id, "embedded", collection_name, now)
        success_count = len(insights)

    except Exception as e:
        logger.error("Batch embedding failed: %s", e)
        for insight in insights:
            _upsert_embedding_status(
                conn, insight.insight_id, "failed", collection_name, now, str(e),
            )
        fail_count = len(insights)

    return success_count, fail_count


def _upsert_embedding_status(
    conn: sqlite3.Connection,
    insight_id: str,
    status: str,
    collection: str,
    timestamp: str,
    error: str | None = None,
) -> None:
    existing = get_row(conn, "insight_embedding", "insight_id", insight_id)
    if existing:
        update_row(conn, "insight_embedding", "insight_id", insight_id, {
            "embedding_status": status,
            "chroma_collection": collection,
            "last_embedded_at": timestamp,
            "error": error,
        })
    else:
        insert_row(conn, "insight_embedding", {
            "insight_id": insight_id,
            "embedding_status": status,
            "chroma_collection": collection,
            "last_embedded_at": timestamp,
            "error": error,
        })


def get_pending_or_failed_insights(
    conn: sqlite3.Connection, domain: str,
) -> list[dict]:
    rows = conn.execute(
        """SELECT i.* FROM insight i
           JOIN insight_embedding ie ON i.insight_id = ie.insight_id
           WHERE i.domain_id = (
               SELECT domain_id FROM domain_brief WHERE domain_name = ?
               UNION SELECT ? WHERE ? IN (SELECT domain_id FROM domain_brief)
           )
           AND ie.embedding_status IN ('pending', 'failed')""",
        (domain, domain, domain),
    ).fetchall()
    return [dict(r) for r in rows]


def reembed_failed(
    domain: str,
    conn: sqlite3.Connection,
    chroma_client: chromadb.ClientAPI,
    openai_client: openai.OpenAI,
    settings: Settings,
) -> tuple[int, int]:
    rows = get_pending_or_failed_insights(conn, domain)
    if not rows:
        return 0, 0

    insights = []
    for row in rows:
        fw = json.loads(row["framework_json"]) if row.get("framework_json") else None
        cl = json.loads(row["claim_json"]) if row.get("claim_json") else None
        insight = Insight(
            insight_id=row["insight_id"],
            content_id=row["content_id"],
            content_item_ref=row.get("content_item_ref", ""),
            source_id=row.get("source_id", ""),
            domain_id=row["domain_id"],
            extracted_at=row["extracted_at"],
            insight_type=row["insight_type"],
            framework=fw,
            claim=cl,
            source_quote_ref=row["source_quote_ref"],
            status=row["status"],
            analyst=row.get("analyst", ""),
            trust_tier=row.get("trust_tier", ""),
            content_source=row.get("content_source", "ra"),
        )
        insights.append(insight)

    return embed_and_store_insights(
        insights, domain, conn, chroma_client, openai_client, settings,
    )

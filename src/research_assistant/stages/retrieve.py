"""Retrieve stage — select content from kb.db for distillation."""

import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from research_assistant.db import insert_row, list_rows


def query_kb_content(
    kb_conn: sqlite3.Connection,
    domain: str,
    trust_tiers: list[str] | None = None,
    analysts: list[str] | None = None,
    source_types: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    sql = "SELECT * FROM content_record WHERE domain = ?"
    params: list = [domain]

    if trust_tiers:
        placeholders = ",".join("?" for _ in trust_tiers)
        sql += f" AND trust_tier IN ({placeholders})"
        params.extend(trust_tiers)

    if analysts:
        placeholders = ",".join("?" for _ in analysts)
        sql += f" AND analyst IN ({placeholders})"
        params.extend(analysts)

    if source_types:
        placeholders = ",".join("?" for _ in source_types)
        sql += f" AND source_type IN ({placeholders})"
        params.extend(source_types)

    if since:
        sql += " AND published_at >= ?"
        params.append(since)

    if until:
        sql += " AND published_at <= ?"
        params.append(until)

    sql += " ORDER BY published_at DESC, ingested_at DESC"

    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    rows = kb_conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_existing_refs(
    conn: sqlite3.Connection, domain: str
) -> set[str]:
    rows = conn.execute(
        "SELECT content_item_ref FROM retrieval_batch "
        "WHERE domain = ? AND distill_status = 'distilled'",
        (domain,),
    ).fetchall()
    return {r[0] for r in rows}


def run_retrieve(
    kb_conn: sqlite3.Connection,
    ra_conn: sqlite3.Connection,
    domain: str,
    trust_tiers: list[str] | None = None,
    analysts: list[str] | None = None,
    source_types: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    matched = query_kb_content(
        kb_conn, domain, trust_tiers, analysts, source_types, since, until, limit,
    )

    if not force:
        already_distilled = get_existing_refs(ra_conn, domain)
        matched = [
            m for m in matched if m["content_id"] not in already_distilled
        ]

    if dry_run:
        return matched

    now = datetime.now(timezone.utc).isoformat()
    batch_id = str(uuid4())

    for item in matched:
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": f"{batch_id}:{item['content_id']}",
            "domain": domain,
            "content_item_ref": item["content_id"],
            "analyst": item.get("analyst", ""),
            "trust_tier": item.get("trust_tier", ""),
            "source_type": item.get("source_type", ""),
            "published_at": item.get("published_at", ""),
            "retrieved_at": now,
            "distill_status": "pending",
        })

    return matched


def list_batch_rows(
    conn: sqlite3.Connection,
    domain: str,
    status: str | None = None,
) -> list[dict]:
    filters: dict = {"domain": domain}
    if status:
        filters["distill_status"] = status
    return list_rows(conn, "retrieval_batch", filters)

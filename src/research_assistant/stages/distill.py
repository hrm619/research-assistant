import json
import logging
import sqlite3
from dataclasses import dataclass
from importlib import resources

import chromadb

from research_assistant.config import Settings
from research_assistant.db import get_row, insert_row, list_rows, resolve_domain
from research_assistant.llm import llm_call_with_list_validation, parse_json_response
from research_assistant.schemas import (
    DomainBriefContent,
    Insight,
    _now_iso,
    _uuid,
)

logger = logging.getLogger(__name__)


@dataclass
class KBContext:
    """Connection context for reading content from the knowledge-base."""

    kb_conn: sqlite3.Connection
    chroma_client: chromadb.ClientAPI
    collection_name: str


def _load_prompt_template() -> str:
    return resources.files("research_assistant.prompts").joinpath("distill.txt").read_text()


def build_distill_prompt(
    raw_text: str,
    domain_brief_json: str,
    mode: str,
    focus: str | None,
) -> tuple[str, str]:
    template = _load_prompt_template()
    system = template

    user_prompt = f"Domain context:\n{domain_brief_json}\n\n"
    user_prompt += f"Extraction mode: {mode}\n"
    if focus:
        user_prompt += f"Operator focus: {focus}\n"
    user_prompt += f"\nCONTENT TO PROCESS:\n{raw_text}"
    return system, user_prompt


def run_distill(
    content_id: str,
    domain_id: str,
    mode: str,
    focus: str | None,
    conn: sqlite3.Connection,
    settings: Settings,
    kb_context: KBContext | None = None,
) -> list[Insight]:
    # --- Resolve domain (always from RA db) ---
    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        raise ValueError(f"Domain not found: {domain_id}")

    domain_row = get_row(conn, "domain_brief", "domain_id", resolved)
    domain_brief_json = domain_row["brief_json"] if domain_row else "{}"

    # --- Get content: KB path or RA path ---
    if kb_context:
        from research_assistant.kb_reader import (
            get_content_record,
            reconstruct_transcript,
        )

        content_record = get_content_record(kb_context.kb_conn, content_id)
        if not content_record:
            raise ValueError(f"KB content not found: {content_id}")

        raw_text = reconstruct_transcript(
            kb_context.chroma_client, kb_context.collection_name, content_id,
        )
        source_id = ""
        analyst = content_record["analyst"]
        trust_tier = content_record["trust_tier"]
        content_source = "kb"
    else:
        content_row = get_row(conn, "content_item", "content_id", content_id)
        if not content_row:
            raise ValueError(f"Content not found: {content_id}")

        raw_text = content_row["raw_text"]
        source_id = content_row["source_id"]
        analyst = ""
        trust_tier = ""
        content_source = "ra"

    # --- Build prompt and call LLM ---
    system, prompt = build_distill_prompt(raw_text, domain_brief_json, mode, focus)

    from research_assistant.llm import call_llm, retry_with_backoff

    def _attempt():
        raw = call_llm(prompt, system, settings)
        data = parse_json_response(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        insights = []
        for item in data:
            item["content_id"] = content_id
            item["content_item_ref"] = content_id
            item["source_id"] = source_id
            item["domain_id"] = resolved
            item["insight_id"] = _uuid()
            item["extracted_at"] = _now_iso()
            item["source_quote_ref"] = item.get("source_quote_ref", "unknown")
            item["status"] = "active"
            item["analyst"] = analyst
            item["trust_tier"] = trust_tier
            item["content_source"] = content_source
            insights.append(Insight.model_validate(item))
        return insights

    return retry_with_backoff(
        _attempt,
        max_retries=settings.llm_max_retries,
        base=settings.llm_backoff_base,
        factor=settings.llm_backoff_factor,
    )


def check_dedup(insight: Insight, conn: sqlite3.Connection) -> bool:
    existing = list_rows(conn, "insight", {"domain_id": insight.domain_id, "insight_type": insight.insight_type})
    for row in existing:
        if insight.insight_type == "framework" and insight.framework:
            existing_fw = json.loads(row["framework_json"]) if row["framework_json"] else {}
            if existing_fw.get("name", "").lower() == insight.framework.name.lower():
                return True
        elif insight.insight_type == "claim" and insight.claim:
            existing_cl = json.loads(row["claim_json"]) if row["claim_json"] else {}
            if existing_cl.get("statement", "").lower() == insight.claim.statement.lower():
                return True
    return False


def save_insights(
    insights: list[Insight],
    conn: sqlite3.Connection,
) -> list[str]:
    ids = []
    has_kb = any(i.content_source == "kb" for i in insights)

    # KB-sourced insights have content_ids that don't exist in RA's content_item
    # table, so we need to temporarily disable FK enforcement.
    if has_kb:
        conn.execute("PRAGMA foreign_keys=OFF")

    try:
        for insight in insights:
            if check_dedup(insight, conn):
                logger.info("Skipping duplicate insight: %s", insight.insight_id)
                continue
            insert_row(conn, "insight", {
                "insight_id": insight.insight_id,
                "content_id": insight.content_id,
                "content_item_ref": insight.content_item_ref or insight.content_id,
                "source_id": insight.source_id,
                "domain_id": insight.domain_id,
                "extracted_at": insight.extracted_at,
                "insight_type": insight.insight_type,
                "framework_json": insight.framework.model_dump_json() if insight.framework else None,
                "claim_json": insight.claim.model_dump_json() if insight.claim else None,
                "source_quote_ref": insight.source_quote_ref,
                "operator_note": insight.operator_note,
                "status": insight.status,
                "analyst": insight.analyst,
                "trust_tier": insight.trust_tier,
                "content_source": insight.content_source,
            })
            ids.append(insight.insight_id)
    finally:
        if has_kb:
            conn.execute("PRAGMA foreign_keys=ON")

    logger.info("Saved %d insights (skipped %d duplicates)", len(ids), len(insights) - len(ids))
    return ids


def run_distill_batch(
    domain: str,
    mode: str,
    focus: str | None,
    conn: sqlite3.Connection,
    settings: Settings,
    kb_conn: sqlite3.Connection,
    chroma_client: chromadb.ClientAPI,
    batch_id: str | None = None,
    limit: int | None = None,
    openai_client: "openai.OpenAI | None" = None,
) -> list[Insight]:
    from research_assistant.kb_reader import get_content_record, reconstruct_transcript
    from research_assistant.db import update_row

    sql = (
        "SELECT * FROM retrieval_batch "
        "WHERE domain = ? AND distill_status = 'pending'"
    )
    params: list = [domain]

    if batch_id:
        sql += " AND batch_id LIKE ?"
        params.append(f"{batch_id}%")

    sql += " ORDER BY retrieved_at DESC"

    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    pending = conn.execute(sql, params).fetchall()
    pending = [dict(r) for r in pending]

    if not pending:
        return []

    resolved = resolve_domain(conn, domain)
    if not resolved:
        raise ValueError(f"Domain not found: {domain}")

    domain_row = get_row(conn, "domain_brief", "domain_id", resolved)
    domain_brief_json = domain_row["brief_json"] if domain_row else "{}"

    collection_name = domain

    all_insights: list[Insight] = []

    for batch_row in pending:
        content_ref = batch_row["content_item_ref"]

        try:
            content_record = get_content_record(kb_conn, content_ref)
            if not content_record:
                update_row(conn, "retrieval_batch", "batch_id", batch_row["batch_id"], {
                    "distill_status": "failed",
                    "distill_error": f"Content not found in kb.db: {content_ref}",
                })
                continue

            raw_text = reconstruct_transcript(chroma_client, collection_name, content_ref)

            system, prompt = build_distill_prompt(raw_text, domain_brief_json, mode, focus)

            from research_assistant.llm import call_llm, retry_with_backoff

            def _attempt():
                raw = call_llm(prompt, system, settings)
                data = parse_json_response(raw)
                if not isinstance(data, list):
                    raise ValueError(f"Expected JSON array, got {type(data).__name__}")

                insights = []
                for item in data:
                    item["content_id"] = content_ref
                    item["content_item_ref"] = content_ref
                    item["source_id"] = ""
                    item["domain_id"] = resolved
                    item["insight_id"] = _uuid()
                    item["extracted_at"] = _now_iso()
                    item["source_quote_ref"] = item.get("source_quote_ref", "unknown")
                    item["status"] = "active"
                    item["analyst"] = batch_row.get("analyst", "")
                    item["trust_tier"] = batch_row.get("trust_tier", "")
                    item["content_source"] = "kb"
                    insights.append(Insight.model_validate(item))
                return insights

            insights = retry_with_backoff(
                _attempt,
                max_retries=settings.llm_max_retries,
                base=settings.llm_backoff_base,
                factor=settings.llm_backoff_factor,
            )

            ids = save_insights(insights, conn)

            if openai_client and ids:
                from research_assistant.insight_embedder import embed_and_store_insights
                saved = [i for i in insights if i.insight_id in ids]
                embed_and_store_insights(
                    saved, domain, conn, chroma_client, openai_client, settings,
                )

            update_row(conn, "retrieval_batch", "batch_id", batch_row["batch_id"], {
                "distill_status": "distilled",
            })

            all_insights.extend(insights)
            logger.info("Distilled %d insights from %s", len(ids), content_ref)

        except Exception as e:
            update_row(conn, "retrieval_batch", "batch_id", batch_row["batch_id"], {
                "distill_status": "failed",
                "distill_error": str(e),
            })
            logger.error("Failed to distill %s: %s", content_ref, e)

    return all_insights


def list_insights(
    domain_id: str,
    conn: sqlite3.Connection,
    filters: dict | None = None,
) -> list[dict]:
    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        return []
    base_filters = {"domain_id": resolved}
    if filters:
        base_filters.update(filters)
    return list_rows(conn, "insight", base_filters)

import json
import logging
import sqlite3
from importlib import resources

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
) -> list[Insight]:
    content_row = get_row(conn, "content_item", "content_id", content_id)
    if not content_row:
        raise ValueError(f"Content not found: {content_id}")

    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        raise ValueError(f"Domain not found: {domain_id}")

    domain_row = get_row(conn, "domain_brief", "domain_id", resolved)
    domain_brief_json = domain_row["brief_json"] if domain_row else "{}"

    system, prompt = build_distill_prompt(
        content_row["raw_text"], domain_brief_json, mode, focus,
    )

    # LLM returns array of partial insight dicts; we need to fill in IDs
    from research_assistant.llm import call_llm, retry_with_backoff

    def _attempt():
        raw = call_llm(prompt, system, settings)
        data = parse_json_response(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        insights = []
        for item in data:
            # Inject required fields that the LLM doesn't generate
            item["content_id"] = content_id
            item["source_id"] = content_row["source_id"]
            item["domain_id"] = resolved
            item["insight_id"] = _uuid()
            item["extracted_at"] = _now_iso()
            item["source_quote_ref"] = item.get("source_quote_ref", "unknown")
            item["status"] = "active"
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


def save_insights(insights: list[Insight], conn: sqlite3.Connection) -> list[str]:
    ids = []
    for insight in insights:
        if check_dedup(insight, conn):
            logger.info("Skipping duplicate insight: %s", insight.insight_id)
            continue
        insert_row(conn, "insight", {
            "insight_id": insight.insight_id,
            "content_id": insight.content_id,
            "source_id": insight.source_id,
            "domain_id": insight.domain_id,
            "extracted_at": insight.extracted_at,
            "insight_type": insight.insight_type,
            "framework_json": insight.framework.model_dump_json() if insight.framework else None,
            "claim_json": insight.claim.model_dump_json() if insight.claim else None,
            "source_quote_ref": insight.source_quote_ref,
            "operator_note": insight.operator_note,
            "status": insight.status,
        })
        ids.append(insight.insight_id)
    logger.info("Saved %d insights (skipped %d duplicates)", len(ids), len(insights) - len(ids))
    return ids


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

import json
import logging
import sqlite3
from importlib import resources

from research_assistant.config import Settings
from research_assistant.db import get_row, insert_row, list_rows, resolve_domain
from research_assistant.llm import call_llm, parse_json_response, retry_with_backoff
from research_assistant.schemas import (
    Hypothesis,
    OperatorContext,
    _now_iso,
    _uuid,
)

logger = logging.getLogger(__name__)


def _load_prompt_template() -> str:
    return resources.files("research_assistant.prompts").joinpath("translate.txt").read_text()


def build_translate_prompt(
    insights_json: str,
    domain_brief_json: str,
    operator_context: OperatorContext,
    mode: str,
) -> tuple[str, str]:
    template = _load_prompt_template()
    system = template

    user_prompt = f"Domain context:\n{domain_brief_json}\n\n"
    user_prompt += f"Operator context:\n{operator_context.model_dump_json()}\n\n"
    user_prompt += f"Translation mode: {mode}\n\n"
    user_prompt += f"INSIGHTS TO TRANSLATE:\n{insights_json}"
    return system, user_prompt


def run_translate(
    insight_ids: list[str],
    domain_id: str,
    mode: str,
    operator_context: OperatorContext,
    conn: sqlite3.Connection,
    settings: Settings,
) -> list[Hypothesis]:
    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        raise ValueError(f"Domain not found: {domain_id}")

    domain_row = get_row(conn, "domain_brief", "domain_id", resolved)
    domain_brief_json = domain_row["brief_json"] if domain_row else "{}"

    # Gather insights
    insights_data = []
    for iid in insight_ids:
        row = get_row(conn, "insight", "insight_id", iid)
        if row:
            insight_dict = dict(row)
            # Parse stored JSON fields for context
            if insight_dict.get("framework_json"):
                insight_dict["framework"] = json.loads(insight_dict["framework_json"])
            if insight_dict.get("claim_json"):
                insight_dict["claim"] = json.loads(insight_dict["claim_json"])
            insights_data.append(insight_dict)

    if not insights_data:
        raise ValueError("No valid insights found for the given IDs")

    system, prompt = build_translate_prompt(
        json.dumps(insights_data, default=str),
        domain_brief_json,
        operator_context,
        mode,
    )

    def _attempt():
        raw = call_llm(prompt, system, settings)
        data = parse_json_response(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        hypotheses = []
        for item in data:
            item["domain_id"] = resolved
            item["hypothesis_id"] = _uuid()
            item["created_at"] = _now_iso()
            item["status"] = "draft"
            hypotheses.append(Hypothesis.model_validate(item))
        return hypotheses

    return retry_with_backoff(
        _attempt,
        max_retries=settings.llm_max_retries,
        base=settings.llm_backoff_base,
        factor=settings.llm_backoff_factor,
    )


def assess_feasibility(hypothesis: Hypothesis, operator_context: OperatorContext) -> Hypothesis:
    available = set(s.lower() for s in operator_context.available_data_sources)
    required = set(s.lower() for s in hypothesis.definition.data_required)
    if required and not required.intersection(available):
        hypothesis.feasibility.estimated_testability = "low"
        hypothesis.definition.data_available = False
    return hypothesis


def save_hypotheses(
    hypotheses: list[Hypothesis],
    insight_ids: list[str],
    conn: sqlite3.Connection,
) -> list[str]:
    ids = []
    for hyp in hypotheses:
        insert_row(conn, "hypothesis", {
            "hypothesis_id": hyp.hypothesis_id,
            "domain_id": hyp.domain_id,
            "created_at": hyp.created_at,
            "status": hyp.status,
            "definition_json": hyp.definition.model_dump_json(),
            "feasibility_json": hyp.feasibility.model_dump_json(),
            "reasoning_chain_json": hyp.reasoning_chain.model_dump_json(),
            "operator_note": hyp.operator_note,
        })
        # Create junction rows
        for iid in insight_ids:
            insert_row(conn, "hypothesis_insight", {
                "hypothesis_id": hyp.hypothesis_id,
                "insight_id": iid,
            })
        ids.append(hyp.hypothesis_id)
    logger.info("Saved %d hypotheses", len(ids))
    return ids


def list_hypotheses(
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
    return list_rows(conn, "hypothesis", base_filters)


def export_for_harness(hypothesis_id: str, conn: sqlite3.Connection) -> dict | None:
    row = get_row(conn, "hypothesis", "hypothesis_id", hypothesis_id)
    if not row:
        return None
    return {
        "hypothesis_id": row["hypothesis_id"],
        "domain_id": row["domain_id"],
        "status": row["status"],
        "definition": json.loads(row["definition_json"]),
        "feasibility": json.loads(row["feasibility_json"]),
        "reasoning_chain": json.loads(row["reasoning_chain_json"]),
    }

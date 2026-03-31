import json
import logging
import sqlite3
from importlib import resources
from pathlib import Path

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


def _build_test_definition_schema(registry: dict) -> str:
    """Build the test_definition prompt section from a domain registry."""
    catalog = ", ".join(f'"{m}"' for m in registry["metrics_catalog"])
    outcomes = ", ".join(f'"{o}"' for o in registry["valid_outcomes"])
    cls_types = ", ".join(f'"{t}"' for t in registry["valid_classification_types"])
    lookbacks = ", ".join(f'"{l}"' for l in registry["valid_lookbacks"])
    stat_tests = ", ".join(f'"{t}"' for t in registry["valid_statistical_tests"])

    return f""",
  "test_definition": {{
    "hypothesis_name": "string (snake_case, unique, suitable as filename)",
    "description": "string (one paragraph description)",
    "version": "1.0.0",
    "metrics": ["string — MUST be from DOMAIN METRICS CATALOG below"],
    "classification": {{
      "type": "one of: {cls_types}",
      "metric": "string — MUST be from DOMAIN METRICS CATALOG below",
      "top_pct": "float or null (required for percentile)",
      "bottom_pct": "float or null (required for percentile)",
      "threshold": "float or null (required for binary)",
      "boundaries": "[float] or null (required for custom)"
    }},
    "outcome": "one of: {outcomes}",
    "lookback": "one of: {lookbacks}",
    "filters": {{
      "seasons": "[int] or null",
      "weeks": "[int] or null",
      "game_type": '["regular"] or null',
      "exclude_week_1": true
    }},
    "min_sample_size": 50,
    "statistical_test": "one of: {stat_tests} (default: binomial)",
    "significance_threshold": 0.05,
    "comparison_buckets": ["Q1", "Q4"]
  }}

DOMAIN METRICS CATALOG — you MUST select metrics ONLY from this list:
[{catalog}]"""


def build_translate_prompt(
    insights_json: str,
    domain_brief_json: str,
    operator_context: OperatorContext,
    mode: str,
    domain_registry: dict | None = None,
) -> tuple[str, str]:
    template = _load_prompt_template()

    if domain_registry:
        schema_section = _build_test_definition_schema(domain_registry)
    else:
        schema_section = ""
    system = template.replace("{test_definition_schema}", schema_section)

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
    domain_registry_path: Path | None = None,
) -> list[Hypothesis]:
    resolved = resolve_domain(conn, domain_id)
    if not resolved:
        raise ValueError(f"Domain not found: {domain_id}")

    domain_row = get_row(conn, "domain_brief", "domain_id", resolved)
    domain_brief_json = domain_row["brief_json"] if domain_row else "{}"

    # Load domain registry if provided
    domain_registry = None
    if domain_registry_path:
        from research_assistant.contracts import load_domain_registry
        domain_registry = load_domain_registry(domain_registry_path)

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
        domain_registry=domain_registry,
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
            hyp = Hypothesis.model_validate(item)

            # Validate test_definition against registry if provided
            if domain_registry and hyp.test_definition:
                from research_assistant.contracts import validate_test_definition
                errors = validate_test_definition(
                    hyp.test_definition.model_dump(), domain_registry
                )
                if errors:
                    raise ValueError(
                        f"test_definition validation failed: {errors}"
                    )

            hypotheses.append(hyp)
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
            "test_definition_json": (
                hyp.test_definition.model_dump_json()
                if hyp.test_definition else None
            ),
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


def export_for_harness(
    hypothesis_id: str,
    conn: sqlite3.Connection,
    domain_registry: dict | None = None,
    output_path: Path | None = None,
) -> dict | None:
    """Export a hypothesis as Contract 1 JSON.

    If domain_registry is provided, validates test_definition against it.
    If output_path is provided, writes JSON to that file.
    """
    row = get_row(conn, "hypothesis", "hypothesis_id", hypothesis_id)
    if not row:
        return None

    definition = json.loads(row["definition_json"])
    feasibility = json.loads(row["feasibility_json"])
    reasoning_chain = json.loads(row["reasoning_chain_json"])

    # Parse test_definition
    test_def_raw = row.get("test_definition_json")
    if not test_def_raw:
        raise ValueError(
            f"Hypothesis {hypothesis_id} has no test_definition. "
            "Re-translate with --domain-registry to generate one."
        )
    test_def = json.loads(test_def_raw)

    # Validate against registry
    if domain_registry:
        from research_assistant.contracts import validate_test_definition
        errors = validate_test_definition(test_def, domain_registry)
        if errors:
            raise ValueError(f"test_definition validation failed: {errors}")

    # Look up domain name
    domain_row = get_row(conn, "domain_brief", "domain_id", row["domain_id"])
    domain_name = domain_row["domain_name"] if domain_row else row["domain_id"]

    contract = {
        "contract_version": "1.0.0",
        "produced_at": _now_iso(),
        "producer": "research-assistant",
        "hypothesis_id": row["hypothesis_id"],
        "domain_id": row["domain_id"],
        "domain_name": domain_name,
        "rich_definition": {
            **definition,
            "feasibility": feasibility,
            "reasoning_chain": reasoning_chain,
        },
        "test_definition": test_def,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(contract, f, indent=2)
        logger.info("Contract 1 written to %s", output_path)

    return contract

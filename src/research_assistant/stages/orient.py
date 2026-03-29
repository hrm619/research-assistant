import json
import logging
import sqlite3
from importlib import resources

from research_assistant.config import Settings
from research_assistant.db import get_row, insert_row
from research_assistant.llm import llm_call_with_validation
from research_assistant.schemas import DomainBriefContent, OrientInput, _now_iso, _uuid

logger = logging.getLogger(__name__)


def _load_prompt_template() -> str:
    return resources.files("research_assistant.prompts").joinpath("orient.txt").read_text()


def build_orient_prompt(input: OrientInput) -> tuple[str, str]:
    template = _load_prompt_template()
    system = template.format(
        operator_known_domains=", ".join(input.operator_known_domains),
    )
    user_prompt = (
        f"Target domain: {input.domain_name}\n"
        f"Market: {input.market_type}\n"
        f"Seed questions: {json.dumps(input.seed_questions)}\n"
    )
    if input.seed_sources:
        user_prompt += f"Seed sources: {json.dumps(input.seed_sources)}\n"
    user_prompt += (
        "\nProduce a DomainBrief for this domain. Generate at least 2 analogies "
        "to the operator's known domains. Identify at least 3 open questions the "
        "operator should investigate before forming hypotheses."
    )
    return system, user_prompt


def run_orient(input: OrientInput, settings: Settings) -> DomainBriefContent:
    system, prompt = build_orient_prompt(input)
    return llm_call_with_validation(prompt, system, DomainBriefContent, settings)


def validate_domain_brief(brief: DomainBriefContent) -> list[str]:
    errors = []
    if len(brief.analogies) < 2:
        errors.append("DomainBrief must contain at least 2 analogies")
    if len(brief.open_questions) < 3:
        errors.append("DomainBrief must contain at least 3 open questions")
    return errors


def save_domain_brief(
    brief_content: DomainBriefContent,
    domain_name: str,
    market_type: str,
    conn: sqlite3.Connection,
) -> str:
    domain_id = _uuid()
    insert_row(conn, "domain_brief", {
        "domain_id": domain_id,
        "domain_name": domain_name,
        "market_type": market_type,
        "created_at": _now_iso(),
        "brief_json": brief_content.model_dump_json(),
        "status": "draft",
    })
    logger.info("Saved DomainBrief %s for domain '%s'", domain_id, domain_name)
    return domain_id


def get_domain_brief(domain_id: str, conn: sqlite3.Connection) -> dict | None:
    return get_row(conn, "domain_brief", "domain_id", domain_id)

import json
from unittest.mock import patch

import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, migrate
from research_assistant.schemas import DomainBriefContent, OrientInput
from research_assistant.stages.orient import (
    build_orient_prompt,
    get_domain_brief,
    run_orient,
    save_domain_brief,
    validate_domain_brief,
)


SAMPLE_BRIEF = {
    "market_mechanics": {
        "instrument_type": "binary",
        "settlement": "cash at expiry",
        "liquidity_profile": "thin",
        "fee_structure": "2% per trade",
        "position_types": ["long", "short"],
        "known_biases": ["retail skew on favorites"],
    },
    "game_theory": {
        "participant_types": ["retail", "market makers"],
        "information_asymmetries": ["insider economic data access"],
        "common_mistakes": ["anchoring to consensus"],
    },
    "current_meta": {
        "dominant_narratives": ["soft landing"],
        "consensus_view": "rates hold through Q3",
        "contrarian_angles": ["recession risk underpriced"],
    },
    "analogies": [
        {
            "known_domain": "sports betting",
            "mapping": "binary outcome markets with public odds",
            "where_analogy_breaks": "no team-level stats equivalent",
        },
        {
            "known_domain": "politics",
            "mapping": "event-driven binary contracts",
            "where_analogy_breaks": "fed decisions have quantitative anchors",
        },
    ],
    "key_data_sources": ["FRED", "CME FedWatch"],
    "open_questions": [
        "What drives volume spikes on Kalshi fed contracts?",
        "How do market makers hedge binary positions?",
        "What is the typical bid-ask spread at different times?",
    ],
    "confidence": "medium",
}


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key", llm_max_retries=1, llm_backoff_base=0.01)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    migrate(c)
    return c


class TestBuildOrientPrompt:
    def test_contains_domain_info(self):
        input = OrientInput(
            domain_name="fed_rate_decisions",
            market_type="kalshi",
            operator_known_domains=["sports", "politics"],
            seed_questions=["How do contracts settle?"],
        )
        system, prompt = build_orient_prompt(input)
        assert "sports, politics" in system
        assert "fed_rate_decisions" in prompt
        assert "kalshi" in prompt
        assert "How do contracts settle?" in prompt


class TestRunOrient:
    @patch("research_assistant.stages.orient.llm_call_with_validation")
    def test_returns_domain_brief(self, mock_llm, settings):
        mock_llm.return_value = DomainBriefContent(**SAMPLE_BRIEF)
        input = OrientInput(
            domain_name="fed_rate_decisions",
            market_type="kalshi",
            operator_known_domains=["sports"],
            seed_questions=["How?"],
        )
        result = run_orient(input, settings)
        assert result.confidence == "medium"
        assert len(result.analogies) == 2


class TestValidateDomainBrief:
    def test_valid_brief(self):
        brief = DomainBriefContent(**SAMPLE_BRIEF)
        errors = validate_domain_brief(brief)
        assert errors == []

    def test_too_few_analogies(self):
        data = {**SAMPLE_BRIEF, "analogies": SAMPLE_BRIEF["analogies"][:1]}
        brief = DomainBriefContent(**data)
        errors = validate_domain_brief(brief)
        assert any("analogies" in e for e in errors)

    def test_too_few_questions(self):
        data = {**SAMPLE_BRIEF, "open_questions": ["one", "two"]}
        brief = DomainBriefContent(**data)
        errors = validate_domain_brief(brief)
        assert any("open questions" in e for e in errors)


class TestSaveAndGet:
    def test_round_trip(self, conn):
        brief = DomainBriefContent(**SAMPLE_BRIEF)
        domain_id = save_domain_brief(brief, "fed_rate_decisions", "kalshi", conn)
        row = get_domain_brief(domain_id, conn)
        assert row is not None
        assert row["domain_name"] == "fed_rate_decisions"
        assert row["status"] == "draft"
        stored = json.loads(row["brief_json"])
        assert stored["confidence"] == "medium"

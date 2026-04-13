import json
from unittest.mock import patch

import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, insert_row, list_rows, migrate
from research_assistant.schemas import Hypothesis, OperatorContext
from research_assistant.stages.translate import (
    assess_feasibility,
    build_translate_prompt,
    export_for_harness,
    list_hypotheses,
    run_translate,
    save_hypotheses,
)


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key", llm_max_retries=1, llm_backoff_base=0.01)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    migrate(c)
    insert_row(c, "domain_brief", {
        "domain_id": "d1",
        "domain_name": "test_domain",
        "market_type": "kalshi",
        "created_at": "2026-01-01T00:00:00Z",
        "brief_json": '{"confidence": "medium"}',
        "status": "active",
    })
    insert_row(c, "insight", {
        "insight_id": "i1",
        "content_id": "c1",
        "content_item_ref": "c1",
        "source_id": "",
        "domain_id": "d1",
        "extracted_at": "2026-01-01T00:00:00Z",
        "insight_type": "framework",
        "framework_json": json.dumps({
            "name": "yield_curve_signal",
            "mechanism": "inversion predicts recession",
            "conditions": ["2y/10y negative"],
            "predictions": ["recession in 18m"],
            "assumptions": ["no QE"],
        }),
        "source_quote_ref": "beginning",
        "status": "active",
    })
    return c


SAMPLE_TEST_DEFINITION = {
    "hypothesis_name": "yield_curve_recession_predictor",
    "description": "Tests whether yield curve inversion predicts recession contract resolution",
    "version": "1.0.0",
    "metrics": ["points_per_game_std"],
    "classification": {"type": "quartile", "metric": "points_per_game_std"},
    "outcome": "ats",
    "lookback": "season_to_date",
}

SAMPLE_HYPOTHESIS = {
    "definition": {
        "name": "yield_curve_recession_predictor",
        "statement": "Yield curve inversion predicts Kalshi recession contract resolution YES",
        "factor": "2y10y spread",
        "classification": "binary (inverted vs not)",
        "outcome_measure": "Kalshi recession contract settles YES within 18 months",
        "timeframe": "18 months",
        "data_required": ["FRED yield curve data", "Kalshi contract prices"],
        "data_available": True,
        "market_expression": "Buy Kalshi recession YES contract when 2y10y inverts",
    },
    "feasibility": {
        "data_gap": [],
        "knowledge_gap": ["Historical hit rate of inversion signal on prediction markets"],
        "minimum_sample_size": 5,
        "estimated_testability": "medium",
    },
    "reasoning_chain": {
        "from_insight": "Expert identifies yield curve inversion as recession signal via lending channel",
        "translation_logic": "If inversion -> recession -> Kalshi YES settles, then buying YES after inversion is positive EV",
        "assumptions_added": ["Kalshi recession contract definition matches NBER definition"],
        "weaknesses": ["Small historical sample of inversions"],
    },
    "test_definition": SAMPLE_TEST_DEFINITION,
}

SAMPLE_LLM_RESPONSE = json.dumps([SAMPLE_HYPOTHESIS])


class TestBuildTranslatePrompt:
    def test_contains_context(self):
        op = OperatorContext(accessible_markets=["kalshi"], known_domains=["sports"])
        system, prompt = build_translate_prompt(
            '[{"insight": "test"}]', '{"confidence": "medium"}', op, "explore",
        )
        assert "hypothesis engineer" in system.lower()
        assert "kalshi" in prompt
        assert "explore" in prompt


class TestRunTranslate:
    @patch("research_assistant.stages.translate.call_llm")
    def test_returns_hypotheses(self, mock_call, conn, settings):
        mock_call.return_value = SAMPLE_LLM_RESPONSE
        op = OperatorContext(accessible_markets=["kalshi"])
        hypotheses = run_translate(["i1"], "d1", "explore", op, conn, settings)
        assert len(hypotheses) == 1
        assert hypotheses[0].definition.name == "yield_curve_recession_predictor"
        assert hypotheses[0].domain_id == "d1"


class TestAssessFeasibility:
    def test_low_testability_when_no_data_overlap(self):
        hyp = Hypothesis(domain_id="d1", **SAMPLE_HYPOTHESIS)
        op = OperatorContext(available_data_sources=["nothing useful"])
        result = assess_feasibility(hyp, op)
        assert result.feasibility.estimated_testability == "low"
        assert result.definition.data_available is False

    def test_keeps_testability_when_data_available(self):
        hyp = Hypothesis(domain_id="d1", **SAMPLE_HYPOTHESIS)
        op = OperatorContext(available_data_sources=["FRED yield curve data"])
        result = assess_feasibility(hyp, op)
        assert result.feasibility.estimated_testability == "medium"


class TestSaveAndList:
    def test_save_and_list_hypotheses(self, conn):
        hyp = Hypothesis(domain_id="d1", **SAMPLE_HYPOTHESIS)
        ids = save_hypotheses([hyp], ["i1"], conn)
        assert len(ids) == 1

        rows = list_hypotheses("d1", conn)
        assert len(rows) == 1

        # Check junction table
        junctions = list_rows(conn, "hypothesis_insight")
        assert len(junctions) == 1
        assert junctions[0]["hypothesis_id"] == hyp.hypothesis_id
        assert junctions[0]["insight_id"] == "i1"


class TestExport:
    def test_export_for_harness(self, conn):
        hyp = Hypothesis(domain_id="d1", **SAMPLE_HYPOTHESIS)
        save_hypotheses([hyp], ["i1"], conn)
        result = export_for_harness(hyp.hypothesis_id, conn)
        assert result is not None
        # Contract 1 format
        assert result["contract_version"] == "1.0.0"
        assert result["producer"] == "research-assistant"
        assert result["rich_definition"]["name"] == "yield_curve_recession_predictor"
        assert result["test_definition"]["hypothesis_name"] == "yield_curve_recession_predictor"
        assert result["domain_name"] == "test_domain"

    def test_export_nonexistent(self, conn):
        assert export_for_harness("nonexistent", conn) is None

    def test_export_without_test_definition(self, conn):
        hyp_data = {k: v for k, v in SAMPLE_HYPOTHESIS.items() if k != "test_definition"}
        hyp = Hypothesis(domain_id="d1", **hyp_data)
        save_hypotheses([hyp], ["i1"], conn)
        with pytest.raises(ValueError, match="no test_definition"):
            export_for_harness(hyp.hypothesis_id, conn)

    def test_export_to_file(self, conn, tmp_path):
        hyp = Hypothesis(domain_id="d1", **SAMPLE_HYPOTHESIS)
        save_hypotheses([hyp], ["i1"], conn)
        out = tmp_path / "output.json"
        result = export_for_harness(hyp.hypothesis_id, conn, output_path=out)
        assert out.exists()
        written = json.loads(out.read_text())
        assert written["contract_version"] == "1.0.0"
        assert written["test_definition"]["hypothesis_name"] == "yield_curve_recession_predictor"

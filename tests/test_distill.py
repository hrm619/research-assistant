import json
from unittest.mock import patch

import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, insert_row, migrate
from research_assistant.schemas import Claim, Framework, Insight
from research_assistant.stages.distill import (
    build_distill_prompt,
    check_dedup,
    list_insights,
    run_distill,
    save_insights,
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
    insert_row(c, "source", {
        "source_id": "s1",
        "source_type": "youtube",
        "url": "https://youtube.com/watch?v=test",
        "author": "Expert",
        "domain_id": "d1",
        "trust_tier": "core",
        "added_at": "2026-01-01T00:00:00Z",
        "active": 1,
    })
    insert_row(c, "content_item", {
        "content_id": "c1",
        "source_id": "s1",
        "ingested_at": "2026-01-01T00:00:00Z",
        "content_type": "transcript",
        "title": "Expert Analysis",
        "author": "Expert",
        "raw_text": "The yield curve inversion signals recession within 18 months because banks tighten lending.",
        "word_count": 13,
        "format_metadata": "{}",
        "processing_status": "success",
    })
    return c


SAMPLE_LLM_RESPONSE = json.dumps([
    {
        "insight_type": "framework",
        "framework": {
            "name": "yield_curve_inversion_signal",
            "author_attribution": "Expert",
            "mechanism": "When yield curve inverts, banks tighten lending, reducing credit availability",
            "conditions": ["2y/10y spread goes negative"],
            "predictions": ["Recession within 18 months"],
            "assumptions": ["Fed does not intervene with QE"],
            "evidence_cited": ["Historical inversions 1980-2020"],
            "confidence_language": "strong historical pattern",
        },
        "source_quote_ref": "near beginning of transcript",
    },
    {
        "insight_type": "claim",
        "claim": {
            "statement": "Recession likely by Q4 2027",
            "reasoning": "Current inversion depth matches 2006 pattern",
            "timeframe": "Q4 2027",
            "falsifiable": True,
            "falsification_trigger": "No NBER recession declared by Q4 2027",
        },
        "source_quote_ref": "middle of transcript",
    },
])


class TestBuildDistillPrompt:
    def test_contains_content(self):
        system, prompt = build_distill_prompt(
            "Some transcript text", '{"confidence": "medium"}', "both", "rate expectations",
        )
        assert "reasoning extraction" in system.lower()
        assert "Some transcript text" in prompt
        assert "rate expectations" in prompt
        assert "both" in prompt


class TestRunDistill:
    @patch("research_assistant.llm.call_llm")
    def test_returns_insights(self, mock_call, conn, settings):
        mock_call.return_value = SAMPLE_LLM_RESPONSE
        insights = run_distill("c1", "d1", "both", None, conn, settings)
        assert len(insights) == 2
        assert insights[0].insight_type == "framework"
        assert insights[0].framework.name == "yield_curve_inversion_signal"
        assert insights[1].insight_type == "claim"
        assert insights[1].claim.falsifiable is True


class TestCheckDedup:
    def test_no_duplicate(self, conn):
        insight = Insight(
            content_id="c1",
            source_id="s1",
            domain_id="d1",
            insight_type="framework",
            framework=Framework(
                name="new_framework",
                author_attribution="Author",
                mechanism="something causes something",
                conditions=["condition"],
                predictions=["prediction"],
                assumptions=["assumption"],
                evidence_cited=["evidence"],
                confidence_language="likely",
            ),
            source_quote_ref="test",
        )
        assert check_dedup(insight, conn) is False

    def test_duplicate_detected(self, conn):
        # Insert an existing insight
        insert_row(conn, "insight", {
            "insight_id": "existing",
            "content_id": "c1",
            "source_id": "s1",
            "domain_id": "d1",
            "extracted_at": "2026-01-01T00:00:00Z",
            "insight_type": "framework",
            "framework_json": json.dumps({"name": "yield_curve_signal"}),
            "source_quote_ref": "test",
            "status": "active",
        })
        insight = Insight(
            content_id="c1",
            source_id="s1",
            domain_id="d1",
            insight_type="framework",
            framework=Framework(
                name="yield_curve_signal",
                author_attribution="Author",
                mechanism="cause and effect",
                conditions=["condition"],
                predictions=["pred"],
                assumptions=["assumption"],
                evidence_cited=["ev"],
                confidence_language="strong",
            ),
            source_quote_ref="test",
        )
        assert check_dedup(insight, conn) is True


class TestSaveAndList:
    def test_save_and_list(self, conn):
        insights = [
            Insight(
                content_id="c1",
                source_id="s1",
                domain_id="d1",
                insight_type="claim",
                claim=Claim(
                    statement="Test claim",
                    reasoning="Because reasons",
                    falsifiable=True,
                    falsification_trigger="If X then wrong",
                ),
                source_quote_ref="end of transcript",
            ),
        ]
        ids = save_insights(insights, conn)
        assert len(ids) == 1

        rows = list_insights("d1", conn)
        assert len(rows) == 1
        assert rows[0]["insight_type"] == "claim"

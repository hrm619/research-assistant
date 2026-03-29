"""End-to-end integration test for the research assistant pipeline.

Mocks LLM and YouTube extractor to test the full pipeline:
orient → ingest → distill → translate → export
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from research_assistant.cli import cli
from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, list_rows, migrate
from research_assistant.schemas import (
    ContentItem,
    DomainBriefContent,
    FormatMetadata,
    OperatorContext,
    OrientInput,
)
from research_assistant.stages.orient import run_orient, save_domain_brief
from research_assistant.stages.ingest import ingest_content, register_source
from research_assistant.stages.distill import run_distill, save_insights
from research_assistant.stages.translate import (
    export_for_harness,
    run_translate,
    save_hypotheses,
)


DOMAIN_BRIEF_DATA = {
    "market_mechanics": {
        "instrument_type": "binary",
        "settlement": "cash",
        "liquidity_profile": "moderate",
        "fee_structure": "2%",
        "position_types": ["long", "short"],
        "known_biases": ["retail skew"],
    },
    "game_theory": {
        "participant_types": ["retail", "institutional"],
        "information_asymmetries": ["economic data access"],
        "common_mistakes": ["anchoring"],
    },
    "current_meta": {
        "dominant_narratives": ["soft landing"],
        "consensus_view": "rates hold",
        "contrarian_angles": ["recession underpriced"],
    },
    "analogies": [
        {"known_domain": "sports", "mapping": "binary outcomes", "where_analogy_breaks": "no stats"},
        {"known_domain": "politics", "mapping": "event contracts", "where_analogy_breaks": "quant anchors"},
    ],
    "key_data_sources": ["FRED", "CME"],
    "open_questions": ["Volume drivers?", "Hedging mechanics?", "Spread dynamics?"],
    "confidence": "medium",
}

DISTILL_RESPONSE = json.dumps([
    {
        "insight_type": "framework",
        "framework": {
            "name": "yield_curve_signal",
            "author_attribution": "Expert",
            "mechanism": "Inversion tightens lending, causing recession",
            "conditions": ["2y/10y goes negative"],
            "predictions": ["Recession in 18 months"],
            "assumptions": ["No QE intervention"],
            "evidence_cited": ["Historical data 1980-2020"],
            "confidence_language": "strong pattern",
        },
        "source_quote_ref": "early in video",
    },
    {
        "insight_type": "claim",
        "claim": {
            "statement": "Fed holds through Q3",
            "reasoning": "Inflation data stabilizing",
            "timeframe": "Q3 2026",
            "falsifiable": True,
            "falsification_trigger": "Rate cut before July 2026",
        },
        "source_quote_ref": "middle of video",
    },
])

TRANSLATE_RESPONSE = json.dumps([
    {
        "definition": {
            "name": "yield_curve_recession_predictor",
            "statement": "Yield curve inversion predicts recession contract YES",
            "factor": "2y10y spread",
            "classification": "binary",
            "outcome_measure": "Kalshi recession YES settles",
            "timeframe": "18 months",
            "data_required": ["FRED yield curve"],
            "data_available": True,
            "market_expression": "Buy Kalshi recession YES on inversion",
        },
        "feasibility": {
            "data_gap": [],
            "knowledge_gap": ["historical hit rate"],
            "minimum_sample_size": 5,
            "estimated_testability": "medium",
        },
        "reasoning_chain": {
            "from_insight": "yield curve inversion signal",
            "translation_logic": "inversion -> recession -> YES settles",
            "assumptions_added": ["Kalshi matches NBER definition"],
            "weaknesses": ["small sample"],
        },
    },
    {
        "definition": {
            "name": "fed_hold_q3_predictor",
            "statement": "Fed holds rates, rates-hold contract settles YES",
            "factor": "fed funds rate",
            "classification": "binary",
            "outcome_measure": "No rate change by Q3 2026",
            "timeframe": "Q3 2026",
            "data_required": ["FOMC schedule", "CME FedWatch"],
            "data_available": True,
            "market_expression": "Buy Kalshi no-change contract",
        },
        "feasibility": {
            "data_gap": [],
            "knowledge_gap": [],
            "minimum_sample_size": None,
            "estimated_testability": "high",
        },
        "reasoning_chain": {
            "from_insight": "Fed holds claim based on stabilizing inflation",
            "translation_logic": "If inflation stable -> Fed holds -> contract settles YES",
            "assumptions_added": ["No external shock"],
            "weaknesses": ["Single expert view"],
        },
    },
])


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="test-key",
        db_path=":memory:",
        llm_max_retries=1,
        llm_backoff_base=0.01,
    )


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    migrate(c)
    return c


class TestFullPipeline:
    """Tests the full pipeline programmatically with mocked externals."""

    def test_end_to_end(self, conn, settings):
        # 1. Orient
        brief = DomainBriefContent(**DOMAIN_BRIEF_DATA)
        domain_id = save_domain_brief(brief, "fed_rate_decisions", "kalshi", conn)
        row = get_row(conn, "domain_brief", "domain_id", domain_id)
        assert row is not None
        assert row["domain_name"] == "fed_rate_decisions"

        # 2. Ingest (mock YouTube)
        sid1 = register_source("youtube", "https://youtube.com/watch?v=vid1", "Expert1", domain_id, "core", conn)
        sid2 = register_source("youtube", "https://youtube.com/watch?v=vid2", "Expert2", domain_id, "core", conn)

        with patch("research_assistant.stages.ingest.extract_youtube") as mock_yt:
            for sid, title in [(sid1, "Video 1"), (sid2, "Video 2")]:
                mock_yt.return_value = ContentItem(
                    source_id=sid,
                    content_type="transcript",
                    title=title,
                    author="Expert",
                    raw_text="The yield curve inversion is a strong signal. Fed will hold through Q3.",
                    word_count=14,
                    format_metadata=FormatMetadata(),
                    processing_status="success",
                )
                content = ingest_content(sid, conn, settings)
                assert content.processing_status == "success"

        content_rows = conn.execute(
            "SELECT * FROM content_item ci JOIN source s ON ci.source_id = s.source_id WHERE s.domain_id = ?",
            (domain_id,),
        ).fetchall()
        assert len(content_rows) == 2

        # 3. Distill (mock LLM)
        with patch("research_assistant.llm.call_llm") as mock_llm:
            mock_llm.return_value = DISTILL_RESPONSE
            cid = content_rows[0]["content_id"]
            insights = run_distill(cid, domain_id, "both", None, conn, settings)
            assert len(insights) == 2
            insight_ids = save_insights(insights, conn)
            assert len(insight_ids) == 2

        insight_rows = list_rows(conn, "insight", {"domain_id": domain_id})
        assert len(insight_rows) == 2
        types = {r["insight_type"] for r in insight_rows}
        assert types == {"framework", "claim"}

        # 4. Translate - explore mode (mock LLM)
        with patch("research_assistant.stages.translate.call_llm") as mock_llm:
            mock_llm.return_value = TRANSLATE_RESPONSE
            op = OperatorContext(
                accessible_markets=["kalshi"],
                available_data_sources=["FRED yield curve"],
            )
            hypotheses = run_translate(insight_ids, domain_id, "explore", op, conn, settings)
            assert len(hypotheses) == 2
            hyp_ids = save_hypotheses(hypotheses, insight_ids, conn)
            assert len(hyp_ids) == 2

        # Verify junction table
        junctions = list_rows(conn, "hypothesis_insight")
        assert len(junctions) == 4  # 2 hypotheses x 2 insights

        # 5. Export
        exported = export_for_harness(hyp_ids[0], conn)
        assert exported is not None
        assert exported["definition"]["name"] == "yield_curve_recession_predictor"
        assert "reasoning_chain" in exported

        # Verify counts
        sources = list_rows(conn, "source", {"domain_id": domain_id})
        assert len(sources) == 2
        hyps = list_rows(conn, "hypothesis", {"domain_id": domain_id})
        assert len(hyps) == 2


class TestCLICommands:
    """Tests CLI commands via Click's test runner."""

    def test_status_command(self, conn, settings):
        # Set up some data
        brief = DomainBriefContent(**DOMAIN_BRIEF_DATA)
        save_domain_brief(brief, "test_domain", "kalshi", conn)

        runner = CliRunner()

        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=conn):
            result = runner.invoke(cli, ["status", "--domain", "test_domain"])

        assert result.exit_code == 0
        assert "test_domain" in result.output
        assert "kalshi" in result.output

    def test_export_not_found(self, conn, settings):
        runner = CliRunner()
        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=conn):
            result = runner.invoke(cli, ["export", "--hypothesis-id", "nonexistent"])
        assert "not found" in result.output.lower()

    def test_list_insights_empty(self, conn, settings):
        brief = DomainBriefContent(**DOMAIN_BRIEF_DATA)
        save_domain_brief(brief, "test_domain", "kalshi", conn)

        runner = CliRunner()
        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=conn):
            result = runner.invoke(cli, ["list", "insights", "--domain", "test_domain"])
        assert result.exit_code == 0
        assert "no insights" in result.output.lower()

"""End-to-end integration test for the refactored pipeline.

Mocks LLM and OpenAI to test: orient → retrieve → distill (batch) → translate (corpus) → export
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from click.testing import CliRunner

from research_assistant.cli import cli
from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, list_rows, migrate
from research_assistant.schemas import DomainBriefContent, OperatorContext
from research_assistant.stages.orient import save_domain_brief
from research_assistant.stages.retrieve import run_retrieve
from research_assistant.stages.distill import run_distill_batch
from research_assistant.stages.translate import run_translate_corpus, save_hypotheses


KB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS content_record (
    content_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    title TEXT NOT NULL,
    analyst TEXT NOT NULL,
    source_type TEXT NOT NULL,
    trust_tier TEXT NOT NULL,
    url TEXT DEFAULT '',
    published_at TEXT DEFAULT '',
    season INTEGER DEFAULT 0,
    content_tag TEXT DEFAULT '',
    raw_text_hash TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
"""

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

DISTILL_RESPONSE_1 = json.dumps([{
    "insight_type": "framework",
    "framework": {
        "name": "yield_curve_signal",
        "author_attribution": "Expert1",
        "mechanism": "Inversion tightens lending, causing recession",
        "conditions": ["2y/10y goes negative"],
        "predictions": ["Recession in 18 months"],
        "assumptions": ["No QE intervention"],
        "evidence_cited": ["Historical data 1980-2020"],
        "confidence_language": "strong pattern",
    },
    "source_quote_ref": "early in video",
}])

DISTILL_RESPONSE_2 = json.dumps([{
    "insight_type": "claim",
    "claim": {
        "statement": "Fed holds through Q3",
        "reasoning": "Inflation data stabilizing",
        "timeframe": "Q3 2026",
        "falsifiable": True,
        "falsification_trigger": "Rate cut before July 2026",
    },
    "source_quote_ref": "middle of video",
}])

TRANSLATE_RESPONSE = json.dumps([{
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
    "supporting_insight_ids": [],
    "contradicting_insight_ids": [],
    "source_coverage": {"analysts": ["Expert1", "Expert2"], "trust_tiers": ["core"], "n_sources": 2},
    "synthesis_note": "Both experts agree on the yield curve signal mechanism.",
    "test_definition": {
        "hypothesis_name": "yield_curve_recession_predictor",
        "description": "Yield curve inversion predicts recession",
        "version": "1.0.0",
        "metrics": ["points_per_game_std"],
        "classification": {"type": "quartile", "metric": "points_per_game_std"},
        "outcome": "ats",
        "lookback": "season_to_date",
    },
}])


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


@pytest.fixture
def kb_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(KB_SCHEMA_SQL)
    for cid, analyst in [("c1", "Expert1"), ("c2", "Expert2")]:
        c.execute(
            """INSERT INTO content_record
               (content_id, domain, title, analyst, source_type, trust_tier,
                url, published_at, season, content_tag, raw_text_hash, word_count, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, "fed_rate_decisions", f"Video by {analyst}", analyst, "youtube",
             "core", f"https://youtube.com/{cid}", "2025-09-01", 2025, "",
             f"hash_{cid}", 5000, "2025-10-01T00:00:00Z"),
        )
    c.commit()
    return c


@pytest.fixture
def kb_chroma():
    client = chromadb.Client()
    coll = client.get_or_create_collection("fed_rate_decisions")
    for cid in ["c1", "c2"]:
        coll.add(
            ids=[f"{cid}-chunk-0"],
            documents=[f"[SOURCE: Expert | DATE: 2025 | TYPE: youtube]\n\nYield curve analysis for {cid}."],
            metadatas=[{"content_id": cid, "chunk_index": 0, "chunk_count": 1}],
        )
    return client


class TestFullPipeline:
    @patch("research_assistant.stages.translate.call_llm")
    @patch("research_assistant.llm.call_llm")
    def test_end_to_end(self, mock_llm, mock_translate_llm, conn, kb_conn, kb_chroma, settings):
        """orient → retrieve → distill (batch) → translate (corpus) → export"""
        # 1. Orient
        brief = DomainBriefContent(**DOMAIN_BRIEF_DATA)
        domain_id = save_domain_brief(brief, "fed_rate_decisions", "kalshi", conn)
        row = get_row(conn, "domain_brief", "domain_id", domain_id)
        assert row is not None

        # 2. Retrieve from kb.db
        matched = run_retrieve(kb_conn, conn, "fed_rate_decisions")
        assert len(matched) == 2

        # 3. Distill (batch-driven, mocked LLM)
        mock_llm.side_effect = [DISTILL_RESPONSE_1, DISTILL_RESPONSE_2]
        insights = run_distill_batch(
            "fed_rate_decisions", "both", None, conn, settings, kb_conn, kb_chroma,
        )
        assert len(insights) == 2

        batch_rows = list_rows(conn, "retrieval_batch", {"domain": "fed_rate_decisions"})
        assert all(r["distill_status"] == "distilled" for r in batch_rows)

        insight_rows = list_rows(conn, "insight", {"domain_id": domain_id})
        assert len(insight_rows) == 2
        assert all(r["content_source"] == "kb" for r in insight_rows)

        # 4. Translate (corpus-aware)
        mock_translate_llm.return_value = TRANSLATE_RESPONSE
        seed_id = insight_rows[0]["insight_id"]

        # Populate insights chroma for corpus retrieval
        insights_coll = kb_chroma.get_or_create_collection(
            "insights_fed_rate_decisions", metadata={"hnsw:space": "cosine"},
        )
        for ir in insight_rows:
            insights_coll.add(
                ids=[ir["insight_id"]],
                documents=[f"[TYPE: {ir['insight_type']}] insight text"],
                metadatas=[{
                    "insight_id": ir["insight_id"],
                    "insight_type": ir["insight_type"],
                    "analyst": ir.get("analyst", ""),
                    "trust_tier": ir.get("trust_tier", ""),
                    "domain": "fed_rate_decisions",
                    "content_item_ref": ir.get("content_item_ref", ""),
                    "status": "active",
                    "content_source": "kb",
                }],
            )

        op = OperatorContext(
            accessible_markets=["kalshi"],
            available_data_sources=["FRED yield curve"],
        )
        hypotheses = run_translate_corpus(
            seed_id, "fed_rate_decisions", "explore", op, conn, settings, kb_chroma,
        )
        assert len(hypotheses) == 1
        h = hypotheses[0]
        assert h.source_coverage is not None
        assert h.synthesis_note is not None

        hyp_ids = save_hypotheses(hypotheses, [seed_id], conn)

        # 5. Export
        from research_assistant.stages.translate import export_for_harness
        exported = export_for_harness(hyp_ids[0], conn)
        assert exported["contract_version"] == "1.0.0"
        assert exported["producer"] == "research-assistant"
        assert exported["domain_name"] == "fed_rate_decisions"


class TestCLICommands:
    def test_status_command(self, conn, settings):
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

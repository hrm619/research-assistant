import json
import sqlite3
from unittest.mock import patch

import chromadb
import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, insert_row, migrate
from research_assistant.schemas import Claim, Framework, Insight
from research_assistant.stages.distill import (
    KBContext,
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


KB_SCHEMA = """
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


@pytest.fixture
def kb_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(KB_SCHEMA)
    c.execute(
        "INSERT INTO content_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "kb-c1", "test_domain", "Expert Analysis",
            "Expert", "youtube", "core", "https://youtube.com/1",
            "2025-06-01", 2025, "", "hash123", 5000,
            "2025-06-01T12:00:00",
        ),
    )
    c.commit()
    return c


@pytest.fixture
def kb_chroma():
    client = chromadb.Client()
    coll = client.get_or_create_collection("test_domain")
    coll.add(
        ids=["chunk-0", "chunk-1"],
        documents=[
            "[SOURCE: Expert | DATE: 2025-06-01 | TYPE: youtube]\n\nThe yield curve inversion signals recession within 18 months.",
            "[SOURCE: Expert | DATE: 2025-06-01 | TYPE: youtube]\n\nBanks tighten lending when curve inverts.",
        ],
        metadatas=[
            {"content_id": "kb-c1", "chunk_index": 0, "chunk_count": 2},
            {"content_id": "kb-c1", "chunk_index": 1, "chunk_count": 2},
        ],
    )
    return client


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
    def test_returns_insights(self, mock_call, conn, settings, kb_conn, kb_chroma):
        mock_call.return_value = SAMPLE_LLM_RESPONSE
        kb_ctx = KBContext(kb_conn=kb_conn, chroma_client=kb_chroma, collection_name="test_domain")
        insights = run_distill("kb-c1", "d1", "both", None, conn, settings, kb_context=kb_ctx)
        assert len(insights) == 2
        assert insights[0].insight_type == "framework"
        assert insights[0].framework.name == "yield_curve_inversion_signal"
        assert insights[1].insight_type == "claim"
        assert insights[1].claim.falsifiable is True


class TestCheckDedup:
    def test_no_duplicate(self, conn):
        insight = Insight(
            content_id="c1",
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
        insert_row(conn, "insight", {
            "insight_id": "existing",
            "content_id": "c1",
            "source_id": "",
            "domain_id": "d1",
            "extracted_at": "2026-01-01T00:00:00Z",
            "insight_type": "framework",
            "framework_json": json.dumps({"name": "yield_curve_signal"}),
            "source_quote_ref": "test",
            "status": "active",
        })
        insight = Insight(
            content_id="c1",
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
                domain_id="d1",
                insight_type="claim",
                claim=Claim(
                    statement="Test claim",
                    reasoning="Because reasons",
                    falsifiable=True,
                    falsification_trigger="If X then wrong",
                ),
                source_quote_ref="end of transcript",
                content_source="kb",
            ),
        ]
        ids = save_insights(insights, conn)
        assert len(ids) == 1

        rows = list_insights("d1", conn)
        assert len(rows) == 1
        assert rows[0]["insight_type"] == "claim"


class TestRunDistillFromKB:
    @patch("research_assistant.llm.call_llm")
    def test_returns_kb_insights(self, mock_call, conn, settings, kb_conn, kb_chroma):
        mock_call.return_value = SAMPLE_LLM_RESPONSE
        kb_ctx = KBContext(kb_conn=kb_conn, chroma_client=kb_chroma, collection_name="test_domain")

        insights = run_distill("kb-c1", "d1", "both", None, conn, settings, kb_context=kb_ctx)

        assert len(insights) == 2
        assert insights[0].content_source == "kb"
        assert insights[0].analyst == "Expert"
        assert insights[0].trust_tier == "core"
        assert insights[0].source_id == ""
        assert insights[0].content_id == "kb-c1"

    @patch("research_assistant.llm.call_llm")
    def test_kb_insights_save_and_round_trip(self, mock_call, conn, settings, kb_conn, kb_chroma):
        mock_call.return_value = SAMPLE_LLM_RESPONSE
        kb_ctx = KBContext(kb_conn=kb_conn, chroma_client=kb_chroma, collection_name="test_domain")

        insights = run_distill("kb-c1", "d1", "both", None, conn, settings, kb_context=kb_ctx)
        ids = save_insights(insights, conn)
        assert len(ids) == 2

        rows = list_insights("d1", conn)
        assert len(rows) == 2
        kb_rows = [r for r in rows if r["content_source"] == "kb"]
        assert len(kb_rows) == 2
        assert kb_rows[0]["analyst"] == "Expert"
        assert kb_rows[0]["trust_tier"] == "core"

    def test_kb_content_not_found(self, conn, settings, kb_conn, kb_chroma):
        kb_ctx = KBContext(kb_conn=kb_conn, chroma_client=kb_chroma, collection_name="test_domain")
        with pytest.raises(ValueError, match="KB content not found"):
            run_distill("nonexistent", "d1", "both", None, conn, settings, kb_context=kb_ctx)

"""Tests for batch-driven distill: retrieve → distill via retrieval_batch."""

import json
import sqlite3
from unittest.mock import patch

import chromadb
import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, insert_row, list_rows, migrate, update_row
from research_assistant.stages.distill import run_distill_batch, save_insights
from research_assistant.stages.retrieve import run_retrieve


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

SAMPLE_LLM_RESPONSE = json.dumps([
    {
        "insight_type": "framework",
        "framework": {
            "name": "workload_value_signal",
            "author_attribution": "barrett",
            "mechanism": "RB snap share above 70% correlates with top-12 finish",
            "conditions": ["snap share > 70%"],
            "predictions": ["Top-12 RB finish"],
            "assumptions": ["No injury"],
            "evidence_cited": ["2024 season data"],
            "confidence_language": "strong pattern",
        },
        "source_quote_ref": "early in video",
    },
    {
        "insight_type": "claim",
        "claim": {
            "statement": "Barkley finishes top 3 in PPR",
            "reasoning": "Eagles offense plus high snap share",
            "timeframe": "2025 season",
            "falsifiable": True,
            "falsification_trigger": "Barkley finishes outside top 3 RB in PPR",
        },
        "source_quote_ref": "middle of video",
    },
])


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="test-key",
        llm_max_retries=1,
        llm_backoff_base=0.01,
    )


@pytest.fixture
def ra_conn():
    c = get_connection(":memory:")
    migrate(c)
    insert_row(c, "domain_brief", {
        "domain_id": "d1",
        "domain_name": "nfl",
        "market_type": "polymarket",
        "created_at": "2026-01-01T00:00:00Z",
        "brief_json": '{"confidence": "medium"}',
        "status": "active",
    })
    return c


@pytest.fixture
def kb_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(KB_SCHEMA_SQL)
    return c


@pytest.fixture
def kb_chroma():
    return chromadb.Client()


def _insert_kb_content(kb_conn, content_id, domain="nfl", analyst="barrett",
                       trust_tier="core"):
    kb_conn.execute(
        """INSERT INTO content_record
           (content_id, domain, title, analyst, source_type, trust_tier,
            url, published_at, season, content_tag, raw_text_hash, word_count, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content_id, domain, f"Title {content_id}", analyst, "youtube",
         trust_tier, f"https://example.com/{content_id}", "2025-09-15",
         2025, "", f"hash_{content_id}", 5000, "2025-10-01T00:00:00Z"),
    )
    kb_conn.commit()


def _add_chroma_chunks(chroma_client, domain, content_id, texts=None):
    coll = chroma_client.get_or_create_collection(domain)
    if texts is None:
        texts = [
            f"[SOURCE: analyst | DATE: 2025-09-15 | TYPE: youtube]\n\nRB workload predicts fantasy value for {content_id}.",
            f"[SOURCE: analyst | DATE: 2025-09-15 | TYPE: youtube]\n\nTarget share matters more than rushing in PPR for {content_id}.",
        ]
    coll.add(
        ids=[f"{content_id}-chunk-{i}" for i in range(len(texts))],
        documents=texts,
        metadatas=[
            {"content_id": content_id, "chunk_index": i, "chunk_count": len(texts)}
            for i in range(len(texts))
        ],
    )


class TestRunDistillBatch:
    @patch("research_assistant.llm.call_llm")
    def test_processes_pending_items(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        _insert_kb_content(kb_conn, "c1")
        _add_chroma_chunks(kb_chroma, "nfl", "c1")
        run_retrieve(kb_conn, ra_conn, "nfl")

        insights = run_distill_batch(
            "nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma,
        )

        assert len(insights) == 2
        assert all(i.content_source == "kb" for i in insights)
        assert all(i.content_item_ref == "c1" for i in insights)
        assert all(i.content_id == "c1" for i in insights)

    @patch("research_assistant.llm.call_llm")
    def test_marks_batch_distilled(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        _insert_kb_content(kb_conn, "c1")
        _add_chroma_chunks(kb_chroma, "nfl", "c1")
        run_retrieve(kb_conn, ra_conn, "nfl")

        run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)

        batch_rows = list_rows(ra_conn, "retrieval_batch", {"domain": "nfl"})
        assert len(batch_rows) == 1
        assert batch_rows[0]["distill_status"] == "distilled"

    @patch("research_assistant.llm.call_llm")
    def test_marks_failed_on_missing_content(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        _insert_kb_content(kb_conn, "c1")
        run_retrieve(kb_conn, ra_conn, "nfl")
        # Don't add chroma chunks — content_record exists but no transcript

        # Remove the content_record so get_content_record returns None
        kb_conn.execute("DELETE FROM content_record WHERE content_id = 'c1'")
        kb_conn.commit()

        insights = run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)

        assert len(insights) == 0
        batch_rows = list_rows(ra_conn, "retrieval_batch", {"domain": "nfl"})
        assert batch_rows[0]["distill_status"] == "failed"
        assert "not found" in batch_rows[0]["distill_error"].lower()

    @patch("research_assistant.llm.call_llm")
    def test_skips_already_distilled(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        _insert_kb_content(kb_conn, "c1")
        _insert_kb_content(kb_conn, "c2")
        _add_chroma_chunks(kb_chroma, "nfl", "c1")
        _add_chroma_chunks(kb_chroma, "nfl", "c2")
        run_retrieve(kb_conn, ra_conn, "nfl")

        # Mark c1 as already distilled
        batch_rows = list_rows(ra_conn, "retrieval_batch", {"domain": "nfl"})
        for r in batch_rows:
            if r["content_item_ref"] == "c1":
                update_row(ra_conn, "retrieval_batch", "batch_id", r["batch_id"], {
                    "distill_status": "distilled",
                })

        insights = run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)

        # Should only process c2
        assert len(insights) == 2
        assert all(i.content_item_ref == "c2" for i in insights)

    @patch("research_assistant.llm.call_llm")
    def test_multiple_items_in_batch(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        for cid in ["c1", "c2", "c3"]:
            _insert_kb_content(kb_conn, cid)
            _add_chroma_chunks(kb_chroma, "nfl", cid)
        run_retrieve(kb_conn, ra_conn, "nfl")

        insights = run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)

        assert len(insights) == 6  # 2 insights per content item x 3 items
        batch_rows = list_rows(ra_conn, "retrieval_batch", {"domain": "nfl"})
        assert all(r["distill_status"] == "distilled" for r in batch_rows)

    @patch("research_assistant.llm.call_llm")
    def test_limit_restricts_items(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        for cid in ["c1", "c2", "c3"]:
            _insert_kb_content(kb_conn, cid)
            _add_chroma_chunks(kb_chroma, "nfl", cid)
        run_retrieve(kb_conn, ra_conn, "nfl")

        insights = run_distill_batch(
            "nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma, limit=1,
        )

        assert len(insights) == 2  # 2 insights from 1 content item

    def test_no_pending_returns_empty(self, ra_conn, kb_conn, kb_chroma, settings):
        insights = run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)
        assert insights == []

    @patch("research_assistant.llm.call_llm")
    def test_insights_have_analyst_from_batch(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        _insert_kb_content(kb_conn, "c1", analyst="winks", trust_tier="supplementary")
        _add_chroma_chunks(kb_chroma, "nfl", "c1")
        run_retrieve(kb_conn, ra_conn, "nfl")

        insights = run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)

        assert all(i.analyst == "winks" for i in insights)
        assert all(i.trust_tier == "supplementary" for i in insights)


class TestInsightContentItemRef:
    @patch("research_assistant.llm.call_llm")
    def test_ref_populated_on_batch_distill(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        mock_llm.return_value = SAMPLE_LLM_RESPONSE

        _insert_kb_content(kb_conn, "c1")
        _add_chroma_chunks(kb_chroma, "nfl", "c1")
        run_retrieve(kb_conn, ra_conn, "nfl")

        insights = run_distill_batch("nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma)
        save_insights(insights, ra_conn)

        rows = list_rows(ra_conn, "insight", {"domain_id": "d1"})
        assert len(rows) >= 2
        for row in rows:
            assert row["content_item_ref"] == "c1"
            assert row["content_source"] == "kb"

    def test_ref_backfilled_on_migration(self):
        c = get_connection(":memory:")
        migrate(c)
        insert_row(c, "domain_brief", {
            "domain_id": "d1", "domain_name": "test",
            "market_type": "kalshi", "created_at": "2026-01-01T00:00:00Z",
            "brief_json": "{}", "status": "draft",
        })
        # Insert an insight with content_id but empty content_item_ref
        insert_row(c, "insight", {
            "insight_id": "i1", "content_id": "old-c1",
            "source_id": "", "domain_id": "d1",
            "extracted_at": "2026-01-01T00:00:00Z",
            "insight_type": "framework",
            "framework_json": '{"name":"test"}',
            "source_quote_ref": "test", "status": "active",
        })
        # Clear the auto-set content_item_ref to simulate old data
        c.execute("UPDATE insight SET content_item_ref = '' WHERE insight_id = 'i1'")
        c.commit()

        # Re-run migration to trigger backfill
        migrate(c)

        row = get_row(c, "insight", "insight_id", "i1")
        assert row["content_item_ref"] == "old-c1"


SAMPLE_LLM_RESPONSE_2 = json.dumps([
    {
        "insight_type": "framework",
        "framework": {
            "name": "target_share_ppr_signal",
            "author_attribution": "winks",
            "mechanism": "Target share above 25% predicts WR1 finish in PPR",
            "conditions": ["target share > 25%"],
            "predictions": ["WR1 finish"],
            "assumptions": ["No injury"],
            "evidence_cited": ["2024 PPR data"],
            "confidence_language": "strong correlation",
        },
        "source_quote_ref": "late in video",
    },
    {
        "insight_type": "claim",
        "claim": {
            "statement": "Chase finishes WR1 overall in 2025",
            "reasoning": "Volume plus efficiency",
            "timeframe": "2025 season",
            "falsifiable": True,
            "falsification_trigger": "Chase finishes outside WR1 in PPR",
        },
        "source_quote_ref": "end of video",
    },
])


class TestEndToEndRetrieveDistill:
    @patch("research_assistant.llm.call_llm")
    def test_retrieve_then_distill(self, mock_llm, ra_conn, kb_conn, kb_chroma, settings):
        """Full e2e: ra retrieve → ra distill (batch-driven)."""
        mock_llm.side_effect = [SAMPLE_LLM_RESPONSE, SAMPLE_LLM_RESPONSE_2]

        # Set up KB content
        _insert_kb_content(kb_conn, "c1", analyst="barrett", trust_tier="core")
        _insert_kb_content(kb_conn, "c2", analyst="winks", trust_tier="supplementary")
        _add_chroma_chunks(kb_chroma, "nfl", "c1")
        _add_chroma_chunks(kb_chroma, "nfl", "c2")

        # Step 1: Retrieve
        matched = run_retrieve(kb_conn, ra_conn, "nfl")
        assert len(matched) == 2

        batch_rows = list_rows(ra_conn, "retrieval_batch", {"domain": "nfl"})
        assert all(r["distill_status"] == "pending" for r in batch_rows)

        # Step 2: Distill from batch
        insights = run_distill_batch(
            "nfl", "both", None, ra_conn, settings, kb_conn, kb_chroma,
        )
        assert len(insights) == 4  # 2 per content item

        # Verify batch status
        batch_rows = list_rows(ra_conn, "retrieval_batch", {"domain": "nfl"})
        assert all(r["distill_status"] == "distilled" for r in batch_rows)

        # Verify insights in DB
        insight_rows = list_rows(ra_conn, "insight", {"domain_id": "d1"})
        assert len(insight_rows) == 4
        analysts = {r["analyst"] for r in insight_rows}
        assert "barrett" in analysts
        assert "winks" in analysts
        refs = {r["content_item_ref"] for r in insight_rows}
        assert refs == {"c1", "c2"}

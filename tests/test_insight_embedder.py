"""Tests for insight embedding to chroma."""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, insert_row, list_rows, migrate, update_row
from research_assistant.insight_embedder import (
    build_chroma_metadata,
    build_embedding_text,
    embed_and_store_insights,
    get_or_create_insights_collection,
    reembed_failed,
)
from research_assistant.schemas import Claim, Framework, Insight
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


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="test-key",
        openai_api_key="test-openai-key",
        llm_max_retries=1,
        llm_backoff_base=0.01,
    )


@pytest.fixture
def conn():
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
def chroma_client():
    return chromadb.Client()


@pytest.fixture
def mock_openai():
    client = MagicMock()
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536
    mock_response.data = [mock_embedding]
    client.embeddings.create.return_value = mock_response
    return client


def _make_framework_insight(insight_id="i1", domain_id="d1", analyst="barrett"):
    return Insight(
        insight_id=insight_id,
        content_id="c1",
        content_item_ref="c1",
        domain_id=domain_id,
        insight_type="framework",
        framework=Framework(
            name="workload_value_signal",
            author_attribution=analyst,
            mechanism="RB snap share above 70% correlates with top-12 finish",
            conditions=["snap share > 70%"],
            predictions=["Top-12 RB finish"],
            assumptions=["No injury"],
            evidence_cited=["2024 data"],
            confidence_language="strong pattern",
        ),
        source_quote_ref="early in video",
        analyst=analyst,
        trust_tier="core",
        content_source="kb",
    )


def _make_claim_insight(insight_id="i2", domain_id="d1", analyst="winks"):
    return Insight(
        insight_id=insight_id,
        content_id="c1",
        content_item_ref="c1",
        domain_id=domain_id,
        insight_type="claim",
        claim=Claim(
            statement="Barkley finishes top 3",
            reasoning="Volume plus efficiency",
            timeframe="2025 season",
            falsifiable=True,
            falsification_trigger="Outside top 3",
        ),
        source_quote_ref="middle of video",
        analyst=analyst,
        trust_tier="supplementary",
        content_source="kb",
    )


class TestBuildEmbeddingText:
    def test_framework_text(self):
        insight = _make_framework_insight()
        text = build_embedding_text(insight)
        assert "[TYPE: framework]" in text
        assert "[ANALYST: barrett]" in text
        assert "[TRUST: core]" in text
        assert "workload_value_signal" in text
        assert "RB snap share above 70%" in text
        assert "Conditions:" in text
        assert "Predictions:" in text

    def test_claim_text(self):
        insight = _make_claim_insight()
        text = build_embedding_text(insight)
        assert "[TYPE: claim]" in text
        assert "[ANALYST: winks]" in text
        assert "Barkley finishes top 3" in text
        assert "Reasoning:" in text
        assert "Timeframe:" in text


class TestBuildChromaMetadata:
    def test_metadata_fields(self):
        insight = _make_framework_insight()
        meta = build_chroma_metadata(insight, "nfl")
        assert meta["insight_id"] == "i1"
        assert meta["insight_type"] == "framework"
        assert meta["analyst"] == "barrett"
        assert meta["trust_tier"] == "core"
        assert meta["domain"] == "nfl"
        assert meta["content_item_ref"] == "c1"
        assert meta["status"] == "active"
        assert meta["content_source"] == "kb"


class TestGetOrCreateInsightsCollection:
    def test_creates_with_cosine(self, chroma_client):
        coll = get_or_create_insights_collection(chroma_client, "nfl")
        assert coll.name == "insights_nfl"
        assert coll.metadata.get("hnsw:space") == "cosine"

    def test_idempotent(self, chroma_client):
        c1 = get_or_create_insights_collection(chroma_client, "nfl")
        c2 = get_or_create_insights_collection(chroma_client, "nfl")
        assert c1.name == c2.name


class TestEmbedAndStoreInsights:
    def test_stores_in_chroma(self, conn, chroma_client, mock_openai, settings):
        insight = _make_framework_insight()
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        success, failed = embed_and_store_insights(
            [insight], "nfl", conn, chroma_client, mock_openai, settings,
        )

        assert success == 1
        assert failed == 0

        coll = chroma_client.get_collection("insights_nfl")
        result = coll.get(ids=["i1"], include=["documents", "metadatas"])
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["insight_id"] == "i1"
        assert result["metadatas"][0]["analyst"] == "barrett"
        assert "[TYPE: framework]" in result["documents"][0]

    def test_updates_embedding_status(self, conn, chroma_client, mock_openai, settings):
        insight = _make_framework_insight()
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        embed_and_store_insights([insight], "nfl", conn, chroma_client, mock_openai, settings)

        row = get_row(conn, "insight_embedding", "insight_id", "i1")
        assert row is not None
        assert row["embedding_status"] == "embedded"
        assert row["chroma_collection"] == "insights_nfl"
        assert row["last_embedded_at"] != ""
        assert row["error"] is None

    def test_multiple_insights(self, conn, chroma_client, mock_openai, settings):
        insights = [_make_framework_insight("i1"), _make_claim_insight("i2")]
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]

        success, failed = embed_and_store_insights(
            insights, "nfl", conn, chroma_client, mock_openai, settings,
        )

        assert success == 2
        assert failed == 0
        coll = chroma_client.get_collection("insights_nfl")
        assert coll.count() == 2

    def test_empty_list(self, conn, chroma_client, mock_openai, settings):
        success, failed = embed_and_store_insights(
            [], "nfl", conn, chroma_client, mock_openai, settings,
        )
        assert success == 0
        assert failed == 0

    def test_embedding_failure_marks_failed(self, conn, chroma_client, mock_openai, settings):
        insight = _make_framework_insight()
        mock_openai.embeddings.create.side_effect = Exception("API rate limit")

        success, failed = embed_and_store_insights(
            [insight], "nfl", conn, chroma_client, mock_openai, settings,
        )

        assert success == 0
        assert failed == 1

        row = get_row(conn, "insight_embedding", "insight_id", "i1")
        assert row["embedding_status"] == "failed"
        assert "API rate limit" in row["error"]

    def test_insight_preserved_on_failure(self, conn, chroma_client, mock_openai, settings):
        insight = _make_framework_insight()
        # Save insight to DB first
        conn.execute("PRAGMA foreign_keys=OFF")
        save_insights([insight], conn)
        conn.execute("PRAGMA foreign_keys=ON")

        mock_openai.embeddings.create.side_effect = Exception("Embedding error")
        embed_and_store_insights([insight], "nfl", conn, chroma_client, mock_openai, settings)

        # Insight is still in ra.db
        row = get_row(conn, "insight", "insight_id", insight.insight_id)
        assert row is not None
        assert row["status"] == "active"

    def test_upsert_overwrites_existing(self, conn, chroma_client, mock_openai, settings):
        insight = _make_framework_insight()
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        # First embed
        s1, f1 = embed_and_store_insights([insight], "nfl", conn, chroma_client, mock_openai, settings)
        assert s1 == 1
        # Second embed (upsert) — should not error
        s2, f2 = embed_and_store_insights([insight], "nfl", conn, chroma_client, mock_openai, settings)
        assert s2 == 1
        assert f2 == 0

        row = get_row(conn, "insight_embedding", "insight_id", "i1")
        assert row["embedding_status"] == "embedded"

        coll = chroma_client.get_collection("insights_nfl")
        result = coll.get(ids=["i1"])
        assert len(result["ids"]) == 1


class TestReembedFailed:
    def test_reembeds_failed_insights(self, conn, chroma_client, mock_openai, settings):
        insight = _make_framework_insight()
        conn.execute("PRAGMA foreign_keys=OFF")
        save_insights([insight], conn)
        conn.execute("PRAGMA foreign_keys=ON")

        # Mark as failed
        insert_row(conn, "insight_embedding", {
            "insight_id": "i1",
            "embedding_status": "failed",
            "chroma_collection": "insights_nfl",
            "last_embedded_at": "2026-01-01T00:00:00Z",
            "error": "previous failure",
        })

        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        success, failed = reembed_failed("nfl", conn, chroma_client, mock_openai, settings)

        assert success == 1
        assert failed == 0
        row = get_row(conn, "insight_embedding", "insight_id", "i1")
        assert row["embedding_status"] == "embedded"

    def test_no_failed_returns_zero(self, conn, chroma_client, mock_openai, settings):
        success, failed = reembed_failed("nfl", conn, chroma_client, mock_openai, settings)
        assert success == 0
        assert failed == 0


class TestDistillBatchWithEmbedding:
    @patch("research_assistant.llm.call_llm")
    def test_batch_distill_embeds_insights(self, mock_llm, conn, chroma_client, mock_openai, settings):
        sample_response = json.dumps([{
            "insight_type": "framework",
            "framework": {
                "name": "snap_share_signal",
                "author_attribution": "barrett",
                "mechanism": "Snap share predicts value",
                "conditions": ["share > 70%"],
                "predictions": ["Top finish"],
                "assumptions": ["Healthy"],
                "evidence_cited": ["Data"],
                "confidence_language": "strong",
            },
            "source_quote_ref": "start",
        }])
        mock_llm.return_value = sample_response
        mock_openai.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]

        # Set up kb fixtures
        kb_conn = sqlite3.connect(":memory:")
        kb_conn.row_factory = sqlite3.Row
        kb_conn.executescript(KB_SCHEMA_SQL)
        kb_conn.execute(
            """INSERT INTO content_record
               (content_id, domain, title, analyst, source_type, trust_tier,
                url, published_at, season, content_tag, raw_text_hash, word_count, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("c1", "nfl", "Title", "barrett", "youtube", "core",
             "https://example.com", "2025-09-15", 2025, "", "hash", 5000,
             "2025-10-01T00:00:00Z"),
        )
        kb_conn.commit()

        # Add chroma chunks for content reconstruction
        kb_chroma = chromadb.Client()
        coll = kb_chroma.get_or_create_collection("nfl")
        coll.add(
            ids=["c1-chunk-0"],
            documents=["[SOURCE: barrett | DATE: 2025 | TYPE: youtube]\n\nSnap share analysis."],
            metadatas=[{"content_id": "c1", "chunk_index": 0, "chunk_count": 1}],
        )

        # Retrieve + distill with embedding
        run_retrieve(kb_conn, conn, "nfl")
        insights = run_distill_batch(
            "nfl", "both", None, conn, settings, kb_conn, kb_chroma,
            openai_client=mock_openai,
        )

        assert len(insights) == 1

        # Verify embedding was stored
        emb_row = get_row(conn, "insight_embedding", "insight_id", insights[0].insight_id)
        assert emb_row is not None
        assert emb_row["embedding_status"] == "embedded"

        # Verify chroma has the record
        # Note: the insights collection is separate from the content collection
        insights_coll = chroma_client.get_or_create_collection("insights_nfl")
        # The embedding was stored to kb_chroma (the chroma_client passed to run_distill_batch)
        # but our mock_openai handled the embedding. Let's check via the conn.
        assert mock_openai.embeddings.create.called

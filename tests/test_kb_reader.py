"""Tests for kb_reader — reading content from knowledge-base stores."""

import sqlite3

import chromadb
import pytest

from research_assistant.kb_reader import (
    _strip_header,
    get_content_record,
    list_kb_content,
    reconstruct_transcript,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(KB_SCHEMA)
    conn.execute(
        "INSERT INTO content_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "kb-001", "fantasy_football_2025", "Top 10 RBs",
            "jj", "youtube", "core", "https://youtube.com/1",
            "2025-06-01", 2025, "preview", "abc123hash", 5000,
            "2025-06-01T12:00:00",
        ),
    )
    conn.execute(
        "INSERT INTO content_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "kb-002", "fantasy_football_2025", "WR Sleepers",
            "winks", "youtube", "supplementary", "https://youtube.com/2",
            "2025-06-15", 2025, "preview", "def456hash", 3000,
            "2025-06-15T12:00:00",
        ),
    )
    conn.execute(
        "INSERT INTO content_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "kb-003", "fantasy_football", "Evergreen Tips",
            "barrett", "youtube", "core", "",
            "2024-01-01", 2024, "evergreen", "ghi789hash", 4000,
            "2024-01-01T12:00:00",
        ),
    )
    conn.commit()
    return conn


@pytest.fixture
def chroma_client():
    return chromadb.Client()


@pytest.fixture
def collection(chroma_client):
    coll = chroma_client.get_or_create_collection("fantasy_football_2025")
    # Add 3 chunks for kb-001, out of order to test sorting
    coll.add(
        ids=["c1", "c2", "c3"],
        documents=[
            "[SOURCE: jj | DATE: 2025-06-01 | TYPE: youtube]\n\nChunk zero content here.",
            "[SOURCE: jj | DATE: 2025-06-01 | TYPE: youtube]\n\nChunk one continues the discussion.",
            "[SOURCE: jj | DATE: 2025-06-01 | TYPE: youtube]\n\nChunk two wraps up.",
        ],
        metadatas=[
            {"content_id": "kb-001", "chunk_index": 0, "chunk_count": 3},
            {"content_id": "kb-001", "chunk_index": 1, "chunk_count": 3},
            {"content_id": "kb-001", "chunk_index": 2, "chunk_count": 3},
        ],
    )
    return coll


# ---------------------------------------------------------------------------
# Tests: header stripping
# ---------------------------------------------------------------------------


class TestStripHeader:
    def test_strips_standard_header(self):
        text = "[SOURCE: jj | DATE: 2025-06-01 | TYPE: youtube]\n\nThe actual content."
        assert _strip_header(text) == "The actual content."

    def test_no_header_unchanged(self):
        text = "Just regular text with no header."
        assert _strip_header(text) == text

    def test_header_with_unknown_date(self):
        text = "[SOURCE: analyst | DATE: unknown | TYPE: pdf]\n\nContent here."
        assert _strip_header(text) == "Content here."


# ---------------------------------------------------------------------------
# Tests: content record queries
# ---------------------------------------------------------------------------


class TestGetContentRecord:
    def test_found(self, kb_conn):
        record = get_content_record(kb_conn, "kb-001")
        assert record is not None
        assert record["title"] == "Top 10 RBs"
        assert record["analyst"] == "jj"
        assert record["trust_tier"] == "core"

    def test_not_found(self, kb_conn):
        assert get_content_record(kb_conn, "nonexistent") is None


class TestListKBContent:
    def test_filters_by_domain(self, kb_conn):
        results = list_kb_content(kb_conn, "fantasy_football_2025")
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"Top 10 RBs", "WR Sleepers"}

    def test_different_domain(self, kb_conn):
        results = list_kb_content(kb_conn, "fantasy_football")
        assert len(results) == 1
        assert results[0]["analyst"] == "barrett"

    def test_empty_domain(self, kb_conn):
        results = list_kb_content(kb_conn, "nonexistent_domain")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: transcript reconstruction
# ---------------------------------------------------------------------------


class TestReconstructTranscript:
    def test_reconstructs_ordered_text(self, chroma_client, collection):
        text = reconstruct_transcript(chroma_client, "fantasy_football_2025", "kb-001")
        assert "Chunk zero content here." in text
        assert "Chunk one continues the discussion." in text
        assert "Chunk two wraps up." in text
        # Verify ordering
        assert text.index("Chunk zero") < text.index("Chunk one")
        assert text.index("Chunk one") < text.index("Chunk two")

    def test_headers_stripped(self, chroma_client, collection):
        text = reconstruct_transcript(chroma_client, "fantasy_football_2025", "kb-001")
        assert "[SOURCE:" not in text

    def test_missing_content_id_raises(self, chroma_client, collection):
        with pytest.raises(ValueError, match="No chunks found"):
            reconstruct_transcript(chroma_client, "fantasy_football_2025", "nonexistent")

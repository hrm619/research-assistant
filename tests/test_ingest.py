from unittest.mock import patch

import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, insert_row, migrate
from research_assistant.schemas import ContentItem, FormatMetadata
from research_assistant.stages.ingest import (
    ingest_content,
    list_content,
    register_source,
)


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key")


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    migrate(c)
    # Insert a test domain
    insert_row(c, "domain_brief", {
        "domain_id": "d1",
        "domain_name": "test_domain",
        "market_type": "kalshi",
        "created_at": "2026-01-01T00:00:00Z",
        "brief_json": "{}",
        "status": "draft",
    })
    return c


class TestRegisterSource:
    def test_register_valid(self, conn):
        sid = register_source("youtube", "https://youtube.com/watch?v=test", "Author", "d1", "core", conn)
        row = get_row(conn, "source", "source_id", sid)
        assert row is not None
        assert row["source_type"] == "youtube"
        assert row["trust_tier"] == "core"

    def test_register_by_domain_name(self, conn):
        sid = register_source("youtube", "https://youtube.com/watch?v=test", "Author", "test_domain", "core", conn)
        row = get_row(conn, "source", "source_id", sid)
        assert row["domain_id"] == "d1"

    def test_register_invalid_domain(self, conn):
        with pytest.raises(ValueError, match="Domain not found"):
            register_source("youtube", "https://youtube.com/watch?v=test", "Author", "nonexistent", "core", conn)


class TestIngestContent:
    @patch("research_assistant.stages.ingest.extract_youtube")
    def test_ingest_youtube(self, mock_extract, conn, settings):
        sid = register_source("youtube", "https://youtube.com/watch?v=test", "Author", "d1", "core", conn)
        mock_extract.return_value = ContentItem(
            source_id=sid,
            content_type="transcript",
            title="Test Video",
            author="Author",
            raw_text="Some transcript text here",
            word_count=4,
            format_metadata=FormatMetadata(),
            processing_status="success",
        )
        content = ingest_content(sid, conn, settings)
        assert content.title == "Test Video"
        assert content.processing_status == "success"
        # Verify saved to DB
        row = get_row(conn, "content_item", "content_id", content.content_id)
        assert row is not None
        assert row["title"] == "Test Video"

    def test_ingest_unknown_source_type(self, conn, settings):
        # Manually insert a substack source
        insert_row(conn, "source", {
            "source_id": "s_sub",
            "source_type": "substack",
            "url": "https://example.substack.com/p/test",
            "author": "Author",
            "domain_id": "d1",
            "trust_tier": "core",
            "added_at": "2026-01-01T00:00:00Z",
            "active": 1,
        })
        with pytest.raises(NotImplementedError, match="not implemented in MVP"):
            ingest_content("s_sub", conn, settings)


class TestListContent:
    @patch("research_assistant.stages.ingest.extract_youtube")
    def test_list_by_domain(self, mock_extract, conn, settings):
        sid = register_source("youtube", "https://youtube.com/watch?v=test", "Author", "d1", "core", conn)
        mock_extract.return_value = ContentItem(
            source_id=sid,
            content_type="transcript",
            title="Video",
            author="Author",
            raw_text="text",
            word_count=1,
        )
        ingest_content(sid, conn, settings)
        results = list_content("d1", conn)
        assert len(results) == 1
        assert results[0]["title"] == "Video"

    def test_list_empty(self, conn):
        results = list_content("d1", conn)
        assert results == []

    def test_list_nonexistent_domain(self, conn):
        results = list_content("nonexistent", conn)
        assert results == []

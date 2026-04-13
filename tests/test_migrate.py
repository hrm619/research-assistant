"""Tests for ra migrate to-kb-ownership."""

import sqlite3

import pytest

from research_assistant.db import get_connection, get_row, insert_row, list_rows, migrate
from research_assistant.stages.migrate import (
    drop_old_tables,
    export_unmatched_content,
    remap_insight_refs,
    run_migration,
)


OLD_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS source (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    url TEXT NOT NULL,
    author TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    trust_tier TEXT NOT NULL,
    added_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS content_item (
    content_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    content_type TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    published_at TEXT,
    raw_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    format_metadata TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    error_detail TEXT
);
"""

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
def legacy_conn():
    """ra.db with old-style tables populated."""
    c = get_connection(":memory:")
    migrate(c)
    # Create old tables for migration testing
    c.executescript(OLD_TABLES_SQL)
    insert_row(c, "domain_brief", {
        "domain_id": "d1",
        "domain_name": "test_domain",
        "market_type": "kalshi",
        "created_at": "2026-01-01T00:00:00Z",
        "brief_json": "{}",
        "status": "draft",
    })
    insert_row(c, "source", {
        "source_id": "s1",
        "source_type": "youtube",
        "url": "https://youtube.com/watch?v=test1",
        "author": "Expert1",
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
        "author": "Expert1",
        "raw_text": "Some content",
        "word_count": 100,
        "format_metadata": "{}",
        "processing_status": "success",
    })
    insert_row(c, "insight", {
        "insight_id": "i1",
        "content_id": "c1",
        "source_id": "s1",
        "domain_id": "d1",
        "extracted_at": "2026-01-01T00:00:00Z",
        "insight_type": "framework",
        "framework_json": '{"name": "test_fw"}',
        "source_quote_ref": "test",
        "status": "active",
    })
    return c


@pytest.fixture
def kb_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(KB_SCHEMA_SQL)
    c.execute(
        """INSERT INTO content_record
           (content_id, domain, title, analyst, source_type, trust_tier,
            url, published_at, season, content_tag, raw_text_hash, word_count, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("kb-c1", "test_domain", "Expert Analysis", "Expert1", "youtube", "core",
         "https://youtube.com/watch?v=test1", "2025-09-01", 2025, "", "hash1",
         100, "2025-10-01T00:00:00Z"),
    )
    c.commit()
    return c


class TestExportUnmatchedContent:
    def test_exports_content_items(self, legacy_conn):
        unmatched = export_unmatched_content(legacy_conn, None)
        assert len(unmatched) == 1
        assert unmatched[0]["content_id"] == "c1"

    def test_empty_when_no_content_table(self):
        c = get_connection(":memory:")
        migrate(c)
        unmatched = export_unmatched_content(c, None)
        assert unmatched == []

    def test_matched_against_kb(self, legacy_conn, kb_conn):
        unmatched = export_unmatched_content(legacy_conn, kb_conn)
        assert len(unmatched) == 1  # URL matching works on content_record


class TestRemapInsightRefs:
    def test_remaps_from_kb(self, legacy_conn, kb_conn):
        stats = remap_insight_refs(legacy_conn, kb_conn)
        row = get_row(legacy_conn, "insight", "insight_id", "i1")
        assert stats["remapped"] == 1 or stats["already_set"] == 1
        assert row["content_item_ref"] != ""

    def test_orphaned_when_no_kb(self, legacy_conn):
        # Clear the content_item_ref first
        legacy_conn.execute("UPDATE insight SET content_item_ref = '' WHERE insight_id = 'i1'")
        legacy_conn.commit()
        stats = remap_insight_refs(legacy_conn, None)
        assert stats["orphaned"] == 1


class TestDropOldTables:
    def test_drops_tables(self, legacy_conn):
        dropped = drop_old_tables(legacy_conn)
        assert "content_item" in dropped
        assert "source" in dropped
        tables = legacy_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "content_item" not in table_names
        assert "source" not in table_names

    def test_insight_survives(self, legacy_conn):
        drop_old_tables(legacy_conn)
        row = get_row(legacy_conn, "insight", "insight_id", "i1")
        assert row is not None
        assert row["status"] == "active"


class TestRunMigration:
    def test_dry_run(self, legacy_conn, kb_conn):
        report = run_migration(legacy_conn, kb_conn, ":memory:", dry_run=True)
        assert report["dry_run"] is True
        assert report["dropped_tables"] == []
        # Tables should still exist
        tables = legacy_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "content_item" in table_names

    def test_full_migration(self, legacy_conn, kb_conn):
        report = run_migration(legacy_conn, kb_conn, ":memory:", dry_run=False)
        assert report["dry_run"] is False
        assert "content_item" in report["dropped_tables"]
        assert "source" in report["dropped_tables"]

        # Insight preserved
        row = get_row(legacy_conn, "insight", "insight_id", "i1")
        assert row is not None

    def test_no_old_tables(self):
        c = get_connection(":memory:")
        migrate(c)
        insert_row(c, "domain_brief", {
            "domain_id": "d1", "domain_name": "test",
            "market_type": "kalshi", "created_at": "2026-01-01T00:00:00Z",
            "brief_json": "{}", "status": "draft",
        })
        report = run_migration(c, None, ":memory:", dry_run=False)
        assert report["unmatched_content"] == []

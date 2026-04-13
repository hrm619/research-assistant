"""Tests for the retrieve stage — selecting kb.db content for distillation."""

import sqlite3
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from research_assistant.cli import cli
from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, insert_row, list_rows, migrate
from research_assistant.stages.retrieve import (
    get_existing_refs,
    list_batch_rows,
    query_kb_content,
    run_retrieve,
)


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

CREATE TABLE IF NOT EXISTS vector_sync (
    content_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    synced_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success'
);
"""


@pytest.fixture
def ra_conn():
    c = get_connection(":memory:")
    migrate(c)
    return c


@pytest.fixture
def kb_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(KB_SCHEMA_SQL)
    return c


def _insert_kb_content(kb_conn, content_id, domain="nfl", analyst="barrett",
                       trust_tier="core", source_type="youtube",
                       published_at="2025-09-15", word_count=5000):
    kb_conn.execute(
        """INSERT INTO content_record
           (content_id, domain, title, analyst, source_type, trust_tier,
            url, published_at, season, content_tag, raw_text_hash, word_count, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content_id, domain, f"Title {content_id}", analyst, source_type,
         trust_tier, f"https://example.com/{content_id}", published_at,
         2025, "", f"hash_{content_id}", word_count, "2025-10-01T00:00:00Z"),
    )
    kb_conn.commit()


class TestQueryKBContent:
    def test_basic_domain_filter(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", domain="nfl")
        _insert_kb_content(kb_conn, "c2", domain="macro")
        result = query_kb_content(kb_conn, "nfl")
        assert len(result) == 1
        assert result[0]["content_id"] == "c1"

    def test_trust_tier_filter(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", trust_tier="core")
        _insert_kb_content(kb_conn, "c2", trust_tier="exploratory")
        result = query_kb_content(kb_conn, "nfl", trust_tiers=["core"])
        assert len(result) == 1
        assert result[0]["content_id"] == "c1"

    def test_multiple_trust_tiers(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", trust_tier="core")
        _insert_kb_content(kb_conn, "c2", trust_tier="supplementary")
        _insert_kb_content(kb_conn, "c3", trust_tier="exploratory")
        result = query_kb_content(kb_conn, "nfl", trust_tiers=["core", "supplementary"])
        assert len(result) == 2

    def test_analyst_filter(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", analyst="barrett")
        _insert_kb_content(kb_conn, "c2", analyst="winks")
        result = query_kb_content(kb_conn, "nfl", analysts=["barrett"])
        assert len(result) == 1
        assert result[0]["analyst"] == "barrett"

    def test_source_type_filter(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", source_type="youtube")
        _insert_kb_content(kb_conn, "c2", source_type="article")
        result = query_kb_content(kb_conn, "nfl", source_types=["youtube"])
        assert len(result) == 1

    def test_since_filter(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", published_at="2025-06-01")
        _insert_kb_content(kb_conn, "c2", published_at="2025-12-01")
        result = query_kb_content(kb_conn, "nfl", since="2025-09-01")
        assert len(result) == 1
        assert result[0]["content_id"] == "c2"

    def test_until_filter(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", published_at="2025-06-01")
        _insert_kb_content(kb_conn, "c2", published_at="2025-12-01")
        result = query_kb_content(kb_conn, "nfl", until="2025-09-01")
        assert len(result) == 1
        assert result[0]["content_id"] == "c1"

    def test_combined_filters(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", analyst="barrett", trust_tier="core", published_at="2025-10-01")
        _insert_kb_content(kb_conn, "c2", analyst="winks", trust_tier="core", published_at="2025-10-01")
        _insert_kb_content(kb_conn, "c3", analyst="barrett", trust_tier="exploratory", published_at="2025-10-01")
        _insert_kb_content(kb_conn, "c4", analyst="barrett", trust_tier="core", published_at="2025-01-01")
        result = query_kb_content(
            kb_conn, "nfl",
            trust_tiers=["core"], analysts=["barrett"], since="2025-06-01",
        )
        assert len(result) == 1
        assert result[0]["content_id"] == "c1"

    def test_limit(self, kb_conn):
        for i in range(10):
            _insert_kb_content(kb_conn, f"c{i}", published_at=f"2025-{i+1:02d}-01")
        result = query_kb_content(kb_conn, "nfl", limit=3)
        assert len(result) == 3

    def test_empty_domain(self, kb_conn):
        _insert_kb_content(kb_conn, "c1", domain="macro")
        result = query_kb_content(kb_conn, "nfl")
        assert len(result) == 0


class TestRunRetrieve:
    def test_dry_run_returns_without_writing(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1")
        _insert_kb_content(kb_conn, "c2")
        result = run_retrieve(kb_conn, ra_conn, "nfl", dry_run=True)
        assert len(result) == 2
        batch_rows = list_rows(ra_conn, "retrieval_batch")
        assert len(batch_rows) == 0

    def test_real_run_populates_batch(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1")
        _insert_kb_content(kb_conn, "c2")
        result = run_retrieve(kb_conn, ra_conn, "nfl")
        assert len(result) == 2
        batch_rows = list_rows(ra_conn, "retrieval_batch")
        assert len(batch_rows) == 2
        assert all(r["distill_status"] == "pending" for r in batch_rows)
        assert all(r["domain"] == "nfl" for r in batch_rows)
        refs = {r["content_item_ref"] for r in batch_rows}
        assert refs == {"c1", "c2"}

    def test_dedup_skips_distilled(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1")
        _insert_kb_content(kb_conn, "c2")
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "old:c1",
            "domain": "nfl",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "distilled",
        })
        result = run_retrieve(kb_conn, ra_conn, "nfl")
        assert len(result) == 1
        assert result[0]["content_id"] == "c2"

    def test_dedup_includes_pending(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1")
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "old:c1",
            "domain": "nfl",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "pending",
        })
        result = run_retrieve(kb_conn, ra_conn, "nfl")
        assert len(result) == 1

    def test_force_overrides_dedup(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1")
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "old:c1",
            "domain": "nfl",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "distilled",
        })
        result = run_retrieve(kb_conn, ra_conn, "nfl", force=True)
        assert len(result) == 1

    def test_batch_rows_have_metadata(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1", analyst="barrett", trust_tier="core", source_type="youtube")
        run_retrieve(kb_conn, ra_conn, "nfl")
        batch_rows = list_rows(ra_conn, "retrieval_batch")
        assert len(batch_rows) == 1
        row = batch_rows[0]
        assert row["analyst"] == "barrett"
        assert row["trust_tier"] == "core"
        assert row["source_type"] == "youtube"
        assert row["content_item_ref"] == "c1"
        assert row["retrieved_at"] != ""

    def test_filters_pass_through(self, kb_conn, ra_conn):
        _insert_kb_content(kb_conn, "c1", trust_tier="core")
        _insert_kb_content(kb_conn, "c2", trust_tier="exploratory")
        result = run_retrieve(kb_conn, ra_conn, "nfl", trust_tiers=["core"])
        assert len(result) == 1
        batch_rows = list_rows(ra_conn, "retrieval_batch")
        assert len(batch_rows) == 1


class TestGetExistingRefs:
    def test_returns_distilled_refs(self, ra_conn):
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "b:c1",
            "domain": "nfl",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "distilled",
        })
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "b:c2",
            "domain": "nfl",
            "content_item_ref": "c2",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "pending",
        })
        refs = get_existing_refs(ra_conn, "nfl")
        assert refs == {"c1"}

    def test_scoped_to_domain(self, ra_conn):
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "b:c1",
            "domain": "macro",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "distilled",
        })
        refs = get_existing_refs(ra_conn, "nfl")
        assert refs == set()


class TestListBatchRows:
    def test_list_by_domain(self, ra_conn):
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "b:c1",
            "domain": "nfl",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "pending",
        })
        rows = list_batch_rows(ra_conn, "nfl")
        assert len(rows) == 1

    def test_list_by_status(self, ra_conn):
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "b:c1",
            "domain": "nfl",
            "content_item_ref": "c1",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "pending",
        })
        insert_row(ra_conn, "retrieval_batch", {
            "batch_id": "b:c2",
            "domain": "nfl",
            "content_item_ref": "c2",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "distill_status": "distilled",
        })
        pending = list_batch_rows(ra_conn, "nfl", status="pending")
        assert len(pending) == 1
        assert pending[0]["content_item_ref"] == "c1"


class TestRetrieveCLI:
    @pytest.fixture
    def settings(self):
        return Settings(
            anthropic_api_key="test-key",
            db_path=":memory:",
            kb_db_path=":memory:",
            llm_max_retries=1,
            llm_backoff_base=0.01,
        )

    def test_dry_run_cli(self, ra_conn, kb_conn, settings):
        _insert_kb_content(kb_conn, "c1")

        runner = CliRunner()
        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=ra_conn), \
             patch("research_assistant.kb_reader.get_kb_connection", return_value=kb_conn):
            result = runner.invoke(cli, ["retrieve", "--domain", "nfl", "--dry-run"])

        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()
        assert "1 content items" in result.output

    def test_real_run_cli(self, ra_conn, kb_conn, settings):
        _insert_kb_content(kb_conn, "c1")
        _insert_kb_content(kb_conn, "c2")

        runner = CliRunner()
        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=ra_conn), \
             patch("research_assistant.kb_reader.get_kb_connection", return_value=kb_conn):
            result = runner.invoke(cli, ["retrieve", "--domain", "nfl"])

        assert result.exit_code == 0
        assert "Retrieved 2" in result.output

    def test_filter_flags_cli(self, ra_conn, kb_conn, settings):
        _insert_kb_content(kb_conn, "c1", trust_tier="core", analyst="barrett")
        _insert_kb_content(kb_conn, "c2", trust_tier="exploratory", analyst="winks")

        runner = CliRunner()
        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=ra_conn), \
             patch("research_assistant.kb_reader.get_kb_connection", return_value=kb_conn):
            result = runner.invoke(cli, [
                "retrieve", "--domain", "nfl",
                "--trust-tier", "core", "--analyst", "barrett",
            ])

        assert result.exit_code == 0
        assert "Retrieved 1" in result.output

    def test_no_match_cli(self, ra_conn, kb_conn, settings):
        runner = CliRunner()
        with patch("research_assistant.cli.get_settings", return_value=settings), \
             patch("research_assistant.cli.get_connection", return_value=ra_conn), \
             patch("research_assistant.kb_reader.get_kb_connection", return_value=kb_conn):
            result = runner.invoke(cli, ["retrieve", "--domain", "nfl"])

        assert result.exit_code == 0
        assert "no matching" in result.output.lower()

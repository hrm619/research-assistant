import sqlite3

import pytest

from research_assistant.db import (
    get_connection,
    get_row,
    insert_row,
    list_rows,
    migrate,
    resolve_domain,
    update_row,
)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    migrate(c)
    return c


def _insert_domain(conn, domain_id="d1", domain_name="test_domain"):
    insert_row(conn, "domain_brief", {
        "domain_id": domain_id,
        "domain_name": domain_name,
        "market_type": "kalshi",
        "created_at": "2026-01-01T00:00:00Z",
        "brief_json": "{}",
        "status": "draft",
    })
    return domain_id


def _insert_source(conn, source_id="s1", domain_id="d1"):
    insert_row(conn, "source", {
        "source_id": source_id,
        "source_type": "youtube",
        "url": "https://youtube.com/watch?v=test",
        "author": "Test Author",
        "domain_id": domain_id,
        "trust_tier": "core",
        "added_at": "2026-01-01T00:00:00Z",
        "active": 1,
    })
    return source_id


class TestMigration:
    def test_creates_all_tables(self, conn):
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = sorted(r["name"] for r in tables)
        assert "content_item" in table_names
        assert "domain_brief" in table_names
        assert "hypothesis" in table_names
        assert "hypothesis_insight" in table_names
        assert "insight" in table_names
        assert "source" in table_names

    def test_idempotent(self, conn):
        migrate(conn)
        migrate(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len([r for r in tables if r["name"] == "domain_brief"]) == 1


class TestCRUD:
    def test_insert_and_get(self, conn):
        _insert_domain(conn)
        row = get_row(conn, "domain_brief", "domain_id", "d1")
        assert row is not None
        assert row["domain_name"] == "test_domain"
        assert row["market_type"] == "kalshi"

    def test_get_nonexistent(self, conn):
        assert get_row(conn, "domain_brief", "domain_id", "nope") is None

    def test_list_rows_no_filter(self, conn):
        _insert_domain(conn, "d1")
        _insert_domain(conn, "d2", "other_domain")
        rows = list_rows(conn, "domain_brief")
        assert len(rows) == 2

    def test_list_rows_with_filter(self, conn):
        _insert_domain(conn, "d1")
        _insert_domain(conn, "d2", "other_domain")
        rows = list_rows(conn, "domain_brief", {"domain_name": "test_domain"})
        assert len(rows) == 1
        assert rows[0]["domain_id"] == "d1"

    def test_update_row(self, conn):
        _insert_domain(conn)
        update_row(conn, "domain_brief", "domain_id", "d1", {"status": "active"})
        row = get_row(conn, "domain_brief", "domain_id", "d1")
        assert row["status"] == "active"


class TestForeignKeys:
    def test_source_requires_valid_domain(self, conn):
        with pytest.raises(sqlite3.IntegrityError):
            _insert_source(conn, domain_id="nonexistent")

    def test_source_with_valid_domain(self, conn):
        _insert_domain(conn)
        _insert_source(conn)
        row = get_row(conn, "source", "source_id", "s1")
        assert row is not None


class TestResolveDomain:
    def test_resolve_by_id(self, conn):
        _insert_domain(conn)
        assert resolve_domain(conn, "d1") == "d1"

    def test_resolve_by_name(self, conn):
        _insert_domain(conn)
        assert resolve_domain(conn, "test_domain") == "d1"

    def test_resolve_not_found(self, conn):
        assert resolve_domain(conn, "nope") is None

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS domain_brief (
    domain_id TEXT PRIMARY KEY,
    domain_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    brief_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft'
);

CREATE TABLE IF NOT EXISTS source (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    url TEXT NOT NULL,
    author TEXT NOT NULL,
    domain_id TEXT NOT NULL REFERENCES domain_brief(domain_id),
    trust_tier TEXT NOT NULL,
    added_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS content_item (
    content_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES source(source_id),
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

CREATE TABLE IF NOT EXISTS insight (
    insight_id TEXT PRIMARY KEY,
    content_id TEXT NOT NULL REFERENCES content_item(content_id),
    source_id TEXT NOT NULL REFERENCES source(source_id),
    domain_id TEXT NOT NULL REFERENCES domain_brief(domain_id),
    extracted_at TEXT NOT NULL,
    insight_type TEXT NOT NULL,
    framework_json TEXT,
    claim_json TEXT,
    source_quote_ref TEXT NOT NULL,
    operator_note TEXT,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS hypothesis (
    hypothesis_id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domain_brief(domain_id),
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    definition_json TEXT NOT NULL,
    feasibility_json TEXT NOT NULL,
    reasoning_chain_json TEXT NOT NULL,
    operator_note TEXT
);

CREATE TABLE IF NOT EXISTS hypothesis_insight (
    hypothesis_id TEXT NOT NULL REFERENCES hypothesis(hypothesis_id),
    insight_id TEXT NOT NULL REFERENCES insight(insight_id),
    PRIMARY KEY (hypothesis_id, insight_id)
);
"""


def get_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def insert_row(conn: sqlite3.Connection, table: str, data: dict) -> str:
    columns = ", ".join(data.keys())
    placeholders = ", ".join("?" for _ in data)
    values = list(data.values())
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    return str(values[0])  # return PK (first column by convention)


def get_row(conn: sqlite3.Connection, table: str, pk_col: str, pk_val: str) -> dict | None:
    row = conn.execute(f"SELECT * FROM {table} WHERE {pk_col} = ?", (pk_val,)).fetchone()
    return dict(row) if row else None


def list_rows(
    conn: sqlite3.Connection,
    table: str,
    filters: dict | None = None,
) -> list[dict]:
    query = f"SELECT * FROM {table}"
    params: list = []
    if filters:
        clauses = []
        for col, val in filters.items():
            clauses.append(f"{col} = ?")
            params.append(val)
        query += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_row(
    conn: sqlite3.Connection,
    table: str,
    pk_col: str,
    pk_val: str,
    data: dict,
) -> None:
    set_clause = ", ".join(f"{col} = ?" for col in data)
    values = list(data.values()) + [pk_val]
    conn.execute(f"UPDATE {table} SET {set_clause} WHERE {pk_col} = ?", values)
    conn.commit()


def resolve_domain(conn: sqlite3.Connection, identifier: str) -> str | None:
    row = get_row(conn, "domain_brief", "domain_id", identifier)
    if row:
        return row["domain_id"]
    row = conn.execute(
        "SELECT domain_id FROM domain_brief WHERE domain_name = ?", (identifier,)
    ).fetchone()
    return row["domain_id"] if row else None

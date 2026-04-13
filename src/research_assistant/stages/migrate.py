"""Migration: move from ra-owned content to kb-owned content."""

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def backup_db(db_path: str) -> str:
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dst = src.with_name(f"{src.stem}.bak-{ts}{src.suffix}")
    shutil.copy2(src, dst)
    logger.info("Backed up %s to %s", src, dst)
    return str(dst)


def export_unmatched_content(
    ra_conn: sqlite3.Connection,
    kb_conn: sqlite3.Connection | None,
) -> list[dict]:
    has_content_item = ra_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='content_item'"
    ).fetchone()
    if not has_content_item:
        return []

    rows = ra_conn.execute("SELECT * FROM content_item").fetchall()
    if not rows:
        return []

    unmatched = []
    for row in rows:
        row = dict(row)
        matched = False
        if kb_conn:
            try:
                kb_row = kb_conn.execute(
                    "SELECT content_id FROM content_record WHERE url = ? OR raw_text_hash = ?",
                    (row.get("url", ""), row.get("checksum", "")),
                ).fetchone()
                if kb_row:
                    matched = True
            except sqlite3.OperationalError:
                pass

        if not matched:
            unmatched.append({
                "content_id": row["content_id"],
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "source_type": "unknown",
                "word_count": row.get("word_count", 0),
            })

    return unmatched


def remap_insight_refs(
    ra_conn: sqlite3.Connection,
    kb_conn: sqlite3.Connection | None,
) -> dict:
    stats = {"remapped": 0, "already_set": 0, "orphaned": 0}

    rows = ra_conn.execute(
        "SELECT insight_id, content_id, content_item_ref FROM insight"
    ).fetchall()

    for row in rows:
        row = dict(row)
        if row.get("content_item_ref") and row["content_item_ref"] != "":
            stats["already_set"] += 1
            continue

        content_id = row["content_id"]
        kb_match = None

        if kb_conn:
            has_content_item = ra_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='content_item'"
            ).fetchone()
            if has_content_item:
                ci_row = ra_conn.execute(
                    "SELECT ci.*, s.url FROM content_item ci "
                    "LEFT JOIN source s ON ci.source_id = s.source_id "
                    "WHERE ci.content_id = ?",
                    (content_id,),
                ).fetchone()
                if ci_row:
                    ci_row = dict(ci_row)
                    try:
                        kb_match = kb_conn.execute(
                            "SELECT content_id FROM content_record WHERE url = ?",
                            (ci_row.get("url", ""),),
                        ).fetchone()
                    except sqlite3.OperationalError:
                        pass

        if kb_match:
            ra_conn.execute(
                "UPDATE insight SET content_item_ref = ? WHERE insight_id = ?",
                (kb_match[0], row["insight_id"]),
            )
            stats["remapped"] += 1
        else:
            ra_conn.execute(
                "UPDATE insight SET content_item_ref = ?, status = 'orphaned' WHERE insight_id = ? AND content_item_ref = ''",
                (content_id, row["insight_id"]),
            )
            stats["orphaned"] += 1

    ra_conn.commit()
    return stats


def drop_old_tables(ra_conn: sqlite3.Connection) -> list[str]:
    dropped = []
    ra_conn.execute("PRAGMA foreign_keys=OFF")
    for table in ["content_item", "source"]:
        try:
            ra_conn.execute(f"DROP TABLE IF EXISTS {table}")
            dropped.append(table)
        except sqlite3.OperationalError as e:
            logger.warning("Could not drop %s: %s", table, e)
    ra_conn.execute("PRAGMA foreign_keys=ON")
    ra_conn.commit()
    return dropped


def run_migration(
    ra_conn: sqlite3.Connection,
    kb_conn: sqlite3.Connection | None,
    db_path: str,
    dry_run: bool = False,
) -> dict:
    report = {
        "backup_path": None,
        "unmatched_content": [],
        "remap_stats": {},
        "dropped_tables": [],
        "dry_run": dry_run,
    }

    if not dry_run:
        if db_path != ":memory:":
            report["backup_path"] = backup_db(db_path)

    report["unmatched_content"] = export_unmatched_content(ra_conn, kb_conn)

    if dry_run:
        return report

    report["remap_stats"] = remap_insight_refs(ra_conn, kb_conn)
    report["dropped_tables"] = drop_old_tables(ra_conn)

    return report

"""Read-only access to the operations database.

Agents can generate SQL, so the database layer does not trust the query:
it must be a single SELECT (or WITH ... SELECT) statement, write keywords are
rejected, the connection itself is opened read-only, and results are capped.
Validation errors are returned as messages instead of raised, so an agent can
read the error and fix its query.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from . import config

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    pass


def validate_query(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Empty query.")
    if ";" in cleaned:
        raise UnsafeQueryError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if FORBIDDEN.search(cleaned):
        raise UnsafeQueryError("Query contains a write or admin keyword, which is not allowed.")
    if not re.search(r"\blimit\s+\d+\s*$", cleaned, re.IGNORECASE):
        cleaned = f"{cleaned} LIMIT {config.SQL_ROW_LIMIT}"
    return cleaned


def _connect() -> sqlite3.Connection:
    if not config.DB_PATH.exists():
        raise FileNotFoundError(
            f"{config.DB_PATH} not found. Run: python scripts/build_database.py"
        )
    return sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)


def run_query(sql: str) -> dict[str, Any]:
    """Runs a validated read-only query. Returns columns + rows, or an error."""
    try:
        safe_sql = validate_query(sql)
    except UnsafeQueryError as exc:
        return {"ok": False, "sql": sql, "error": str(exc)}
    try:
        with _connect() as con:
            cur = con.execute(safe_sql)
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchall()]
    except sqlite3.Error as exc:
        return {"ok": False, "sql": safe_sql, "error": f"SQL error: {exc}"}
    return {"ok": True, "sql": safe_sql, "columns": columns, "rows": rows}


def describe_schema() -> str:
    with _connect() as con:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return "\n\n".join(r[0].strip() for r in rows)


def customer_names() -> list[str]:
    with _connect() as con:
        return [r[0] for r in con.execute("SELECT name FROM customers ORDER BY name")]

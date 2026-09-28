"""LangChain tools the specialist agents can call.

Tools return JSON strings, because that is what goes back to the model as a
ToolMessage. The specialist nodes parse the same JSON to record findings.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from . import config, sql_tools
from .retrieval import get_retriever


@tool
def search_docs(query: str) -> str:
    """Search internal company documents (refund, support SLA, shipping, data access,
    incident response and IT onboarding policies). Returns the most relevant
    paragraphs with their source. Use specific keywords, for example
    'Enterprise first response time high priority'."""
    hits = get_retriever().search(query, k=config.TOP_K)
    return json.dumps(
        [
            {"source": c.source, "doc": c.doc, "text": c.text, "score": round(s, 3)}
            for c, s in hits
        ]
    )


@tool
def describe_database() -> str:
    """Return the CREATE TABLE statements of the operations database
    (customers, orders, tickets). Call this before writing SQL."""
    return sql_tools.describe_schema()


@tool
def run_sql(query: str) -> str:
    """Run one read-only SQLite SELECT query against the operations database and
    return columns and rows as JSON. Write statements are rejected. If the
    result contains an error, fix the query and try again."""
    return json.dumps(sql_tools.run_query(query), default=str)


DOC_TOOLS = [search_docs]
DATA_TOOLS = [describe_database, run_sql]

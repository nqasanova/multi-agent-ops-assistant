from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class OpsState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation, kept per thread
    question: str
    route: list[str]           # specialists chosen by the supervisor
    plan: list[str]            # specialists still to run, in order
    route_reason: str
    doc_findings: list[dict]   # {id, source, doc, text, score}
    data_findings: list[dict]  # {id, label, sql, columns, rows}
    answer: str
    review: dict
    feedback: str              # reviewer issues passed back to the writer
    attempts: int
    status: str                # approved | needs_human_review
    trace: list[str]           # readable log of what each node did

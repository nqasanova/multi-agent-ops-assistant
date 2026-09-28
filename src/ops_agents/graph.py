"""Builds the LangGraph workflow.

    START -> supervisor -> [data_agent] -> [docs_agent] -> writer -> reviewer
                                                             ^          |
                                                             +-- retry -+-> finalize -> END
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    finalize,
    make_after_review,
    make_data_agent,
    make_docs_agent,
    make_supervisor,
    make_writer,
    next_step,
    reviewer,
)
from .state import OpsState

ROUTES = {"data_agent": "data_agent", "docs_agent": "docs_agent", "writer": "writer"}


def build_graph(model: BaseChatModel | None = None, checkpointer=None):
    g = StateGraph(OpsState)
    g.add_node("supervisor", make_supervisor(model))
    g.add_node("data_agent", make_data_agent(model))
    g.add_node("docs_agent", make_docs_agent(model))
    g.add_node("writer", make_writer(model))
    g.add_node("reviewer", reviewer)
    g.add_node("finalize", finalize)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", next_step, ROUTES)
    g.add_conditional_edges("data_agent", next_step, ROUTES)
    g.add_conditional_edges("docs_agent", next_step, ROUTES)
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer", make_after_review(model), {"writer": "writer", "finalize": "finalize"}
    )
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def ask(graph, question: str, thread_id: str = "default", callbacks=None) -> dict:
    config = {"configurable": {"thread_id": thread_id}, "metadata": {"langfuse_session_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    return graph.invoke({"question": question, "trace": []}, config)

"""Graph nodes: supervisor, two specialist agents, writer, reviewer, finalizer.

Each node works with or without a chat model. With a model, the supervisor
uses structured output to route, the specialists run their own tool-calling
loop (a small compiled LangGraph subgraph), and the writer drafts the answer.
Without one, the same nodes fall back to the rules in offline.py.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from . import config, offline, sql_tools
from .retrieval import get_retriever
from .review import review_answer
from .state import OpsState
from .tools import DATA_TOOLS, DOC_TOOLS

# ---------------------------------------------------------------- supervisor

SUPERVISOR_PROMPT = """You route questions for an internal operations assistant.
Two specialists are available:
- data: queries the operations database (customers with their tier, orders, support tickets).
- docs: searches internal policy documents (refunds, support SLAs, shipping, data access,
  incident response, IT onboarding).
Decide which specialists are needed. Pick both when the answer depends on facts about a
specific customer AND on a policy (for example an SLA that depends on the customer's tier).
Pick neither if the question has nothing to do with company data or policies."""


class RoutePlan(BaseModel):
    needs_data: bool = Field(description="True if the operations database is needed.")
    needs_docs: bool = Field(description="True if internal policy documents are needed.")
    reason: str = Field(description="One short sentence explaining the choice.")


def make_supervisor(model: BaseChatModel | None):
    router = model.with_structured_output(RoutePlan) if model else None

    def supervisor(state: OpsState) -> dict:
        question = state["question"]
        trace = list(state.get("trace", []))
        if router is not None:
            try:
                history = state.get("messages", [])[-6:]
                plan: RoutePlan = router.invoke(
                    [SystemMessage(SUPERVISOR_PROMPT), *history, HumanMessage(question)]
                )
                needs_data, needs_docs, reason = plan.needs_data, plan.needs_docs, plan.reason
            except Exception as exc:  # keep answering if the router call fails
                trace.append(f"supervisor: LLM routing failed ({exc!r}), using rule-based router")
                needs_data, needs_docs, reason = offline.route(question)
        else:
            needs_data, needs_docs, reason = offline.route(question)

        # Data runs first, so the docs agent can use what it found (e.g. the customer's tier).
        steps = [s for s, needed in (("data_agent", needs_data), ("docs_agent", needs_docs)) if needed]
        trace.append(f"supervisor: plan={steps or ['answer directly']} ({reason})")
        return {
            "route": steps,
            "plan": steps,
            "route_reason": reason,
            "doc_findings": [],
            "data_findings": [],
            "attempts": 0,
            "feedback": "",
            "trace": trace,
        }

    return supervisor


def next_step(state: OpsState) -> str:
    plan = state.get("plan") or []
    return plan[0] if plan else "writer"


# ------------------------------------------------- specialist tool-call loop

class SpecialistState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: list[dict]


def build_specialist(model: BaseChatModel, tools: list, system_prompt: str):
    """A compiled subgraph: the model calls tools until it stops asking for them."""
    bound = model.bind_tools(tools)
    by_name = {t.name: t for t in tools}

    def call_model(state: SpecialistState) -> dict:
        return {"messages": [bound.invoke([SystemMessage(system_prompt), *state["messages"]])]}

    def call_tools(state: SpecialistState) -> dict:
        last = state["messages"][-1]
        messages, results = [], list(state.get("tool_results", []))
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                output = json.dumps({"ok": False, "error": f"Unknown tool {call['name']}"})
            else:
                output = tool.invoke(call["args"])
            results.append({"tool": call["name"], "args": call["args"], "output": output})
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": messages, "tool_results": results}

    def should_continue(state: SpecialistState) -> str:
        last = state["messages"][-1]
        steps = sum(isinstance(m, AIMessage) for m in state["messages"])
        if getattr(last, "tool_calls", None) and steps <= config.MAX_TOOL_STEPS:
            return "tools"
        return END

    g = StateGraph(SpecialistState)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g.compile()


def _context_from_data(state: OpsState) -> str:
    parts = []
    for f in state.get("data_findings", []):
        parts.append(f"{f['label']}: columns {f['columns']}, rows {f['rows'][:10]}")
    return "\n".join(parts)


# ------------------------------------------------------------- data agent

DATA_PROMPT = """You are the data specialist. Answer the user's question by querying the
operations database with run_sql. Call describe_database first if you are unsure about
table or column names. Use exact customer names from the database. Revenue means the sum
of amount_eur over orders whose status is not 'refunded'. If a query fails, read the error
and fix it. When you have the data you need, reply with one short line saying so."""


def make_data_agent(model: BaseChatModel | None):
    specialist = build_specialist(model, DATA_TOOLS, DATA_PROMPT) if model else None

    def data_agent(state: OpsState) -> dict:
        question = state["question"]
        trace = list(state.get("trace", []))
        findings: list[dict[str, Any]] = []
        needs_docs_after = "docs_agent" in (state.get("plan") or [])

        if specialist is not None:
            prompt = question
            if needs_docs_after:
                prompt += "\n(A policy lookup follows, so also fetch the customer's tier if a customer is mentioned.)"
            out = specialist.invoke(
                {"messages": [HumanMessage(prompt)], "tool_results": []},
                {"recursion_limit": 4 * config.MAX_TOOL_STEPS},
            )
            for r in out["tool_results"]:
                if r["tool"] != "run_sql":
                    continue
                result = json.loads(r["output"])
                if result.get("ok"):
                    findings.append({"label": "Query result", **{k: result[k] for k in ("sql", "columns", "rows")}})
                else:
                    trace.append(f"data_agent: query rejected or failed, model retried ({result['error']})")
        else:
            for label, sql in offline.plan_sql(question, with_policy_context=needs_docs_after):
                result = sql_tools.run_query(sql)
                if result["ok"]:
                    findings.append({"label": label, **{k: result[k] for k in ("sql", "columns", "rows")}})
                else:
                    trace.append(f"data_agent: {result['error']}")

        for i, f in enumerate(findings, start=1):
            f["id"] = f"Q{i}"
        trace.append(f"data_agent: {len(findings)} successful quer{'y' if len(findings) == 1 else 'ies'}")
        return {"data_findings": findings, "plan": state["plan"][1:], "trace": trace}

    return data_agent


# ------------------------------------------------------------- docs agent

DOCS_PROMPT = """You are the policy specialist. Search the internal documents with
search_docs to find the passages that answer the user's question. Search more than once
with different keywords if the first results do not cover the question. Use the database
context, if given, to make the search specific (for example the customer's tier).
When you have found the relevant passages, reply with one short line saying so."""


def make_docs_agent(model: BaseChatModel | None):
    specialist = build_specialist(model, DOC_TOOLS, DOCS_PROMPT) if model else None

    def docs_agent(state: OpsState) -> dict:
        question = state["question"]
        context = _context_from_data(state)
        trace = list(state.get("trace", []))
        hits: list[dict] = []

        if specialist is not None:
            prompt = question + (f"\n\nDatabase context:\n{context}" if context else "")
            out = specialist.invoke(
                {"messages": [HumanMessage(prompt)], "tool_results": []},
                {"recursion_limit": 4 * config.MAX_TOOL_STEPS},
            )
            for r in out["tool_results"]:
                hits.extend(json.loads(r["output"]))
            queries = [r["args"].get("query", "") for r in out["tool_results"]]
        else:
            # Offline: add tier and priority values found by the data agent to the query.
            extra = []
            for f in state.get("data_findings", []):
                for row in f["rows"]:
                    extra += [v for v in row if isinstance(v, str) and v in {
                        "Enterprise", "Business", "Standard", "high", "medium", "low"}]
            query = " ".join([question, *dict.fromkeys(extra)])
            hits = [
                {"source": c.source, "doc": c.doc, "text": c.text, "score": round(s, 3)}
                for c, s in get_retriever().search(query)
            ]
            queries = [query]

        findings, seen = [], set()
        for h in sorted(hits, key=lambda h: -h["score"]):
            key = (h["source"], h["text"])
            if key not in seen:
                seen.add(key)
                findings.append({"id": f"D{len(findings) + 1}", **h})
        trace.append(f"docs_agent: {len(queries)} search(es), {len(findings)} passages")
        return {"doc_findings": findings, "plan": state["plan"][1:], "trace": trace}

    return docs_agent


# ----------------------------------------------------------------- writer

WRITER_PROMPT = """You write the final answer for an internal operations assistant.
Use only the findings below. After every claim, cite the finding it comes from, like [D1]
or [Q2]. Copy numbers exactly as they appear in the findings. Keep it short and direct.
If the findings do not answer the question, say what is missing instead of guessing."""


def _findings_block(state: OpsState) -> str:
    lines = []
    for f in state.get("data_findings", []):
        lines.append(f"[{f['id']}] {f['label']}\nSQL: {f['sql']}\ncolumns: {f['columns']}\nrows: {f['rows']}")
    for f in state.get("doc_findings", []):
        lines.append(f"[{f['id']}] {f['source']}: {f['text']}")
    return "\n\n".join(lines) or "(no findings)"


def make_writer(model: BaseChatModel | None):
    def writer(state: OpsState) -> dict:
        question = state["question"]
        attempts = state.get("attempts", 0) + 1
        trace = list(state.get("trace", []))
        if model is not None:
            prompt = f"Question: {question}\n\nFindings:\n{_findings_block(state)}"
            if state.get("feedback"):
                prompt += f"\n\nYour previous draft was rejected by the reviewer: {state['feedback']} Fix these problems."
            answer = model.invoke([SystemMessage(WRITER_PROMPT), HumanMessage(prompt)]).content
            if isinstance(answer, list):  # some providers return content blocks
                answer = "".join(b.get("text", "") for b in answer if isinstance(b, dict))
        else:
            answer = offline.write_answer(
                question, state.get("doc_findings", []), state.get("data_findings", [])
            )
        trace.append(f"writer: draft {attempts}")
        return {"answer": answer.strip(), "attempts": attempts, "trace": trace}

    return writer


# --------------------------------------------------------------- reviewer

def reviewer(state: OpsState) -> dict:
    result = review_answer(
        state["answer"], state["question"],
        state.get("doc_findings", []), state.get("data_findings", []),
    )
    trace = list(state.get("trace", []))
    trace.append("reviewer: approved" if result["approved"] else f"reviewer: rejected ({' '.join(result['issues'])})")
    return {"review": result, "feedback": " ".join(result["issues"]), "trace": trace}


def make_after_review(model: BaseChatModel | None):
    def after_review(state: OpsState) -> str:
        if state["review"]["approved"]:
            return "finalize"
        # The offline writer is deterministic, so a retry would produce the same draft.
        if model is not None and state.get("attempts", 0) < config.MAX_WRITE_ATTEMPTS:
            return "writer"
        return "finalize"

    return after_review


def finalize(state: OpsState) -> dict:
    status = "approved" if state["review"]["approved"] else "needs_human_review"
    trace = list(state.get("trace", []))
    trace.append(f"finalize: {status}")
    return {
        "status": status,
        "messages": [HumanMessage(state["question"]), AIMessage(state["answer"])],
        "trace": trace,
    }

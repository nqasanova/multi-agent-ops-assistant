"""Runs the graph in LLM mode with a scripted chat model.

The fake model returns pre-written messages, including tool calls, so the
tool-calling subgraphs, SQL error recovery and the reviewer retry loop are
tested without an API key.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ops_agents.agents import RoutePlan
from ops_agents.graph import ask, build_graph


class ScriptedModel(GenericFakeChatModel):
    route: RoutePlan | None = None

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: self.route)


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def test_llm_mode_end_to_end_with_sql_retry_and_review_retry():
    script = iter([
        # data agent: bad query first, then a fixed one, then done
        call("run_sql", {"query": "SELECT tier FROM customer WHERE name = 'Vantara Medical'"}, 1),
        call("run_sql", {"query": "SELECT name, tier FROM customers WHERE name = 'Vantara Medical'"}, 2),
        AIMessage(content="Found the tier."),
        # docs agent: one search, then done
        call("search_docs", {"query": "Enterprise express delivery cost"}, 3),
        AIMessage(content="Found the policy."),
        # writer: first draft invents a number, second draft is grounded
        AIMessage(content="Vantara Medical is Enterprise [Q1] and saves 49 EUR [D1]."),
        AIMessage(content="Vantara Medical is an Enterprise customer [Q1], so express delivery is free [D1]."),
    ])
    model = ScriptedModel(messages=script, route=RoutePlan(needs_data=True, needs_docs=True, reason="test"))
    result = ask(build_graph(model), "Is Vantara Medical entitled to free express delivery?", "llm-1")

    assert result["route"] == ["data_agent", "docs_agent"]
    assert len(result["data_findings"]) == 1          # failed query not kept as a finding
    assert result["data_findings"][0]["rows"] == [["Vantara Medical", "Enterprise"]]
    assert any("model retried" in t for t in result["trace"])
    assert any(f["doc"] == "shipping_policy.md" for f in result["doc_findings"])
    assert result["attempts"] == 2                    # reviewer rejected the first draft
    assert result["status"] == "approved"
    assert "express delivery is free" in result["answer"]


def test_router_failure_falls_back_to_rules():
    class BrokenRouter(ScriptedModel):
        def with_structured_output(self, schema, **kwargs):
            def fail(_):
                raise RuntimeError("provider down")
            return RunnableLambda(fail)

    model = BrokenRouter(messages=iter([AIMessage(content="No findings to use.")]))
    result = ask(build_graph(model), "What is the capital of France?", "llm-2")
    assert any("using rule-based router" in t for t in result["trace"])
    assert result["route"] == []

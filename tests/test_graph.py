from ops_agents.graph import ask, build_graph

graph = build_graph(model=None)


def test_mixed_question_uses_both_agents_in_order():
    result = ask(graph, "What is the refund window for Mistral Aero Parts?", "t1")
    assert result["route"] == ["data_agent", "docs_agent"]
    assert "Enterprise" in result["answer"] and "60 day" in result["answer"]
    assert result["status"] == "approved"


def test_data_only_question():
    result = ask(graph, "Who are our top 3 customers by revenue in 2025?", "t2")
    assert result["route"] == ["data_agent"]
    assert len(result["data_findings"][0]["rows"]) == 3


def test_out_of_scope_question_is_not_answered_from_sources():
    result = ask(graph, "What is the capital of France?", "t3")
    assert result["route"] == []
    assert "could not find" in result["answer"]


def test_conversation_is_kept_per_thread():
    ask(graph, "How long does standard delivery to France take?", "t4")
    result = ask(graph, "Who has to approve a refund of 2,000 EUR?", "t4")
    assert len(result["messages"]) == 4

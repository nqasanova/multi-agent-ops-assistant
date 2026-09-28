from ops_agents.retrieval import get_retriever, load_chunks


def test_chunks_keep_section_context():
    chunks = load_chunks()
    assert chunks
    assert all(c.section and c.text for c in chunks)
    assert any(c.source == "refund_policy.md > Approval limits" for c in chunks)


def test_search_finds_the_right_document():
    hits = get_retriever().search("who approves a refund above 5,000 EUR")
    assert hits[0][0].doc == "refund_policy.md"
    hits = get_retriever().search("on-call engineer acknowledge SEV1")
    assert hits[0][0].doc == "incident_response.md"

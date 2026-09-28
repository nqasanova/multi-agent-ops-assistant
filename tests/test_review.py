from ops_agents.review import review_answer

DOCS = [{"id": "D1", "source": "shipping_policy.md > Shipping costs", "doc": "shipping_policy.md",
         "text": "Shipping is free for orders above 250 EUR within the EU.", "score": 0.5}]
DATA = [{"id": "Q1", "label": "Revenue", "sql": "SELECT 1", "columns": ["revenue_eur"], "rows": [[12345.67]]}]


def test_grounded_answer_is_approved():
    answer = "Revenue was 12,345.67 EUR [Q1]. Shipping is free above 250 EUR [D1]."
    assert review_answer(answer, "q", DOCS, DATA)["approved"]


def test_rejects_invented_numbers():
    result = review_answer("Revenue was 99,000 EUR [Q1].", "q", DOCS, DATA)
    assert not result["approved"] and "99000" in result["issues"][0]


def test_rejects_unknown_or_missing_citations():
    assert not review_answer("Shipping is free [D7].", "q", DOCS, DATA)["approved"]
    assert not review_answer("Shipping is free above 250 EUR.", "q", DOCS, DATA)["approved"]

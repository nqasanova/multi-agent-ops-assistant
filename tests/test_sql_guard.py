import pytest

from ops_agents.sql_tools import UnsafeQueryError, run_query, validate_query


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "DROP TABLE customers",
    "UPDATE orders SET amount_eur = 0",
    "SELECT * FROM orders; DROP TABLE orders",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "",
])
def test_rejects_unsafe_queries(sql):
    with pytest.raises(UnsafeQueryError):
        validate_query(sql)


def test_adds_row_limit():
    assert validate_query("SELECT * FROM orders").endswith("LIMIT 50")
    assert validate_query("SELECT * FROM orders LIMIT 5").endswith("LIMIT 5")


def test_errors_are_returned_not_raised():
    result = run_query("SELECT nope FROM customers")
    assert result["ok"] is False and "SQL error" in result["error"]
    result = run_query("DELETE FROM customers")
    assert result["ok"] is False


def test_valid_query_returns_rows():
    result = run_query("SELECT COUNT(*) AS n FROM customers")
    assert result["ok"] and result["columns"] == ["n"] and result["rows"][0][0] == 16

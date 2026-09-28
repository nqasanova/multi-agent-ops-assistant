"""Deterministic fallbacks used when no LLM is configured.

They cover the same three jobs the models do in LLM mode: routing a question,
turning it into SQL, and writing a cited answer. They handle a fixed set of
question patterns, which is enough to run the whole graph, the tests and the
eval without an API key. The router is also used as a fallback if the LLM
router call fails.
"""

from __future__ import annotations

import re
from typing import Any

from . import sql_tools

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "what", "which", "who", "how", "do", "does",
    "did", "for", "of", "to", "in", "on", "at", "and", "or", "our", "we", "us", "it",
    "its", "their", "they", "can", "be", "by", "with", "has", "have", "any", "this",
    "that", "there", "from", "should", "would", "much", "many", "get", "i", "my",
    "me", "you", "your", "if", "as", "about", "customer", "customers",
}

DATA_PATTERNS = re.compile(
    r"\b(how many (?!(business )?(days|hours))|number of|count|revenue|spent|spend|"
    r"top \d+|total|open tickets|refunded orders|orders by|breakdown|largest|biggest)\b",
    re.IGNORECASE,
)
DOCS_PATTERNS = re.compile(
    r"\b(policy|policies|sla|response (time|target)|refund window|approve|approval|allowed|"
    r"entitled|must|how long|shipping|express|delivery time|incident|sev\d|onboarding|"
    r"access|gdpr|breach|restocking|escalat\w*|support hours|lost|mfa|training|rule|rules|"
    r"refunds?|deliver\w*|ship\w*|pay back|paid back|how many (business )?(days|hours))\b",
    re.IGNORECASE,
)
TIER_DEPENDENT = re.compile(
    r"\b(sla|response|refund|express|entitled|support hours|escalat\w*)\b", re.IGNORECASE
)


# ---------- routing ----------

def find_customers(question: str) -> list[str]:
    q = question.lower()
    found = []
    for name in sql_tools.customer_names():
        first = name.split()[0].lower()
        if name.lower() in q or re.search(rf"\b{re.escape(first)}\b", q):
            found.append(name)
    return found


def route(question: str) -> tuple[bool, bool, str]:
    """Returns (needs_data, needs_docs, reason)."""
    customers = find_customers(question)
    needs_data = bool(customers) or bool(DATA_PATTERNS.search(question))
    needs_docs = bool(DOCS_PATTERNS.search(question))
    reasons = []
    if customers:
        reasons.append(f"mentions customer {', '.join(customers)}")
    if DATA_PATTERNS.search(question):
        reasons.append("asks for figures from the database")
    if needs_docs:
        reasons.append("asks about a policy or process")
    return needs_data, needs_docs, "; ".join(reasons) or "no internal source matches"


# ---------- SQL planning ----------

def _q(value: str) -> str:
    return value.replace("'", "''")


def plan_sql(question: str, with_policy_context: bool = False) -> list[tuple[str, str]]:
    """Maps a question to one or more (label, SQL) pairs."""
    q = question.lower()
    year = re.search(r"\b(20\d\d)\b", q)
    year_filter = f" AND strftime('%Y', o.order_date) = '{year.group(1)}'" if year else ""
    year_label = f" in {year.group(1)}" if year else ""
    prio = re.search(r"\b(high|medium|low)\b", q)
    plans: list[tuple[str, str]] = []

    for name in find_customers(question):
        n = _q(name)
        if with_policy_context or TIER_DEPENDENT.search(q) or "tier" in q:
            plans.append((
                f"Customer record for {name}",
                f"SELECT name, tier, country FROM customers WHERE name = '{n}'",
            ))
        if "ticket" in q:
            status = "" if re.search(r"\b(all|closed)\b", q) else " AND t.status = 'open'"
            p = f" AND t.priority = '{prio.group(1)}'" if prio else ""
            what = "Tickets" if not status else "Open tickets"
            label = f"{what} for {name}" + (f" with {prio.group(1)} priority" if prio else "")
            plans.append((
                label,
                "SELECT t.id, t.priority, t.status, t.opened_at, t.subject FROM tickets t "
                f"JOIN customers c ON c.id = t.customer_id WHERE c.name = '{n}'{status}{p} "
                "ORDER BY t.opened_at",
            ))
        if re.search(r"\b(revenue|spent|spend|order value|total)\b", q) or re.search(
            r"how many orders", q
        ):
            plans.append((
                f"Order summary for {name}{year_label} (refunded orders excluded)",
                "SELECT COUNT(*) AS orders, ROUND(SUM(o.amount_eur), 2) AS revenue_eur "
                f"FROM orders o JOIN customers c ON c.id = o.customer_id "
                f"WHERE c.name = '{n}' AND o.status != 'refunded'{year_filter}",
            ))
        if "refunded" in q:
            plans.append((
                f"Refunded orders for {name}",
                "SELECT o.id, o.order_date, o.amount_eur FROM orders o "
                f"JOIN customers c ON c.id = o.customer_id WHERE c.name = '{n}' "
                "AND o.status = 'refunded' ORDER BY o.order_date",
            ))

    if plans:
        return plans

    top = re.search(r"\btop (\d+)\b", q)
    if top or re.search(r"\b(largest|biggest)\b", q):
        limit = int(top.group(1)) if top else 5
        plans.append((
            f"Top {limit} customers by revenue{year_label} (refunded orders excluded)",
            "SELECT c.name, c.tier, ROUND(SUM(o.amount_eur), 2) AS revenue_eur "
            "FROM orders o JOIN customers c ON c.id = o.customer_id "
            f"WHERE o.status != 'refunded'{year_filter} GROUP BY c.name, c.tier "
            f"ORDER BY revenue_eur DESC LIMIT {limit}",
        ))
    elif "ticket" in q and "tier" in q:
        plans.append((
            "Open tickets by customer tier",
            "SELECT c.tier, COUNT(*) AS open_tickets FROM tickets t "
            "JOIN customers c ON c.id = t.customer_id WHERE t.status = 'open' "
            "GROUP BY c.tier ORDER BY open_tickets DESC",
        ))
    elif "ticket" in q:
        plans.append((
            "Open tickets by priority",
            "SELECT priority, COUNT(*) AS open_tickets FROM tickets WHERE status = 'open' "
            "GROUP BY priority ORDER BY open_tickets DESC",
        ))
    elif "revenue" in q and "tier" in q:
        plans.append((
            f"Revenue by customer tier{year_label} (refunded orders excluded)",
            "SELECT c.tier, ROUND(SUM(o.amount_eur), 2) AS revenue_eur FROM orders o "
            f"JOIN customers c ON c.id = o.customer_id WHERE o.status != 'refunded'{year_filter} "
            "GROUP BY c.tier ORDER BY revenue_eur DESC",
        ))
    elif "order" in q and ("status" in q or "refunded" in q):
        plans.append((
            "Orders by status",
            "SELECT status, COUNT(*) AS orders FROM orders GROUP BY status ORDER BY orders DESC",
        ))
    elif "revenue" in q or "total" in q:
        plans.append((
            f"Total revenue{year_label} (refunded orders excluded)",
            "SELECT COUNT(*) AS orders, ROUND(SUM(o.amount_eur), 2) AS revenue_eur "
            f"FROM orders o WHERE o.status != 'refunded'{year_filter}",
        ))
    return plans


# ---------- answer writing ----------

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w.rstrip("s") for w in words if w not in STOPWORDS and len(w) > 1}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _describe_rows(finding: dict) -> list[str]:
    cols, rows, fid = finding["columns"], finding["rows"], finding["id"]
    label = finding["label"]
    if not rows:
        return [f"{label}: no matching records [{fid}]."]
    if len(rows) == 1:
        pairs = ", ".join(f"{c.replace('_', ' ')} {_fmt(v)}" for c, v in zip(cols, rows[0]))
        return [f"{label}: {pairs} [{fid}]."]
    lines = [f"{label}, {len(rows)} rows [{fid}]:"]
    for row in rows[:10]:
        lines.append("- " + ", ".join(f"{c.replace('_', ' ')}: {_fmt(v)}" for c, v in zip(cols, row)))
    if len(rows) > 10:
        lines.append(f"- ... {len(rows) - 10} more rows")
    return lines


def write_answer(question: str, doc_findings: list[dict], data_findings: list[dict]) -> str:
    lines: list[str] = []
    for f in data_findings:
        lines.extend(_describe_rows(f))

    if doc_findings:
        q_tokens = _tokens(question)
        # Values from the data (a customer's tier, a ticket priority) make the
        # matching policy sentence more relevant, so they are added to the query.
        context_tokens = set()
        for f in data_findings:
            for row in f["rows"]:
                for v in row:
                    if isinstance(v, str) and v.lower() in {
                        "enterprise", "business", "standard", "high", "medium", "low"
                    }:
                        context_tokens.add(v.lower())
        tiers = {"enterprise", "business", "standard"}
        known_tiers = context_tokens & tiers
        scored = []
        for f in doc_findings:
            for sentence in re.split(r"(?<=[.!?])\s+", f["text"]):
                s_tokens = _tokens(sentence)
                # Skip sentences about a different tier than the customer's.
                if known_tiers and (s_tokens & tiers) and not (s_tokens & known_tiers):
                    continue
                score = len(q_tokens & s_tokens) + 1.5 * len(context_tokens & s_tokens)
                score += 0.5 * f.get("score", 0)
                if score > 0:
                    scored.append((score, sentence.strip(), f["id"]))
        scored.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for score, sentence, fid in scored:
            if sentence in seen:
                continue
            if picked and (score < 0.75 * picked[0][0] or len(q_tokens & _tokens(sentence)) < 2):
                break
            picked.append((score, sentence, fid))
            seen.add(sentence)
            if len(picked) == 2:
                break
        for _, sentence, fid in picked:
            lines.append(f"{sentence} [{fid}]")

    if not lines:
        return (
            "I could not find anything in the internal documents or the operations "
            "database that answers this question."
        )
    return "\n".join(lines)

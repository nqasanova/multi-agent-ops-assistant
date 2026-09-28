"""Runs the assistant on evals/cases.jsonl and writes evals/results.md.

Metrics
  routing accuracy   supervisor picked exactly the expected specialists
  doc recall         an expected document is among the retrieved passages
  answer correct     every expected phrase and every value from the
                     reference SQL query appears in the answer
  grounded           the reviewer approved the answer on its own checks
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ops_agents import config  # noqa: E402
from ops_agents.graph import ask, build_graph  # noqa: E402
from ops_agents.llm import get_chat_model, resolve_provider  # noqa: E402
from ops_agents.review import extract_numbers  # noqa: E402
from ops_agents.tracing import flush, langfuse_callbacks  # noqa: E402


def reference_values(sql: str | None) -> list:
    if not sql:
        return []
    with sqlite3.connect(config.DB_PATH) as con:
        return [v for row in con.execute(sql).fetchall() for v in row]


def value_in_answer(value, answer: str) -> bool:
    if isinstance(value, (int, float)):
        return round(float(value), 2) in extract_numbers(answer)
    return str(value).lower() in answer.lower()


def main() -> None:
    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text().splitlines() if l.strip()]
    graph = build_graph(get_chat_model())
    callbacks = langfuse_callbacks()
    rows, totals = [], {"route": 0, "doc": 0, "doc_n": 0, "correct": 0, "grounded": 0}
    start = time.time()

    for case in cases:
        result = ask(graph, case["question"], f"eval-{case['id']}", callbacks)
        answer = result["answer"]
        route = sorted(s.replace("_agent", "") for s in result["route"])
        route_ok = route == sorted(case["expected_route"])

        doc_ok = None
        if case["expected_docs"]:
            got = {f["doc"] for f in result.get("doc_findings", [])}
            doc_ok = bool(got & set(case["expected_docs"]))
            totals["doc_n"] += 1
            totals["doc"] += doc_ok

        phrases_ok = all(p.lower() in answer.lower() for p in case["must_include"])
        values_ok = all(value_in_answer(v, answer) for v in reference_values(case["reference_sql"]))
        correct = phrases_ok and values_ok
        grounded = result["status"] == "approved"

        totals["route"] += route_ok
        totals["correct"] += correct
        totals["grounded"] += grounded
        rows.append((case["id"], route_ok, doc_ok, correct, grounded))

    n = len(cases)
    mark = lambda v: "n/a" if v is None else ("yes" if v else "no")
    lines = [
        "# Evaluation results",
        "",
        f"Provider: `{resolve_provider()}`, {n} cases, {time.time() - start:.1f}s total.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Routing accuracy | {totals['route']}/{n} ({totals['route'] / n:.0%}) |",
        f"| Doc recall (expected document retrieved) | {totals['doc']}/{totals['doc_n']} ({totals['doc'] / totals['doc_n']:.0%}) |",
        f"| Answer correct (phrases + reference SQL values) | {totals['correct']}/{n} ({totals['correct'] / n:.0%}) |",
        f"| Passed grounding review | {totals['grounded']}/{n} ({totals['grounded'] / n:.0%}) |",
        "",
        "| Case | Routing | Doc recall | Correct | Grounded |",
        "|---|---|---|---|---|",
        *[f"| {cid} | {mark(r)} | {mark(d)} | {mark(c)} | {mark(g)} |" for cid, r, d, c, g in rows],
    ]
    out = ROOT / "evals" / "results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:11]))
    print(f"\nFull table: {out.relative_to(ROOT)}")
    flush()


if __name__ == "__main__":
    main()

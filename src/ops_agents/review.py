"""Grounding checks run on every draft answer before it is returned.

The checks are deterministic on purpose: they do not rely on a second model
call, so they cost nothing and behave the same way every time.
  1. Every citation points to a finding that was actually retrieved.
  2. A draft built from findings cites at least one of them.
  3. Every number in the answer appears in the findings or the question.
"""

from __future__ import annotations

import json
import re

CITATION = re.compile(r"\[([DQ]\d+)\]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> set[float]:
    values = set()
    for raw in NUMBER.findall(text):
        try:
            values.add(round(float(raw.replace(",", "")), 2))
        except ValueError:
            continue
    return values


def review_answer(answer: str, question: str, doc_findings: list[dict], data_findings: list[dict]) -> dict:
    issues: list[str] = []
    known_ids = {f["id"] for f in doc_findings} | {f["id"] for f in data_findings}
    cited = set(CITATION.findall(answer))

    unknown = sorted(cited - known_ids)
    if unknown:
        issues.append(f"Cites sources that were never retrieved: {', '.join(unknown)}.")
    if known_ids and not cited:
        issues.append("The answer uses retrieved findings but cites none of them.")

    evidence = question + " " + json.dumps(doc_findings, default=str) + " " + json.dumps(
        data_findings, default=str
    )
    allowed = extract_numbers(evidence)
    allowed |= {float(len(f.get("rows", []))) for f in data_findings}  # row counts
    unsupported = sorted(
        n for n in extract_numbers(CITATION.sub("", answer)) if n not in allowed
    )
    if unsupported:
        shown = ", ".join(f"{n:g}" for n in unsupported[:5])
        issues.append(f"Numbers not found in any finding: {shown}.")

    return {"approved": not issues, "issues": issues, "citations": sorted(cited)}

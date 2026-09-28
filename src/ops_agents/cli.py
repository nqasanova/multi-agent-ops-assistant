"""Command line entry point.

    python -m ops_agents.cli ask "question"      one question
    python -m ops_agents.cli chat                multi-turn session (same thread)
    python -m ops_agents.cli graph               print the graph as Mermaid
"""

from __future__ import annotations

import argparse
import uuid

from .graph import ask, build_graph
from .llm import get_chat_model, resolve_provider
from .tracing import flush, langfuse_callbacks


def _print_result(result: dict, verbose: bool) -> None:
    print(result["answer"])
    print(f"\nstatus: {result['status']}")
    if verbose:
        print("\ntrace:")
        for line in result["trace"]:
            print(f"  {line}")
        for f in result.get("data_findings", []):
            print(f"  [{f['id']}] SQL: {f['sql']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ops-agents")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("-v", "--verbose", action="store_true")
    p_chat = sub.add_parser("chat")
    p_chat.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("graph")
    args = parser.parse_args()

    graph = build_graph(get_chat_model())
    if args.command == "graph":
        print(graph.get_graph().draw_mermaid())
        return

    callbacks = langfuse_callbacks()
    print(f"[provider: {resolve_provider()} | tracing: {'langfuse' if callbacks else 'off'}]\n")
    if args.command == "ask":
        _print_result(ask(graph, args.question, str(uuid.uuid4()), callbacks), args.verbose)
    else:
        thread = str(uuid.uuid4())
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                _print_result(ask(graph, question, thread, callbacks), args.verbose)
    flush()


if __name__ == "__main__":
    main()

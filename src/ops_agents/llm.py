"""Chat model selection.

LLM_PROVIDER can be anthropic, openai or offline. If it is not set, the
provider is picked from whichever API key is present, and with no key the
graph runs in offline mode (rule-based routing, template SQL, extractive
answers). Both modes run the same LangGraph graph.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "offline"


def get_chat_model() -> BaseChatModel | None:
    provider = resolve_provider()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"), temperature=0
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "offline":
        return None
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

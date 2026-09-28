"""Optional Langfuse tracing.

If LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, every graph run is sent
to Langfuse as one trace, with a span per node, tool call and model call.
Otherwise tracing is off and nothing is imported.
"""

from __future__ import annotations

import os


def langfuse_callbacks() -> list:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def flush() -> None:
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client

        get_client().flush()

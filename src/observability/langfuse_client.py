"""Langfuse tracing setup, shared by the CLI app, Streamlit UI, and evaluation harness."""

import os
from functools import lru_cache

from src.config.config import Config


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """Return a Langfuse LangChain CallbackHandler, or None if Langfuse isn't configured."""
    if not (Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY):
        return None

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", Config.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", Config.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", Config.LANGFUSE_HOST)

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def with_langfuse(config: dict | None = None) -> dict:
    """Merge a Langfuse callback into a LangChain/LangGraph `config` dict, if configured."""
    config = dict(config or {})
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [*config.get("callbacks", []), handler]
    return config

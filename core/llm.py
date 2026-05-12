"""
Groq LLM wrapper via LangChain.
"""
import os
import logging
from functools import lru_cache

from langchain_groq import ChatGroq
from config import GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    # Read dynamically so sidebar key updates take effect after cache_clear()
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set — paste it in the sidebar.")
    return ChatGroq(
        api_key=key,
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

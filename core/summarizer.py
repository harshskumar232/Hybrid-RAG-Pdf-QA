"""
Auto-document summarization using map-reduce over top chunks.
"""
import logging

from langchain_core.messages import HumanMessage
from core.llm import get_llm
from config import SUMMARY_MAX_CHUNKS

logger = logging.getLogger(__name__)

MAP_PROMPT = """Summarize the following text excerpt in 2-3 sentences.
Be concise and focus on key facts and arguments.

Text:
{text}

Summary:"""

REDUCE_PROMPT = """You are given several partial summaries of a document called "{title}".
Combine them into one coherent summary of 4-6 sentences covering the main topics,
key arguments, and important findings. Write in the same language as the summaries.

Partial summaries:
{summaries}

Final summary:"""


def summarize_document(
    chunks_text: list[str],
    doc_title: str,
    language: str = "en",
) -> str:
    llm = get_llm()
    selected = chunks_text[:SUMMARY_MAX_CHUNKS]

    # Map: summarize each chunk
    partial_summaries = []
    for text in selected:
        prompt = MAP_PROMPT.format(text=text[:1500])
        resp = llm.invoke([HumanMessage(content=prompt)])
        partial_summaries.append(resp.content.strip())

    # Reduce: combine partial summaries
    combined = "\n\n".join(f"- {s}" for s in partial_summaries)
    reduce_prompt = REDUCE_PROMPT.format(title=doc_title, summaries=combined)
    final = llm.invoke([HumanMessage(content=reduce_prompt)])

    logger.info(f"Generated summary for '{doc_title}'")
    return final.content.strip()

"""
RAG chain with conversation memory and source tracking.
"""
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory

from core.llm import get_llm
from core.retriever import retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert document analyst. Answer questions using ONLY the provided context.
Always cite your sources by mentioning the document name and page number.
If the context does not contain enough information, say so clearly.
Respond in the same language as the question (English or German)."""

QA_PROMPT_TEMPLATE = """{system}

--- CONTEXT ---
{context}
--- END CONTEXT ---

Conversation so far:
{history}

Question: {question}

Answer (cite sources as [filename, p.X]):"""


def _format_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        meta = h["metadata"]
        parts.append(
            f"[{meta.get('filename','?')}, p.{meta.get('page_number','?')}]\n{h['text']}"
        )
    return "\n\n".join(parts)


def _format_history(messages: list) -> str:
    if not messages:
        return "(no prior conversation)"
    lines = []
    for m in messages[-6:]:   # last 3 turns
        role = "Human" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def answer(
    question: str,
    all_chunks_text: list[str],
    all_chunks_meta: list[dict],
    memory: ConversationBufferWindowMemory,
    doc_ids: Optional[list[str]] = None,
) -> dict:
    """
    Run one QA turn.

    Returns:
        answer   str  — LLM answer
        sources  list[dict] — retrieved chunks with metadata
    """
    llm = get_llm()

    # Retrieve relevant chunks
    hits = retrieve(question, all_chunks_text, all_chunks_meta, doc_ids=doc_ids)
    context = _format_context(hits)

    # Build prompt
    history_str = _format_history(memory.chat_memory.messages)
    prompt = QA_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        history=history_str,
        question=question,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    answer_text = response.content.strip()

    # Save to memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer_text)

    return {"answer": answer_text, "sources": hits}


def make_memory(window: int = 5) -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(k=window, return_messages=True)

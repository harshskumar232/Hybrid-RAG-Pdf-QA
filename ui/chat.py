"""
Chat UI — Q&A with conversation memory and source highlighting.
"""
import time
import logging
import streamlit as st
from langdetect import detect, LangDetectException

from core.chain import answer
from analytics.database import get_all_documents
from analytics.tracker import track_query

logger = logging.getLogger(__name__)


def _detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} source(s) used", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src["metadata"]
            score = src.get("score", 0)
            st.markdown(
                f"**[{i}] {meta.get('filename', '?')}** — "
                f"Page **{meta.get('page_number', '?')}**, "
                f"Block {meta.get('block_number', '?')} | "
                f"Relevance: `{score:.2f}`"
            )
            st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))
            if i < len(sources):
                st.divider()


def render_chat(session_state) -> None:
    docs = get_all_documents()

    if not docs:
        st.info("Upload at least one PDF in the **Documents** tab to start chatting.")
        return

    # Document selector
    doc_options = {d["filename"]: d["doc_id"] for d in docs}
    selected_names = st.multiselect(
        "Query documents",
        options=list(doc_options.keys()),
        default=list(doc_options.keys()),
        help="Select one or more documents to search across.",
    )
    selected_doc_ids = [doc_options[n] for n in selected_names] if selected_names else None

    st.divider()

    # Chat history
    for msg in session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # Input
    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    if not selected_names:
        st.warning("Select at least one document above.")
        return

    # Show user message
    session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer…"):
            t0 = time.time()
            try:
                result = answer(
                    question=question,
                    all_chunks_text=session_state.all_chunks_text,
                    all_chunks_meta=session_state.all_chunks_meta,
                    memory=session_state.memory,
                    doc_ids=selected_doc_ids,
                )
                latency_ms = int((time.time() - t0) * 1000)
                lang = _detect_lang(question)

                st.markdown(result["answer"])
                _render_sources(result["sources"])

                session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

                track_query(
                    session_id=session_state.session_id,
                    question=question,
                    answer=result["answer"],
                    sources=result["sources"],
                    doc_ids=selected_doc_ids or [],
                    language=lang,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Chain error: {e}", exc_info=True)

    # Clear conversation button
    if session_state.chat_history:
        if st.button("🗑 Clear conversation"):
            session_state.chat_history = []
            session_state.memory = __import__(
                "core.chain", fromlist=["make_memory"]
            ).make_memory()
            st.rerun()

"""
PDF upload panel — ingest, chunk, embed, summarise, manage documents.
"""
import shutil
import logging
import streamlit as st

from config import UPLOADS_DIR
from core.ingestion import ingest_pdf
from core.chunking import chunk_document
from core.vectorstore import add_chunks, delete_document, get_all_chunks
from core.summarizer import summarize_document
from analytics.database import get_all_documents
from analytics.tracker import track_document

logger = logging.getLogger(__name__)


def _refresh_bm25_cache(session_state) -> None:
    texts, metas = get_all_chunks()
    session_state.all_chunks_text = texts
    session_state.all_chunks_meta = metas


def render_upload_panel(session_state) -> None:
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports German and English documents.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            dest = UPLOADS_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.read())

            with st.status(f"Processing **{uf.name}**…", expanded=True) as status:
                try:
                    st.write("📄 Extracting text…")
                    doc = ingest_pdf(dest)

                    st.write("✂️ Chunking…")
                    chunks = chunk_document(doc)

                    st.write("🔢 Embedding & storing…")
                    add_chunks(chunks)

                    st.write("📝 Summarising…")
                    try:
                        chunk_texts = [c.text for c in chunks]
                        summary = summarize_document(chunk_texts, doc.title, doc.language)
                    except Exception as llm_err:
                        summary = ""
                        st.warning(f"Summary skipped (LLM error): {llm_err}")

                    track_document(doc, chunks, summary)
                    _refresh_bm25_cache(session_state)

                    status.update(
                        label=f"✅ {uf.name} ready ({doc.num_pages} pages, {len(chunks)} chunks)",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label=f"❌ Failed: {e}", state="error")
                    logger.error(f"Upload failed for {uf.name}: {e}", exc_info=True)

    st.divider()
    _render_document_library(session_state)


def _render_document_library(session_state) -> None:
    st.subheader("Document Library")
    docs = get_all_documents()

    if not docs:
        st.info("No documents yet. Upload a PDF above.")
        return

    for doc in docs:
        with st.expander(
            f"📄 {doc['filename']}  —  {doc['num_pages']} pages  |  "
            f"lang: {doc['language'].upper()}  |  {doc['num_chunks']} chunks",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Title:** {doc['title']}")
                st.markdown(f"**Author:** {doc['author']}")
                st.markdown(f"**Words:** {doc['word_count']:,}")
                st.markdown(f"**Uploaded:** {doc['uploaded_at']}")
                if doc.get("summary"):
                    st.markdown("**Auto-Summary:**")
                    st.info(doc["summary"])
            with col2:
                if st.button("🗑 Delete", key=f"del_{doc['doc_id']}"):
                    delete_document(doc["doc_id"])
                    _refresh_bm25_cache(session_state)
                    st.success(f"Deleted {doc['filename']}")
                    st.rerun()

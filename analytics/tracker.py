"""
High-level tracking helpers called from the Streamlit app.
"""
import json
import time
import logging
from analytics.database import insert_document, insert_query
from core.ingestion import DocumentData
from core.chunking import Chunk

logger = logging.getLogger(__name__)


def track_document(doc: DocumentData, chunks: list[Chunk], summary: str) -> None:
    insert_document({
        "doc_id":     doc.doc_id,
        "filename":   doc.filename,
        "title":      doc.title,
        "author":     doc.author,
        "language":   doc.language,
        "num_pages":  doc.num_pages,
        "num_chunks": len(chunks),
        "word_count": doc.word_count,
        "summary":    summary,
    })


def track_query(
    session_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    doc_ids: list[str],
    language: str,
    latency_ms: int,
) -> None:
    insert_query(
        q={
            "session_id": session_id,
            "question":   question,
            "answer":     answer,
            "doc_ids":    json.dumps(doc_ids),
            "num_sources":len(sources),
            "language":   language,
            "latency_ms": latency_ms,
        },
        sources=sources,
    )

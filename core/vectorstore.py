"""
ChromaDB vector store — one collection per project (all docs together).
Chunks are tagged with doc_id so per-document filtering is possible.
"""
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, TOP_K_DENSE
from core.chunking import Chunk
from core.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_documents"


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    client = _get_client()
    collection = _get_collection(client)

    ids        = [c.chunk_id for c in chunks]
    texts      = [c.text for c in chunks]
    metadatas  = [c.to_metadata() for c in chunks]
    embeddings = embed_texts(texts)

    # upsert — safe to re-add same doc
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(chunks)} chunks into ChromaDB")
    return len(chunks)


def search(
    query: str,
    top_k: int = TOP_K_DENSE,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Returns list of dicts with keys: text, metadata, score.
    Optionally filter to specific doc_ids.
    """
    client = _get_client()
    collection = _get_collection(client)

    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

    q_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=min(top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": text, "metadata": meta, "score": 1 - dist})

    return hits


def delete_document(doc_id: str) -> int:
    client = _get_client()
    collection = _get_collection(client)
    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted all chunks for doc_id={doc_id}")
    return 1


def get_all_chunks() -> tuple[list[str], list[dict]]:
    """Return (texts, metadatas) for every chunk — used to build BM25 index."""
    client = _get_client()
    collection = _get_collection(client)
    if collection.count() == 0:
        return [], []
    result = collection.get(include=["documents", "metadatas"])
    return result["documents"], result["metadatas"]


def list_doc_ids() -> list[str]:
    client = _get_client()
    collection = _get_collection(client)
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    return list({m["doc_id"] for m in all_meta})

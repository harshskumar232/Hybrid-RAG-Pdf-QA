"""
Hybrid retrieval: dense (ChromaDB) + sparse (BM25), fused with Reciprocal Rank Fusion.
"""
import logging
from typing import Optional

from rank_bm25 import BM25Okapi

from config import TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL
from core import vectorstore

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using RRF. Returns hits sorted by fused score."""
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for rank, hit in enumerate(dense_hits):
        key = hit["metadata"]["chunk_id"] if "chunk_id" in hit["metadata"] else hit["text"][:60]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key] = hit

    for rank, hit in enumerate(bm25_hits):
        key = hit["metadata"].get("chunk_id", hit["text"][:60])
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in docs:
            docs[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[k] for k, _ in ranked[:TOP_K_FINAL]]


def retrieve(
    query: str,
    all_chunks_text: list[str],
    all_chunks_meta: list[dict],
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Hybrid retrieval.

    all_chunks_text / all_chunks_meta: all chunk texts + metadata already in memory
    (used for BM25; fetched once per session from ChromaDB get()).
    """
    # 1. Dense retrieval
    dense_hits = vectorstore.search(query, top_k=TOP_K_DENSE, doc_ids=doc_ids)

    # 2. BM25 sparse retrieval
    if all_chunks_text:
        filtered_texts = all_chunks_text
        filtered_meta  = all_chunks_meta

        if doc_ids:
            pairs = [
                (t, m) for t, m in zip(all_chunks_text, all_chunks_meta)
                if m.get("doc_id") in doc_ids
            ]
            filtered_texts = [p[0] for p in pairs]
            filtered_meta  = [p[1] for p in pairs]

        tokenized = [t.lower().split() for t in filtered_texts]
        if tokenized:
            bm25    = BM25Okapi(tokenized)
            scores  = bm25.get_scores(query.lower().split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_BM25]
            bm25_hits = [
                {"text": filtered_texts[i], "metadata": filtered_meta[i], "score": float(scores[i])}
                for i in top_idx if scores[i] > 0
            ]
        else:
            bm25_hits = []
    else:
        bm25_hits = []

    # 3. Fuse
    if bm25_hits:
        return _reciprocal_rank_fusion(dense_hits, bm25_hits)
    return dense_hits[:TOP_K_FINAL]

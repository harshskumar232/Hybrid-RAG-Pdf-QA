"""
Local sentence-transformers embedder.
Uses paraphrase-multilingual-MiniLM-L12-v2 — supports German + English.
"""
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]

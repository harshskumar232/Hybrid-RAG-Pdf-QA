"""
Sentence-aware chunking that preserves page + paragraph metadata on every chunk.
"""
import re
import logging
from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP
from core.ingestion import DocumentData, PageBlock

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str           # "{doc_id}-{index}"
    doc_id: str
    text: str
    page_number: int
    block_number: int
    chunk_index: int        # global index within document
    char_start: int         # character offset in block text
    language: str
    filename: str

    def to_metadata(self) -> dict:
        return {
            "doc_id":       self.doc_id,
            "filename":     self.filename,
            "page_number":  self.page_number,
            "block_number": self.block_number,
            "chunk_index":  self.chunk_index,
            "char_start":   self.char_start,
            "language":     self.language,
        }


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, preserving the delimiter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_block(
    block: PageBlock,
    doc_id: str,
    filename: str,
    language: str,
    start_index: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    sentences = _split_sentences(block.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    char_offset = 0
    idx = start_index

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            text = " ".join(current)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}-{idx}",
                doc_id=doc_id,
                text=text,
                page_number=block.page_number,
                block_number=block.block_number,
                chunk_index=idx,
                char_start=char_offset,
                language=language,
                filename=filename,
            ))
            idx += 1
            # overlap: keep last N characters worth of sentences
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            char_offset += current_len - overlap_len
            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}-{idx}",
            doc_id=doc_id,
            text=" ".join(current),
            page_number=block.page_number,
            block_number=block.block_number,
            chunk_index=idx,
            char_start=char_offset,
            language=language,
            filename=filename,
        ))

    return chunks


def chunk_document(doc: DocumentData) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    global_idx = 0

    for block in doc.blocks:
        if len(block.text) < 20:   # skip very short blocks (headers, page numbers)
            continue
        new_chunks = _chunk_block(
            block=block,
            doc_id=doc.doc_id,
            filename=doc.filename,
            language=doc.language,
            start_index=global_idx,
        )
        all_chunks.extend(new_chunks)
        global_idx += len(new_chunks)

    logger.info(f"Chunked '{doc.filename}': {len(all_chunks)} chunks from {len(doc.blocks)} blocks")
    return all_chunks

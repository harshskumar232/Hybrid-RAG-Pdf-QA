"""
PDF ingestion using PyMuPDF.
Extracts text with page, paragraph, and block-level metadata.
"""
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


@dataclass
class PageBlock:
    text: str
    page_number: int        # 1-indexed
    block_number: int       # block index within page
    bbox: tuple             # (x0, y0, x1, y1) bounding box
    block_type: str         # "text" | "image"


@dataclass
class DocumentData:
    doc_id: str
    filename: str
    filepath: str
    num_pages: int
    language: str
    title: str
    author: str
    blocks: list[PageBlock] = field(default_factory=list)
    raw_text: str = ""

    @property
    def word_count(self) -> int:
        return len(self.raw_text.split())


def _compute_doc_id(filepath: str) -> str:
    """SHA-256 of file content — stable across re-uploads of the same file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _detect_language(text: str) -> str:
    sample = text[:2000].strip()
    if not sample:
        return "en"
    try:
        lang = detect(sample)
        return lang if lang in ("en", "de") else "en"
    except LangDetectException:
        return "en"


def _extract_metadata(doc: fitz.Document) -> dict:
    meta = doc.metadata or {}
    return {
        "title":  meta.get("title", "").strip() or "Untitled",
        "author": meta.get("author", "").strip() or "Unknown",
    }


def ingest_pdf(filepath: str | Path) -> DocumentData:
    """
    Parse a PDF and return a DocumentData with all text blocks
    tagged with page number, block index, and bounding box.
    """
    filepath = str(filepath)
    doc = fitz.open(filepath)
    meta = _extract_metadata(doc)
    doc_id = _compute_doc_id(filepath)

    blocks: list[PageBlock] = []
    all_text_parts: list[str] = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        raw_blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,bno,btype)

        for b in raw_blocks:
            x0, y0, x1, y1, text, bno, btype = b
            text = text.strip()
            if not text or btype != 0:   # skip empty and image blocks
                continue
            block = PageBlock(
                text=text,
                page_number=page_num,
                block_number=bno,
                bbox=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                block_type="text",
            )
            blocks.append(block)
            all_text_parts.append(text)

    raw_text  = "\n".join(all_text_parts)
    language  = _detect_language(raw_text)
    num_pages = doc.page_count   # save before close

    doc.close()

    logger.info(
        f"Ingested '{Path(filepath).name}': "
        f"{num_pages} pages, {len(blocks)} blocks, lang={language}"
    )

    return DocumentData(
        doc_id=doc_id,
        filename=Path(filepath).name,
        filepath=filepath,
        num_pages=num_pages,
        language=language,
        title=meta["title"],
        author=meta["author"],
        blocks=blocks,
        raw_text=raw_text,
    )


def ingest_multiple(filepaths: list[str | Path]) -> list[DocumentData]:
    docs = []
    for fp in filepaths:
        try:
            docs.append(ingest_pdf(fp))
        except Exception as e:
            logger.error(f"Failed to ingest {fp}: {e}")
    return docs

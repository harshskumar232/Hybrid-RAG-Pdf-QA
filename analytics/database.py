"""
SQLite analytics database — tracks documents, queries, and sessions.
"""
import sqlite3
import logging
from contextlib import contextmanager
from config import ANALYTICS_DB

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    title         TEXT,
    author        TEXT,
    language      TEXT,
    num_pages     INTEGER,
    num_chunks    INTEGER,
    word_count    INTEGER,
    summary       TEXT,
    uploaded_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queries (
    query_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT,
    doc_ids       TEXT,           -- JSON list of doc_ids queried
    num_sources   INTEGER,
    language      TEXT,
    latency_ms    INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_sources (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id      INTEGER REFERENCES queries(query_id),
    doc_id        TEXT,
    filename      TEXT,
    page_number   INTEGER,
    chunk_index   INTEGER,
    score         REAL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(ANALYTICS_DB))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    logger.info("Analytics DB initialised")


def insert_document(doc: dict) -> None:
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO documents
                (doc_id, filename, title, author, language,
                 num_pages, num_chunks, word_count, summary)
            VALUES (:doc_id,:filename,:title,:author,:language,
                    :num_pages,:num_chunks,:word_count,:summary)
        """, doc)


def insert_query(q: dict, sources: list[dict]) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO queries
                (session_id, question, answer, doc_ids, num_sources, language, latency_ms)
            VALUES (:session_id,:question,:answer,:doc_ids,:num_sources,:language,:latency_ms)
        """, q)
        query_id = cur.lastrowid
        for s in sources:
            conn.execute("""
                INSERT INTO query_sources
                    (query_id, doc_id, filename, page_number, chunk_index, score)
                VALUES (?,?,?,?,?,?)
            """, (query_id,
                  s["metadata"].get("doc_id"),
                  s["metadata"].get("filename"),
                  s["metadata"].get("page_number"),
                  s["metadata"].get("chunk_index"),
                  s.get("score", 0)))
        return query_id


def get_all_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_query_stats() -> dict:
    with get_conn() as conn:
        total_q  = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        total_d  = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        avg_lat  = conn.execute("SELECT AVG(latency_ms) FROM queries").fetchone()[0] or 0
        by_lang  = conn.execute(
            "SELECT language, COUNT(*) as cnt FROM queries GROUP BY language"
        ).fetchall()
        by_doc   = conn.execute("""
            SELECT qs.filename, COUNT(*) as hits
            FROM query_sources qs
            GROUP BY qs.filename ORDER BY hits DESC LIMIT 10
        """).fetchall()
        recent   = conn.execute("""
            SELECT question, created_at, num_sources, latency_ms
            FROM queries ORDER BY created_at DESC LIMIT 20
        """).fetchall()
    return {
        "total_queries":    total_q,
        "total_documents":  total_d,
        "avg_latency_ms":   round(avg_lat, 1),
        "queries_by_lang":  [dict(r) for r in by_lang],
        "top_documents":    [dict(r) for r in by_doc],
        "recent_queries":   [dict(r) for r in recent],
    }

# RAG Document Q&A

An advanced Retrieval-Augmented Generation (RAG) system for multi-PDF question answering with conversation memory, source highlighting, auto-summarization, and multilingual support (German + English).

## Features

| Feature | Details |
|---------|---------|
| Multi-PDF upload | Upload and query multiple PDFs simultaneously |
| Hybrid retrieval | Dense (ChromaDB) + Sparse (BM25) fused with Reciprocal Rank Fusion |
| Source highlighting | Every answer cites filename, page number, and block index |
| Conversation memory | 5-turn rolling window memory per session |
| Auto-summarization | Map-reduce summary generated on every upload |
| Multilingual | German + English via `paraphrase-multilingual-MiniLM-L12-v2` |
| Analytics dashboard | SQLite-backed tracking of queries, documents, latency, and language |

## Architecture

```
PDF Upload
    │
    ▼
PyMuPDF (text + page/block metadata)
    │
    ▼
Sentence-aware Chunking (512 chars, 80 overlap)
    │
    ├──► ChromaDB (dense embeddings via sentence-transformers)
    │
    └──► BM25 Index (sparse keyword search)
              │
              ▼
    Reciprocal Rank Fusion (top 5 results)
              │
              ▼
    Groq LLM (llama-3.3-70b) + Conversation Memory
              │
              ▼
    Streamlit UI (Chat · Documents · Analytics)
```

## Tech Stack

| Layer | Tool |
|-------|------|
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` (local, free) |
| Vector store | ChromaDB (persistent) |
| Sparse search | BM25 (rank-bm25) |
| LLM | Groq API — llama-3.3-70b-versatile (free tier) |
| Orchestration | LangChain |
| Analytics DB | SQLite |
| UI | Streamlit |

## Project Structure

```
rag-document-qa/
├── app.py                        # Main Streamlit app (3 tabs)
├── config.py                     # Central config (paths, models, params)
├── requirements.txt
├── core/
│   ├── ingestion.py              # PyMuPDF → page blocks with metadata
│   ├── chunking.py               # Sentence-aware chunking, preserves page/block
│   ├── embeddings.py             # Multilingual sentence-transformers embedder
│   ├── vectorstore.py            # ChromaDB upsert + cosine search
│   ├── retriever.py              # Dense + BM25 hybrid, RRF fusion
│   ├── llm.py                    # Groq ChatGroq wrapper (LangChain)
│   ├── chain.py                  # RAG chain with conversation memory
│   └── summarizer.py             # Map-reduce auto-summarization
├── analytics/
│   ├── database.py               # SQLite schema + CRUD
│   └── tracker.py                # track_document() + track_query()
├── ui/
│   ├── upload.py                 # PDF upload panel + document library
│   ├── chat.py                   # Chat UI with source cards
│   └── analytics_dash.py         # Analytics dashboard (Plotly charts)
└── data/
    ├── uploads/                  # Uploaded PDFs (gitignored)
    ├── chroma_db/                # ChromaDB persistence (gitignored)
    └── analytics.db              # SQLite analytics (gitignored)
```

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Add your free Groq API key from console.groq.com
```

Or paste the key directly in the app sidebar after launch.

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 4. Use it

1. Go to **Documents** tab → upload one or more PDFs
2. Auto-summary is generated on upload
3. Switch to **Chat** tab → select documents → ask questions
4. Sources are shown below each answer (filename, page, relevance score)
5. Check **Analytics** tab for query history and document stats

## Key Differentiators

- **Hybrid search** — dense + sparse BM25 fused with RRF outperforms either alone
- **Page-level citations** on every answer (filename + page number + block)
- **Multilingual** — same model handles German and English without switching
- **Conversation memory** — follow-up questions use prior context
- **Analytics tracking** — every query logged with latency, language, and sources
- **Local embeddings** — no API cost for embedding (sentence-transformers runs on CPU)

Project Screenshots
<img width="1728" height="959" alt="Screenshot 2026-05-17 at 8 52 30 AM" src="https://github.com/user-attachments/assets/c26e81d9-fd79-4b37-981f-21dae542c66d" />
<img width="1728" height="959" alt="Screenshot 2026-05-17 at 8 52 39 AM" src="https://github.com/user-attachments/assets/61131525-9859-4959-a6f0-87375765bd99" />
<img width="1728" height="959" alt="Screenshot 2026-05-17 at 8 52 49 AM" src="https://github.com/user-attachments/assets/69e3d6b1-3428-480c-8d27-13af2a8a5d55" />
<img width="1728" height="959" alt="Screenshot 2026-05-17 at 9 16 32 AM" src="https://github.com/user-attachments/assets/104566f2-a2f8-4526-b961-dc8107451c49" />
<img width="1728" height="959" alt="Screenshot 2026-05-17 at 9 16 42 AM" src="https://github.com/user-attachments/assets/08c2eede-f789-453e-bd34-66590067e106" />


import os
import uuid
import logging
import streamlit as st

from analytics.database import init_db
from core.chain import make_memory
from core.vectorstore import get_all_chunks
from ui.upload import render_upload_panel
from ui.chat import render_chat
from ui.analytics_dash import render_analytics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide",
)

# ── One-time DB init ──────────────────────────────────────────────────────────
init_db()

# ── Groq API key — sidebar input overrides .env ───────────────────────────────
st.sidebar.title("⚙️ Settings")
current_key = os.environ.get("GROQ_API_KEY", "")
key_display = f"{current_key[:8]}…" if len(current_key) > 8 else "(not set)"
st.sidebar.caption(f"Current key: `{key_display}`")

new_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_…",
    help="Get a free key at console.groq.com",
)
if new_key and new_key != current_key:
    os.environ["GROQ_API_KEY"] = new_key
    # clear LRU cache so the next LLM call picks up the new key
    from core.llm import get_llm
    get_llm.cache_clear()
    st.sidebar.success("Key updated ✓")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "memory" not in st.session_state:
    st.session_state.memory = make_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "all_chunks_text" not in st.session_state:
    texts, metas = get_all_chunks()
    st.session_state.all_chunks_text = texts
    st.session_state.all_chunks_meta = metas

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 RAG Document Q&A")
st.caption("Multi-PDF question answering with source highlighting · German & English · Conversation memory")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_analytics = st.tabs(["💬 Chat", "📄 Documents", "📊 Analytics"])

with tab_chat:
    render_chat(st.session_state)

with tab_docs:
    render_upload_panel(st.session_state)

with tab_analytics:
    render_analytics()

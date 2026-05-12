"""
Analytics dashboard — queries, documents, latency, language breakdown.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.database import get_query_stats, get_all_documents


def render_analytics() -> None:
    st.subheader("Pipeline Analytics")

    stats = get_query_stats()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Queries",    stats["total_queries"])
    k2.metric("Documents Loaded", stats["total_documents"])
    k3.metric("Avg Latency",      f"{stats['avg_latency_ms']:.0f} ms")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Top documents by hit count ─────────────────────────────────────────────
    with col_left:
        st.markdown("**Most Retrieved Documents**")
        top_docs = stats["top_documents"]
        if top_docs:
            df_docs = pd.DataFrame(top_docs)
            fig = px.bar(
                df_docs, x="hits", y="filename", orientation="h",
                color="hits", color_continuous_scale="Blues",
                labels={"hits": "Retrieval Hits", "filename": ""},
            )
            fig.update_layout(
                template="plotly_white", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False, coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query data yet.")

    # ── Query language breakdown ───────────────────────────────────────────────
    with col_right:
        st.markdown("**Query Language Distribution**")
        by_lang = stats["queries_by_lang"]
        if by_lang:
            df_lang = pd.DataFrame(by_lang)
            df_lang["language"] = df_lang["language"].map(
                {"en": "English", "de": "German"}
            ).fillna(df_lang["language"])
            fig2 = px.pie(
                df_lang, names="language", values="cnt",
                color_discrete_sequence=["#1f5fa6", "#90bce8"],
                hole=0.45,
            )
            fig2.update_traces(textinfo="percent+label")
            fig2.update_layout(
                template="plotly_white", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No query data yet.")

    st.divider()

    # ── Document library table ─────────────────────────────────────────────────
    st.markdown("**All Documents**")
    docs = get_all_documents()
    if docs:
        df = pd.DataFrame(docs)[
            ["filename", "language", "num_pages", "num_chunks", "word_count", "uploaded_at"]
        ]
        df.columns = ["Filename", "Lang", "Pages", "Chunks", "Words", "Uploaded"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # ── Recent queries table ───────────────────────────────────────────────────
    st.markdown("**Recent Queries**")
    recent = stats["recent_queries"]
    if recent:
        df_q = pd.DataFrame(recent)[["question", "created_at", "num_sources", "latency_ms"]]
        df_q.columns = ["Question", "Time", "Sources", "Latency (ms)"]
        st.dataframe(df_q, use_container_width=True, hide_index=True)
    else:
        st.info("No queries yet.")

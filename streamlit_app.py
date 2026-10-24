"""Streamlit Web UI for Enterprise RAG Document Assistant."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from ragassistant.config import load_config
from ragassistant.generation.generator import AnswerGenerator
from ragassistant.generation.llm import get_llm_client
from ragassistant.ingestion.pipeline import ingest_directory, ingest_file
from ragassistant.observability.tracer import get_tracer
from ragassistant.retrieval.engine import RetrievalEngine

# Page Configuration
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-family: monospace;
        color: #334155;
        margin-top: 8px;
    }
    .badge-refused {
        background-color: #FEF2F2;
        border-color: #FECACA;
        color: #991B1B;
    }
    .badge-grounded {
        background-color: #F0FDF4;
        border-color: #BBF7D0;
        color: #166534;
    }
    .citation-box {
        border-left: 3px solid #3B82F6;
        background-color: #F8FAFC;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .quote-text {
        font-style: italic;
        color: #1E293B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing retrieval and embedding engine...")
def load_app_resources(
    provider: str = "groq",
    model: str = "llama-3.1-8b-instant",
    api_key: str | None = None,
) -> tuple[AnswerGenerator, RetrievalEngine]:
    """Initialize and cache retrieval engine and answer generator."""
    cfg = load_config()
    retrieval_engine = RetrievalEngine.from_config(cfg)

    # Ingest default sample docs if store is empty
    if retrieval_engine.vector_store.count() == 0:
        docs_dir = Path(cfg.get("storage", {}).get("sample_docs_dir", "data/sample_docs"))
        if docs_dir.exists():
            ingest_directory(docs_dir, retrieval_engine.vector_store)

    llm_client = get_llm_client(
        provider=provider,
        model=model,
        api_key=api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        fallback_on_missing_key=True,
    )

    generator = AnswerGenerator(
        retrieval_engine=retrieval_engine,
        llm_client=llm_client,
        model_name=model,
        tracer=get_tracer(),
    )
    return generator, retrieval_engine


# --- SIDEBAR: Document Management & Retention Controls ---
with st.sidebar:
    st.title("📂 Document Repository")
    st.markdown("Enterprise document ingestion, active index tracking, and retention controls.")

    # LLM Provider Configuration
    with st.expander("⚙️ Provider & LLM Settings", expanded=False):
        selected_provider = st.selectbox(
            "Inference Provider",
            options=["groq", "openai", "deterministic"],
            index=0,
            help="Select Groq for LPU inference, OpenAI, or Deterministic for offline evaluation.",
        )
        selected_model = (
            "llama-3.1-8b-instant"
            if selected_provider == "groq"
            else ("gpt-4o-mini" if selected_provider == "openai" else "deterministic")
        )
        user_api_key = st.text_input(
            "API Key (Optional)",
            type="password",
            help="Leave blank to use environment keys or deterministic offline extraction.",
        )

    generator, retrieval_engine = load_app_resources(
        provider=selected_provider,
        model=selected_model,
        api_key=user_api_key.strip() if user_api_key else None,
    )
    vector_store = retrieval_engine.vector_store

    st.divider()

    # Document Upload Section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, Markdown, or TXT documents",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Files are indexed immediately with chunking and semantic embeddings.",
    )

    if uploaded_files:
        if st.button("🚀 Ingest Uploaded Documents", use_container_width=True):
            with st.spinner("Ingesting and embedding documents..."):
                total_chunks_added = 0
                for file_obj in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(file_obj.name).suffix,
                    ) as tmp:
                        tmp.write(file_obj.getvalue())
                        tmp_path = Path(tmp.name)

                    try:
                        stats = ingest_file(tmp_path, vector_store)
                        # Patch doc_name to original uploaded name
                        vector_store.collection.update(
                            where={"doc_name": tmp_path.name},
                            metadatas=[{"doc_name": file_obj.name}] * stats.total_chunks,
                        )
                        total_chunks_added += stats.total_chunks
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()

                st.success(f"Successfully ingested {len(uploaded_files)} files ({total_chunks_added} chunks).")
                st.rerun()

    st.divider()

    # Active Documents & Retention Management
    active_docs = vector_store.list_documents()
    total_chunk_count = vector_store.count()

    st.subheader(f"Active Documents ({len(active_docs)})")
    st.caption(f"Total Chunks Indexed: **{total_chunk_count}**")

    if active_docs:
        for doc_name in active_docs:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.write(f"📄 `{doc_name}`")
            with col_del:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Delete {doc_name} from index"):
                    vector_store.delete_document(doc_name)
                    st.toast(f"Deleted {doc_name} under retention policy.")
                    st.rerun()
    else:
        st.info("No documents currently indexed.")

    st.divider()

    # Retention Lifecycle Actions
    st.subheader("Data Retention Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Re-index Samples", use_container_width=True, help="Reload built-in corporate policies"):
            with st.spinner("Re-indexing sample documents..."):
                cfg = load_config()
                sample_dir = Path(cfg.get("storage", {}).get("sample_docs_dir", "data/sample_docs"))
                ingest_directory(sample_dir, vector_store)
                st.toast("Re-indexed sample corporate documents.")
                st.rerun()
    with c2:
        if st.button("⚠️ Wipe Index", use_container_width=True, help="Compliance wipe of all chunks"):
            vector_store.cleanup_retention()
            st.toast("Index wiped completely.")
            st.rerun()


# --- CENTER PANEL: Chat Conversation Interface ---
st.markdown('<div class="main-header">📚 Enterprise Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Retrieval-augmented AI assistant with verbatim citations, automatic refusal, and cost tracking.</div>',
    unsafe_allow_html=True,
)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Query Suggestions
st.markdown("**Quick Prompts:**")
suggested_queries = [
    "What are the standard core working hours at Acme Technologies?",
    "How many days of remote work are employees permitted per week?",
    "What are the RPO and RTO disaster recovery targets for CloudScale?",
    "What is the gym reimbursement amount in Tokyo?",  # Adversarial query
]

cols = st.columns(len(suggested_queries))
prompt_to_submit = None

for i, query_text in enumerate(suggested_queries):
    with cols[i]:
        label = query_text if len(query_text) < 45 else query_text[:42] + "..."
        if st.button(label, key=f"sug_{i}", use_container_width=True):
            prompt_to_submit = query_text

# Render Existing Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Status Badge
            refused = msg.get("refused", False)
            conf = msg.get("confidence", 0.0)

            if refused:
                st.markdown(
                    '<span class="badge-pill badge-refused">🛡️ REFUSED: Query outside verified documentation</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge-pill badge-grounded">✅ GROUNDED (Confidence: {conf:.1%})</span>',
                    unsafe_allow_html=True,
                )

            # Collapsible Citations Accordion
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"🔍 Source Citations ({len(citations)})", expanded=False):
                    for idx, c in enumerate(citations, 1):
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <b>Citation #{idx}</b>: <code>{c['doc_name']}</code> (Page {c['page']})<br>
                                <span class="quote-text">"{c['quote']}"</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Floating Telemetry Pill
            trace = msg.get("trace")
            if trace:
                st.markdown(
                    f"""
                    <div class="badge-pill">
                        <span>⚡ <b>{trace['total_latency_ms']:.0f} ms</b> (Retrieval: {trace['retrieval_latency_ms']:.0f}ms | Gen: {trace['generation_latency_ms']:.0f}ms)</span>
                        <span>🪙 <b>{trace['total_tokens']} tokens</b> ({trace['input_tokens']} in / {trace['output_tokens']} out)</span>
                        <span>💵 <b>${trace['estimated_cost_usd']:.6f}</b> (${trace['cost_per_1k_queries_usd']:.2f}/1k queries)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Chat Input & Execution
chat_input_val = st.chat_input("Ask a question about employee policies, cloud architecture, or contracts...")
active_query = prompt_to_submit or chat_input_val

if active_query:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and validating citations..."):
            answer, trace = generator.generate_with_trace(active_query, use_reranker=True)

        st.markdown(answer.answer)

        # Status badge
        if answer.refused:
            st.markdown(
                '<span class="badge-pill badge-refused">🛡️ REFUSED: Query outside verified documentation</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="badge-pill badge-grounded">✅ GROUNDED (Confidence: {answer.confidence:.1%})</span>',
                unsafe_allow_html=True,
            )

        # Citations Accordion
        citations_data = [
            {"doc_name": c.doc_name, "page": c.page, "quote": c.quote}
            for c in answer.citations
        ]
        if citations_data:
            with st.expander(f"🔍 Source Citations ({len(citations_data)})", expanded=True):
                for idx, c in enumerate(citations_data, 1):
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <b>Citation #{idx}</b>: <code>{c['doc_name']}</code> (Page {c['page']})<br>
                            <span class="quote-text">"{c['quote']}"</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Telemetry Pill
        trace_data = trace.model_dump()
        st.markdown(
            f"""
            <div class="badge-pill">
                <span>⚡ <b>{trace_data['total_latency_ms']:.0f} ms</b> (Retrieval: {trace_data['retrieval_latency_ms']:.0f}ms | Gen: {trace_data['generation_latency_ms']:.0f}ms)</span>
                <span>🪙 <b>{trace_data['total_tokens']} tokens</b> ({trace_data['input_tokens']} in / {trace_data['output_tokens']} out)</span>
                <span>💵 <b>${trace_data['estimated_cost_usd']:.6f}</b> (${trace_data['cost_per_1k_queries_usd']:.2f}/1k queries)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.answer,
            "citations": citations_data,
            "refused": answer.refused,
            "confidence": answer.confidence,
            "trace": trace_data,
        }
    )

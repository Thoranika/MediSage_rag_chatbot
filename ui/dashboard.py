"""Home Dashboard view with KPI summary cards, quick-access panels, and runtime health."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st

from rag.safety import EDUCATIONAL_DISCLAIMER
from scripts.generate_sample_docs import generate_all_samples
from ui.styles import render_kpi_card, status_badge


def render_dashboard(services: dict[str, Any]) -> None:
    """Render the MediSage Home Dashboard."""
    db = services["db"]
    chroma = services["chroma"]
    embeddings = services["embeddings"]
    ingestion = services["ingestion"]
    settings = services["settings"]

    st.markdown(
        """
        <div class="med-hero">
            <span class="med-hero-pill">Clinical Reference Intelligence</span>
            <div class="med-hero-title">MediSage — Evidence-Grounded Medical Reference Assistant</div>
            <p class="med-hero-sub">
                Search permitted clinical guidelines, pharmacotherapy summaries, and medical reference
                documents with transparent page-level citations, local CPU embeddings, and strict
                educational safety guardrails.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="med-disclaimer"><strong>Educational Notice:</strong> {html.escape(EDUCATIONAL_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    stats = db.get_knowledge_stats()
    vector_count = chroma.count()
    docs = db.list_documents()
    last_doc = docs[0] if docs else None
    last_status_label = (
        f"{last_doc['original_filename']} ({last_doc['processing_status']})"
        if last_doc
        else "No documents ingested yet"
    )

    eval_path = Path("evaluation/eval_results.json")
    eval_recall = "Not run"
    eval_sub = "Run evaluation benchmark"
    if eval_path.exists():
        try:
            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
            summ = eval_data.get("summary", {})
            eval_recall = f"{summ.get('recall_at_k', 0.0) * 100:.0f}%"
            eval_sub = f"Mean retrieval: {summ.get('mean_retrieval_latency_ms', 0.0):.1f} ms"
        except Exception:
            pass

    # Top KPI Summary Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Indexed Documents",
            stats["indexed_documents"],
            f"{stats['total_pages']} extracted reference pages",
            icon="📄",
            accent="teal",
        )
    with c2:
        render_kpi_card(
            "Vector Chunks",
            vector_count,
            "Persistent ChromaDB index",
            icon="🧩",
            accent="lavender",
        )
    with c3:
        render_kpi_card(
            "Conversations",
            stats["conversation_count"],
            f"{stats['message_count']} persisted chat messages",
            icon="💬",
            accent="coral",
        )
    with c4:
        render_kpi_card(
            "Page Recall@K",
            eval_recall,
            eval_sub,
            icon="🎯",
            accent="green",
        )

    st.markdown("<div style='height: 0.9rem'></div>", unsafe_allow_html=True)

    # Quick-Access Feature Cards
    q1, q2, q3 = st.columns(3, gap="medium")
    with q1:
        st.markdown(
            """
            <div class="med-feature-card">
                <div class="med-feature-head">📚 Knowledge Library</div>
                <p class="med-feature-desc">
                    Upload PDF, DOCX, TXT, or Markdown references, inspect extracted chunks,
                    and manage indexed documents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Knowledge Library", use_container_width=True, key="qa_lib"):
            st.session_state["nav_page"] = "Knowledge Library"
            st.rerun()

    with q2:
        st.markdown(
            """
            <div class="med-feature-card">
                <div class="med-feature-head">🩺 Medical Assistant</div>
                <p class="med-feature-desc">
                    Ask evidence-grounded clinical reference questions with verified page citations
                    and follow-up memory.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Launch Medical Assistant",
            type="primary",
            use_container_width=True,
            key="qa_chat",
        ):
            st.session_state["nav_page"] = "Medical Assistant"
            st.rerun()

    with q3:
        st.markdown(
            """
            <div class="med-feature-card">
                <div class="med-feature-head">📊 Evaluation & Tuning</div>
                <p class="med-feature-desc">
                    Measure Recall@K, MRR, and abstention accuracy across chunking, Top-K,
                    and threshold settings.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Evaluation & Tuning", use_container_width=True, key="qa_eval"):
            st.session_state["nav_page"] = "Evaluation & Tuning"
            st.rerun()

    st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1.0], gap="medium")

    with col_left:
        with st.container(border=True):
            st.subheader("Evidence-Grounded RAG Workflow")
            st.markdown(
                """
                1. **Document Ingestion:** Upload permitted guidelines in **Knowledge Library** with MIME, size, and SHA-256 validation.
                2. **CPU Indexing:** Text is cleaned, chunked along headings and tables, embedded via `all-MiniLM-L6-v2`, and persisted in ChromaDB.
                3. **Grounded Generation:** **Medical Assistant** retrieves the top passages, applies safety boundaries, and validates `[Source: filename, page N]` citations.
                4. **Quality Verification:** **Evaluation & Tuning** benchmarks retrieval precision and out-of-knowledge-base abstention.
                """
            )
            st.caption(f"Last ingestion activity: `{last_status_label}`")

        with st.container(border=True):
            st.subheader("Quick Start: Educational Sample Corpus")
            st.write(
                "Load three non-sensitive educational clinical reference documents "
                "(Adult Hypertension PDF, Type 2 Diabetes PDF, Acute Respiratory Stewardship DOCX)."
            )
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button(
                    "Load Sample Medical Library",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Loading and indexing sample reference documents..."):
                        sample_files = generate_all_samples(settings.samples_dir)
                        results = []
                        for spath in sample_files:
                            outcome = ingestion.ingest_file_path(
                                spath, source_origin="sample"
                            )
                            results.append(outcome)
                    indexed_n = sum(1 for r in results if r.status == "indexed")
                    dup_n = sum(1 for r in results if r.status == "duplicate")
                    st.success(
                        f"Sample library ready: {indexed_n} newly indexed, {dup_n} already present."
                    )
                    st.rerun()
            with b_col2:
                if st.button("View Conversation History", use_container_width=True):
                    st.session_state["nav_page"] = "Conversation History"
                    st.rerun()

    with col_right:
        with st.container(border=True):
            st.subheader("System Health & Runtime Status")

            db_ok = db.health_check()
            chroma_ok = chroma.health_check()
            emb_warm = embeddings.is_loaded()
            api_ok = settings.has_api_key()
            active_model = db.get_setting("llm_model_id", settings.llm_model_id)

            st.markdown(
                f"**SQLite Metadata Store**  \n{status_badge('Connected' if db_ok else 'Error', 'ok' if db_ok else 'err')}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**ChromaDB Vector Index**  \n{status_badge(f'Ready ({vector_count} chunks)' if chroma_ok else 'Error', 'ok' if chroma_ok else 'err')}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**CPU Embeddings (`{settings.embedding_model_name}`)**  \n"
                + status_badge(
                    f"Warm ({embeddings.cold_start_ms:.0f} ms)" if emb_warm else "Ready (Lazy CPU)",
                    "ok" if emb_warm else "info",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Cloud LLM (`{settings.llm_provider} / {active_model}`)**  \n"
                + status_badge(
                    f"API Connected ({settings.llm_api_version})"
                    if api_ok
                    else "API Key Missing (.env)",
                    "ok" if api_ok else "warn",
                ),
                unsafe_allow_html=True,
            )
            if not api_ok:
                st.info(
                    "Local CPU retrieval and chunk inspection remain available offline. "
                    "Configure `GEMINI_API_KEY` in `.env` to enable Gemini synthesis."
                )
            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
            if st.button("Open Settings & Security", use_container_width=True):
                st.session_state["nav_page"] = "Settings & Security"
                st.rerun()

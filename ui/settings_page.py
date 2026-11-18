"""Settings, Security, Cache & Privacy Management page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.llm_client import test_llm_connection
from ui.styles import render_kpi_card, render_page_header, status_badge
from utils.cache import get_shared_cache
from utils.validators import (
    SUPPORTED_MODELS,
    SUPPORTED_RESPONSE_STYLES,
    validate_settings_payload,
)


def render_settings(services: dict[str, Any]) -> None:
    """Render validated settings UI with modern cards, toggles, and security overview."""
    db = services["db"]
    chroma = services["chroma"]
    ingestion = services["ingestion"]
    settings = services["settings"]
    cache = get_shared_cache()

    render_page_header(
        "Application Settings & Security",
        "Manage cloud Gemini model parameters, retrieval tuning defaults, cache controls, and data governance.",
    )

    st.markdown(
        """
        <div class="med-privacy-warn">
            <strong>Medical Privacy & Authorization Notice:</strong> Only index permitted,
            non-sensitive educational medical references. Never upload identifiable patient records (PHI)
            without formal authorization and safeguards.
        </div>
        """,
        unsafe_allow_html=True,
    )

    persisted = db.get_all_settings()
    cur_provider = persisted.get("llm_provider", settings.llm_provider)
    cur_model = persisted.get("llm_model_id", settings.llm_model_id)
    cur_top_k = int(persisted.get("retrieval_top_k", settings.retrieval_top_k))
    cur_threshold = float(
        persisted.get("similarity_threshold", settings.similarity_threshold)
    )
    cur_style = persisted.get("response_style", settings.response_style)
    cur_temp = float(persisted.get("llm_temperature", settings.llm_temperature))
    cur_max_tokens = int(
        persisted.get("llm_max_output_tokens", settings.llm_max_output_tokens)
    )
    cur_chunk_size = int(
        persisted.get("chunk_size_tokens", settings.chunk_size_tokens)
    )
    cur_chunk_overlap = int(
        persisted.get("chunk_overlap_tokens", settings.chunk_overlap_tokens)
    )

    # Top Runtime & Security Status Cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_kpi_card(
            "API Key Status",
            "Configured" if settings.has_api_key() else "Missing",
            settings.masked_api_key(),
            icon="🔐",
            accent="green" if settings.has_api_key() else "coral",
        )
    with s2:
        render_kpi_card(
            "Selected LLM",
            cur_model,
            f"{cur_provider} ({settings.llm_api_version})",
            icon="✨",
            accent="lavender",
        )
    with s3:
        render_kpi_card(
            "Embedding Model",
            settings.embedding_model_name,
            f"Local CPU ({settings.embedding_device})",
            icon="🧠",
            accent="teal",
        )
    with s4:
        render_kpi_card(
            "Vector DB",
            "ChromaDB Ready" if chroma.health_check() else "Unavailable",
            f"{chroma.count()} stored vectors",
            icon="🗄️",
            accent="teal",
        )

    st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)

    # Standard Settings Card
    with st.container(border=True):
        with st.form("standard_settings_form"):
            st.subheader("Model, Retrieval & Response Preferences")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                provider = st.selectbox("Cloud LLM Provider", ["GEMINI"], index=0)
                model_choice = st.selectbox(
                    "Gemini Model ID",
                    options=SUPPORTED_MODELS + ["Custom..."],
                    index=SUPPORTED_MODELS.index(cur_model)
                    if cur_model in SUPPORTED_MODELS
                    else len(SUPPORTED_MODELS),
                )
                custom_model = st.text_input(
                    "Custom Model ID (used if 'Custom...' selected)",
                    value=cur_model if cur_model not in SUPPORTED_MODELS else "",
                )
                response_style = st.selectbox(
                    "Response Style",
                    options=SUPPORTED_RESPONSE_STYLES,
                    index=SUPPORTED_RESPONSE_STYLES.index(cur_style)
                    if cur_style in SUPPORTED_RESPONSE_STYLES
                    else 0,
                )
            with c2:
                top_k = st.slider(
                    "Top-K Retrieved Passages", min_value=1, max_value=12, value=cur_top_k
                )
                similarity_threshold = st.slider(
                    "Minimum Similarity Threshold",
                    min_value=0.0,
                    max_value=0.90,
                    value=cur_threshold,
                    step=0.05,
                )
                temperature = st.slider(
                    "LLM Temperature (lower = stricter grounding)",
                    min_value=0.0,
                    max_value=1.0,
                    value=cur_temp,
                    step=0.05,
                )
                max_output_tokens = st.number_input(
                    "Max Output Tokens",
                    min_value=128,
                    max_value=4096,
                    value=cur_max_tokens,
                    step=128,
                )

            submitted = st.form_submit_button("Save Standard Settings", type="primary")
            if submitted:
                chosen_model = (
                    custom_model.strip() if model_choice == "Custom..." else model_choice
                )
                ok, errors, cleaned = validate_settings_payload(
                    {
                        "llm_provider": provider,
                        "llm_model_id": chosen_model,
                        "retrieval_top_k": top_k,
                        "similarity_threshold": similarity_threshold,
                        "llm_temperature": temperature,
                        "llm_max_output_tokens": max_output_tokens,
                        "response_style": response_style,
                        "chunk_size_tokens": cur_chunk_size,
                        "chunk_overlap_tokens": cur_chunk_overlap,
                    }
                )
                if ok:
                    db.set_settings_bulk(cleaned)
                    st.success("Settings validated and saved persistently.")
                else:
                    for err in errors:
                        st.error(err)

    # Cloud API Credentials & Verification Card
    with st.container(border=True):
        st.subheader("Cloud API Credentials & Connectivity")
        key_badge = (
            status_badge("Configured (" + settings.masked_api_key() + ")", "ok")
            if settings.has_api_key()
            else status_badge("Not Configured", "err")
        )
        st.markdown(
            f"**GEMINI_API_KEY:** {key_badge} &nbsp; "
            f"{status_badge('API ' + settings.llm_api_version, 'lavender')}",
            unsafe_allow_html=True,
        )
        st.caption(
            "API credentials are loaded exclusively from `.env` or environment variables and are masked in all logs."
        )

        col_btn, col_env = st.columns([1.0, 1.8])
        with col_btn:
            if st.button("Test Gemini API Connection", type="primary", use_container_width=True):
                with st.spinner("Verifying Gemini API connection..."):
                    res = test_llm_connection(
                        {
                            "llm_provider": cur_provider,
                            "llm_model_id": cur_model,
                            "llm_api_version": settings.llm_api_version,
                            "api_key": settings.get_api_key(),
                        }
                    )
                if res.success:
                    st.success(
                        f"Connected to {res.provider} ({res.model_id}) in {res.latency_ms:.0f} ms."
                    )
                else:
                    st.warning(f"Connection check ({res.error_type}): {res.error_message}")
        with col_env:
            with st.expander("Environment Configuration Instructions (.env)", expanded=False):
                st.markdown(
                    """
                    1. Copy `.env.example` to `.env` in the project root.
                    2. Set `GEMINI_API_KEY=<your_api_key>`, `LLM_MODEL_ID=gemini-2.5-flash`, and `GEMINI_API_VERSION=v1beta`.
                    3. Restart Streamlit to reload updated `.env` values; `.env` is ignored by Git.
                    """
                )

    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)

    # Advanced Settings separated cleanly from standard actions
    with st.expander("⚙️ Advanced Settings: Chunking, Cache & Data Governance", expanded=False):
        with st.form("advanced_chunking_form"):
            st.markdown("**Document Chunking Configuration (Applies to New / Re-indexed Docs)**")
            ac1, ac2 = st.columns(2)
            with ac1:
                chunk_size = st.number_input(
                    "Chunk Size (approx. tokens)",
                    min_value=150,
                    max_value=1500,
                    value=cur_chunk_size,
                    step=50,
                )
            with ac2:
                chunk_overlap = st.number_input(
                    "Chunk Overlap (approx. tokens)",
                    min_value=0,
                    max_value=400,
                    value=cur_chunk_overlap,
                    step=20,
                )
            save_adv = st.form_submit_button("Save Chunking Parameters")
            if save_adv:
                ok, errors, cleaned = validate_settings_payload(
                    {
                        "llm_provider": cur_provider,
                        "llm_model_id": cur_model,
                        "retrieval_top_k": cur_top_k,
                        "similarity_threshold": cur_threshold,
                        "llm_temperature": cur_temp,
                        "llm_max_output_tokens": cur_max_tokens,
                        "response_style": cur_style,
                        "chunk_size_tokens": chunk_size,
                        "chunk_overlap_tokens": chunk_overlap,
                    }
                )
                if ok:
                    db.set_settings_bulk(
                        {
                            "chunk_size_tokens": cleaned["chunk_size_tokens"],
                            "chunk_overlap_tokens": cleaned["chunk_overlap_tokens"],
                        }
                    )
                    st.success("Chunking parameters updated.")
                else:
                    for err in errors:
                        st.error(err)

        st.divider()
        st.markdown("**Deterministic Retrieval & Embedding Cache**")
        c_stats = cache.stats()
        st.markdown(
            status_badge(f"Index v{c_stats['index_version']}", "info")
            + status_badge(f"{c_stats['embedding_entries']} cached embeddings", "neutral")
            + status_badge(f"{c_stats['retrieval_entries']} cached queries", "neutral")
            + status_badge(f"Hits/Misses: {c_stats['hits']}/{c_stats['misses']}", "lavender"),
            unsafe_allow_html=True,
        )
        if st.button("Invalidate & Clear Caches"):
            cache.clear_all()
            st.success("Caches cleared.")
            st.rerun()

        st.divider()
        st.markdown("**Privacy & Data Purge Controls**")
        confirm_purge_chat = st.toggle(
            "Confirm deletion of all saved conversations and feedback"
        )
        if st.button("Purge All Conversations & Feedback", disabled=not confirm_purge_chat):
            removed = db.delete_all_conversations_and_feedback()
            st.session_state.pop("active_conversation_id", None)
            st.success(f"Deleted {removed} conversation(s) and associated feedback.")
            st.rerun()

        confirm_purge_kb = st.toggle(
            "Confirm deletion of all uploaded documents and vector embeddings"
        )
        if st.button("Purge Entire Knowledge Base", disabled=not confirm_purge_kb):
            docs = db.list_documents()
            for d in docs:
                ingestion.delete_document_completely(d["document_id"])
            chroma.clear_collection()
            st.success("All documents, uploaded files, and vector records removed.")
            st.rerun()

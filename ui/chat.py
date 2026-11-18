"""Interactive Medical AI Chat view with distinct bubbles, clickable sample prompts, citations, and retry."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from rag.safety import EDUCATIONAL_DISCLAIMER
from ui.styles import render_empty_state, render_page_header, status_badge

_SAMPLE_PROMPTS = [
    {
        "title": "Hypertension Stages",
        "icon": "🫀",
        "question": "What blood pressure ranges define Stage 1 and Stage 2 hypertension?",
    },
    {
        "title": "Metformin Renal Limits",
        "icon": "💊",
        "question": "At what eGFR threshold is metformin contraindicated in Type 2 diabetes?",
    },
    {
        "title": "CURB-65 Pneumonia Score",
        "icon": "🫁",
        "question": "What are the five CURB-65 scoring criteria and how do scores guide care?",
    },
]


def _render_citations_panel(
    citations: list[dict[str, Any]],
    *,
    retrieval_status: str,
    rewritten_query: str = "",
    latency_ms: float = 0.0,
    timestamp_display: str = "",
) -> None:
    """Render status badges, response time, timestamp, and expandable verified citations."""
    badge_map = {
        "ok": ("Grounded Retrieval", "ok"),
        "low_confidence": ("Moderate Confidence", "warn"),
        "no_evidence": ("Insufficient KB Evidence", "warn"),
        "safety_blocked": ("Safety Escalation", "err"),
        "llm_error": ("LLM Unavailable", "err"),
    }
    label, variant = badge_map.get(retrieval_status, (retrieval_status or "Ready", "neutral"))
    badges_html = status_badge(label, variant)
    if latency_ms > 0:
        badges_html += status_badge(f"⚡ {latency_ms:.0f} ms", "info")
    if rewritten_query:
        badges_html += status_badge("Context Follow-Up Rewritten", "lavender")
    time_html = (
        f'<span class="med-msg-time">{html.escape(timestamp_display)}</span>'
        if timestamp_display
        else ""
    )

    st.markdown(
        f'<div class="med-msg-meta-bar">{badges_html}{time_html}</div>',
        unsafe_allow_html=True,
    )

    if rewritten_query:
        st.caption(f"Standalone retrieval query: `{rewritten_query}`")

    if citations:
        with st.expander(f"📚 View Verified Sources & Excerpts ({len(citations)})", expanded=False):
            for idx, cit in enumerate(citations, start=1):
                fname = html.escape(str(cit.get("filename", "Unknown")))
                page_n = cit.get("page_number", 1)
                sec = html.escape(str(cit.get("section_title", "")))
                sim = float(cit.get("similarity_score", 0.0))
                excerpt = html.escape(str(cit.get("excerpt", "")))
                sec_part = f" • {sec}" if sec else ""
                st.markdown(
                    f"""
                    <div class="med-citation">
                        <div class="med-citation-head">
                            <span>[{idx}] {fname} — Page {page_n}{sec_part}</span>
                            <span>{status_badge(f"Similarity {sim:.2f}", "info")}</span>
                        </div>
                        <div class="med-citation-body">{excerpt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _execute_chat_turn(
    chat: Any,
    conversation_id: str,
    question: str,
    metadata_filter: dict[str, Any] | None,
) -> None:
    """Run RAG pipeline via ChatManager and update latest retrieval inspector state."""
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Searching medical references and generating grounded answer..."):
            _, _, rag_resp = chat.ask(
                conversation_id=conversation_id,
                user_question=question,
                metadata_filters=metadata_filter,
            )
        st.session_state["last_rag_chunks"] = [
            {
                "chunk_id": c.chunk_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "similarity": c.similarity,
                "distance": c.distance,
                "text": c.text,
            }
            for c in rag_resp.retrieved_chunks
        ]
    st.rerun()


def render_chat(services: dict[str, Any]) -> None:
    """Render multi-turn Medical Assistant chat page."""
    db = services["db"]
    chat = services["chat"]
    settings = services["settings"]

    render_page_header(
        "Medical Knowledge Assistant",
        "Ask evidence-grounded educational questions backed by page-level citations from your Knowledge Library.",
    )

    indexed_docs = db.list_documents(status="indexed")
    if not indexed_docs:
        render_empty_state(
            "📄",
            "No Indexed Documents Available",
            "Load the sample reference library from Home or upload PDF/DOCX documents in Knowledge Library before asking questions.",
        )

    # Active conversation management
    convs = chat.list_conversations()
    if "active_conversation_id" not in st.session_state or not db.get_conversation(
        st.session_state["active_conversation_id"]
    ):
        if convs:
            st.session_state["active_conversation_id"] = convs[0]["conversation_id"]
        else:
            created = chat.create_conversation("New Medical Query")
            st.session_state["active_conversation_id"] = created["conversation_id"]
            convs = chat.list_conversations()

    active_cid = st.session_state["active_conversation_id"]

    # Unified Toolbar Container
    with st.container(border=True):
        bar1, bar2, bar3, bar4 = st.columns([2.2, 1.6, 0.85, 0.85])
        with bar1:
            conv_map = {
                f"{c['title']} ({c.get('message_count', 0)} msgs)": c["conversation_id"]
                for c in convs
            }
            labels = list(conv_map.keys())
            current_idx = next(
                (i for i, lbl in enumerate(labels) if conv_map[lbl] == active_cid), 0
            )
            chosen_lbl = st.selectbox(
                "Active Conversation",
                labels,
                index=current_idx,
                label_visibility="collapsed",
            )
            if conv_map[chosen_lbl] != active_cid:
                st.session_state["active_conversation_id"] = conv_map[chosen_lbl]
                st.rerun()

        with bar2:
            doc_filter_options = {"All Indexed Documents": None}
            for d in indexed_docs:
                doc_filter_options[d["original_filename"]] = {
                    "document_id": d["document_id"]
                }
            chosen_filter_lbl = st.selectbox(
                "Document Scope",
                list(doc_filter_options.keys()),
                label_visibility="collapsed",
            )
            metadata_filter = doc_filter_options[chosen_filter_lbl]

        with bar3:
            if st.button("＋ New Chat", type="primary", use_container_width=True):
                new_conv = chat.create_conversation("New Medical Query")
                st.session_state["active_conversation_id"] = new_conv["conversation_id"]
                st.rerun()

        with bar4:
            if st.button("Clear Chat", use_container_width=True):
                chat.clear_conversation(active_cid)
                st.session_state.pop("last_rag_chunks", None)
                st.rerun()

    if not settings.has_api_key():
        st.warning(
            "GEMINI_API_KEY is not configured in `.env`. Queries will run local CPU semantic retrieval "
            "and surface matching source excerpts while reporting the missing key."
        )

    messages = chat.get_messages(active_cid)
    feedback_map = db.get_feedback_for_conversation(active_cid)

    # Empty state with clickable sample educational question cards
    if not messages:
        st.markdown(
            """
            <div class="med-card" style="margin-bottom:0.75rem;">
                <div class="med-card-title">Start an Evidence-Grounded Conversation</div>
                <p style="font-size:0.88rem;color:#748196;margin:0;">
                    Select a sample educational question below or type your own medical reference query.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        scols = st.columns(3, gap="medium")
        for idx, sample in enumerate(_SAMPLE_PROMPTS):
            with scols[idx]:
                st.markdown(
                    f"""
                    <div class="med-feature-card">
                        <div class="med-feature-head">{sample['icon']} {html.escape(sample['title'])}</div>
                        <p class="med-feature-desc">{html.escape(sample['question'])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Ask This Question",
                    key=f"sample_q_{idx}",
                    use_container_width=True,
                    disabled=not bool(indexed_docs),
                ):
                    _execute_chat_turn(
                        chat, active_cid, sample["question"], metadata_filter
                    )

    # Conversation Message History
    last_user_question = ""
    for msg in messages:
        role = msg["role"]
        if role == "user":
            last_user_question = msg["content"]
        ts_raw = str(msg.get("created_at", ""))
        ts_display = f"{ts_raw[:19].replace('T', ' ')} UTC" if ts_raw else ""

        with st.chat_message(role, avatar="🧑‍⚕️" if role == "user" else "🩺"):
            st.markdown(msg["content"])
            if role == "user":
                if ts_display:
                    st.caption(ts_display)
            else:
                _render_citations_panel(
                    msg.get("citations", []),
                    retrieval_status=msg.get("retrieval_status", ""),
                    rewritten_query=msg.get("rewritten_query", ""),
                    latency_ms=float(msg.get("latency_ms", 0.0)),
                    timestamp_display=ts_display,
                )
                mid = msg["message_id"]
                existing_rating = feedback_map.get(mid)

                if msg.get("retrieval_status") == "llm_error" and last_user_question:
                    if st.button("↻ Retry Generation", key=f"retry_{mid}"):
                        _execute_chat_turn(
                            chat, active_cid, last_user_question, metadata_filter
                        )

                fb1, fb2, _ = st.columns([0.25, 0.29, 0.46])
                with fb1:
                    if st.button(
                        "👍 Helpful ✓" if existing_rating == "helpful" else "👍 Helpful",
                        key=f"up_{mid}",
                        use_container_width=True,
                    ):
                        chat.submit_feedback(
                            message_id=mid, conversation_id=active_cid, rating="helpful"
                        )
                        st.rerun()
                with fb2:
                    if st.button(
                        "👎 Not Helpful ✓"
                        if existing_rating == "not_helpful"
                        else "👎 Not Helpful",
                        key=f"down_{mid}",
                        use_container_width=True,
                    ):
                        chat.submit_feedback(
                            message_id=mid,
                            conversation_id=active_cid,
                            rating="not_helpful",
                        )
                        st.rerun()

    # Developer Retrieval Inspector
    last_chunks = st.session_state.get("last_rag_chunks") or []
    if last_chunks:
        with st.expander("🔎 Developer Retrieval Inspector (Latest Query)", expanded=False):
            for rank, item in enumerate(last_chunks, start=1):
                st.markdown(
                    f"**Rank {rank}** | `{item['filename']}` (Page {item['page_number']}, "
                    f"Section: `{item['section_title']}`) | Similarity: `{item['similarity']:.4f}` | "
                    f"Distance: `{item['distance']:.4f}` | Chunk ID: `{item['chunk_id']}`"
                )
                st.text(item["text"])

    st.markdown(
        f'<div class="med-disclaimer" style="margin-top:0.75rem;margin-bottom:0.5rem;">'
        f'<strong>Educational Disclaimer:</strong> {html.escape(EDUCATIONAL_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    prompt = st.chat_input("Ask an evidence-grounded question about your medical documents...")
    if prompt:
        _execute_chat_turn(chat, active_cid, prompt, metadata_filter)

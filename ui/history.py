"""Conversation History & Feedback Management page with modern cards and export."""

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
import streamlit as st

from ui.styles import render_empty_state, render_kpi_card, render_page_header, status_badge


def render_history(services: dict[str, Any]) -> None:
    """Render conversation search, cards, rename, reopen, export, delete, and feedback log."""
    db = services["db"]
    chat = services["chat"]

    render_page_header(
        "Conversation History & Feedback",
        "Review past evidence-grounded conversations, rename or export transcripts, and audit helpfulness feedback.",
    )

    fb_stats = db.get_feedback_stats()
    all_convs = chat.list_conversations()

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card(
            "Saved Conversations",
            len(all_convs),
            "Persisted in local SQLite",
            icon="🗂️",
            accent="teal",
        )
    with k2:
        render_kpi_card(
            "Helpful Ratings",
            fb_stats["helpful"],
            f"Out of {fb_stats['total']} total ratings",
            icon="👍",
            accent="green",
        )
    with k3:
        render_kpi_card(
            "Needs Review",
            fb_stats["not_helpful"],
            "Marked not helpful by users",
            icon="📝",
            accent="coral",
        )

    st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)

    search_q = st.text_input(
        "Search conversations by title",
        placeholder="🔍 Search saved conversations...",
        label_visibility="collapsed",
    )
    convs = chat.list_conversations(search_query=search_q)

    if not convs:
        render_empty_state(
            "💬",
            "No Conversations Found",
            "Start a conversation in Medical Assistant or clear your search filter to view saved sessions.",
        )
    else:
        st.subheader("Saved Conversations")
        for c in convs:
            cid = c["conversation_id"]
            created_str = c["created_at"][:19].replace("T", " ")
            updated_str = c["updated_at"][:19].replace("T", " ")
            msg_cnt = c.get("message_count", 0)

            card_col, act_open, act_del = st.columns([3.2, 0.85, 0.75], gap="small")
            with card_col:
                st.markdown(
                    f"""
                    <div class="med-conv-card">
                        <div class="med-doc-title">💬 {html.escape(c['title'])}</div>
                        <div>
                            {status_badge(f"{msg_cnt} messages", "info")}
                            {status_badge(f"Updated {updated_str} UTC", "neutral")}
                        </div>
                        <div class="med-doc-meta">Created: {html.escape(created_str)} UTC &bull; ID: {html.escape(cid)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with act_open:
                st.markdown("<div style='height: 0.9rem'></div>", unsafe_allow_html=True)
                if st.button("Open Chat", key=f"open_{cid}", type="primary", use_container_width=True):
                    st.session_state["active_conversation_id"] = cid
                    st.session_state["nav_page"] = "Medical Assistant"
                    st.rerun()
            with act_del:
                st.markdown("<div style='height: 0.9rem'></div>", unsafe_allow_html=True)
                if st.button("Delete", key=f"del_{cid}", use_container_width=True):
                    chat.delete_conversation(cid)
                    if st.session_state.get("active_conversation_id") == cid:
                        st.session_state.pop("active_conversation_id", None)
                    st.success("Conversation deleted.")
                    st.rerun()

        with st.container(border=True):
            st.subheader("Manage, Rename & Export Selected Conversation")
            conv_options = {
                f"{c['title']} ({c['conversation_id']})": c for c in convs
            }
            chosen_key = st.selectbox("Select Conversation", list(conv_options.keys()))
            chosen = conv_options[chosen_key]
            cid = chosen["conversation_id"]

            col1, col2 = st.columns([2.2, 0.9])
            with col1:
                new_title = st.text_input(
                    "Rename conversation", value=chosen["title"], key=f"ren_{cid}"
                )
            with col2:
                st.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
                if st.button("Save Title", use_container_width=True):
                    if chat.rename_conversation(cid, new_title):
                        st.success("Conversation renamed.")
                        st.rerun()
                    else:
                        st.warning("Title cannot be empty.")

            msgs = chat.get_messages(cid)
            export_payload = json.dumps(
                {"conversation": chosen, "messages": msgs}, indent=2
            )

            a1, a2 = st.columns(2)
            with a1:
                st.download_button(
                    "⬇ Export Conversation JSON",
                    data=export_payload,
                    file_name=f"{cid}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with a2:
                if st.button("Clear Messages Only", use_container_width=True):
                    chat.clear_conversation(cid)
                    st.success("Conversation messages cleared.")
                    st.rerun()

            with st.expander(f"Preview Transcript ({len(msgs)} messages)", expanded=False):
                if not msgs:
                    st.caption("No messages in this conversation.")
                for m in msgs:
                    st.markdown(f"**{m['role'].capitalize()}** *({m['created_at'][:19]} UTC)*")
                    st.write(m["content"])
                    st.divider()

    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("User Feedback Log")
        fb_items = db.list_feedback(limit=50)
        if fb_items:
            fb_rows = [
                {
                    "Feedback ID": item["feedback_id"],
                    "Rating": item["rating"],
                    "Conversation": item.get("conversation_title") or item["conversation_id"],
                    "Response Preview": (item.get("message_preview") or "")[:90],
                    "Timestamp": item["created_at"][:19].replace("T", " "),
                }
                for item in fb_items
            ]
            st.dataframe(pd.DataFrame(fb_rows), use_container_width=True, hide_index=True)

            del_fb_id = st.selectbox(
                "Select feedback record to delete",
                [item["feedback_id"] for item in fb_items],
            )
            if st.button("Delete Selected Feedback"):
                db.delete_feedback(del_fb_id)
                st.success("Feedback record deleted.")
                st.rerun()
        else:
            st.caption("No feedback ratings recorded yet.")

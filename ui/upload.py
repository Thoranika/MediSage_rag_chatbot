"""Knowledge Library & Document Upload page with modern cards, validation, and chunk inspection."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from ui.styles import render_empty_state, render_page_header, status_badge
from utils.helpers import format_file_size


def render_upload_library(services: dict[str, Any]) -> None:
    """Render document upload, validation, status display, chunk inspection, and deletion."""
    db = services["db"]
    chroma = services["chroma"]
    ingestion = services["ingestion"]
    settings = services["settings"]

    render_page_header(
        "Knowledge Library & Document Ingestion",
        "Upload permitted medical reference documents, monitor indexing status, and inspect extracted chunks.",
    )

    st.markdown(
        """
        <div class="med-privacy-warn">
            <strong>Privacy & Authorization Notice:</strong> Upload only permitted educational medical
            references or clinical guidelines. Do <strong>not</strong> upload identifiable patient
            records (PHI) unless you hold formal authorization and safeguards.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Upload Reference Documents")
        st.markdown(
            status_badge("PDF (Text-Searchable)", "info")
            + status_badge("DOCX", "lavender")
            + status_badge("TXT / Markdown", "neutral")
            + status_badge(f"Max {settings.max_upload_size_mb} MB / file", "neutral"),
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Drag and drop medical reference files here",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="medisage_uploader",
        )
        u_col1, u_col2 = st.columns([1.6, 1.0])
        with u_col1:
            replace_dups = st.toggle(
                "Re-index if an identical SHA-256 file hash already exists",
                value=False,
            )
        with u_col2:
            process_clicked = st.button(
                "Upload & Index Documents",
                type="primary",
                use_container_width=True,
                disabled=not bool(uploaded_files),
            )

        if uploaded_files and process_clicked:
            progress_bar = st.progress(0, text="Starting document validation & indexing...")
            total = len(uploaded_files)
            for idx, up in enumerate(uploaded_files, start=1):
                progress_bar.progress(
                    int(((idx - 1) / total) * 100),
                    text=f"Processing {up.name} ({idx}/{total})...",
                )
                raw_bytes = up.getvalue()
                outcome = ingestion.ingest_bytes(
                    filename=up.name,
                    content=raw_bytes,
                    mime_type=getattr(up, "type", None),
                    source_origin="upload",
                    replace_existing=replace_dups,
                )
                if outcome.status == "indexed":
                    st.success(outcome.message)
                elif outcome.status == "duplicate":
                    st.info(outcome.message)
                else:
                    st.error(f"{up.name}: {outcome.message}")
            progress_bar.progress(100, text="Document ingestion complete.")

    st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)
    st.subheader("Indexed Knowledge Base")

    docs = db.list_documents()
    if not docs:
        render_empty_state(
            "📚",
            "Your Knowledge Library is Empty",
            "Upload a medical reference PDF, DOCX, TXT, or Markdown file above, or load the "
            "built-in educational sample library from the Home dashboard.",
        )
        return

    f1, f2, f3 = st.columns([2.0, 1.0, 0.9])
    with f1:
        search_filter = st.text_input(
            "Search documents",
            placeholder="Search by filename, format, or status...",
            label_visibility="collapsed",
        ).strip().lower()
    with f2:
        status_filter = st.selectbox(
            "Filter status",
            ["All Statuses", "indexed", "failed", "processing"],
            label_visibility="collapsed",
        )
    with f3:
        view_mode = st.selectbox(
            "View style",
            ["Card Grid", "Table View"],
            label_visibility="collapsed",
        )

    filtered_docs = [
        d
        for d in docs
        if (
            not search_filter
            or search_filter in d["original_filename"].lower()
            or search_filter in d["processing_status"].lower()
            or search_filter in d["file_type"].lower()
        )
        and (
            status_filter == "All Statuses"
            or d["processing_status"] == status_filter
        )
    ]

    if not filtered_docs:
        render_empty_state(
            "🔍",
            "No Matching Documents Found",
            "Try clearing your search query or switching the status filter above.",
        )
    elif view_mode == "Card Grid":
        cols = st.columns(2, gap="medium")
        for idx, d in enumerate(filtered_docs):
            status_val = d["processing_status"]
            variant = (
                "ok"
                if status_val == "indexed"
                else ("err" if status_val == "failed" else "warn")
            )
            origin_badge = (
                status_badge("Sample", "lavender")
                if d.get("source_origin") == "sample"
                else status_badge("Uploaded", "info")
            )
            err_html = (
                f'<div style="margin-top:0.35rem;font-size:0.79rem;color:#A63F37;">'
                f'Error: {html.escape(d["error_details"])}</div>'
                if d["error_details"]
                else ""
            )
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="med-doc-card">
                        <div class="med-doc-title">📄 {html.escape(d['original_filename'])}</div>
                        <div>
                            {status_badge(status_val.upper(), variant)}
                            {status_badge(d['file_type'].upper(), 'neutral')}
                            {origin_badge}
                        </div>
                        <div class="med-doc-meta">
                            <strong>{d['page_count']}</strong> pages &bull;
                            <strong>{d['chunk_count']}</strong> chunks &bull;
                            <strong>{format_file_size(d['file_size_bytes'])}</strong><br/>
                            Uploaded: {html.escape(d['upload_timestamp'][:19].replace('T', ' '))} UTC
                        </div>
                        {err_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        table_rows = [
            {
                "Document": d["original_filename"],
                "Type": d["file_type"].upper(),
                "Status": d["processing_status"],
                "Pages": d["page_count"],
                "Chunks": d["chunk_count"],
                "Size": format_file_size(d["file_size_bytes"]),
                "Uploaded (UTC)": d["upload_timestamp"][:19].replace("T", " "),
                "Errors": d["error_details"] or "—",
            }
            for d in filtered_docs
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Document Actions & Chunk Inspector")
        doc_labels = {
            f"{d['original_filename']} [{d['processing_status']}] ({d['document_id']})": d
            for d in docs
        }
        selected_label = st.selectbox("Select a document to manage", list(doc_labels.keys()))
        selected_doc = doc_labels[selected_label]
        doc_id = selected_doc["document_id"]

        act1, act2, act3 = st.columns([1, 1, 1.5])
        with act1:
            if st.button("Re-index Selected Document", use_container_width=True):
                with st.spinner("Re-chunking and re-embedding document..."):
                    res = ingestion.reindex_document(doc_id)
                if res.success:
                    st.success(res.message)
                    st.rerun()
                else:
                    st.error(res.message)
        with act2:
            if st.button(
                "Delete Document & Vectors",
                type="secondary",
                use_container_width=True,
            ):
                deleted = ingestion.delete_document_completely(doc_id)
                if deleted:
                    st.success(
                        f"Removed '{selected_doc['original_filename']}', stored file, and ChromaDB vectors."
                    )
                    st.rerun()
                else:
                    st.error("Unable to delete document.")
        with act3:
            st.caption(
                f"SHA-256: `{selected_doc['file_hash'][:16]}...` | "
                f"Origin: `{selected_doc['source_origin']}`"
            )

        if selected_doc["error_details"]:
            st.warning(f"Last parse/index error: {selected_doc['error_details']}")

        if selected_doc["processing_status"] == "indexed":
            chunks = chroma.get_document_chunks(doc_id)
            with st.expander(f"Inspect Extracted Chunks ({len(chunks)} stored)", expanded=False):
                page_nums = sorted({c["page_number"] for c in chunks})
                chosen_page = st.selectbox(
                    "Filter chunks by page",
                    ["All Pages"] + [f"Page {p}" for p in page_nums],
                )
                for ch in chunks:
                    if chosen_page != "All Pages" and f"Page {ch['page_number']}" != chosen_page:
                        continue
                    st.markdown(
                        f"**Chunk #{ch['chunk_index']}** | Page {ch['page_number']} | "
                        f"Section: `{ch['section_title']}` | ~{ch['token_count']} tokens"
                    )
                    st.text(ch["text"])
                    st.divider()

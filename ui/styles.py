"""Design system, custom CSS, and reusable UI primitives for MediSage."""

from __future__ import annotations

import html

import streamlit as st

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --ms-primary: #269C96;
        --ms-primary-hover: #1E827D;
        --ms-primary-soft: #E6F5F4;
        --ms-secondary: #8B82C8;
        --ms-secondary-soft: #F0EFF9;
        --ms-accent: #F28B82;
        --ms-accent-soft: #FDF0EF;
        --ms-bg: #F3F7FA;
        --ms-sidebar: #F0EFF9;
        --ms-card: #FFFFFF;
        --ms-text: #26334A;
        --ms-text-muted: #748196;
        --ms-border: #DCE5ED;
        --ms-success: #78B99A;
        --ms-success-soft: #EDF7F2;
        --ms-warning: #F2C879;
        --ms-warning-soft: #FEF8EC;
        --ms-shadow-sm: 0 2px 8px rgba(38, 51, 74, 0.04);
        --ms-shadow-md: 0 6px 20px rgba(38, 51, 74, 0.06);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--ms-text);
    }

    h1, h2, h3, h4, .med-hero-title, .med-page-title, .med-kpi-value, .med-brand-name {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        color: var(--ms-text);
        letter-spacing: -0.015em;
    }

    .stApp {
        background-color: var(--ms-bg);
        color: var(--ms-text);
    }

    /* Main Layout Container */
    .main .block-container {
        max-width: 1240px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        overflow-x: hidden;
    }

    /* Top Streamlit Toolbar subtle tint */
    header[data-testid="stHeader"] {
        background: rgba(243, 247, 250, 0.85);
        backdrop-filter: blur(8px);
    }

    /* ---------------- Sidebar Redesign ---------------- */
    [data-testid="stSidebar"] {
        background-color: var(--ms-sidebar);
        border-right: 1px solid var(--ms-border);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    .med-brand-box {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFC 100%);
        border: 1px solid var(--ms-border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: var(--ms-shadow-sm);
    }
    .med-brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        background: linear-gradient(135deg, #269C96 0%, #8B82C8 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
        box-shadow: 0 3px 10px rgba(38, 156, 150, 0.22);
    }
    .med-brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--ms-text);
        line-height: 1.15;
    }
    .med-brand-sub {
        font-size: 0.74rem;
        color: var(--ms-text-muted);
        margin-top: 0.15rem;
        line-height: 1.25;
    }
    .med-nav-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ms-text-muted);
        margin: 0.25rem 0 0.45rem 0.25rem;
    }

    /* Style sidebar radio as modern navigation pills */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 11px;
        padding: 0.55rem 0.8rem;
        margin: 0;
        transition: all 0.18s ease;
        cursor: pointer;
        width: 100%;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 0.91rem;
        font-weight: 500;
        color: var(--ms-text);
        margin: 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.85);
        border-color: var(--ms-border);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #269C96 0%, #2C8F96 100%);
        border-color: #218A85;
        box-shadow: 0 3px 10px rgba(38, 156, 150, 0.22);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    .med-sidebar-stats {
        background: #FFFFFF;
        border: 1px solid var(--ms-border);
        border-radius: 13px;
        padding: 0.85rem 0.95rem;
        margin-top: 1rem;
        box-shadow: var(--ms-shadow-sm);
    }
    .med-sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.32rem 0;
        font-size: 0.84rem;
        border-bottom: 1px solid #F1F5F8;
    }
    .med-sidebar-stat-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .med-sidebar-stat-label {
        color: var(--ms-text-muted);
    }
    .med-sidebar-stat-val {
        font-weight: 700;
        color: var(--ms-text);
        background: var(--ms-bg);
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        font-size: 0.78rem;
        border: 1px solid var(--ms-border);
    }
    .med-sidebar-disclaimer {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--ms-border);
        border-left: 3px solid var(--ms-secondary);
        border-radius: 10px;
        padding: 0.68rem 0.8rem;
        margin-top: 0.9rem;
        font-size: 0.75rem;
        color: var(--ms-text-muted);
        line-height: 1.45;
    }

    /* ---------------- Buttons & Inputs ---------------- */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        border-radius: 11px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.48rem 1rem !important;
        transition: all 0.18s ease !important;
        border: 1px solid var(--ms-border) !important;
    }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color: var(--ms-primary) !important;
        color: #FFFFFF !important;
        border-color: var(--ms-primary) !important;
        box-shadow: 0 2px 8px rgba(38, 156, 150, 0.20) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: var(--ms-primary-hover) !important;
        border-color: var(--ms-primary-hover) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"],
    .stDownloadButton > button,
    button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: var(--ms-text) !important;
        border: 1px solid var(--ms-border) !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background-color: var(--ms-primary-soft) !important;
        border-color: var(--ms-primary) !important;
        color: var(--ms-primary-hover) !important;
    }

    /* Inputs & Selectboxes */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        background-color: #FFFFFF !important;
        border-color: var(--ms-border) !important;
        border-radius: 10px !important;
        color: var(--ms-text) !important;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--ms-primary) !important;
        box-shadow: 0 0 0 2px rgba(38, 156, 150, 0.14) !important;
    }

    /* Containers with border=True */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--ms-card);
        border: 1px solid var(--ms-border) !important;
        border-radius: 14px !important;
        box-shadow: var(--ms-shadow-sm);
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, #FAFCFE 0%, #F3F8FA 100%) !important;
        border: 1.5px dashed #9AC7C5 !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--ms-primary) !important;
        background: var(--ms-primary-soft) !important;
    }

    /* ---------------- Page & Hero Headers ---------------- */
    .med-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #EDF7F6 58%, #F0EFF9 100%);
        border: 1px solid var(--ms-border);
        border-radius: 16px;
        padding: 1.55rem 1.8rem;
        margin-bottom: 1.1rem;
        box-shadow: var(--ms-shadow-md);
    }
    .med-hero-pill {
        display: inline-block;
        background: var(--ms-primary-soft);
        color: var(--ms-primary-hover);
        border: 1px solid #BEE4E2;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.55rem;
    }
    .med-hero-title {
        font-size: 1.72rem;
        font-weight: 700;
        color: var(--ms-text);
        margin: 0 0 0.4rem 0;
        line-height: 1.25;
    }
    .med-hero-sub {
        font-size: 0.96rem;
        color: var(--ms-text-muted);
        margin: 0;
        line-height: 1.55;
        max-width: 820px;
    }

    .med-page-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F9FC 65%, #F0EFF9 100%);
        border: 1px solid var(--ms-border);
        border-left: 4px solid var(--ms-primary);
        border-radius: 14px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: var(--ms-shadow-sm);
    }
    .med-page-title {
        font-size: 1.38rem;
        font-weight: 700;
        color: var(--ms-text);
        margin: 0 0 0.2rem 0;
    }
    .med-page-sub {
        font-size: 0.89rem;
        color: var(--ms-text-muted);
        margin: 0;
        line-height: 1.45;
    }

    /* ---------------- Cards & KPIs ---------------- */
    .med-card {
        background: var(--ms-card);
        border: 1px solid var(--ms-border);
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 0.95rem;
        box-shadow: var(--ms-shadow-sm);
    }
    .med-card-title {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-size: 1.03rem;
        font-weight: 700;
        color: var(--ms-text);
        margin: 0 0 0.5rem 0;
    }

    .med-kpi-card {
        background: var(--ms-card);
        border: 1px solid var(--ms-border);
        border-top: 3px solid var(--ms-primary);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: var(--ms-shadow-sm);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        height: 100%;
    }
    .med-kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--ms-shadow-md);
    }
    .kpi-lavender { border-top-color: var(--ms-secondary); }
    .kpi-coral { border-top-color: var(--ms-accent); }
    .kpi-green { border-top-color: var(--ms-success); }
    .kpi-amber { border-top-color: var(--ms-warning); }

    .med-kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.35rem;
    }
    .med-kpi-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ms-text-muted);
    }
    .med-kpi-icon {
        font-size: 1rem;
        padding: 0.25rem 0.45rem;
        border-radius: 8px;
        background: var(--ms-bg);
    }
    .med-kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--ms-text);
        line-height: 1.2;
    }
    .med-kpi-sub {
        font-size: 0.8rem;
        color: var(--ms-text-muted);
        margin-top: 0.25rem;
    }

    /* Quick Access Feature Cards */
    .med-feature-card {
        background: var(--ms-card);
        border: 1px solid var(--ms-border);
        border-radius: 14px;
        padding: 1.05rem 1.15rem;
        box-shadow: var(--ms-shadow-sm);
        margin-bottom: 0.55rem;
    }
    .med-feature-head {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: var(--ms-text);
        margin-bottom: 0.3rem;
    }
    .med-feature-desc {
        font-size: 0.85rem;
        color: var(--ms-text-muted);
        line-height: 1.45;
        margin: 0;
    }

    /* Document & Conversation Cards */
    .med-doc-card, .med-conv-card {
        background: var(--ms-card);
        border: 1px solid var(--ms-border);
        border-radius: 13px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.65rem;
        box-shadow: var(--ms-shadow-sm);
        transition: border-color 0.16s ease, box-shadow 0.16s ease;
    }
    .med-doc-card:hover, .med-conv-card:hover {
        border-color: #B4CED9;
        box-shadow: var(--ms-shadow-md);
    }
    .med-doc-title {
        font-weight: 600;
        font-size: 0.94rem;
        color: var(--ms-text);
        margin-bottom: 0.3rem;
        word-break: break-word;
    }
    .med-doc-meta {
        font-size: 0.8rem;
        color: var(--ms-text-muted);
        margin-top: 0.35rem;
    }

    /* Empty State */
    .med-empty-state {
        background: #FFFFFF;
        border: 1.5px dashed var(--ms-border);
        border-radius: 15px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 0.75rem 0 1rem 0;
    }
    .med-empty-icon {
        font-size: 2rem;
        margin-bottom: 0.45rem;
    }
    .med-empty-title {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.04rem;
        color: var(--ms-text);
        margin-bottom: 0.3rem;
    }
    .med-empty-desc {
        font-size: 0.88rem;
        color: var(--ms-text-muted);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Notices & Disclaimers */
    .med-disclaimer {
        background: #FFFFFF;
        border: 1px solid var(--ms-border);
        border-left: 4px solid var(--ms-secondary);
        border-radius: 11px;
        padding: 0.72rem 1rem;
        font-size: 0.84rem;
        color: var(--ms-text-muted);
        margin-bottom: 0.95rem;
        line-height: 1.48;
    }
    .med-disclaimer strong {
        color: var(--ms-text);
    }
    .med-privacy-warn {
        background: var(--ms-warning-soft);
        border: 1px solid #EFE0B8;
        border-left: 4px solid var(--ms-warning);
        border-radius: 11px;
        padding: 0.75rem 1rem;
        font-size: 0.86rem;
        color: #6B4E16;
        margin-bottom: 1rem;
        line-height: 1.48;
    }

    /* Badges */
    .med-badge {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.35;
        margin-right: 0.35rem;
        margin-bottom: 0.2rem;
        white-space: nowrap;
    }
    .badge-ok { background: var(--ms-success-soft); color: #2B6D4E; border: 1px solid #BFE3D0; }
    .badge-info { background: var(--ms-primary-soft); color: #1A6E69; border: 1px solid #B8E1DF; }
    .badge-lavender { background: var(--ms-secondary-soft); color: #5B529E; border: 1px solid #D2CEEC; }
    .badge-warn { background: var(--ms-warning-soft); color: #7C5A16; border: 1px solid #F3DCAA; }
    .badge-err { background: var(--ms-accent-soft); color: #A63F37; border: 1px solid #F6C5C0; }
    .badge-neutral { background: #F1F5F8; color: var(--ms-text-muted); border: 1px solid var(--ms-border); }

    /* ---------------- Medical Assistant Chat Experience ---------------- */
    [data-testid="stChatMessage"] {
        border-radius: 15px !important;
        padding: 0.95rem 1.1rem !important;
        margin-bottom: 0.85rem !important;
        border: 1px solid var(--ms-border);
        box-shadow: var(--ms-shadow-sm);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #EAF6F5 0%, #F4FAFA 100%) !important;
        border-color: #C4E5E3 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #FFFFFF !important;
        border-color: var(--ms-border) !important;
        border-left: 4px solid var(--ms-secondary) !important;
    }

    /* Chat Input */
    [data-testid="stChatInput"] > div {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD8E4 !important;
        border-radius: 16px !important;
        box-shadow: 0 3px 14px rgba(38, 51, 74, 0.06) !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--ms-primary) !important;
        box-shadow: 0 4px 16px rgba(38, 156, 150, 0.14) !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background-color: var(--ms-primary) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    /* Citation Cards inside Chat */
    .med-citation {
        background: #F8FBFD;
        border: 1px solid var(--ms-border);
        border-left: 3px solid var(--ms-primary);
        border-radius: 10px;
        padding: 0.72rem 0.95rem;
        margin-bottom: 0.55rem;
        font-size: 0.86rem;
    }
    .med-citation-head {
        font-weight: 600;
        color: var(--ms-primary-hover);
        margin-bottom: 0.28rem;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .med-citation-body {
        color: var(--ms-text);
        line-height: 1.5;
        word-break: break-word;
    }

    .med-msg-meta-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.45rem;
        margin-bottom: 0.4rem;
    }
    .med-msg-time {
        font-size: 0.75rem;
        color: var(--ms-text-muted);
        margin-left: auto;
    }

    /* Expanders */
    details[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid var(--ms-border) !important;
        border-radius: 12px !important;
    }

    /* Responsive Design (Tablet & Mobile) */
    @media (max-width: 1024px) {
        .main .block-container {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
        .med-kpi-value { font-size: 1.38rem; }
        .med-hero-title { font-size: 1.48rem; }
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        .med-hero {
            padding: 1.15rem 1.15rem;
        }
        .med-hero-title { font-size: 1.3rem; }
        .med-page-header { padding: 0.9rem 1rem; }
        .med-msg-time { margin-left: 0; width: 100%; }
    }
</style>
"""


def inject_custom_css() -> None:
    """Inject MediSage custom soft-teal and lavender design system CSS."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def status_badge(label: str, variant: str = "neutral") -> str:
    """Render an inline HTML status pill."""
    cls = {
        "ok": "badge-ok",
        "info": "badge-info",
        "lavender": "badge-lavender",
        "warn": "badge-warn",
        "err": "badge-err",
    }.get(variant, "badge-neutral")
    return f'<span class="med-badge {cls}">{html.escape(label)}</span>'


def render_page_header(title: str, subtitle: str) -> None:
    """Render a consistent styled page header card."""
    st.markdown(
        f"""
        <div class="med-page-header">
            <div class="med-page-title">{html.escape(title)}</div>
            <p class="med-page-sub">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str | int,
    subtitle: str,
    *,
    icon: str = "📌",
    accent: str = "teal",
) -> None:
    """Render an individual KPI summary card."""
    accent_cls = {
        "teal": "",
        "lavender": "kpi-lavender",
        "coral": "kpi-coral",
        "green": "kpi-green",
        "amber": "kpi-amber",
    }.get(accent, "")
    st.markdown(
        f"""
        <div class="med-kpi-card {accent_cls}">
            <div class="med-kpi-top">
                <span class="med-kpi-label">{html.escape(label)}</span>
                <span class="med-kpi-icon">{icon}</span>
            </div>
            <div class="med-kpi-value">{html.escape(str(value))}</div>
            <div class="med-kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, title: str, description: str) -> None:
    """Render a centered, friendly empty-state panel."""
    st.markdown(
        f"""
        <div class="med-empty-state">
            <div class="med-empty-icon">{icon}</div>
            <div class="med-empty-title">{html.escape(title)}</div>
            <p class="med-empty-desc">{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

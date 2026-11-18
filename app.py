"""MediSage — Evidence-Grounded Medical Knowledge Assistant (Streamlit Entry Point)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.chat_manager import ChatManager
from config.settings import ensure_directories, get_settings
from core.llm_client import GeminiLLMClient
from core.logging_utils import configure_logging
from database.db import DatabaseManager
from ingestion.embeddings import EmbeddingService
from ingestion.ingestion_service import IngestionService
from rag.llm_service import LLMService
from rag.rag_pipeline import RAGPipeline
from rag.retriever import SemanticRetriever
from ui.chat import render_chat
from ui.dashboard import render_dashboard
from ui.evaluation import render_evaluation
from ui.history import render_history
from ui.settings_page import render_settings
from ui.styles import inject_custom_css
from ui.upload import render_upload_library
from vectorstore.chroma_manager import ChromaManager

st.set_page_config(
    page_title="MediSage | Evidence-Grounded Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing MediSage local services...")
def init_services() -> dict:
    """Initialize and cache singleton application services across reruns."""
    settings = ensure_directories(get_settings())
    configure_logging(log_level=settings.log_level, log_dir=settings.log_dir)

    db = DatabaseManager(settings.sqlite_db_path)
    chroma = ChromaManager(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    embeddings = EmbeddingService(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )
    ingestion = IngestionService(
        db=db, chroma=chroma, embeddings=embeddings, settings=settings
    )
    ingestion.verify_and_repair_consistency()
    retriever = SemanticRetriever(chroma=chroma, embeddings=embeddings)
    llm_service = LLMService(client=GeminiLLMClient(), settings=settings)
    rag_pipeline = RAGPipeline(
        retriever=retriever, llm_service=llm_service, db=db, settings=settings
    )
    chat_manager = ChatManager(
        db=db,
        rag_pipeline=rag_pipeline,
        max_history_messages=settings.max_history_turns * 2,
    )
    return {
        "settings": settings,
        "db": db,
        "chroma": chroma,
        "embeddings": embeddings,
        "ingestion": ingestion,
        "retriever": retriever,
        "llm_service": llm_service,
        "rag_pipeline": rag_pipeline,
        "chat": chat_manager,
    }


PAGES = [
    "Home",
    "Knowledge Library",
    "Medical Assistant",
    "Conversation History",
    "Evaluation & Tuning",
    "Settings & Security",
]

NAV_LABELS = {
    "Home": "🏠  Home",
    "Knowledge Library": "📚  Knowledge Library",
    "Medical Assistant": "🩺  Medical Assistant",
    "Conversation History": "🕒  Conversation History",
    "Evaluation & Tuning": "📊  Evaluation & Tuning",
    "Settings & Security": "⚙️  Settings & Security",
}


def main() -> None:
    inject_custom_css()
    services = init_services()
    db = services["db"]

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Home"

    with st.sidebar:
        st.markdown(
            """
            <div class="med-brand-box">
                <div class="med-brand-icon">🩺</div>
                <div>
                    <div class="med-brand-name">MediSage</div>
                    <div class="med-brand-sub">Evidence-Grounded Medical Reference Assistant</div>
                </div>
            </div>
            <div class="med-nav-label">Workspace Navigation</div>
            """,
            unsafe_allow_html=True,
        )

        current_page = (
            st.session_state["nav_page"]
            if st.session_state["nav_page"] in PAGES
            else "Home"
        )
        selected = st.radio(
            "Navigation",
            PAGES,
            index=PAGES.index(current_page),
            format_func=lambda p: NAV_LABELS.get(p, p),
            label_visibility="collapsed",
        )
        if selected != st.session_state["nav_page"]:
            st.session_state["nav_page"] = selected
            st.rerun()

        stats = db.get_knowledge_stats()
        st.markdown(
            f"""
            <div class="med-sidebar-stats">
                <div class="med-nav-label" style="margin-top:0;">Knowledge Base Summary</div>
                <div class="med-sidebar-stat-row">
                    <span class="med-sidebar-stat-label">Indexed Documents</span>
                    <span class="med-sidebar-stat-val">{stats['indexed_documents']}</span>
                </div>
                <div class="med-sidebar-stat-row">
                    <span class="med-sidebar-stat-label">Vector Chunks</span>
                    <span class="med-sidebar-stat-val">{stats['total_chunks']}</span>
                </div>
                <div class="med-sidebar-stat-row">
                    <span class="med-sidebar-stat-label">Conversations</span>
                    <span class="med-sidebar-stat-val">{stats['conversation_count']}</span>
                </div>
            </div>
            <div class="med-sidebar-disclaimer">
                <strong>Educational Reference Only:</strong> Not for clinical diagnosis,
                prescriptions, or personalized treatment decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    page = st.session_state["nav_page"]
    if page == "Home":
        render_dashboard(services)
    elif page == "Knowledge Library":
        render_upload_library(services)
    elif page == "Medical Assistant":
        render_chat(services)
    elif page == "Conversation History":
        render_history(services)
    elif page == "Evaluation & Tuning":
        render_evaluation(services)
    elif page == "Settings & Security":
        render_settings(services)


if __name__ == "__main__":
    main()

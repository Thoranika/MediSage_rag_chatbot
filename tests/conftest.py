"""Shared pytest fixtures for MediSage tests (no live Gemini API calls)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf as fitz
import pytest
from docx import Document as DocxDocument

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from core.llm_client import BaseLLMClient, LLMResponse
from database.db import DatabaseManager
from ingestion.embeddings import EmbeddingService
from ingestion.ingestion_service import IngestionService
from utils.cache import get_shared_cache
from vectorstore.chroma_manager import ChromaManager


@pytest.fixture(scope="session")
def shared_embedding_service() -> EmbeddingService:
    """Cache all-MiniLM-L6-v2 CPU model once per pytest session."""
    svc = EmbeddingService(model_name="all-MiniLM-L6-v2", device="cpu")
    svc.ensure_loaded()
    return svc


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Create isolated SQLite, ChromaDB, and upload directories per test."""
    get_shared_cache().clear_all()
    cfg = Settings(
        llm_provider="GEMINI",
        llm_model_id="gemini-2.5-flash",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_device="cpu",
        chroma_persist_dir=tmp_path / "chroma_db",
        chroma_collection_name="test_medisage_chunks",
        sqlite_db_path=tmp_path / "test_medisage.db",
        upload_dir=tmp_path / "uploads",
        samples_dir=tmp_path / "samples",
        log_dir=tmp_path / "logs",
        max_upload_size_mb=5,
        chunk_size_tokens=300,
        chunk_overlap_tokens=50,
        retrieval_top_k=4,
        similarity_threshold=0.20,
    )
    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    cfg.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    db = DatabaseManager(cfg.sqlite_db_path)
    chroma = ChromaManager(cfg.chroma_persist_dir, cfg.chroma_collection_name)
    return cfg, db, chroma


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a valid multi-page medical PDF with repeated header and clinical content."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 40), "Clinical Header Standard", fontsize=9)
    p1.insert_text((50, 80), "Hypertension Reference", fontsize=14)
    p1.insert_text(
        (50, 110),
        "Stage 1 hypertension is defined as systolic blood pressure 130-139 mmHg "
        "or diastolic 80-89 mmHg. Stage 2 hypertension is >= 140/90 mmHg.",
        fontsize=11,
    )
    p1.insert_text((50, 760), "Page 1 of 3", fontsize=9)

    p2 = doc.new_page()
    p2.insert_text((50, 40), "Clinical Header Standard", fontsize=9)
    p2.insert_text((50, 80), "Pregnancy Contraindications", fontsize=14)
    p2.insert_text(
        (50, 110),
        "ACE inhibitors such as lisinopril and ARBs such as losartan are strictly "
        "contraindicated during pregnancy due to fetal renal injury risk.",
        fontsize=11,
    )
    p2.insert_text((50, 760), "Page 2 of 3", fontsize=9)

    p3 = doc.new_page()
    p3.insert_text((50, 40), "Clinical Header Standard", fontsize=9)
    p3.insert_text((50, 80), "Lifestyle Therapy", fontsize=14)
    p3.insert_text(
        (50, 110),
        "Reducing dietary sodium to less than 2 grams per day lowers blood pressure.",
        fontsize=11,
    )
    p3.insert_text((50, 760), "Page 3 of 3", fontsize=9)

    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sample_docx_bytes() -> bytes:
    """Generate a valid DOCX containing headings, paragraphs, and a clinical table."""
    doc = DocxDocument()
    doc.add_heading("Pneumonia Severity Assessment", level=1)
    doc.add_paragraph(
        "The CURB-65 score evaluates five bedside criteria for community-acquired pneumonia."
    )
    tbl = doc.add_table(rows=3, cols=2)
    tbl.rows[0].cells[0].text = "Criterion"
    tbl.rows[0].cells[1].text = "Threshold"
    tbl.rows[1].cells[0].text = "Confusion"
    tbl.rows[1].cells[1].text = "New mental confusion (1 point)"
    tbl.rows[2].cells[0].text = "Respiratory Rate"
    tbl.rows[2].cells[1].text = ">= 30 breaths per minute (1 point)"
    doc.add_paragraph("Scores of 3 to 5 indicate severe pneumonia requiring hospitalization.")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """Create a 1-page PDF with vector drawings/images and zero extractable text."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    data = doc.tobytes()
    doc.close()
    return data


class MockGroundedLLM(BaseLLMClient):
    """Mock LLM returning caller-defined text without making external API calls."""

    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.calls: list[str] = []

    def generate(self, prompt: str, settings=None) -> LLMResponse:
        self.calls.append(prompt)
        return LLMResponse(
            success=True,
            text=self.reply_text,
            provider="GEMINI",
            model_id="mock-gemini-2.5-flash",
            latency_ms=1.5,
        )

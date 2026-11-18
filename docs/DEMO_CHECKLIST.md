# MediSage Demonstration Checklist & Presentation Outline

## Pre-Demo Verification Checklist

1. **Environment & Dependencies**
   - [ ] Python 3.10+ environment active (`python --version`).
   - [ ] All packages installed (`pip install -r requirements.txt`).
   - [ ] `.env` created from `.env.example` with a valid `GEMINI_API_KEY` (or intentionally omitted to demonstrate offline CPU retrieval + explicit API key status handling).
   - [ ] Local `.env`, `data/uploads/`, `data/chroma_db/`, and `*.db` confirmed ignored by `.gitignore`.
2. **Pre-Run Automated Verification**
   - [ ] `python -m pytest -v` passes all unit & integration tests cleanly.
   - [ ] `python -m evaluation.evaluator` executes in one command and writes `evaluation/eval_results.json`.
   - [ ] `python -m evaluation.tuning` executes and updates `evaluation/tuning_results.json`.
3. **Application Launch**
   - [ ] Run `python -m streamlit run app.py` from `c:\Users\THORANIKA\OneDrive\Desktop\RAG_MEDISAGE`.
   - [ ] Confirm Home Dashboard loads cleanly with zero horizontal overflow and shows green status for SQLite, ChromaDB, and `all-MiniLM-L6-v2` (CPU).

---

## Live Demo Script (5–7 Minutes)

1. **Home Dashboard (`Home`)**
   - Explain project objective: evidence-grounded educational medical knowledge assistant running on an ordinary CPU laptop with cloud LLM generation.
   - Highlight the prominent **Educational Only / Not for Diagnosis** notice and System Health panel.
   - Click **Load Sample Medical Library** (or verify pre-loaded sample docs) to demonstrate PDF + DOCX ingestion.
2. **Knowledge Library & Validation (`Knowledge Library`)**
   - Show indexed documents with page counts, chunk counts, SHA-256 hashes, and upload timestamps.
   - Open **Inspect Extracted Chunks** to verify page numbers, section headings, and preserved clinical tables (e.g., BP Categories Table or CURB-65 Criteria).
   - Optionally demonstrate duplicate detection or rejection of an invalid/scanned file.
3. **Medical Assistant (`Medical Assistant`)**
   - Ask a grounded clinical reference question (e.g., *"What systolic and diastolic blood pressure ranges define Stage 1 and Stage 2 hypertension?"*).
   - Expand **View Verified Sources & Excerpts** and **Developer Retrieval Inspector** to prove exact `[Source: filename, page N]` grounding.
   - Ask a contextual follow-up (e.g., *"What lifestyle changes lower it?"*) to show standalone query rewriting without altering the user's displayed question.
   - Demonstrate safety guardrails:
     - Out-of-KB question (*"pediatric chemotherapy protocol for ALL"*) -> honest abstention.
     - Diagnosis/dosing question (*"Diagnose me and tell me what lisinopril dose to take"*) -> explicit refusal + clinician referral.
     - Emergency query (*"crushing chest pain and difficulty breathing"*) -> immediate emergency escalation.
   - Click **Helpful** feedback button to verify SQLite persistence.
4. **Conversation History & Governance (`Conversation History` & `Settings & Security`)**
   - Show saved conversation persistence, search, rename, JSON export, and feedback summary.
   - Show **Settings & Security**: configurable Gemini model ID, `top_k`, `similarity_threshold`, `response_style`, API connection check, and separated advanced chunking/cache/purge controls.
5. **RAG Evaluation & Tuning (`Evaluation & Tuning`)**
   - Present the 16-question benchmark metrics (`Recall@K`, `Source Hit Rate`, `MRR`, `Context Relevance`, `Citation Correctness`, `OOKB Abstention`, `Mean Retrieval Latency`) and the Before/After Retrieval Tuning comparison chart.

---

## Presentation Slide Outline

1. **Problem Statement & Motivation** — Why clinical reference Q&A needs strict grounding, page citations, and safety boundaries rather than ungrounded LLM chat.
2. **System Architecture & Stack** — CPU-friendly local ingestion (`PyMuPDF`, `python-docx`, `all-MiniLM-L6-v2`, `ChromaDB`, `SQLite`) + cloud `google-genai` Gemini generation + `Streamlit`.
3. **Milestone 1: Ingestion & Storage** — Page/table extraction, scanned PDF detection, heading-aware chunking, SHA-256 deduplication, and synchronized SQLite/Chroma deletion.
4. **Milestone 2: RAG Pipeline & Medical Safety** — Query rewriting, deduplicated semantic retrieval, strict citation validation, emergency triage, and multi-turn persistence.
5. **Milestone 3: Evaluation, Tuning & Security** — 16-question curated dataset, before/after chunk & threshold tuning gains, path-traversal/log-redaction controls, and pytest suite.
6. **Live Demo & Future Enhancements** — Live walkthrough + future roadmap (hybrid BM25 + dense fusion, optional vetted OCR pipeline, multi-user auth).

# MediSage — Evidence-Grounded Medical Knowledge Assistant

**MediSage** is a lightweight, CPU-friendly Retrieval-Augmented Generation (RAG) workspace built for querying permitted educational medical reference documents (PDF, DOCX, TXT, Markdown) with verifiable page-level source citations, persistent conversation history, medical safety boundaries, and repeatable RAG evaluation.

> **Educational Reference Notice:** MediSage is strictly an educational document-reference assistant. It is **not** a diagnostic tool, does not prescribe medications or personalized dosages, and is **not** a substitute for a qualified healthcare professional.

---

## Key Features Across Milestones 1, 2 & 3

- **Knowledge Ingestion (Milestone 1):**
  - Drag-and-drop multi-file upload supporting **PDF** (`PyMuPDF`), **DOCX** (`python-docx` preserving reading order & tables), **TXT**, and **Markdown**.
  - Strict security validation: extension whitelist, MIME & container magic-byte checks (`%PDF-`, `PK\x03\x04`), file size cap (`20 MB`), traversal-safe UUID filenames, and honest rejection of encrypted, corrupt, empty, and scanned/image-only PDFs (no fabricated OCR).
  - Heading- and paragraph-aware chunking (`600` tokens default, `100` overlap) with full metadata (`document_id`, `filename`, `page_number`, `section_title`, `chunk_index`, `file_hash`, `ingestion_timestamp`).
  - Local CPU embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) persisted in **ChromaDB** (`cosine` space) and tracked in **SQLite** with SHA-256 duplicate prevention and synchronized deletion.
- **Grounded RAG & Safety (Milestone 2):**
  - Cloud LLM adapter (`core/llm_client.py`) using the official `google-genai` SDK (default `gemini-2.5-flash`, provider/model configurable via `.env` and Settings UI) with bounded retries and clean handling for missing keys, timeouts, rate limits, and malformed responses.
  - Ranked, deduplicated semantic retrieval (`rag/retriever.py`, `retrieval.py`) with `top_k`, similarity threshold (`0.32`), metadata filtering, document diversity, and explicit `no_evidence` / `low_confidence` handling.
  - Standalone follow-up query rewriting using bounded recent turns without modifying the user's displayed question.
  - Inline `[Source: filename, page N]` citation parsing and validation (`rag/citations.py`) stripping any ungrounded references.
  - Medical safety gate (`rag/safety.py`) for immediate emergency triage escalation and diagnosis/dosage/cessation boundaries, plus persistent **Helpful / Not Helpful** feedback controls.
- **Evaluation, Security & Polish (Milestone 3):**
  - 16-question curated medical benchmark (`evaluation/eval_dataset.json`) covering direct, paraphrased, table/multi-section, follow-up, safety, and out-of-knowledge-base queries—executed in an isolated temporary vector index separate from user conversations.
  - Evidence-based retrieval tuning (`evaluation/tuning.py`) comparing chunk sizes, thresholds, deduplication, and query rewriting.
  - Secret-redacted logging (`core/logging_utils.py`), parameterized SQL, path-traversal defense, index-versioned query/embedding caching (`utils/cache.py`), 21 automated `pytest` tests, and a polished light-mode 6-view Streamlit interface.

---

## Project Architecture

```text
RAG_MEDISAGE/
├── app.py                          # Main Streamlit application & sidebar navigation
├── retrieval.py                    # Top-level retrieval interface alias
├── requirements.txt                # Tested Python dependencies
├── .env.example                    # Safe environment variable template
├── .gitignore                      # Excludes .env, uploads, chroma_db, *.db, logs, caches
├── .streamlit/config.toml          # Light medical UI theme
├── config/ & core/                 # Centralized Settings, secret-redacted logging, Gemini LLM client
├── database/ & storage/            # SQLite schema.sql & parameterized DatabaseManager CRUD
├── ingestion/                      # Loaders (PDF/DOCX/TXT), cleaner, chunker, CPU embeddings, ingestion service
├── vectorstore/                    # Persistent ChromaDB manager (cosine index + cleanup)
├── rag/                            # Safety gate, query rewriter, retriever, prompt builder, citations, pipeline
├── chatbot/                        # Multi-turn conversation persistence & bounded context manager
├── evaluation/                     # 16-case eval_dataset.json, metrics, evaluator CLI, tuning experiments
├── ui/                             # 6 modular Streamlit views + custom CSS
├── utils/                          # File/settings validators, SHA-256/token/error helpers, RAG cache
├── scripts/                        # Sample medical guideline generator, local bootstrapper, CPU benchmark
├── docs/                           # Architecture, Security, Evaluation & Tuning, Test Report, Demo Checklist
├── tests/                          # 21 automated pytest tests (mocked LLM)
└── data/
    ├── samples/                    # Redistributable non-sensitive educational medical guidelines
    ├── uploads/                    # Safe UUID-stored uploaded files (gitignored)
    └── chroma_db/                  # Persistent ChromaDB storage (gitignored)
```

For full architecture diagrams, SQLite schema, ChromaDB metadata, and RAG data-flow details, see:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/EVALUATION_AND_TUNING.md](docs/EVALUATION_AND_TUNING.md)
- [docs/SECURITY.md](docs/SECURITY.md)
- [docs/TEST_REPORT.md](docs/TEST_REPORT.md)
- [docs/DEMO_CHECKLIST.md](docs/DEMO_CHECKLIST.md)

---

## Local Setup & Execution (Windows PowerShell)

### 1. Prerequisites
- Windows 10/11 with **Python 3.10+** (verified on Python 3.13.1, CPU-only—no GPU required).

### 2. Create Virtual Environment & Install Dependencies
```powershell
cd "c:\Users\THORANIKA\OneDrive\Desktop\RAG_MEDISAGE"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure Gemini API Key (`.env`)
Copy `.env.example` to `.env` and set your Google Gemini API key:
```powershell
Copy-Item .env.example .env
notepad .env
```
Edit `.env`:
```dotenv
LLM_PROVIDER=GEMINI
LLM_MODEL_ID=gemini-2.5-flash
GEMINI_API_VERSION=v1beta
GEMINI_API_KEY=your_actual_gemini_api_key
```
> **Note:** `GEMINI_API_VERSION=v1beta` is required so the `google-genai` SDK sends `system_instruction` (developer instructions) cleanly to Gemini 2.5 Flash. Never commit `.env` to Git. If `GEMINI_API_KEY` is left unset, local CPU ingestion, ChromaDB semantic retrieval, source inspection, and evaluation benchmarks still run cleanly, while chat responses transparently report `missing_api_key` alongside retrieved passages.

### Six Application Pages
1. **Home** — Project overview, medical educational notice, live document/vector/conversation metrics, System Health status, quick-navigation buttons, and 1-click sample library loader.
2. **Knowledge Library** — Multi-file upload (PDF, DOCX, TXT, Markdown), strict format/magic-byte validation, SHA-256 deduplication, chunk inspection, re-indexing, and synchronized deletion.
3. **Medical Assistant** — Evidence-grounded conversational RAG chatbot with Gemini `v1beta` synthesis, document/all-docs filter, follow-up query rewriting, timestamps, expandable source citations, developer retrieval inspector, and feedback controls.
4. **Conversation History** — Persistent SQLite conversation search, rename, reopen, transcript preview, JSON export, clear/delete actions, and feedback log management.
5. **Evaluation & Tuning** — Isolated 16-case benchmark (`Recall@K`, `Source Hit`, `MRR`, `Context Relevance`, `Citation Correctness`, `OOKB Abstention`, `Latency`) and before/after chunking, Top-K, and threshold comparisons.
6. **Settings & Security** — API configuration status (masked key), embedding & LLM model indicators, ChromaDB health, validated tuning sliders, live Gemini connection test, cache clear, and purge controls.

### 4. (Optional) Pre-Load Sample Medical Reference Library
```powershell
python scripts/bootstrap_local_store.py
```
You can also click **Load Sample Medical Library** directly from the **Home** page inside the Streamlit app.

### 5. Launch the Streamlit Application
```powershell
streamlit run app.py
```
*(If Python's user `Scripts` folder is not on your `PATH`, use `python -m streamlit run app.py`.)*

---

## Running Evaluation, Tuning & Automated Tests

```powershell
# Run all 23 automated unit and integration tests (mocked LLM; 0 API credits used)
python -m pytest -v

# Run the 16-question RAG evaluation benchmark (isolated index)
python -m evaluation.evaluator

# Run baseline vs. tuned chunking/threshold experiments
python -m evaluation.tuning

# Measure CPU embedding cold-start, retrieval, and pipeline latency
python scripts/benchmark_performance.py
```

---

## Troubleshooting

- **`streamlit` or `pytest` not recognized in PowerShell:** Run `python -m streamlit run app.py` or `python -m pytest -v` (or activate `.venv\Scripts\Activate.ps1`).
- **`missing_api_key` or `auth_error` in Chat:** Ensure `.env` exists in the project root with a valid `GEMINI_API_KEY` (without quotes or placeholder text) and verify connectivity in **Settings & Security → Test Gemini API Connection**.
- **`Scanned or image-only PDF detected`:** MediSage intentionally disables OCR for predictable CPU performance; upload a text-searchable PDF or `.docx` version of the reference document.
- **First-run model download:** On very first launch, `all-MiniLM-L6-v2` (~90 MB) downloads once into the local HuggingFace cache (`~/.cache/huggingface`) and runs offline thereafter.

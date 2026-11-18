# MediSage System Architecture & Data Flow

## 1. High-Level Architecture Overview

MediSage is structured as a modular, CPU-friendly local Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, SQLite, ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`), and Google Gemini via the official `google-genai` SDK.

```text
+-----------------------------------------------------------------------------+
|                         Streamlit Presentation Layer (ui/)                   |
|  Home | Knowledge Library | Medical Assistant | History | Eval | Settings   |
+------------------------------------+----------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+----------------------------------+    +------------------------------------+
|   Ingestion Pipeline (ingestion) |    |     RAG & Chat Pipeline (rag/)     |
| 1. Security & Magic-Byte Checks  |    | 1. Emergency & Clinical Safety Gate|
| 2. Safe UUID Storage (uploads/)  |    | 2. Bounded Follow-Up Rewriter      |
| 3. PyMuPDF / python-docx Parser  |    | 3. Query Embedding (CPU Cached)    |
| 4. Header/Footer Cleaning        |    | 4. ChromaDB Top-K + Threshold      |
| 5. Heading/Table-Aware Chunker   |    | 5. Deduplication & Doc Diversity   |
| 6. all-MiniLM-L6-v2 CPU Embed    |    | 6. Grounded Prompt + Gemini API    |
+----------------+-----------------+    | 7. Citation Verification Engine    |
                 |                      +------------------+-----------------+
                 |                                         |
                 +-------------------+---------------------+
                                     |
         +---------------------------+---------------------------+
         v                                                       v
+----------------------------------+    +------------------------------------+
|   SQLite Metadata & Chat Store   |    |    ChromaDB Persistent Vectors     |
|  - documents (hash, status, err) |    |  - Collection: medisage_chunks     |
|  - conversations, messages       |    |  - Space: cosine (384-dim)         |
|  - feedback, app_settings        |    |  - Metadata: doc_id, filename,     |
|  - evaluation_runs               |    |    page_number, section, chunk_idx |
+----------------------------------+    +------------------------------------+
```

---

## 2. Module Responsibilities

| Layer | Module(s) | Responsibility |
| --- | --- | --- |
| **Entry Point** | `app.py` | Configures page layout, caches singleton services via `@st.cache_resource`, injects CSS, and dispatches sidebar routes. |
| **Configuration & Core** | `config/settings.py`, `core/config.py`, `core/logging_utils.py`, `core/llm_client.py` | Loads `.env` securely, validates runtime parameters, redacts API keys in logs, and implements the provider-configurable Gemini API adapter. |
| **Security & Cache** | `utils/validators.py`, `utils/helpers.py`, `utils/cache.py` | Enforces file extension/MIME/magic-byte validation, traversal-safe storage paths, error sanitization, and index-versioned deterministic caching. |
| **Database** | `database/schema.sql`, `database/db.py` | Parameterized SQLite CRUD for documents, conversations, messages, feedback, settings, and evaluation runs (`PRAGMA foreign_keys = ON`). |
| **Ingestion** | `ingestion/document_loader.py`, `ingestion/text_cleaner.py`, `ingestion/chunking.py`, `ingestion/embeddings.py`, `ingestion/ingestion_service.py` | Extracts page/section text and tables, rejects scanned/corrupt/encrypted files honestly, chunks along heading/paragraph boundaries, embeds locally on CPU, and prevents duplicate ingestion via SHA-256 hashes. |
| **Vector Store** | `vectorstore/chroma_manager.py` | Manages persistent ChromaDB collection (`hnsw:space=cosine`), idempotent `document_id` re-indexing, metadata filtering, and synchronized vector deletion. |
| **RAG & Safety** | `rag/safety.py`, `rag/query_rewriter.py`, `rag/retriever.py`, `rag/prompt_builder.py`, `rag/citations.py`, `rag/llm_service.py`, `rag/rag_pipeline.py` | Screens for emergencies and personal diagnosis/dosing requests, rewrites follow-ups, retrieves ranked/deduplicated passages, constructs strict context-only prompts, and validates `[Source: filename, page N]` citations. |
| **Chatbot** | `chatbot/chat_manager.py` | Orchestrates multi-turn conversation persistence, auto-titling, bounded history windowing, and user feedback capture. |
| **Evaluation** | `evaluation/eval_dataset.json`, `evaluation/metrics.py`, `evaluation/evaluator.py`, `evaluation/tuning.py` | Runs repeatable isolated retrieval/safety/citation benchmarks and comparative chunk/threshold tuning experiments. |
| **UI Views** | `ui/styles.py`, `ui/dashboard.py`, `ui/upload.py`, `ui/chat.py`, `ui/history.py`, `ui/evaluation.py`, `ui/settings_page.py` | Implements the light medical dashboard, knowledge library manager, multi-turn source-inspectable chat, history manager, evaluation dashboard, and validated settings. |

---

## 3. SQLite Database Schema (`database/schema.sql`)

- **`documents`**: Tracks `document_id` (PK), `stored_filename`, `original_filename`, `file_hash` (UNIQUE SHA-256), `file_type`, `file_size_bytes`, `page_count`, `chunk_count`, `processing_status` (`pending`/`processing`/`indexed`/`failed`), `error_details`, `source_origin`, `upload_timestamp`, `updated_timestamp`.
- **`conversations`**: Tracks `conversation_id` (PK), `title`, `created_at`, `updated_at`.
- **`messages`**: Tracks `message_id` (PK), `conversation_id` (FK -> `conversations` `ON DELETE CASCADE`), `role` (`user`/`assistant`), `content`, `rewritten_query`, `retrieval_status`, `citations_json` (compact citation metadata + <=450-char excerpts only; never full document text), `safety_flags_json`, `latency_ms`, `created_at`.
- **`feedback`**: Tracks `feedback_id` (PK), `message_id` (FK `ON DELETE CASCADE`), `conversation_id` (FK `ON DELETE CASCADE`), `rating` (`helpful`/`not_helpful`), `comment`, `created_at`.
- **`app_settings`**: Key-value JSON persistence for validated user settings (`llm_model_id`, `retrieval_top_k`, `similarity_threshold`, `chunk_size_tokens`, `chunk_overlap_tokens`, `response_style`, etc.).
- **`evaluation_runs`**: Historical evaluation benchmark snapshots (`run_id`, `run_label`, `config_json`, `metrics_json`, `created_at`).

---

## 4. Vector Metadata Schema (ChromaDB `medisage_chunks`)

Each indexed vector record stores:
- **`chunk_id`** (`str`): Unique ID formatted `<document_id>_p<page_number>_c<chunk_index>`
- **`document_id`** (`str`): Foreign reference matching SQLite `documents.document_id`
- **`filename`** (`str`): Original display filename (for human-readable citations)
- **`page_number`** (`int`): 1-indexed page number (or logical page index for DOCX/TXT/MD)
- **`section_title`** (`str`): Nearest preceding section or heading title
- **`chunk_index`** (`int`): Sequential chunk index within the document
- **`token_count`** (`int`): Estimated token length
- **`ingestion_timestamp`** (`str`): ISO-8601 UTC timestamp
- **`file_hash`** (`str`): SHA-256 hash of the source file

---

## 5. End-to-End RAG Data Flow

1. **Safety Gate (`rag/safety.py`)**: Scans the query for acute emergencies (e.g., chest pain, severe dyspnea, stroke signs, anaphylaxis, overdose, self-harm). If detected, returns an immediate urgent escalation response without calling retrieval or the LLM. Flags diagnosis/dosage/cessation requests with strict clinical boundary directives.
2. **Standalone Follow-Up Rewriting (`rag/query_rewriter.py`)**: If the query contains unresolved references (e.g., `"it"`, `"that medication"`) and recent turns exist, appends a concise topic anchor solely for retrieval while keeping the user's displayed message untouched.
3. **CPU Semantic Retrieval (`rag/retriever.py`)**: Encodes the query once via cached `all-MiniLM-L6-v2` (384-dim L2-normalized), queries ChromaDB (`top_k * 3` candidates), filters by `similarity_threshold`, removes `>78%` shingle-overlapping duplicate chunks, applies per-document diversity capping, and classifies outcome (`ok`, `low_confidence`, `no_evidence`).
4. **Grounded Prompt & Cloud LLM (`rag/prompt_builder.py`, `core/llm_client.py`)**: Sends only the retrieved excerpts, bounded recent history (max 4 turns), and safety instructions to Google Gemini via `google-genai` with bounded retries.
5. **Citation Verification (`rag/citations.py`)**: Parses every `[Source: filename, page N]` tag and verifies it against the actual retrieved `(filename, page_number)` pairs; any fabricated citation tags are stripped before display.
